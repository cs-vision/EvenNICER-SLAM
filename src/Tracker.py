import copy
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

# NOTE : 
import torch.nn.functional as F

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera, 
                        get_samples_event,
                        get_rays_from_uv,
                        quad2rotation,get_tensor_from_camera_in_pytorch)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer

from pose_net import transNet,quatsNet

#wandb
# TODO : wandb should be in Visualizer class
import wandb

class Tracker(object):
    def __init__(self, cfg, args, slam
                 ):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']

        self.idx = slam.idx
        self.nice = slam.nice
        self.bound = slam.bound
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.shared_c = slam.shared_c
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list

        self.cam_lr = cfg['tracking']['lr']
        self.device = cfg['tracking']['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.seperate_LR = cfg['tracking']['seperate_LR']
        self.w_color_loss = cfg['tracking']['w_color_loss']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.handle_dynamic = cfg['tracking']['handle_dynamic']
        self.use_color_in_tracking = cfg['tracking']['use_color_in_tracking']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(
            self.frame_reader, batch_size=1, shuffle=False, num_workers=0) # to avoid "RuntimeError: unable to mmap 128 bytes from file </torch_6409_1637078694_63344>: Cannot allocate memory (12)"
        
        self.slam = slam

        # wandb
        self.experiment = slam.experiment

        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, 
                                     experiment = self.experiment, # wabdb
                                     device=self.device,
                                     stage = 'tracker')
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

        # RGBD available condition
        self.rgbd_every_frame = cfg['event']['rgbd_every_frame']

        # NOTE : PoseNet 
        self.transNet = transNet(self.cfg)
        self.quatsNet = quatsNet(self.cfg)

    def init_posenet_train(self):
        self.optim_trans_init = torch.optim.Adam([dict(params=self.transNet.parameters(), lr = self.cam_lr*1)])
        self.optim_quats_init = torch.optim.Adam([dict(params=self.quatsNet.parameters() , lr = self.cam_lr*0.2)])

    # TODO : import as  e-sim class?
    def rgb_to_luma(self, rgb, esim=True):
        """
        Input:
        :rgb torch.Tensor (N_evs, 3)

        Output:
        :luma torch.Tensor (N_evs, 1)
        :esim Use luma-conversion-coefficients from esim, else from v2e-paper.
        "ITU-R recommendation BT.709 digital video (linear, non-gamma-corrected) color space conversion"
        see https://gist.github.com/yohhoy/dafa5a47dade85d8b40625261af3776a
        or https://mymusing.co/bt-709-yuv-to-rgb-conversion-color/ for numbers
        """
        if esim:
            #  https://github.com/uzh-rpg/rpg_esim/blob/4cf0b8952e9f58f674c3098f1b027a4b6db53427/event_camera_simulator/imp/imp_opengl_renderer/src/opengl_renderer.cpp#L319-L321
            #  image format esim: https://github.com/uzh-rpg/rpg_esim/blob/4cf0b8952e9f58f674c3098f1b027a4b6db53427/event_camera_simulator/esim_visualization/src/ros_utils.cpp#L29-L36
            #  color conv factorsr rgb->gray: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
            r = 0.299
            g = 0.587
            b = 0.114
        else:
            r = 0.2126
            g = 0.7152
            b = 0.0722

        factors = torch.Tensor([r, g, b]).to(self.device)# .cpu()  # (3)
        luma = torch.sum(rgb * factors[None, :], axis=-1)  # (N_evs, 3) * (1, 3) => (N_evs)
        return luma[..., None]  # (N_evs, 1)
    
    def lin_log(self, color, linlog_thres=20):
        """
        Input: 
        :color torch.Tensor of (N_rand_events, 1 or 3). 1 if use_luma, else 3 (rgb).
            We pass rgb here, if we want to treat r,g,b separately in the loss (each pixel must obey event constraint).
        """
        # Compute the required slope for linear region (below luma_thres)
        # we need natural log (v2e writes ln and "it comes from exponential relation")
        lin_slope = np.log(linlog_thres) / linlog_thres
        # Peform linear-map for smaller thres, and log-mapping for above thresh
        lin_log_rgb = torch.where(color < linlog_thres, lin_slope * color, torch.log(color))
        return lin_log_rgb
    
    def inverse_lin_log(self, lin_log_rgb, linlog_thres=20):
        lin_slope = np.log(linlog_thres) / linlog_thres
        # Perform inverse linear mapping for values below linlog_thres
        inverse_lin_log_rgb = torch.where(
            lin_log_rgb < lin_slope * linlog_thres,
            lin_log_rgb / lin_slope,
            torch.exp(lin_log_rgb)
        )
        return inverse_lin_log_rgb
    
    def asynchronous_event_sampling_optimize(self, idx_tensor, camera_tensor,
                                             pre_gt_color, gt_depth, gt_event, # NOTE : gt_event : txt
                                             optim_quats_init, optim_trans_init,
                                             estimated_new_cam_c2w,
                                             gt_color): # NOTE : to compare 
        device = self.device
        H, W, fx, fy, cx, cy = self.H// 4, self.W//4, self.fx//4, self.fy//4, self.cx//4, self.cy//4
        current_min_loss_event = 10000000000.
        candidate_cam_tensor = camera_tensor.clone().detach()

        gt_depth = F.interpolate(gt_depth.unsqueeze(0).unsqueeze(0), (H,W)).squeeze()
        pre_gt_color = F.interpolate(pre_gt_color.permute(2, 0, 1).unsqueeze(0), (H, W)).squeeze().permute(1, 2, 0)
        pre_gt_gray = self.rgb_to_luma(pre_gt_color, esim=True)
        gt_gray = self.rgb_to_luma()
        gt_event_array = gt_event.cpu().numpy()

        events_array = np.zeros((self.H, self.W, 2))
        # NOTE : how many pixels which achive 10 events
        count = 0
        loss_event = torch.tensor(0.0).unsqueeze(0).unsqueeze(0).to(device)

        for event in gt_event_array:
            i = int(event[0])
            j = int(event[1])
            event_value = float(event[2])
            events_array[j][i][1] += event[3]
            #max_events = np.max(np.abs(events_array))
            if (events_array[j][i][1] == 10 or events_array[j][i][1] == -10) and events_array[j][i][0] == 0 :
                i_tensor= torch.tensor(i).to(device)
                j_tensor = torch.tensor(j).to(device)
                # NOTE : interbal is set to be 120
                time = torch.tensor(event_value*120, dtype=torch.float32, device=device).unsqueeze(0)
                self.transNet.float()
                self.quatsNet.float()
                accurate_cam_trans = self.transNet.forward(time).unsqueeze(0)
                accurate_cam_quad = self.quatsNet.forward(time).unsqueeze(0)
                accurate_cam_rots = quad2rotation(accurate_cam_quad)
                accurate_cam_c2w = torch.concat([accurate_cam_rots, accurate_cam_trans[...,None] ],dim=-1).squeeze()
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(torch.float32).to(device)
                accurate_cam_c2w_homogeneous= torch.cat([accurate_cam_c2w, bottom], dim=0)
                accurate_compose_pose = torch.matmul(estimated_new_cam_c2w.float(), accurate_cam_c2w_homogeneous)[:3, :]
                accurate_camera_tensor = get_tensor_from_camera_in_pytorch(accurate_compose_pose) 
                c2w = get_camera_from_tensor(accurate_camera_tensor)

                ray_o, ray_d = get_rays_from_uv(i_tensor, j_tensor, c2w, self.H,self.W, self.fx, self.fy, self.cx, self.cy, self.device)
                ret = self.renderer.render_batch_ray(self.c, self.decoders, ray_d, ray_o,  device, stage='color',  gt_depth=gt_depth[j, i])
                _, _, rendered_color = ret
                rendered_gray = torch.clamp(self.rgb_to_luma(rendered_color, esim=True), 0, 1)
                #print(f"rendered_gray:{rendered_gray*255}\n")
                pre_gt_gray_pixel = pre_gt_gray[j, i]
                pre_loggray_pixel = self.lin_log(pre_gt_gray_pixel*255, linlog_thres=20)
                #print(f"pre_loggray_pixel:{pre_loggray_pixel}\n")

                # NOTE : gt_grayと比較してみる
                

                C_thres = 0.1
                loggray_add_events = pre_loggray_pixel + events_array[j][i][1]*C_thres
                #print(f"loggray_add_events:{loggray_add_events}\n") # 0ばっかり
                inverse_loggray_pixel = torch.clamp(self.inverse_lin_log(loggray_add_events), 0, 255)
                #print(f"inverse_loggray_pixel:{inverse_loggray_pixel}\n") #60~150あたり
                loss_event += torch.abs(inverse_loggray_pixel - rendered_gray*255).sum()
                events_array[j][i][0] = 1
                events_array[j][i][1] = 0
                count += 1

            if count == 50:
                # TODO : non_event_lossを定義
                loss_event.backward()
                loss_event_item = loss_event.item()
                print("loss_event backward\n")
                print(f"loss_event_loss : {loss_event_item}")
                if loss_event_item < current_min_loss_event:
                    current_min_loss_event = loss_event_item
                    candidate_cam_tensor = camera_tensor.clone().detach()

                optim_quats_init.step()
                optim_trans_init.step()
                optim_quats_init.zero_grad()
                optim_trans_init.zero_grad()

                estimated_correct_cam_trans = self.transNet.forward(idx_tensor).unsqueeze(0)
                estimated_new_cam_quad = self.quatsNet.forward(idx_tensor).unsqueeze(0)
                estimated_correct_cam_rots = quad2rotation(estimated_new_cam_quad)
                estimated_correct_new_cam_c2w = torch.concat([estimated_correct_cam_rots, estimated_correct_cam_trans[...,None] ],dim=-1).squeeze()
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(torch.float32).to(device)
                estimated_correct_new_cam_c2w_homogeneous= torch.cat([estimated_correct_new_cam_c2w, bottom], dim=0)
                compose_pose = torch.matmul(estimated_new_cam_c2w, estimated_correct_new_cam_c2w_homogeneous)[:3, :]
                camera_tensor = get_tensor_from_camera_in_pytorch(compose_pose) 

                count = 0
                loss_event = torch.tensor(0.0).unsqueeze(0).unsqueeze(0).to(device)

        optim_quats_init.zero_grad()
        optim_trans_init.zero_grad()

        return candidate_cam_tensor
    
    def optimize_cam_event(self, camera_tensor, 
                              gt_color, gt_depth, gt_event,
                              batch_size, 
                              optim_quats_init, optim_trans_init, # PoseNet
                              pre_gt_color,
                              pre_gt_depth):
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optim_quats_init.zero_grad()
        optim_trans_init.zero_grad()
        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H
        batch_rays_o_event, batch_rays_d_event, batch_gt_depth, batch_pre_gt_color, batch_gt_event = get_samples_event(
        Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, pre_gt_color, gt_event, device)
        ret_event = self.renderer.render_batch_ray(
        self.c, self.decoders, batch_rays_d_event, batch_rays_o_event,  device, stage='color',  gt_depth=batch_gt_depth)
        _, event_uncertainty, rendered_color = ret_event

        # 1. pre_gt_color(batch), rendered_color are converted to gray scale and log scale
        rendered_gray = self.rgb_to_luma(rendered_color, esim=True)
        pre_gt_gray = self.rgb_to_luma(batch_pre_gt_color, esim=True)
        pre_gt_loggray = self.lin_log(pre_gt_gray*255, linlog_thres=20)

        # 2. add events to pre_gt_gray(batch)
        gt_pos = torch.unsqueeze(batch_gt_event[:, 0] , dim = 1)
        gt_neg = torch.unsqueeze(batch_gt_event[:, 1] , dim = 1)
        C_thres = 0.1
        gt_loggray_events = pre_gt_loggray -  gt_pos * C_thres + gt_neg * C_thres
        gt_inverse_loggray = self.inverse_lin_log(gt_loggray_events)

        # 3. define event_loss
        loss_event = torch.abs(gt_inverse_loggray - rendered_gray*255).sum()
        balancer = self.cfg['event']['balancer'] # coefficient to balance event loss and rgbd loss
        loss_event = loss_event * balancer
        
        optim_quats_init.step()
        optim_trans_init.step()
        optim_quats_init.zero_grad()
        optim_trans_init.zero_grad()

        loss_event_item = loss_event.item()

        return loss_event_item
    
    def optimize_cam_rgbd(self, camera_tensor, gt_color, gt_depth, batch_size, optim_quats_init, optim_trans_init):
        """
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            camera_tensor (tensor): camera tensor.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optim_quats_init.zero_grad()
        optim_trans_init.zero_grad()
        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, device)
        if self.nice:
            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(device)-det_rays_o)/det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

        ret = self.renderer.render_batch_ray(
            self.c, self.decoders, batch_rays_d, batch_rays_o,  device, stage='color',  gt_depth=batch_gt_depth)
        depth, uncertainty, color = ret
        uncertainty = uncertainty.detach()
        if self.handle_dynamic:
            tmp = torch.abs(batch_gt_depth-depth)/torch.sqrt(uncertainty+1e-10)
            mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0)
        else:
            mask = batch_gt_depth > 0

        loss_rgbd = (torch.abs(batch_gt_depth-depth) /
                torch.sqrt(uncertainty+1e-10))[mask].sum()

        if self.use_color_in_tracking:
            color_loss = torch.abs(
                batch_gt_color - color)[mask].sum()
            loss_rgbd += self.w_color_loss*color_loss

        loss_rgbd.backward()   

        optim_quats_init.step()
        optim_trans_init.step()
        optim_quats_init.zero_grad()
        optim_trans_init.zero_grad()

        return loss_rgbd.item()

    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.

        """
        # shared_decoders, shared_c are shared in "tracking" and "mapping"
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')
            self.decoders = copy.deepcopy(self.shared_decoders).to(self.device) 
            for key, val in self.shared_c.items():
                val = val.clone().to(self.device)
                self.c[key] = val
            self.prev_mapping_idx = self.mapping_idx[0].clone()
    
    def run(self):
        device = self.device

        # NOTE : pretrain PoseNet
        idx_total = torch.arange(start=0, end=self.n_img).to(device).reshape(self.n_img, -1)
        self.init_posenet_train()
        for init_i in range(50): # TODO : 0→50 にする
            estimated_new_cam_trans = self.transNet.forward(idx_total)
            estimated_new_cam_quad = self.quatsNet.forward(idx_total)
            loss_trans = torch.abs(estimated_new_cam_trans - torch.tensor([0, 0, 0]).to(device)).mean()
            loss_trans.backward()

            loss_quad = torch.abs(estimated_new_cam_quad - torch.tensor([1, 0, 0, 0]).to(device)).mean()
            loss_quad.backward()

            self.optim_trans_init.step()
            self.optim_quats_init.step()
            self.optim_trans_init.zero_grad()
            self.optim_quats_init.zero_grad()

        self.c = {}
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)

        for idx, gt_color, gt_depth, gt_event, gt_c2w in pbar:
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")

            idx = idx[0]
            gt_depth = gt_depth[0]
            gt_color = gt_color[0]
            gt_event = gt_event[0]
            gt_c2w = gt_c2w[0]

            if self.sync_method == 'strict':
                # strictly mapping and then tracking
                # initiate mapping every self.every_frame frames
                if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                    while self.mapping_idx[0] != idx-1:
                        time.sleep(0.1)
                # NOTE : 基準となるposeを5個前にする
                    #pre_c2w = self.estimate_c2w_list[idx-1].to(device) 
                    gt_event_integrate = gt_event
                elif idx > 0 and idx % self.every_frame != 1:
                    gt_event_integrate = torch.cat((gt_event_integrate, gt_event), dim = 0)
            
            elif self.sync_method == 'loose':
                # mapping idx can be later than tracking idx is within the bound of
                # [-self.every_frame-self.every_frame//2, -self.every_frame+self.every_frame//2]
                while self.mapping_idx[0] < idx-self.every_frame-self.every_frame//2:
                    time.sleep(0.1)
            elif self.sync_method == 'free':
                # pure parallel, if mesh/vis happens may cause inbalance
                pass

            self.update_para_from_mapping()

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)

            if idx == 0 or self.gt_camera:
                c2w = gt_c2w
                if not self.no_vis_on_first_frame:
                    self.visualizer.vis(
                        idx, 0, gt_depth, gt_color, c2w, self.c, self.decoders)
                # added (6/30)
                pre_c2w = gt_c2w

            else:
                gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                if self.const_speed_assumption and idx-2 >= 0:
                    pre_c2w = pre_c2w.float()
                    delta = pre_c2w@self.estimate_c2w_list[idx-2].to(
                        device).float().inverse()
                    estimated_new_cam_c2w = delta@pre_c2w
                # NOTE : PoseNetならこっち
                else:
                    estimated_new_cam_c2w = pre_c2w

                # Initialization
                idx_tensor = torch.tensor(idx).unsqueeze(0).to(device)
                estimated_correct_cam_trans = self.transNet.forward(idx_tensor).unsqueeze(0)
                estimated_new_cam_quad = self.quatsNet.forward(idx_tensor).unsqueeze(0)
                estimated_correct_cam_rots = quad2rotation(estimated_new_cam_quad)
                estimated_correct_new_cam_c2w = torch.concat([estimated_correct_cam_rots, estimated_correct_cam_trans[...,None] ],dim=-1).squeeze()
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(torch.float32).to(device)
                estimated_correct_new_cam_c2w_homogeneous = torch.cat([estimated_correct_new_cam_c2w, bottom], dim=0)
                compose_pose = torch.matmul(estimated_new_cam_c2w, estimated_correct_new_cam_c2w_homogeneous)[:3, :]
                camera_tensor = get_tensor_from_camera_in_pytorch(compose_pose)
        
                initial_loss_camera_tensor = torch.abs(
                    gt_camera_tensor.to(device)-camera_tensor).mean().item()
                #candidate_cam_tensor = None
                candidate_cam_tensor = camera_tensor.clone().detach()
                current_min_loss = 10000000000.

                # candidate_cam_tensor  = self.asynchronous_event_sampling_optimize(idx_tensor, camera_tensor,  
                #                                                             pre_gt_color, gt_depth, gt_event,
                #                                                             self.optim_quats_init, self.optim_trans_init,
                #                                                             estimated_new_cam_c2w)
                
                # print(f"loss_camera_tensor: {loss_camera_tensor}\n")
                
                rgbd_available = (idx % self.rgbd_every_frame == 0)
                #rgbd_available = True
                loss_event_backward = False
                if rgbd_available:
                    if loss_event_backward:

                        loss_camera_tensor = torch.abs(gt_camera_tensor.to(device)- candidate_cam_tensor).mean().item()
                        print(f"loss_camera_tensor before loss_event backward: {loss_camera_tensor}\n")

                        candidate_cam_tensor  = self.asynchronous_event_sampling_optimize(idx_tensor, camera_tensor,  
                                                                            pre_gt_color, gt_depth, gt_event_integrate,
                                                                            self.optim_quats_init, self.optim_trans_init,
                                                                            estimated_new_cam_c2w,
                                                                            gt_color)       

                        loss_camera_tensor = torch.abs(gt_camera_tensor.to(device)- candidate_cam_tensor).mean().item()
                        print(f"loss_camera_tensor after loss_event backward: {loss_camera_tensor}\n")

                    for cam_iter in range(self.num_cam_iters):
                        self.visualizer.vis(
                            idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)

                        estimated_correct_cam_trans = self.transNet.forward(idx_tensor).unsqueeze(0)
                        estimated_new_cam_quad = self.quatsNet.forward(idx_tensor).unsqueeze(0)
                        estimated_correct_cam_rots = quad2rotation(estimated_new_cam_quad)
                        estimated_correct_new_cam_c2w = torch.concat([estimated_correct_cam_rots, estimated_correct_cam_trans[...,None] ],dim=-1).squeeze()
                        bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(torch.float32).to(device)
                        estimated_correct_new_cam_c2w_homogeneous= torch.cat([estimated_correct_new_cam_c2w, bottom], dim=0)
                        compose_pose = torch.matmul(estimated_new_cam_c2w, estimated_correct_new_cam_c2w_homogeneous)[:3, :]
                        camera_tensor = get_tensor_from_camera_in_pytorch(compose_pose)
                        loss_rgbd  = self.optimize_cam_rgbd(camera_tensor, gt_color, gt_depth, self.tracking_pixels,
                                                            self.optim_quats_init, self.optim_trans_init)
                        
                        print("RGBD loss: ", loss_rgbd)
                        #print("Event loss : ", loss_event)
                        print("\n")
                    
                        loss_camera_tensor = torch.abs(gt_camera_tensor.to(device)-camera_tensor).mean().item()
                        print("camera tensor error: ", loss_camera_tensor)
                        print("\n")

                        if cam_iter == 0:
                            initial_loss_rgbd = loss_rgbd
                            #initial_loss_event = loss_event
                    
                        if self.verbose:
                            if cam_iter == self.num_cam_iters-1:
                                # if rgbd_available:
                                print(
                                    f'RGBD loss: {initial_loss_rgbd:.2f}->{loss_rgbd:.2f} ' +
                                    #f'event loss: {initial_loss_event:.2f}->{loss_event:.2f} ' + 
                                    f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')
                                
                                # wandb logging
                                dict_log = {
                                    'RGBD loss' : loss_rgbd,
                                    'RGBD loss improvement': initial_loss_rgbd - loss_rgbd,
                                    #'Event loss' : loss_event,
                                    #'Event loss improvement': initial_loss_event - loss_event,
                                    'Camera error': loss_camera_tensor,
                                    'Camera error improvement': initial_loss_camera_tensor - loss_camera_tensor,
                                    'Frame': idx
                                }

                                self.experiment.log(dict_log)

                    # use event loss as criteria because it is always available
                    # NOTE : rgbd_availableのときは，loss_rgbd使った方がいいかも？？
                        if loss_rgbd < current_min_loss:
                            current_min_loss = loss_rgbd
                            candidate_cam_tensor = camera_tensor.clone().detach()

                    bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                        [1, 4])).type(torch.float32).to(self.device)
                    print("candidate_cam_tensor:", candidate_cam_tensor)
                    c2w = get_camera_from_tensor(
                    candidate_cam_tensor.clone().detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                    self.estimate_c2w_list[idx] = c2w.clone().cpu()
                    self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
                    pre_c2w = c2w.clone()
            self.idx[0] = idx
            if self.low_gpu_mem:
                torch.cuda.empty_cache()
            
            # store previous gt color
            if idx % self.every_frame == 0:
                pre_gt_depth = gt_depth
                pre_gt_color = gt_color