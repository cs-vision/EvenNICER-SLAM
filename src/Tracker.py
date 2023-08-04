import copy
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera, 
                        get_samples_event,
                        get_rays_from_uv,
                        quad2rotation,get_tensor_from_camera_in_pytorch)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer

# PoseNet 
from pose_net import transNet,quatsNet

import random

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
            self.frame_reader, batch_size=1, shuffle=False, num_workers=1)
        
        self.slam = slam

        # wandb
        self.experiment = slam.experiment

        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, 
                                     experiment=self.experiment, # wandb
                                     device=self.device,
                                     stage = 'tracker')
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

        # RGBD available condition
        self.rgbd_every_frame = cfg['event']['rgbd_every_frame']

        # NOTE : PoseNet 
        self.transNet = transNet(self.cfg)
        self.quatsNet = quatsNet(self.cfg)
        self.fps = 120
        self.use_last = False

    def init_posenet_train(self, scale=0.6):
        self.optim_trans_init = torch.optim.Adam([dict(params=self.transNet.parameters(), lr = self.cam_lr*1*scale)])
        self.optim_quats_init = torch.optim.Adam([dict(params=self.quatsNet.parameters() , lr = self.cam_lr*0.2*scale)])

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
    
    def get_camera_pose(self, idx, pre_c2w, device):
        self.transNet.float()
        self.quatsNet.float()
        ret_cam_trans = self.transNet.forward(idx).unsqueeze(2)
        ret_cam_quad = self.quatsNet.forward(idx)
        cam_rots = quad2rotation(ret_cam_quad)
        estimated_cam_c2w = torch.concat([cam_rots, ret_cam_trans], dim=2)
        bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(torch.float32).to(device)
        bottoms = torch.stack([bottom]*idx.shape[0])
        estimated_cam_c2w_homogeneous = torch.cat([estimated_cam_c2w, bottoms], dim=1)
        estimated_c2w = torch.matmul(pre_c2w, estimated_cam_c2w_homogeneous)[:, :3, :]
        estimated_tensor = get_tensor_from_camera_in_pytorch(estimated_c2w)
        return estimated_c2w, estimated_tensor
    
    def get_event_rays(self, i, j, time, pre_c2w, H, W, fx, fy, cx, cy, device, fix=False):
        idx = time*self.fps/100
        c2w, _ = self.get_camera_pose(idx, pre_c2w, device)
        if fix:
            c2w = c2w.clone().detach()
        dirs = torch.stack(
                [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
        dirs = dirs.reshape(-1, 1, 3)
        rays_d = torch.sum(dirs * c2w[:, :3, :3], -1)
        rays_o = c2w[:, :3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    def optimize_after_sampling_pixels(self, idx, pre_c2w, gt_c2w, camera_tensor,
                                       evs_dict_xy, pos_evs_dict_xy, neg_evs_dict_xy, no_evs_pixels, 
                                       pre_gt_color, pre_gt_depth, fixed_pre_log_gray,
                                       gt_color, gt_depth, gt_event,
                                       last_evs_time,
                                       optim_quats_init, optim_trans_init):
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        
        optim_quats_init.zero_grad()
        optim_trans_init.zero_grad()

        pre_gt_depth = F.interpolate(pre_gt_depth.unsqueeze(0).unsqueeze(0), (H,W)).squeeze()
        gt_depth = F.interpolate(gt_depth.unsqueeze(0).unsqueeze(0), (H,W)).squeeze()
        pre_gt_color = F.interpolate(pre_gt_color.permute(2, 0, 1).unsqueeze(0), (H, W)).squeeze().permute(1, 2, 0)
        gt_color = F.interpolate(gt_color.permute(2, 0, 1).unsqueeze(0), (H, W)).squeeze().permute(1, 2, 0)
        pre_gt_gray = self.rgb_to_luma(pre_gt_color)

        # NOTE : negative sampling
        N_noevs = 200
        condition = (W//4 < no_evs_pixels[:, 0]) & (no_evs_pixels[:, 0] < W - W//4) & (H//4 < no_evs_pixels[:, 1]) & (no_evs_pixels[:, 1] < H - H//4)
        indices = np.where(condition)[0]
        selected_indices = np.random.choice(indices, size=N_noevs, replace=False)
        sampled_no_evs_xys = torch.tensor(no_evs_pixels[selected_indices]).to(device)
        noevs_i_tensor = sampled_no_evs_xys[:, 0].long() # W
        noevs_j_tensor = sampled_no_evs_xys[:, 1].long() # H

        c2w = get_camera_from_tensor(camera_tensor)
        noevs_ray_o, noevs_ray_d = get_rays_from_uv(noevs_i_tensor, noevs_j_tensor, c2w, H, W, fx, fy, cx, cy, device)
        noevs_gt_depth = gt_depth[noevs_j_tensor, noevs_i_tensor]
        noevs_ret = self.renderer.render_batch_ray(self.c, self.decoders, noevs_ray_d, noevs_ray_o, device, stage='color', gt_depth=noevs_gt_depth)
        _, _, noevs_color = noevs_ret
        noevs_gray = self.rgb_to_luma(noevs_color)
        noevs_pre_gray = self.rgb_to_luma(pre_gt_color)[noevs_j_tensor, noevs_i_tensor]
        loss_events = torch.abs(noevs_gray*255 - noevs_pre_gray*255).sum()
        print(loss_events.item())
    
        # NOTE : active sampling
        N_evs = 100
        xys_mtNevs = np.array(list(evs_dict_xy.keys()))
        condition = (W//4 < xys_mtNevs[:, 0]) & (xys_mtNevs[:, 0] < W - W//4) & (H//4 < xys_mtNevs[:, 1]) & (xys_mtNevs[:, 1] < H - H//4)
        indices = np.where(condition)[0]
        selected_indices = np.random.choice(indices, size=N_evs, replace=False)
        sampled_xys =  xys_mtNevs[selected_indices]
        sampled_xys = [tuple(row) for row in sampled_xys]
        num_pos_evs_at_xy = np.asarray([len(pos_evs_dict_xy.get(xy, [])) for xy in sampled_xys])
        num_neg_evs_at_xy = np.asarray([len(neg_evs_dict_xy.get(xy, [])) for xy in sampled_xys])
       
        sampled_tensor = torch.tensor(sampled_xys).view(N_evs, -1).to(device)
        i_tensor = sampled_tensor[:, 0].long()
        j_tensor = sampled_tensor[:, 1].long()
        # events_first_time = []
        # events_last_time = []
        # first_events_polarity = []
        # for xy in sampled_xys:
        #     events_time_stamps = []
        #     events_time_stamps.append([item[2] for item in evs_dict_xy[xy]])
        #     events_first_time.append(events_time_stamps[0][0])
        #     events_last_time.append(events_time_stamps[0][-1])

        #     events_polarities = []
        #     events_polarities.append([item[3] for item in evs_dict_xy[xy]])
        #     first_events_polarity.append(events_polarities[0][0])

        evs_at_xy = num_pos_evs_at_xy*0.1 - num_neg_evs_at_xy*0.1 
        evs_at_xy = torch.tensor(evs_at_xy).unsqueeze(1).to(device)

        #events_last_time = torch.tensor(events_last_time, dtype=torch.float32).reshape(N_evs, -1).to(device)
        events_last_time = last_evs_time[j_tensor, i_tensor].view(N_evs, -1)
        # NOTE : last_time(semi-asynchronous)
        ray_o, ray_d = self.get_event_rays(i_tensor, j_tensor, events_last_time, pre_c2w, H, W, fx, fy, cx, cy, device)
        # NOTE : c2w
        #ray_o, ray_d = get_rays_from_uv(i_tensor, j_tensor, c2w, H, W, fx, fy, cx, cy, device)

        evs_gt_depth = gt_depth[j_tensor, i_tensor]
        ret = self.renderer.render_batch_ray(self.c, self.decoders, ray_d, ray_o, device, stage='color', gt_depth=evs_gt_depth)
        _, _, rendered_color = ret
        rendered_gray = self.rgb_to_luma(rendered_color, esim=True)

        #pre_gray = pre_gt_gray[j_tensor, i_tensor]
        #pre_log_gray = self.lin_log(pre_gray*255)

        # TODO : fixed_pre_log_grayを先に求めておく
        fixed_pre_log_gray = fixed_pre_log_gray[j_tensor, i_tensor]
        expected_gray = self.inverse_lin_log(fixed_pre_log_gray + evs_at_xy)
        
        active_sampling  = True
        if active_sampling:
            loss_events += torch.abs(expected_gray - rendered_gray*255).sum()
        print(loss_events.item())

        loss_events = loss_events*0.025
        loss_events.backward(retain_graph = True)
        return loss_events.item()

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
        for init_i in range(50): 
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

        for idx, gt_color, gt_depth, gt_event, gt_c2w  in pbar:
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
                    #pre_c2w = self.estimate_c2w_list[idx-1].to(device)
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
            
            else:
                gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                if self.const_speed_assumption and idx-2 >= 0:
                    pre_c2w = pre_c2w.float()
                    delta = pre_c2w@self.estimate_c2w_list[idx-2].to(
                        device).float().inverse()
                    estimated_new_cam_c2w = delta@pre_c2w
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
                candidate_cam_tensor = None
                current_min_loss = 10000000000.
                current_min_loss_events = 10000000000.
                # NOTE : accumulate event 
                gt_event_integrate = torch.cat((gt_event_integrate, gt_event), dim = 0)
                if idx % 5 == 0:
                    events_in = gt_event_integrate.cpu().numpy()
                    pos_evs_dict_xy = {}
                    neg_evs_dict_xy = {}
                    evs_dict_xy = {}

                    # NOTE: get fixed_pre_log_gray
                    idx_time = torch.full((self.H, self.W), (idx-1)/self.fps*100).to(self.device)
                    first_evs_time = torch.zeros(self.H, self.W).to(self.device)
                    last_evs_time = torch.full((self.H, self.W), idx/self.fps*100).to(self.device)
                    first_evs_pol = torch.zeros(self.H, self.W).to(self.device)

                    for ev in events_in:
                        key_xy = (ev[0], ev[1])
                        if key_xy in evs_dict_xy.keys():
                            evs_dict_xy[key_xy].append(ev.tolist())
                        else:
                            evs_dict_xy[key_xy] = [ev.tolist()]
                        polarity = ev[3]
                        if polarity == 1.0 and key_xy in pos_evs_dict_xy.keys():
                            pos_evs_dict_xy[key_xy].append(ev.tolist())
                        elif polarity == -1.0 and key_xy in neg_evs_dict_xy.keys():
                            neg_evs_dict_xy[key_xy].append(ev.tolist())
                        elif polarity == 1.0:
                            pos_evs_dict_xy[key_xy] = [ev.tolist()]
                        elif polarity == -1.0:
                            neg_evs_dict_xy[key_xy] = [ev.tolist()]      
                        
                        x, y, t, p = int(ev[0]), int(ev[1]), ev[2], ev[3]
                        last_evs_time[y, x] = t*100
                        if first_evs_pol[y, x] == 0:
                            first_evs_time[y, x] = t*100
                            first_evs_pol[y, x] = p
                    
                    # NOTE: get fixed_pre_log_gray
                    pre_gt_gray = self.rgb_to_luma(pre_gt_color)
                    fixed_pre_log_gray = self.lin_log(pre_gt_gray*255)
                    if idx >= 10:
                        residual_events = (idx_time - pre_last_evs_time)*first_evs_pol / (first_evs_time - pre_last_evs_time)
                        residual_events = residual_events.unsqueeze(-1)
                        fixed_pre_log_gray -= residual_events*0.1

                    evs_dict_xy = dict((k, v) for k, v in evs_dict_xy.items() if len(v) > 1) 
                    pos_evs_dict_xy = dict((k, v) for k, v in pos_evs_dict_xy.items() if len(v) > 1) 
                    neg_evs_dict_xy = dict((k, v) for k, v in neg_evs_dict_xy.items() if len(v) > 1) 
                    x = np.arange(self.W)
                    y = np.arange(self.H)
                    no_evs_pixels = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
                    no_evs_set = set(map(tuple, no_evs_pixels))
                    evs_set = set(evs_dict_xy.keys())
                    no_evs_set -= evs_set
                    no_evs_pixels = np.array(list(no_evs_set))
                
                    for cam_iter in range(self.num_cam_iters):
                        estimated_correct_cam_trans = self.transNet.forward(idx_tensor).unsqueeze(0)
                        estimated_new_cam_quad = self.quatsNet.forward(idx_tensor).unsqueeze(0)
                        estimated_correct_cam_rots = quad2rotation(estimated_new_cam_quad)
                        estimated_correct_new_cam_c2w = torch.concat([estimated_correct_cam_rots, estimated_correct_cam_trans[...,None] ],dim=-1).squeeze()
                        bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(torch.float32).to(device)
                        estimated_correct_new_cam_c2w_homogeneous= torch.cat([estimated_correct_new_cam_c2w, bottom], dim=0)
                        compose_pose = torch.matmul(estimated_new_cam_c2w, estimated_correct_new_cam_c2w_homogeneous)[:3, :]
                        camera_tensor = get_tensor_from_camera_in_pytorch(compose_pose)
                    
                        loss_events = self.optimize_after_sampling_pixels(idx, pre_c2w, gt_c2w, camera_tensor,
                                                                      evs_dict_xy, pos_evs_dict_xy, neg_evs_dict_xy, no_evs_pixels,
                                                                      pre_gt_color, pre_gt_depth, fixed_pre_log_gray,
                                                                      gt_color, gt_depth, gt_event,
                                                                      last_evs_time,
                                                                      self.optim_quats_init, self.optim_trans_init)

                        loss_rgbd  = self.optimize_cam_rgbd(camera_tensor, gt_color, gt_depth, self.tracking_pixels,
                                                                self.optim_quats_init, self.optim_trans_init)
                        
                        self.optim_quats_init.step()
                        self.optim_trans_init.step()
                        self.optim_quats_init.zero_grad()
                        self.optim_trans_init.zero_grad()
                        
                        print("Event Loss", loss_events)
                        print(f"RGBD loss:{loss_rgbd}\n")

                        c2w = get_camera_from_tensor(camera_tensor)
                        loss_camera_tensor = torch.abs(gt_camera_tensor.to(device)-camera_tensor).mean().item()

                        fixed_camera_tensor = camera_tensor.clone().detach()
                        if camera_tensor[0] < 0:
                            fixed_camera_tensor[:4] *= -1
                        fixed_camera_error = torch.abs(gt_camera_tensor.to(device)- fixed_camera_tensor).mean().item()
                
                        print(f"camera tensor error: {loss_camera_tensor}\n")
                        print(f"camera tensor: {camera_tensor}\n")

    
                        if self.verbose:
                            if cam_iter == self.num_cam_iters-1:
                                    # wandb logging
                                    dict_log = {
                                        'Event loss' : loss_events,
                                        #'Event loss improvement': initial_loss_event - loss_event,
                                        'Camera error': loss_camera_tensor,
                                        'Fixed camera error' : fixed_camera_error,
                                        #'Camera error improvement': initial_loss_camera_tensor - loss_camera_tensor,
                                        'Frame': idx,
                                        'first element of gt quaternion' : gt_camera_tensor[0],
                                        'first element of quaternion' : camera_tensor[0],
                                        'first element of translation' : camera_tensor[4],
                                        'first element of gt translation' : gt_camera_tensor[4]
                                    }

                                    self.experiment.log(dict_log)

                        # if loss_events < current_min_loss_events:
                        #     current_min_loss_events = loss_events
                        #     candidate_cam_tensor = camera_tensor.clone().detach()
                        #     if not self.use_last:
                        #         candidate_transNet_para = self.transNet.state_dict()
                        #         candidate_quatsNet_para = self.quatsNet.state_dict()

                        if loss_rgbd < current_min_loss:
                            current_min_loss = loss_rgbd
                            candidate_cam_tensor = camera_tensor.clone().detach()
                            if not self.use_last:
                                candidate_transNet_para = self.transNet.state_dict()
                                candidate_quatsNet_para = self.quatsNet.state_dict()

                    bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                            [1, 4])).type(torch.float32).to(self.device)
                    print("candidate_cam_tensor:", candidate_cam_tensor)
                    if self.use_last:
                        c2w = get_camera_from_tensor(
                            camera_tensor.clone().detach()
                        )
                        c2w = torch.cat([c2w, bottom], dim=0)
                    else:
                        c2w = get_camera_from_tensor(
                            candidate_cam_tensor.clone().detach()
                        )
                        c2w = torch.cat([c2w, bottom], dim=0)   

                        self.transNet.load_state_dict(candidate_transNet_para)
                        self.quatsNet.load_state_dict(candidate_quatsNet_para)

                        del candidate_transNet_para
                        del candidate_quatsNet_para       

            
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
                # NOTE : insert dummy event
                gt_event_integrate = torch.tensor([0, 0, 0, 0]).unsqueeze(0).to(device)
                if idx >= 5:
                    pre_last_evs_time = last_evs_time