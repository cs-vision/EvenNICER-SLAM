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

# # PoseNet 
# from pose_net import transNet,quatsNet

# PoseGrid
from src.pose_grid import PoseGrid_decoder
import transform3d.euler as txe
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


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

        # # NOTE : PoseNet 
        # self.transNet = transNet(self.cfg)
        # self.quatsNet = quatsNet(self.cfg)
        # self.fps = 120
        # self.use_last = False

        # TODO : PoseGrid (refer to self.c init) and decoder init
        self.pose_decoder = PoseGrid_decoder(self.cfg)
        self.load_pretrained(self.cfg)
        self.posegrid_init(self.cfg)
        self.hidden_dim = 32 # length of pose encoding for each DoF is 32, size of B is (32, 32)

        # decide if tracker fits directly to gt pose
        self.fit_gt = False

    # def init_posenet_train(self, scale=1.0):
    #     self.optim_trans_init = torch.optim.Adam([dict(params=self.transNet.parameters(), lr = self.cam_lr*1*scale)])
    #     self.optim_quats_init = torch.optim.Adam([dict(params=self.quatsNet.parameters() , lr = self.cam_lr*0.2*scale)])

    def load_pretrained(self, cfg):
        self.pose_decoder.load_state_dict(torch.load(cfg['PoseGrid']['pretrained_decoder']))
        self.B_stack = torch.from_numpy(np.load(cfg['PoseGrid']['pretrained_B']))
        self.B_theta, self.B_phi, self.B_gamma = self.B_stack[0], self.B_stack[1], self.B_stack[2]
        self.B_x = self.B_stack[3]
        if cfg['PoseGrid']['only_Box']:
            self.B_y = self.B_x.clone()
            self.B_z = self.B_x.clone()
        else:
            self.B_y = self.B_stack[4]
            self.B_z = self.B_stack[5]
        self.zeropose_encoding = torch.from_numpy(np.load(cfg['PoseGrid']['startpose_encoding']))
      
    def posegrid_init(self, cfg):
        self.encoding_interval = cfg['PoseGrid']['encoding_interval']
        self.encoding_dim = 199 # 32*6 + 7, 32 each DoF, 7 for translation & quaternion prime
        self.n_encoding = int(np.ceil(self.n_img / self.encoding_interval) + 1) # make sure the first and the last frame has its encoding
        self.posegrid = torch.cat([self.zeropose_encoding, torch.zeros(7)]).view(self.encoding_dim, 1).repeat(1, self.n_encoding)

      

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
    
    # NOTE : need to be changed from posenet to posegrid
    # def get_camera_pose(self, idx, pre_c2w, device):
    #     self.transNet.float()
    #     self.quatsNet.float()
    #     ret_cam_trans = self.transNet.forward(idx).unsqueeze(2)
    #     ret_cam_quad = self.quatsNet.forward(idx)
    #     cam_rots = quad2rotation(ret_cam_quad)
    #     estimated_cam_c2w = torch.concat([cam_rots, ret_cam_trans], dim=2)
    #     bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(torch.float32).to(device)
    #     bottoms = torch.stack([bottom]*idx.shape[0])
    #     estimated_cam_c2w_homogeneous = torch.cat([estimated_cam_c2w, bottoms], dim=1)
    #     estimated_c2w = torch.matmul(pre_c2w, estimated_cam_c2w_homogeneous)[:, :3, :]
    #     #estimated_c2w_new = torch.matmul(estimated_cam_c2w_homogeneous, pre_c2w)[:, :3, :]
    #     estimated_tensor = get_tensor_from_camera_in_pytorch(estimated_c2w)
    #     return estimated_c2w, estimated_tensor
    
    def get_event_rays(self, idx_tensor, i, j, time, pre_c2w, H, W, fx, fy, cx, cy, device, fix=False):
        idx = time*self.fps
        c2w, _ = self.get_camera_pose(idx, pre_c2w, device)
        if fix:
            c2w = c2w.clone().detach()
        dirs = torch.stack(
                [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
        dirs = dirs.reshape(-1, 1, 3)
        rays_d = torch.sum(dirs * c2w[:, :3, :3], -1)
        rays_o = c2w[:, :3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    # NOTE : need to be changed from posenet to posegrid
    def optimize_after_sampling_pixels(self, idx, pre_c2w, gt_c2w, camera_tensor,
                                       evs_dict_xy, pre_evs_dict_xy, gt_event_images, no_evs_pixels, 
                                       pre_gt_color, pre_gt_depth,
                                       gt_color, gt_depth, gt_event,
                                       optim_quats_init, optim_trans_init):
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        
        optim_quats_init.zero_grad()
        optim_trans_init.zero_grad()

        # NOTE : negative sampling
        start = time.time()
        N_noevs = 50
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
        end = time.time()
        print("negative sampling = {:0.5f}".format(end-start))
    
        # NOTE : active sampling
        start = time.time()
        N_evs = 150
        xys_mtNevs = np.array(list(evs_dict_xy.keys()))
        condition = (W//4 < xys_mtNevs[:, 0]) & (xys_mtNevs[:, 0] < W - W//4) & (H//4 < xys_mtNevs[:, 1]) & (xys_mtNevs[:, 1] < H - H//4)
        indices = np.where(condition)[0]
        if len(indices) > 0:
            selected_indices = np.random.choice(indices, size=N_evs, replace=True)
        else:
            condition = (W//16 < xys_mtNevs[:, 0]) & (xys_mtNevs[:, 0] < W - W//16) & (H//16 < xys_mtNevs[:, 1]) & (xys_mtNevs[:, 1] < H - H//16)
            indices = np.where(condition)[0]
            selected_indices = np.random.choice(indices, size=N_evs, replace=True)
        sampled_xys =  xys_mtNevs[selected_indices]
        sampled_xys = [tuple(row) for row in sampled_xys]
        # num_pos_evs_at_xy = np.asarray([len(pos_evs_dict_xy.get(xy, [])) for xy in sampled_xys])
        # num_neg_evs_at_xy = np.asarray([len(neg_evs_dict_xy.get(xy, [])) for xy in sampled_xys])
       
        sampled_tensor = torch.tensor(sampled_xys).view(N_evs, -1).to(device)
        i_tensor = sampled_tensor[:, 0].long()
        j_tensor = sampled_tensor[:, 1].long()

        events_first_time = []
        events_last_time = []
        first_event_polarity = []
        events_pre_last_time = []
        for xy in sampled_xys:
            events_time_stamps = []
            events_polarities = []
            events_time_stamps.append([item[2] for item in evs_dict_xy[xy]])
            events_polarities.append([item[3] for item in evs_dict_xy[xy]])
            events_last_time.append(events_time_stamps[0][-1])
            events_first_time.append(events_time_stamps[0][0])
            first_event_polarity.append(events_polarities[0][0])
            if xy in pre_evs_dict_xy.keys():
                events_pre_time_stamps = []
                events_pre_time_stamps.append([item[2] for item in pre_evs_dict_xy[xy]])
                events_pre_last_time.append(events_pre_time_stamps[0][-1])
            elif idx >= 10:
                events_pre_last_time.append((idx-10)/self.fps)

        events_last_time = torch.tensor(events_last_time, dtype=torch.float32).reshape(N_evs, -1).to(device)
        events_first_time = torch.tensor(events_first_time, dtype=torch.float32).reshape(N_evs, -1).to(device)
        first_event_polarity = torch.tensor(first_event_polarity, dtype=torch.float32).reshape(N_evs, -1).to(device)
        events_pre_last_time = torch.tensor(events_pre_last_time, dtype=torch.float32).reshape(N_evs, -1).to(device)
        # evs_at_xy = num_pos_evs_at_xy*0.1 - num_neg_evs_at_xy*0.1 
        # evs_at_xy = torch.tensor(evs_at_xy).unsqueeze(1).to(device)

        evs_at_xy = gt_event_images.squeeze()[j_tensor, i_tensor]
    
        # NOTE : last_time
        last_time = False
        idx_tensor = torch.tensor(idx).unsqueeze(0).to(device)
        ray_o, ray_d = self.get_event_rays(idx_tensor.unsqueeze(0), i_tensor, j_tensor, events_last_time, pre_c2w, H, W, fx, fy, cx, cy, device)

        evs_gt_depth = gt_depth[j_tensor, i_tensor]
        ret = self.renderer.render_batch_ray(self.c, self.decoders, ray_d, ray_o, device, stage='color', gt_depth=evs_gt_depth)
        _, _, rendered_color = ret
        rendered_gray = self.rgb_to_luma(rendered_color, esim=True)

        if last_time == False:
            ray_o, ray_d = get_rays_from_uv(i_tensor, j_tensor, c2w, H, W, fx, fy, cx, cy, device)
            evs_gt_depth = gt_depth[j_tensor, i_tensor]
            ret = self.renderer.render_batch_ray(self.c, self.decoders, ray_d, ray_o, device, stage='color', gt_depth=evs_gt_depth)
            _, _, rendered_color = ret
            rendered_gray = self.rgb_to_luma(rendered_color)

        # NOTE: get fixed_pre_log_gray
        pre_gt_gray = self.rgb_to_luma(pre_gt_color)[j_tensor, i_tensor]
        fixed_pre_log_gray_new = self.lin_log(pre_gt_gray*255)
        idx_time = torch.full((N_evs, 1), (idx-5)/self.fps, dtype=torch.float64).to(self.device)
        if idx >= 10 and False:
            residual_events = (idx_time - events_pre_last_time)*first_event_polarity / (events_first_time - events_pre_last_time)
            fixed_pre_log_gray_new -= residual_events*0.1

        expected_gray = self.inverse_lin_log(fixed_pre_log_gray_new + evs_at_xy*0.1)
        
        active_sampling  = True
        if active_sampling:
            loss_events += torch.abs(expected_gray - rendered_gray*255).sum()
        print(loss_events.item())
        end = time.time()
        print("positive samping = {:0.5f}".format(end-start))

        loss_events = loss_events*0.025
        loss_events.backward(retain_graph = True)
        return loss_events.item()

    def optimize_cam_rgbd(self, camera_tensor, gt_color, gt_depth, batch_size, optimizer):
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
        # optim_quats_init.zero_grad()
        # optim_trans_init.zero_grad()
        optimizer.zero_grad() 
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

    # The new methods below are modified from https://github.com/AlvinZhuyx/camera_pose_representation

    def rotate_by_B(self, base_enc, residual, B):
        M = self.get_M(B, residual)
        new_enc = self.motion_model(M, base_enc)
        return new_enc
      
    def get_M(self, B, a):
        B_re = torch.unsqueeze(B, 0)
        a_re = a.view(-1, 1, 1)
        M = torch.unsqueeze(torch.eye(self.hidden_dim), 0).to(self.device)
        return M
    
    def motion_model(self, M, base_enc):
        new_enc = torch.matmul(M, torch.unsqueeze(base_enc, -1))
        new_enc = new_enc.view(-1, self.hidden_dim)
        return new_enc
    
    def regress_prime(self, ys, len_xs, interval=1):
        xs = torch.linspace(0, len_xs-1, len_xs)
        xs_demeaned = xs - xs.mean()
        xs_demeaned2 = xs_demeaned**2
        ys_demeaned = ys - ys.mean(dim=0)
        prime = ys_demeaned.transpose(0,1) @ xs_demeaned / xs_demeaned2.sum()
        return prime

    def run(self):
        device = self.device
        
        cfg = self.cfg
        self.fix_decoder = cfg['PoseGrid']['fix_decoder']
        self.min_locs = self.bound[:, 0].to(device)
        self.boxing_scales = torch.from_numpy(np.array(cfg['PoseGrid']['boxing_scales'])).to(device)

        self.c = {}
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)

        for idx, gt_color, gt_depth, gt_event, gt_c2w, gt_event_image  in pbar:
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")

            idx = idx[0]
            gt_depth = gt_depth[0]
            gt_color = gt_color[0]
            gt_event = gt_event[0]
            gt_c2w = gt_c2w[0]

            # indices for the encodings
            _ = idx / self.encoding_interval
            idx_enc_prev, idx_enc_next = int(torch.floor(_)), int(torch.ceil(_))
            idx_enc_prev_slam, idx_enc_next_slam = torch.tensor(idx_enc_prev*self.encoding_interval), torch.tensor(idx_enc_next*self.encoding_interval)

            if self.sync_method == 'strict':
                # strictly mapping and then tracking
                # initiate mapping every self.every_frame frames
                if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                    while self.mapping_idx[0] != idx-1:
                        time.sleep(0.1)
                    pre_c2w = self.estimate_c2w_list[idx-1].to(device)
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
                # if self.const_speed_assumption and idx-2 >= 0:
                #     pre_c2w = pre_c2w.float()
                #     delta = pre_c2w@self.estimate_c2w_list[idx-2].to(
                #         device).float().inverse()
                #     estimated_new_cam_c2w = delta@pre_c2w
                # else:
                #     estimated_new_cam_c2w = pre_c2w

                t = torch.tensor(idx).unsqueeze(0).to(device)
                t1, t2 = idx_enc_prev_slam.to(device), idx_enc_next_slam.to(device)
                # NOTE : 以下わからない
                enc1, enc2 = self.posegrid[:, idx_enc_prev].clone().to(device), self.posegrid[:, idx_enc_next].clone().to(device)
                pose_enc1, pose_enc2 = enc1[None, :-7], enc2[None, :-7]
                prime1, prime2 = enc1[None, -7:], enc2[None, -7:]
                with torch.no_grad():
                    pred_trans, pred_quat = self.pose_decoder.forward(t, t1, t2, pose_enc1, pose_enc2, prime1, prime2)
                    # NOTE: 以下の必要性
                    pred_trans_unboxed = pred_trans / self.boxing_scales + self.min_locs
                    pred_quat = F.normalize(pred_quat, p=2, dim=-1)
                    pred_rot = F.normalize(pred_quat)
                    pred_c2w = torch.cat([pred_rot, pred_trans_unboxed[..., None]], dim=-1).squeeze()
                    pred_c2w_homo = torch.cat([pred_c2w, torch.tensor([[0,0,0,1]], device=device)], dim=0)
                    camera_tensor_init = get_tensor_from_camera_in_pytorch(pred_c2w_homo)
                
                initial_loss_camera_tensor = torch.abs(
                    gt_camera_tensor.to(device)-camera_tensor_init).mean().item()

                # specify trainable variables
                trans_enc1, trans_enc2 = pose_enc1[:96], pose_enc2[:96]
                quat_enc1, quat_enc2 = pose_enc1[96:], pose_enc2[96:]
                trans_prime1, trans_prime2 = prime1[:3], prime2[:3]
                quat_prime1, quat_prime2 = prime1[3:], prime2[3:]

                trans_enc1_grad, trans_enc2_grad = Variable(trans_enc1, requires_grad=True), Variable(trans_enc2, requires_grad=True) 
                quat_enc1_grad, quat_enc2_grad = Variable(quat_enc1, requires_grad=True), Variable(quat_enc2, requires_grad=True)
                trans_prime1_grad, trans_prime2_grad = Variable(trans_prime1, requires_grad=True), Variable(trans_prime2, requires_grad=True)
                quat_prime1_grad, quat_prime2_grad = Variable(quat_prime1, requires_grad=True), Variable(quat_prime2, requires_grad=True)

                trans_enc_para_list = [trans_enc1_grad, trans_enc2_grad]
                quat_enc_para_list = [quat_enc1_grad, quat_enc2_grad]
                trans_prime_para_list = [trans_prime1_grad, trans_prime2_grad]
                quat_prime_para_list = [quat_prime1_grad, quat_prime2_grad]
                decoder_para_list = []
                # default set to True
                if not self.fix_decoder:
                    decoder_para_list += list(self.pose_decoder.parameters())
                
                # set up optimizer
                optimizer = torch.optim.Adam([{'params': decoder_para_list, 'lr': 0},
                                              {'params': trans_enc_para_list, 'lr': 0},
                                              {'params': quat_enc_para_list, 'lr': 0},
                                              {'params': trans_prime_para_list, 'lr': 0},
                                              {'params': quat_prime_para_list, 'lr': 0}])

                optimizer.param_groups[0]['lr'] = cfg['PoseGrid']['decoder_lr']
                optimizer.param_group[1]['lr'] = cfg['PoseGrid']['trans_enc_lr']
                optimizer.param_group[2]['lr'] = cfg['PoseGrid']['quat_enc_lr']
                optimizer.param_groups[3]['lr'] = cfg['PoseGrid']['trans_prime_lr']
                optimizer.param_groups[4]['lr'] = cfg['PoseGrid']['quat_prime_lr']          
                
                candidate_cam_tensor = None
                current_min_loss = 100000000000.
                # NOTE : accumulate event 
                gt_event_integrate = torch.cat((gt_event_integrate, gt_event), dim = 0)
                gt_event_images += gt_event_image

                for cam_iter in range(self.num_cam_iters):
                    # cat back together 
                    pose_enc1_grad, pose_enc2_grad = 

                # NOTE : backward every frame
                # every_frame_backward = False
                if idx % 5 == 0:
                    events_in = gt_event_integrate.cpu().numpy()
                    evs_dict_xy = {}

                    if self.event == True:
                        start = time.time()
                        for ev in events_in:
                            key_xy = (ev[0], ev[1])
                            if key_xy in evs_dict_xy.keys():
                                evs_dict_xy[key_xy].append(ev.tolist())
                            else:
                                evs_dict_xy[key_xy] = [ev.tolist()]

                        if idx < 10 :
                            pre_evs_dict_xy = dict((k, v) for k, v in evs_dict_xy.items() if len(v) > 0) 

                        evs_dict_xy = dict((k, v) for k, v in evs_dict_xy.items() if len(v) > 0) 
                        x = np.arange(self.W)
                        y = np.arange(self.H)
                        no_evs_pixels = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
                        no_evs_set = set(map(tuple, no_evs_pixels))
                        evs_set = set(evs_dict_xy.keys())
                        no_evs_set -= evs_set
                        no_evs_pixels = np.array(list(no_evs_set))
                        end = time.time()
                        print("read events= {:0.5f}".format(end-start))
                
                    for cam_iter in range(self.num_cam_iters):
                        if every_frame_backward == True:
                            idx_tensor = torch.tensor(idx%5).unsqueeze(0).to(device)
                        else:
                            idx_tensor = torch.tensor(5).unsqueeze(0).to(device)
                        estimated_correct_cam_trans = self.transNet.forward(idx_tensor).unsqueeze(0)
                        estimated_correct_cam_quad = self.quatsNet.forward(idx_tensor).unsqueeze(0)
                        estimated_correct_cam_rots = quad2rotation(estimated_correct_cam_quad)
                        estimated_correct_new_cam_c2w = torch.concat([estimated_correct_cam_rots, estimated_correct_cam_trans[...,None] ],dim=-1).squeeze()
                        bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(torch.float32).to(device)
                        estimated_correct_new_cam_c2w_homogeneous= torch.cat([estimated_correct_new_cam_c2w, bottom], dim=0)
                        compose_pose = torch.matmul(previous_c2w, estimated_correct_new_cam_c2w_homogeneous)[:3, :]
                        camera_tensor = get_tensor_from_camera_in_pytorch(compose_pose)

                        #if self.event == True:
                        loss_events = self.optimize_after_sampling_pixels(idx, estimated_new_cam_c2w, gt_c2w, camera_tensor,
                                                                      evs_dict_xy, pre_evs_dict_xy, gt_event_images, no_evs_pixels,
                                                                      pre_gt_color, pre_gt_depth,
                                                                      gt_color, gt_depth, gt_event,
                                                                      self.optim_quats_init, self.optim_trans_init)

                        if idx % 5 == 0:
                            loss_rgbd  = self.optimize_cam_rgbd(camera_tensor, gt_color, gt_depth, self.tracking_pixels,
                                                                self.optim_quats_init, self.optim_trans_init)
                        
                        self.optim_quats_init.step()
                        self.optim_trans_init.step()
                        self.optim_quats_init.zero_grad()
                        self.optim_trans_init.zero_grad()
                     
                        # print(f"Event loss:{loss_events}\n")
                        if idx % 5 == 0:
                            # print("RGBD Loss", loss_rgbd)
                            loss_events += loss_rgbd
                        c2w = get_camera_from_tensor(camera_tensor)
                        loss_camera_tensor = torch.abs(gt_camera_tensor.to(device)-camera_tensor).mean().item()

                        fixed_camera_tensor = camera_tensor.clone().detach()
                        if camera_tensor[0] < 0:
                            fixed_camera_tensor[:4] *= -1
                        fixed_camera_error = torch.abs(gt_camera_tensor.to(device)- fixed_camera_tensor).mean().item()
                
                        print(f"camera tensor error: {loss_camera_tensor}\n")
                        # print(f"camera tensor: {camera_tensor}\n")

    
                        if self.verbose:
                            if cam_iter == self.num_cam_iters-1:
                                    # wandb logging
                                    if self.event == True:
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
                                    else:
                                        dict_log = {
                                            #'Event loss' : loss_events,
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

                        
                        if loss_events < current_min_loss:
                            current_min_loss = loss_events
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
                # NOTE : update pose every 5 frame 
                previous_c2w = c2w.clone()
                # NOTE : insert dummy event
                gt_event_integrate = torch.tensor([0, 0, 0, 0]).unsqueeze(0).to(device)
                gt_event_images = torch.zeros_like(gt_event_image)
                if idx >= 5:
                    pre_evs_dict_xy = dict(evs_dict_xy)
                    # NOTE : initialize PoseNet
                    idx_total = torch.arange(start=0, end=5).to(device).reshape(5, -1)
                    self.init_posenet_train()
                    for init_i in range(50): 
                        estimated_new_cam_trans = self.transNet.forward(idx_total)
                        estimated_new_cam_quad = self.quatsNet.forward(idx_total)
                        loss_trans = torch.abs(estimated_new_cam_trans - estimated_correct_cam_trans.clone().detach()).to(device).mean()
                        loss_trans.backward()

                        loss_quad = torch.abs(estimated_new_cam_quad - estimated_correct_cam_quad.clone().detach()).to(device).mean()
                        loss_quad.backward()

                        self.optim_trans_init.step()
                        self.optim_quats_init.step()
                        self.optim_trans_init.zero_grad()
                        self.optim_quats_init.zero_grad()