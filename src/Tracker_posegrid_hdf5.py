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

# PoseGrid
from src.pose_grid import PoseGrid_decoder
import transforms3d.euler as txe
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

        # TODO : PoseGrid (refer to self.c init) and decoder init
        self.pose_decoder = PoseGrid_decoder(self.cfg)
        self.load_pretrained(self.cfg)
        self.posegrid_init(self.cfg)
        self.hidden_dim = 32 # length of pose encoding for each DoF is 32, size of B is (32, 32)

        # decide if tracker fits directly to gt pose
        self.fit_gt = False

    def load_pretrained(self, cfg):
        self.pose_decoder.load_state_dict(torch.load(cfg['PoseGrid']['pretrained_decoder']))
        self.B_stack = torch.from_numpy(np.load(cfg['PoseGrid']['pretrained_B']))
        self.B_theta, self.B_phi, self.B_gamma = self.B_stack[0], self.B_stack[1], self.B_stack[2]
        self.B_x = self.B_stack[3]
        if cfg['PoseGrid']['only_Bx']:
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
    def optimize_posegrid(self,no_evs_pixels, evs_dict_xy, camera_tensor_idx, gt_depth, pre_gt_color) :
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        # NOTE : negative sampling
        N_noevs = 50
        condition = (W//4 < no_evs_pixels[:, 0]) & (no_evs_pixels[:, 0] < W - W//4) & (H//4 < no_evs_pixels[:, 1]) & (no_evs_pixels[:, 1] < H - H//4)
        indices = np.where(condition)[0]
        selected_indices = np.random.choice(indices, size=N_noevs, replace=False)
        sampled_no_evs_xys = torch.tensor(no_evs_pixels[selected_indices]).to(device)
        noevs_i_tensor = sampled_no_evs_xys[:, 0].long() # W
        noevs_j_tensor = sampled_no_evs_xys[:, 1].long() # H

        # NOTE : calculate loss
        c2w = get_camera_from_tensor(camera_tensor_idx)
        noevs_ray_o, noevs_ray_d = get_rays_from_uv(noevs_i_tensor, noevs_j_tensor, c2w, H, W, fx, fy, cx, cy, device)
        noevs_gt_depth = gt_depth[noevs_j_tensor, noevs_i_tensor]
        noevs_ret = self.renderer.render_batch_ray(self.c, self.decoders, noevs_ray_d, noevs_ray_o, device, stage='color', gt_depth=noevs_gt_depth)
        _, _, noevs_color = noevs_ret
        noevs_gray = self.rgb_to_luma(noevs_color)
        noevs_pre_gray = self.rgb_to_luma(pre_gt_color)[noevs_j_tensor, noevs_i_tensor]
        loss_events = torch.abs(noevs_gray*255 - noevs_pre_gray*255).sum()

        # NOTE : active sampling
        N_evs = 150
        xys_mtNevs = np.array(list(evs_dict_xy.keys()))
        condition = (W//16 < xys_mtNevs[:, 0]) & (xys_mtNevs[:, 0] < W - W//16) & (H//16 < xys_mtNevs[:, 1]) & (xys_mtNevs[:, 1] < H - H//16)
        indices = np.where(condition)[0]
        sampled_xys = xys_mtNevs[selected_indices]
        sampled_xys = [tuple(row) for row in sampled_xys]
        sampled_tensor = torch.tensor(sampled_xys).view(N_evs, -1).to(device)
        i_tensor = sampled_tensor[:, 0].long()
        j_tensor = sampled_tensor[:, 1].long()

        # NOTE : calculate loss
        events_random_time = []
        for xy in sampled_xys:
            events_time_stamps = []
            events_polarities = []
            events_time_stamps.append([item[2] for item in evs_dict_xy[xy]])
            events_polarities.append([item[3] for item in evs_dict_xy[xy]])

            print(events_polarities)
            # sample event time randomly
            events_sum = torch.cumsum(events_polarities, dim=0)
            event_random_time = random.choice(events_time_stamps[0])
            events_random_index = events_time_stamps[0].index(event_random_time)
            events_random_time.append(event_random_time)
            events_random_time_sum = events_sum[events_random_index]
            print(xy)
            print(events_random_time[-1])
        
        # evs_gt_depth = gt_depth[j_tensor, i_tensor]
        # ret = self.renderer.render_batch_ray(self.c, self.decoders, ray_d, ray_o, device, stage='color', gt_depth=evs_gt_depth)
        # _, _, rendered_color = ret
        # rendered_gray = self.rgb_to_luma(rendered_color, esim=True)

    def optimize_cam_in_batch(self, camera_tensor, gt_color, gt_depth, batch_size, optimizer):
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
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
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
            self.c, self.decoders, batch_rays_d, batch_rays_o,  self.device, stage='color',  gt_depth=batch_gt_depth)
        depth, uncertainty, color = ret

        uncertainty = uncertainty.detach()
        if self.handle_dynamic:
            tmp = torch.abs(batch_gt_depth-depth)/torch.sqrt(uncertainty+1e-10)
            mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0)
        else:
            mask = batch_gt_depth > 0

        loss = (torch.abs(batch_gt_depth-depth) /
               torch.sqrt(uncertainty+1e-10))[mask].sum()

        if self.use_color_in_tracking:
            color_loss = torch.abs(
                batch_gt_color - color)[mask].sum()
            loss += self.w_color_loss*color_loss
        loss.backward()   
        optimizer.step()
        optimizer.zero_grad()

        return color_loss.item()
    
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

                # NOTE: event processing
                events_in = gt_event_integrate.cpu().numpy()
                evs_dict_xy = {}
                for ev in events_in:
                    key_xy = (ev[0], ev[1])
                    if key_xy in evs_dict_xy.keys():
                        evs_dict_xy[key_xy].append(ev)
                    else:
                        evs_dict_xy[key_xy] = [ev]
                   
                evs_dict_xy = dict((k, v) for k, v in evs_dict_xy.items() if len(v) > 0) 
                x = np.arange(self.W)
                y = np.arange(self.H)
                no_evs_pixels = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
                no_evs_set = set(map(tuple, no_evs_pixels))
                evs_set = set(evs_dict_xy.keys())
                no_evs_set -= evs_set
                no_evs_pixels = np.array(list(no_evs_set))

                for cam_iter in range(self.num_cam_iters):

                    # cat back together 
                    pose_enc1_grad, pose_enc2_grad = torch.cat([trans_enc1_grad, quat_enc1_grad]), torch.cat([trans_enc2_grad, quat_enc2_grad])
                    prime1_grad, prime2_grad = torch.cat([trans_prime1_grad, quat_prime1_grad]), torch.cat([trans_prime2_grad, quat_prime2_grad])

                    pred_trans, pred_quat = self.pose_decoder.forward(t, t1, t2, pose_enc1_grad, pose_enc2_grad, prime1_grad, prime2_grad)  
                    pred_trans_unboxed = pred_trans / self.boxing_scales + self.min_locs
                    pred_quat = F.normalize(pred_quat, p=2, dim=-1)
                    pred_rot = quad2rotation(pred_quat)
                    pred_c2w = torch.cat([pred_rot, pred_trans_unboxed[..., None]], dim=-1).squeeze()
                    pred_c2w_homo = torch.cat([pred_c2w, torch.tensor([[0,0,0,1]], device=device)], dim=0)
                    camera_tensor = get_tensor_from_camera_in_pytorch(pred_c2w_homo)

                    if self.fit_gt:
                        # Here learn gt pose directly to test if PoseGrid successfully fits it
                        loss = torch.abs(gt_camera_tensor.to(device) - camera_tensor).mean()
                        loss.backward()

                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        # TODO : switch to event loss
                        loss = self.optimize_cam_in_batch(camera_tensor, gt_color, gt_depth, self.tracking_pixels, optimizer)
                    
                    if cam_iter == 0:
                        initial_loss = loss
                    
                    if self.fit_gt:
                        loss_camera_tensor = loss.item()
                    else:
                        loss_camera_tensor = torch.abs(
                            gt_camera_tensor.to(device)-camera_tensor.to(device)).mean().item()
                    
                    fixed_camera_tensor = camera_tensor.clone().detach()
                    if camera_tensor[0] < 0 :
                        fixed_camera_tensor[:4] *= -1
                    fixed_camera_error = torch.abs(gt_camera_tensor.to(device)- fixed_camera_tensor).mean().item()

                    if self.verbose:
                        if cam_iter == self.num_cam_iters-1:
                            print(
                                f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')
                            dict_log = {
                                'Camera error': loss_camera_tensor,
                                'Fixed camera error' : fixed_camera_error,
                                'Camera error improvement': initial_loss_camera_tensor - loss_camera_tensor,
                                'Frame': idx,
                                'pred quat w' : camera_tensor[0], 
                                'init quat w' : camera_tensor_init[0], 
                                'gt quat w' : gt_camera_tensor[0],
                                'pred quat x' : camera_tensor[1], 
                                'init quat x' : camera_tensor_init[1], 
                                'gt quat x' : gt_camera_tensor[1],
                                'pred quat y' : camera_tensor[2], 
                                'init quat y' : camera_tensor_init[2], 
                                'gt quat y' : gt_camera_tensor[2],
                                'pred quat z' : camera_tensor[3], 
                                'init quat z' : camera_tensor_init[3], 
                                'gt quat z' : gt_camera_tensor[3],
                                'pred trans x' : camera_tensor[4],
                                'init trans x' : camera_tensor_init[4], 
                                'gt trans x' : gt_camera_tensor[4],
                                'pred trans y' : camera_tensor[5],
                                'init trans y' : camera_tensor_init[5],
                                'gt trans y' : gt_camera_tensor[5],
                                'pred trans z' : camera_tensor[6],
                                'init trans z' : camera_tensor_init[6],
                                'gt trans z' : gt_camera_tensor[6]
                                }
                            self.experiment.log(dict_log)

                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_tensor = camera_tensor.to(device)

                        # store pred_trans & pred_quat
                        candidate_pred_trans, candidate_pred_quat = pred_trans, pred_quat

                        if not self.fix_decoder:
                            candidate_decoder_para = self.PoseGrid_decoder.state_dict()
                        candidate_trans_enc_para_list = trans_enc_para_list
                        candidate_quat_enc_para_list = quat_enc_para_list
                        candidate_trans_prime_para_list = trans_prime_para_list
                        candidate_quat_prime_para_list = quat_prime_para_list

                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                    [1, 4])).type(torch.float32).to(self.device)
                print("candidate_cam_tensor", candidate_cam_tensor)

                c2w = get_camera_from_tensor(
                    candidate_cam_tensor.clone().detach())
                c2w = torch.cat([c2w, bottom], dim=0)

                if not self.fix_decoder:
                    self.pose_decoder.load_state_dict(candidate_decoder_para)
                    del candidate_decoder_para
                
                candidate_trans_enc1_grad, candidate_trans_enc2_grad = tuple(candidate_trans_enc_para_list)
                candidate_quat_enc1_grad, candidate_quat_enc2_grad = tuple(candidate_quat_enc_para_list)
                candidate_trans_prime1_grad, candidate_trans_prime2_grad = tuple(candidate_trans_prime_para_list)
                candidate_quat_prime1_grad, candidate_quat_prime2_grad = tuple(candidate_quat_prime_para_list)
                del candidate_trans_enc_para_list, candidate_quat_enc_para_list, candidate_trans_prime_para_list, candidate_quat_prime_para_list
                candidate_pose_enc1 = torch.cat([candidate_trans_enc1_grad, candidate_quat_enc1_grad])
                candidate_pose_enc2 = torch.cat([candidate_trans_enc2_grad, candidate_quat_enc2_grad])
                candidate_prime1 = torch.cat([candidate_trans_prime1_grad, candidate_quat_prime1_grad])
                candidate_prime2 = torch.cat([candidate_trans_prime2_grad, candidate_quat_prime2_grad])
                candidate_enc1 = torch.cat([candidate_pose_enc1, candidate_prime1], dim=-1)
                candidate_enc2 = torch.cat([candidate_pose_enc2, candidate_prime2], dim=-1)
                self.posegrid[:, idx_enc_prev] = candidate_enc1.clone().detach()
                self.posegrid[:, idx_enc_next] = candidate_enc2.clone().detach()
                                  

            self.estimate_c2w_list[idx] = c2w.clone().detach()
            self.gt_c2w_list[idx] = gt_c2w.clone().detach()

            # TODO : manually set primes if not learned
            if not self.learn_prime and idx >= 1:
                if idx >= self.len_avg_prime - 1:
                    len_avg_prime = self.len_avg_prime
                else:
                    len_avg_prime = idx+1

                avg_c2w_list = self.estimate_c2w_list[idx-len_avg_prime+1:idx+1]
                avg_tensor_list = [get_tensor_from_camera(c2w.to(device).float(), Tquad=True) for c2w in avg_c2w_list]
                avg_tensors = torch.vstack(avg_tensor_list)
                prime = self.regress_prime(avg_tensors, len_avg_prime)
                self.posegrid[-7:, idx_enc_prev] = prime.clone().detach()

            pre_c2w = c2w.clone().detach()
            self.idx[0] = idx

            # TODO: constant speed assumption after optimizing grid point encoding
            print(f'idx_enc_prev: {idx_enc_prev}, idx_enc_next: {idx_enc_next}, is equal: {idx_enc_prev == idx_enc_next}')
            if self.const_speed_assumption and idx_enc_prev == idx_enc_next and not idx == 0:
                print("Is doing constant speed assumption!!!!!")
                with torch.no_grad():
                    pose_enc_now = self.posegrid[:-7, idx_enc_next].clone().to(device)
                    prime_now = self.posegrid[-7:, idx_enc_next].clone().to(device)
                    delta = (prime_now * self.encoding_interval)
                    delta_trans = delta[:3]
                    delta_quat = delta[3:]

                    # trans_now = candidate_pred_trans # this translation is bounded by the 4x4x4 box
                    quat_now = F.normalize(candidate_pred_quat, p=2, dim=-1)
                    quat_next = quat_now + delta_quat
                    quat_next = F.normalize(quat_next, p=2, dim=-1)

                    # NOTE: zero rotation (0, 0, 0) is mapped to (pi, pi/2, pi) for encoding, the following real world Euler angles are bounded by (+-pi, +-pi/2, +-pi)
                    theta_now, phi_now, gamma_now = txe.quat2euler(quat_now.squeeze().cpu())
                    theta_next, phi_next, gamma_next = txe.quat2euler(quat_next.squeeze().cpu())
                    delta_euler = torch.tensor([theta_next-theta_now, phi_next-phi_now, gamma_next-gamma_now])
                    delta_euler /= np.pi/1.8 # need to rescale like this to use B
                    delta_euler = delta_euler.to(device)

                    # get a good initialization of the next encodings using pretrained B's
                    delta_transeuler = torch.cat([delta_trans, delta_euler], dim=-1)
                    B_list = [self.B_x, self.B_y, self.B_z, self.B_theta, self.B_phi, self.B_gamma]
                    pose_enc_new = []
                    n_dof = 6
                    for dof in range(n_dof):
                        pose_enc_dof = pose_enc_now[dof*self.hidden_dim:(dof+1)*self.hidden_dim]
                        pose_enc_new.append(self.rotate_by_B(pose_enc_dof, delta_transeuler[dof], B_list[dof].to(device)))
                    pose_enc_new = torch.cat(pose_enc_new, dim=-1).squeeze()
                    prime_new = prime_now
                    enc_new = torch.cat([pose_enc_new, prime_new], dim=-1)
                    self.posegrid[:, idx_enc_next+1] = enc_new.clone()
            else:
                print("NOT doing constant speed assumption!!!!!")

            if self.low_gpu_mem:
                torch.cuda.empty_cache()
            
            if idx % self.every_frame == 0:
                pre_gt_depth = gt_depth
                pre_gt_color = gt_color
                gt_event_integrate = torch.tensor([0, 0, 0, 0]).unsqueeze(0).to(device)
                gt_event_images = torch.zeros_like(gt_event_image)





            #     # NOTE : backward every frame
            #     # every_frame_backward = False
            #     if idx % 5 == 0:
            #         events_in = gt_event_integrate.cpu().numpy()
            #         evs_dict_xy = {}

            #         if self.event == True:
            #             start = time.time()
            #             for ev in events_in:
            #                 key_xy = (ev[0], ev[1])
            #                 if key_xy in evs_dict_xy.keys():
            #                     evs_dict_xy[key_xy].append(ev.tolist())
            #                 else:
            #                     evs_dict_xy[key_xy] = [ev.tolist()]

            #             if idx < 10 :
            #                 pre_evs_dict_xy = dict((k, v) for k, v in evs_dict_xy.items() if len(v) > 0) 

            #             evs_dict_xy = dict((k, v) for k, v in evs_dict_xy.items() if len(v) > 0) 
            #             x = np.arange(self.W)
            #             y = np.arange(self.H)
            #             no_evs_pixels = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
            #             no_evs_set = set(map(tuple, no_evs_pixels))
            #             evs_set = set(evs_dict_xy.keys())
            #             no_evs_set -= evs_set
            #             no_evs_pixels = np.array(list(no_evs_set))
            #             end = time.time()
            #             print("read events= {:0.5f}".format(end-start))
                
            #         for cam_iter in range(self.num_cam_iters):
            #             if every_frame_backward == True:
            #                 idx_tensor = torch.tensor(idx%5).unsqueeze(0).to(device)
            #             else:
            #                 idx_tensor = torch.tensor(5).unsqueeze(0).to(device)
            #             estimated_correct_cam_trans = self.transNet.forward(idx_tensor).unsqueeze(0)
            #             estimated_correct_cam_quad = self.quatsNet.forward(idx_tensor).unsqueeze(0)
            #             estimated_correct_cam_rots = quad2rotation(estimated_correct_cam_quad)
            #             estimated_correct_new_cam_c2w = torch.concat([estimated_correct_cam_rots, estimated_correct_cam_trans[...,None] ],dim=-1).squeeze()
            #             bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(torch.float32).to(device)
            #             estimated_correct_new_cam_c2w_homogeneous= torch.cat([estimated_correct_new_cam_c2w, bottom], dim=0)
            #             compose_pose = torch.matmul(previous_c2w, estimated_correct_new_cam_c2w_homogeneous)[:3, :]
            #             camera_tensor = get_tensor_from_camera_in_pytorch(compose_pose)

            #             #if self.event == True:
            #             loss_events = self.optimize_after_sampling_pixels(idx, estimated_new_cam_c2w, gt_c2w, camera_tensor,
            #                                                           evs_dict_xy, pre_evs_dict_xy, gt_event_images, no_evs_pixels,
            #                                                           pre_gt_color, pre_gt_depth,
            #                                                           gt_color, gt_depth, gt_event,
            #                                                           self.optim_quats_init, self.optim_trans_init)

            #             if idx % 5 == 0:
            #                 loss_rgbd  = self.optimize_cam_rgbd(camera_tensor, gt_color, gt_depth, self.tracking_pixels,
            #                                                     self.optim_quats_init, self.optim_trans_init)
                        
            #             self.optim_quats_init.step()
            #             self.optim_trans_init.step()
            #             self.optim_quats_init.zero_grad()
            #             self.optim_trans_init.zero_grad()
                     
            #             # print(f"Event loss:{loss_events}\n")
            #             if idx % 5 == 0:
            #                 # print("RGBD Loss", loss_rgbd)
            #                 loss_events += loss_rgbd
            #             c2w = get_camera_from_tensor(camera_tensor)
            #             loss_camera_tensor = torch.abs(gt_camera_tensor.to(device)-camera_tensor).mean().item()

            #             fixed_camera_tensor = camera_tensor.clone().detach()
            #             if camera_tensor[0] < 0:
            #                 fixed_camera_tensor[:4] *= -1
            #             fixed_camera_error = torch.abs(gt_camera_tensor.to(device)- fixed_camera_tensor).mean().item()
                
            #             print(f"camera tensor error: {loss_camera_tensor}\n")
            #             # print(f"camera tensor: {camera_tensor}\n")

    
            #             if self.verbose:
            #                 if cam_iter == self.num_cam_iters-1:
            #                         # wandb logging
            #                         if self.event == True:
            #                             dict_log = {
            #                                 'Event loss' : loss_events,
            #                                 #'Event loss improvement': initial_loss_event - loss_event,
            #                                 'Camera error': loss_camera_tensor,
            #                                 'Fixed camera error' : fixed_camera_error,
            #                                 #'Camera error improvement': initial_loss_camera_tensor - loss_camera_tensor,
            #                                 'Frame': idx,
            #                                 'first element of gt quaternion' : gt_camera_tensor[0],
            #                                 'first element of quaternion' : camera_tensor[0],
            #                                 'first element of translation' : camera_tensor[4],
            #                                 'first element of gt translation' : gt_camera_tensor[4]
            #                             }
            #                         else:
            #                             dict_log = {
            #                                 #'Event loss' : loss_events,
            #                                 #'Event loss improvement': initial_loss_event - loss_event,
            #                                 'Camera error': loss_camera_tensor,
            #                                 'Fixed camera error' : fixed_camera_error,
            #                                 #'Camera error improvement': initial_loss_camera_tensor - loss_camera_tensor,
            #                                 'Frame': idx,
            #                                 'first element of gt quaternion' : gt_camera_tensor[0],
            #                                 'first element of quaternion' : camera_tensor[0],
            #                                 'first element of translation' : camera_tensor[4],
            #                                 'first element of gt translation' : gt_camera_tensor[4]
            #                             }                                       
            #                         self.experiment.log(dict_log)

                        
            #             if loss_events < current_min_loss:
            #                 current_min_loss = loss_events
            #                 candidate_cam_tensor = camera_tensor.clone().detach()
            #                 if not self.use_last:
            #                     candidate_transNet_para = self.transNet.state_dict()
            #                     candidate_quatsNet_para = self.quatsNet.state_dict()

            #         bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
            #                 [1, 4])).type(torch.float32).to(self.device)
            #         print("candidate_cam_tensor:", candidate_cam_tensor)
            #         if self.use_last:
            #             c2w = get_camera_from_tensor(
            #                 camera_tensor.clone().detach()
            #             )
            #             c2w = torch.cat([c2w, bottom], dim=0)
            #         else:
            #             c2w = get_camera_from_tensor(
            #                 candidate_cam_tensor.clone().detach()
            #             )
            #             c2w = torch.cat([c2w, bottom], dim=0)   

            #             self.transNet.load_state_dict(candidate_transNet_para)
            #             self.quatsNet.load_state_dict(candidate_quatsNet_para)

            #             del candidate_transNet_para
            #             del candidate_quatsNet_para       

            
            # self.estimate_c2w_list[idx] = c2w.clone().cpu()
            # self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
            # pre_c2w = c2w.clone()
            # self.idx[0] = idx
            # if self.low_gpu_mem:
            #     torch.cuda.empty_cache()
                
            # # store previous gt color
            # if idx % self.every_frame == 0:
            #     pre_gt_depth = gt_depth
            #     pre_gt_color = gt_color
            #     # NOTE : update pose every 5 frame 
            #     previous_c2w = c2w.clone()
            #     # NOTE : insert dummy event
            #     gt_event_integrate = torch.tensor([0, 0, 0, 0]).unsqueeze(0).to(device)
            #     gt_event_images = torch.zeros_like(gt_event_image)