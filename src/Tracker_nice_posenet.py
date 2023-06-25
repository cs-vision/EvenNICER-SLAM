import copy
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera, quad2rotation)
from src.utils.datasets import get_dataset
from src.utils.Visualizer_nice import Visualizer

# PoseNet 
from pose_net import transNet,quatsNet

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
        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

        # NOTE : PoseNet
        self.transNet = transNet(self.cfg)
        self.quatsNet = quatsNet(self.cfg)
        # self.quaternion = slam.quaternion
        # self.pose = slam.pose

    def init_posenet_train(self):
        self.optim_trans_init = torch.optim.Adam([dict(params=self.transNet.parameters(), lr = self.cam_lr*1)]) # TODO ; 10→1
        self.optim_quats_init = torch.optim.Adam([dict(params=self.quatsNet.parameters() , lr = self.cam_lr*0.2)])
        #self.optim_trans_init.zero_grad()
        #self.optim_quats_init.zero_grad()

    def optimize_trans_in_batch(self, translation, quaternion, gt_color, gt_depth, batch_size, optim_trans_init):
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optim_trans_init.zero_grad()
        quaternion = quaternion.clone().detach().to(device)
        rotation = quad2rotation(quaternion)
        c2w = torch.concat([rotation,translation[...,None]],dim=-1).squeeze()

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

        optim_trans_init.step()
        optim_trans_init.zero_grad()

        return loss.item()
    
    def optimize_quats_in_batch(self, translation, quaternion, gt_color, gt_depth, batch_size, optim_quats_init):
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optim_quats_init.zero_grad()
        translation = translation.clone().detach().to(device)
        rotation = quad2rotation(quaternion)
        c2w = torch.concat([rotation,translation[...,None] ],dim=-1).squeeze()

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

        optim_quats_init.step()
        optim_quats_init.zero_grad()

        return loss.item()

    def optimize_cam_in_batch(self, camera_tensor, gt_color, gt_depth, batch_size):
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
        #optim_quats_init.zero_grad()
        #optim_trans_init.zero_grad()
        #optimizer.zero_grad()
        

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

        #loss.backward(retain_graph=True)
        
        # optim_quats_init.step()
        # optim_trans_init.step()

        # optim_quats_init.zero_grad()
        # optim_trans_init.zero_grad()
        
        return loss

    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.

        """
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
            loss_trans = torch.abs(estimated_new_cam_trans - torch.tensor([0, 0, 0]).to(self.device)).mean()
            loss_trans.backward()

            loss_quad = torch.abs(estimated_new_cam_quad - torch.tensor([1, 0, 0, 0]).to(self.device)).mean()
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

        for idx, gt_color, gt_depth, gt_c2w in pbar:
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")

            idx = idx[0]
            gt_depth = gt_depth[0]
            gt_color = gt_color[0]
            gt_c2w = gt_c2w[0]

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
                if self.const_speed_assumption and idx-2 >= 0:
                    pre_c2w = pre_c2w.float()
                    delta = pre_c2w@self.estimate_c2w_list[idx-2].to(
                        device).float().inverse()
                    estimated_new_cam_c2w = delta@pre_c2w
                else:
                    estimated_new_cam_c2w = pre_c2w

                idx_tensor = torch.tensor(idx).unsqueeze(0).to(device)
                estimated_correct_cam_trans = self.transNet.forward(idx_tensor).unsqueeze(0)
                estimated_new_cam_quad = self.quatsNet.forward(idx_tensor).unsqueeze(0)
                estimated_correct_cam_rots = quad2rotation(estimated_new_cam_quad)
                # estimated_correct_cam_rots = self.quaternion.q_to_R(estimated_new_cam_quad)
                estimated_correct_new_cam_c2w = torch.concat([estimated_correct_cam_rots, estimated_correct_cam_trans [...,None] ],dim=-1).squeeze()
    
                # NOTE : what is pose.compose??
                #compose_pose = self.pose.compose([estimated_correct_new_cam_c2w, estimated_new_cam_c2w.detach()[:3,:]]).squeeze()

                camera_tensor = get_tensor_from_camera(
                    estimated_correct_new_cam_c2w).to(device)

                # if self.seperate_LR:
                #     camera_tensor = camera_tensor.to(device).detach()
                #     T = camera_tensor[-3:]
                #     quad = camera_tensor[:4]
                #     cam_para_list_quad = [quad]
                #     quad = Variable(quad, requires_grad=True)
                #     T = Variable(T, requires_grad=True)
                #     camera_tensor = torch.cat([quad, T], 0)
                #     cam_para_list_T = [T]
                #     cam_para_list_quad = [quad]
                #     optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr},
                #                                          {'params': cam_para_list_quad, 'lr': self.cam_lr*0.2}])
                # else:
                #     camera_tensor = Variable(
                #         camera_tensor.to(device), requires_grad=True)
                #     cam_para_list = [camera_tensor]
                #     optimizer_camera = torch.optim.Adam(
                #         cam_para_list, lr=self.cam_lr)

                initial_loss_camera_tensor = torch.abs(
                    gt_camera_tensor.to(device)- camera_tensor).mean().item()
                candidate_cam_tensor = None
                current_min_loss = 10000000000.
                #self.init_posenet_train()
                for cam_iter in range(self.num_cam_iters):
                    # if self.seperate_LR:
                    #     camera_tensor = torch.cat([quad, T], 0).to(self.device)
                    self.visualizer.vis(
                        idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)
                    
                    estimated_correct_cam_trans = self.transNet.forward(idx_tensor).unsqueeze(0)
                    estimated_new_cam_quad = self.quatsNet.forward(idx_tensor).unsqueeze(0)
                    estimated_correct_cam_rots = quad2rotation(estimated_new_cam_quad)

                    loss = self.optimize_quats_in_batch(estimated_correct_cam_trans, estimated_new_cam_quad, gt_color, gt_depth, self.tracking_pixels, self.optim_quats_init)
                    loss += self.optimize_trans_in_batch(estimated_correct_cam_trans, estimated_new_cam_quad, gt_color, gt_depth, self.tracking_pixels, self.optim_trans_init)
            
                    estimated_correct_cam_rots = quad2rotation(estimated_new_cam_quad.clone().detach())
                    camera_tensor = get_tensor_from_camera(torch.concat([estimated_correct_cam_rots, estimated_correct_cam_trans.clone().detach()[...,None] ],dim=-1).squeeze())


                    if cam_iter == 0:
                        initial_loss = loss

                    loss_camera_tensor = torch.abs(
                        gt_camera_tensor.to(device)-camera_tensor.to(device)).mean().item()

                    if self.verbose:
                        if cam_iter == self.num_cam_iters-1:
                            print(
                                f'Re-rendering loss: {initial_loss:.2f}->{loss:.2f} ' +
                                f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_tensor = camera_tensor.to(device)
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                    [1, 4])).type(torch.float32).to(self.device)
                c2w = get_camera_from_tensor(
                    candidate_cam_tensor.clone().detach())
                c2w = torch.cat([c2w, bottom], dim=0)
            self.estimate_c2w_list[idx] = c2w.clone().cpu()
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
            pre_c2w = c2w.clone()
            self.idx[0] = idx
            if self.low_gpu_mem:
                torch.cuda.empty_cache()
