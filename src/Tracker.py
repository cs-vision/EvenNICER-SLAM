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
                        get_samples_noevent,
                        get_samples_exit_event,
                        get_rays_from_uv,
                        quad2rotation,get_tensor_from_camera_in_pytorch)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer

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
            self.frame_reader, batch_size=1, shuffle=False, num_workers=0) # to avoid "RuntimeError: unable to mmap 128 bytes from file </torch_6409_1637078694_63344>: Cannot allocate memory (12)"
        
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


        # NOTE :event
        self.event = True

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
   
    def optimize_cam_event(self, camera_tensor, gt_depth, gt_event, pre_gt_color, batch_size, optimizer):
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optimizer.zero_grad()

        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H

        # NOTE : noevent sampling
        batch_i, batch_j, batch_rays_o, batch_rays_d, batch_gt_depth, batch_pre_gt_color, batch_gt_event = get_samples_noevent(
            Hedge, H-Hedge, Wedge, W-Wedge, int(batch_size*0.25), H, W, fx, fy, cx, cy, c2w, gt_depth, pre_gt_color, gt_event, device)
        ray_event = self.renderer.render_batch_ray(
            self.c, self.decoders, batch_rays_d, batch_rays_o,  device, stage='color', gt_depth=batch_gt_depth)
        _, event_uncertainty, rendered_color = ray_event

        rendered_gray = self.rgb_to_luma(rendered_color, esim=True)
        rendered_gray_log = self.lin_log(rendered_gray*255, 20)

        batch_pre_gt_gray = self.rgb_to_luma(batch_pre_gt_color, esim=True)
        batch_pre_gt_loggray = self.lin_log(batch_pre_gt_gray*255, 20)

        # 2. add events to pre_gt_gray
        gt_posneg = torch.unsqueeze(batch_gt_event[:, 0], dim =1)

        C_thres = 0.1
        batch_gt_loggray_events = batch_pre_gt_loggray + gt_posneg * C_thres

        batch_gt_inverse_loggray = self.inverse_lin_log(batch_gt_loggray_events)

        # 3. define event loss
        loss_event = torch.abs(batch_gt_inverse_loggray - rendered_gray*255).sum()

        # NOTE : exit event sampling
        batch_i, batch_j, batch_rays_o, batch_rays_d, batch_gt_depth, batch_pre_gt_color, batch_gt_event = get_samples_exit_event(
            Hedge, H-Hedge, Wedge, W-Wedge, int(batch_size*0.75), H, W, fx, fy, cx, cy, c2w, gt_depth, pre_gt_color, gt_event, device)
        ray_event = self.renderer.render_batch_ray(
            self.c, self.decoders, batch_rays_d, batch_rays_o,  device, stage='color', gt_depth=batch_gt_depth) 
        _, event_uncertainty, rendered_color = ray_event

        batch_pre_gt_gray = self.rgb_to_luma(batch_pre_gt_color, esim=True)
        batch_pre_gt_loggray = self.lin_log(batch_pre_gt_gray*255, 20)

        gt_posneg = torch.unsqueeze(batch_gt_event[:, 0], dim =1)

        C_thres = 0.1
        batch_gt_loggray_events = batch_pre_gt_loggray + gt_posneg * C_thres

        batch_gt_inverse_loggray = self.inverse_lin_log(batch_gt_loggray_events)
        
        loss_event += torch.abs(batch_gt_inverse_loggray - rendered_gray*255).sum()
        

        balancer = self.cfg['event']['balancer']
        loss_event = loss_event * balancer
        loss_event.backward(retain_graph = True)

        return loss_event.item()

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

    def run(self):
        device = self.device
        
        
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
            #gt_event_image = gt_event_image[0]

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

                camera_tensor = get_tensor_from_camera(
                    estimated_new_cam_c2w.detach())
                if self.seperate_LR:
                    camera_tensor = camera_tensor.to(device).detach()
                    T = camera_tensor[-3:]
                    quad = camera_tensor[:4]
                    cam_para_list_quad = [quad]
                    quad = Variable(quad, requires_grad=True)
                    T = Variable(T, requires_grad=True)
                    camera_tensor = torch.cat([quad, T], 0)
                    cam_para_list_T = [T]
                    cam_para_list_quad = [quad]
                    optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr},
                                                         {'params': cam_para_list_quad, 'lr': self.cam_lr*0.2}])
                else:
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    cam_para_list = [camera_tensor]
                    optimizer_camera = torch.optim.Adam(
                        cam_para_list, lr=self.cam_lr)
                
                initial_loss_camera_tensor = torch.abs(
                    gt_camera_tensor.to(device)-camera_tensor).mean().item()
                candidate_cam_tensor = None
                current_min_loss_event = 100000000000.
                       
                # NOTE : accumulate event 
                gt_event_integrate = torch.cat((gt_event_integrate, gt_event), dim = 0)
                gt_event_images += gt_event_image

    
                for cam_iter in range(self.num_cam_iters):
                    # loss_events = self.optimize_after_sampling_pixels(idx, estimated_new_cam_c2w, camera_tensor,
                    #                                                   evs_dict_xy, gt_event_images, no_evs_pixels,
                    #                                                   gt_color, gt_depth,
                    #                                                   pre_gt_color, pre_gt_depth,
                    #                                                   optimizer_camera)
                    
                    loss_events = self.optimize_cam_event(camera_tensor, gt_depth, gt_event_images.squeeze(0), pre_gt_color, self.tracking_pixels, optimizer=optimizer_camera)

                    if idx % 5 == 0:
                        loss_rgbd  = self.optimize_cam_rgbd(camera_tensor, gt_color, gt_depth, self.tracking_pixels,
                                                                optimizer_camera)
                        
                        print("RGBD loss: ", loss_rgbd)
                    optimizer_camera.step()
                    optimizer_camera.zero_grad()

                    print("Event loss : ", loss_events)
                    print("\n")

                    self.visualizer.vis(
                      idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)
                     
                       
                    c2w = get_camera_from_tensor(camera_tensor)
                    loss_camera_tensor = torch.abs(gt_camera_tensor.to(device)-camera_tensor).mean().item()

                    fixed_camera_tensor = camera_tensor.clone().detach()
                    if camera_tensor[0] < 0:
                        fixed_camera_tensor[:4] *= -1
                    fixed_camera_error = torch.abs(gt_camera_tensor.to(device)- fixed_camera_tensor).mean().item()
                
                    print(f"camera tensor error: {loss_camera_tensor}\n")
        

    
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

                    if loss_events < current_min_loss_event:
                        current_min_loss_event = loss_events
                        candidate_cam_tensor = camera_tensor.clone().detach()

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
                    
            # store previous gt color
            if idx % self.every_frame == 0:
                pre_gt_depth = gt_depth
                pre_gt_color = gt_color
                # NOTE : insert dummy event
                gt_event_integrate = torch.tensor([0, 0, 0, 0]).unsqueeze(0).to(device)
                gt_event_images = torch.zeros_like(gt_event_image)
                