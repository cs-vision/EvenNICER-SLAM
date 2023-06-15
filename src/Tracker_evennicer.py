import copy
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

from src.common import (get_camera_from_tensor, get_samples, get_samples_event,
                        get_tensor_from_camera)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer

# event net
from src.event_net import inference_event, event_to_image


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
            # self.frame_reader, batch_size=1, shuffle=False, num_workers=1)
            self.frame_reader, batch_size=1, shuffle=False, num_workers=0) # to avoid "RuntimeError: unable to mmap 128 bytes from file </torch_6409_1637078694_63344>: Cannot allocate memory (12)"

        self.slam = slam
        self.scale_factor = slam.scale_factor

        # wandb
        self.experiment = slam.experiment

        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], 
                                    #  inside_freq=cfg['tracking']['vis_inside_freq'],
                                     inside_freq=2*cfg['tracking']['vis_inside_freq']-1, # to see start and end
                                     vis_dir=os.path.join(self.output, 'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, 
                                     experiment=self.experiment, # wandb
                                     device=self.device, 
                                     stage='tracker')
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

        # event-net loading
        self.event_net = slam.event_net

        # a switch to turn on/off events
        self.activate_events = cfg['event']['activate_events']

        # event image Gaussian blur
        self.blur = cfg['event']['blur']
        self.kernel_sizes = cfg['event']['kernel_sizes']
        self.unblurred_weight = cfg['event']['unblurred_weight']
        self.kernel_weights = cfg['event']['kernel_weights']

        # RGBD available condition
        self.rgbd_every_frame = cfg['event']['rgbd_every_frame']


    def  optimize_cam_in_batch(self, camera_tensor, 
                              pre_c2w,
                              gt_color, gt_depth, gt_event,
                              gt_mask,
                              batch_size, optimizer, 
                              idx, iter, # for rendered image saving
                              pre_gt_color, 
                              rgbd=True, event=True, scale_factor=0.1):
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

        # gt event rescale
        if event:
            gt_event = gt_event.permute(2, 0, 1)
            p, h, w = gt_event.shape
            h_new, w_new = int(scale_factor * h), int(scale_factor * w)
            assert w_new > 0 and h_new > 0, 'Scale is too small, resized images would have no pixels'
            # transform = transforms.Resize((h_new, w_new), interpolation=transforms.InterpolationMode.BILINEAR)
            transform = transforms.Resize((h_new, w_new), interpolation=transforms.InterpolationMode.NEAREST)
            gt_event = transform(gt_event).permute(1, 2, 0)

            # gt mask rescale
            gt_mask = transform(gt_mask[None, :, :]).permute(1, 2, 0) # consider max pooling here?

        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H

        if event:
            full_color_previous = transform(pre_gt_color.permute(2, 0, 1)).permute(1, 2, 0) # previous available rgb image
            if self.low_gpu_mem:
                torch.cuda.empty_cache()
            _, _, full_color_current = self.renderer.render_img_rescale(self.c, self.decoders, c2w, self.device, stage='color', gt_depth=gt_depth, scale_factor=scale_factor) # these rendered images are [0, 1]

        if event:
            full_event, event_mask = inference_event(net=self.event_net, img1=full_color_previous, img2=full_color_current, device=device, scale_factor=1.0, out_threshold=0.5)

            # save fresh prediction for debugging
            # event_img = event_to_image(full_event.clone().detach().cpu().numpy())
            # event_img.save('fresh_event_prediction.png')


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

        # loss definition
        if rgbd:
            loss_rgbd = (torch.abs(batch_gt_depth-depth) /
                    torch.sqrt(uncertainty+1e-10))[mask].sum()

            if self.use_color_in_tracking:
                color_loss = torch.abs(
                    batch_gt_color - color)[mask].sum()
                loss_rgbd += self.w_color_loss*color_loss # self.w_color_loss = 0.2 by default

            if event:
                loss_rgbd.backward(retain_graph=True)
            else:
                loss_rgbd.backward(retain_graph=False)

            loss_rgbd_item = loss_rgbd.item()
        else:
            loss_rgbd_item = None
        if event:
            # loss_event = torch.abs(gt_event - full_event).sum() # l1 loss
            loss_event = ((gt_event - full_event)**2).sum() # l2 loss

            if self.blur:
                gts_event_list = []
                preds_event_list = []
                losses_event_list = [self.unblurred_weight * loss_event]
                for kernel_size, kernel_weight in zip(self.kernel_sizes, self.kernel_weights):
                    gt_event_tmp = transforms.functional.gaussian_blur(gt_event.permute(2, 0, 1), kernel_size=kernel_size).permute(1, 2, 0)
                    full_event_tmp = transforms.functional.gaussian_blur(full_event.permute(2, 0, 1), kernel_size=kernel_size).permute(1, 2, 0)

                    loss_event_tmp = ((gt_event_tmp - full_event_tmp)**2).sum() # blur both, l2 loss. Try other types of loss?
                    loss_event += kernel_weight * loss_event_tmp
                    gts_event_list.append(gt_event_tmp)
                    preds_event_list.append(full_event_tmp)
                    losses_event_list.append(loss_event_tmp.item())

            # event existence mask. not used, just for reference
            criterion_ce = torch.nn.CrossEntropyLoss()
            loss_mask = criterion_ce(event_mask, gt_mask.permute(2, 0, 1))

            # balancer = batch_size / (w_new*h_new) # coefficient to balance event loss and rgbd loss
            balancer = self.cfg['event']['balancer'] # coefficient to balance event loss and rgbd loss
            loss_event = loss_event * balancer

            if self.activate_events:
                loss_event.backward()

            loss_event_item = loss_event.item()
            loss_mask_item = loss_mask.item()
        else:
            loss_event_item = None
            loss_mask_item = None

        optimizer.step()
        optimizer.zero_grad()

        if event and (not self.blur):
            return loss_rgbd_item, loss_event_item, loss_mask_item, gt_event, full_event, gt_mask, event_mask[0][1]
        else:
            return loss_rgbd_item, loss_event_item, loss_mask_item, gt_event, full_event, gts_event_list, preds_event_list, losses_event_list, gt_mask, event_mask[0][1]


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
        self.c = {}
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)

        for idx, gt_color, gt_depth, gt_event, gt_mask, gt_c2w in pbar:
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")

            idx = idx[0]
            gt_depth = gt_depth[0]
            gt_color = gt_color[0]
            gt_event = gt_event[0]
            gt_mask = gt_mask[0]
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

                # initialize accumulated events
                gt_event_integrate = torch.zeros_like(gt_event)

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
                # current_min_loss = 10000000000.
                current_min_loss_event = 10000000000.

                # add up gt_event_integrate asap
                gt_event_integrate += gt_event

                for cam_iter in range(self.num_cam_iters):
                    if self.seperate_LR:
                        camera_tensor = torch.cat([quad, T], 0).to(self.device)

                    rgbd_available = (idx % self.rgbd_every_frame == 0) # align with when mapping is dispatched
                    if rgbd_available:
                        loss_rgbd, loss_event, loss_mask, gt_event_lores, pred_event, gts_event_list, preds_event_list, losses_event_list, gt_mask_lores, pred_mask = self.optimize_cam_in_batch(
                            camera_tensor, pre_c2w, 
                            gt_color, gt_depth, gt_event_integrate, 
                            gt_mask, 
                            self.tracking_pixels, optimizer_camera, 
                            idx, cam_iter, 
                            pre_gt_color,
                            rgbd=True, event=True, scale_factor=self.scale_factor)
                    else:
                        loss_rgbd, loss_event, loss_mask, gt_event_lores, pred_event, gts_event_list, preds_event_list, losses_event_list, gt_mask_lores, pred_mask = self.optimize_cam_in_batch(
                            camera_tensor, pre_c2w, 
                            gt_color, gt_depth, gt_event_integrate, 
                            gt_mask, 
                            self.tracking_pixels, optimizer_camera, 
                            idx, cam_iter, 
                            pre_gt_color,
                            rgbd=False, event=True, scale_factor=self.scale_factor)

                    print("RGBD loss:", loss_rgbd)
                    print("Event loss:", loss_event)
                    print("Mask loss:", loss_mask)
                    print('\n')

                    if cam_iter == 0:
                        initial_loss_rgbd = loss_rgbd
                        initial_loss_event = loss_event
                        initial_losses_event_blur = losses_event_list
                        initial_loss_mask = loss_mask

                    loss_camera_tensor = torch.abs(
                        gt_camera_tensor.to(device)-camera_tensor).mean().item()
                    if self.verbose:
                        if cam_iter == self.num_cam_iters-1:
                            if rgbd_available:
                                print(
                                    f'RGBD loss: {initial_loss_rgbd:.2f}->{loss_rgbd:.2f} ' +
                                    f'event loss: {initial_loss_event:.2f}->{loss_event:.2f} ' +
                                    f'mask loss: {initial_loss_mask:.2f}->{loss_mask:.2f} ' +
                                    f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')

                                # wandb logging
                                dict_log = {
                                    'RGBD loss': loss_rgbd,
                                    'RGBD loss improvement': initial_loss_rgbd - loss_rgbd,
                                    'Event loss': loss_event,
                                    'Event loss improvement': initial_loss_event - loss_event,
                                    'Mask loss': loss_mask,
                                    'Mask loss improvement': initial_loss_mask - loss_mask,
                                    'Camera error': loss_camera_tensor,
                                    'Camera error improvement': initial_loss_camera_tensor - loss_camera_tensor,
                                    'Frame': idx
                                }
                                for blur_level, loss_event_blur in enumerate(losses_event_list):
                                    dict_log[f'Event loss blur {blur_level}'] = loss_event_blur
                                    dict_log[f'Event loss blur {blur_level} improvement'] = initial_losses_event_blur[blur_level] - loss_event_blur

                                self.experiment.log(dict_log)
                            else:
                                print(
                                    f'RGBD loss: None ' +
                                    f'event loss: {initial_loss_event:.2f}->{loss_event:.2f} ' +
                                    f'mask loss: {initial_loss_mask:.2f}->{loss_mask:.2f} ' +
                                    f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')

                                # wandb logging
                                dict_log = {
                                    'Event loss': loss_event,
                                    'Event loss improvement': initial_loss_event - loss_event,
                                    'Mask loss': loss_mask,
                                    'Mask loss improvement': initial_loss_mask - loss_mask,
                                    'Camera error': loss_camera_tensor,
                                    'Camera error improvement': initial_loss_camera_tensor - loss_camera_tensor,
                                    'Frame': idx
                                }
                                for blur_level, loss_event_blur in enumerate(losses_event_list):
                                    dict_log[f'Event loss blur {blur_level}'] = loss_event_blur
                                    dict_log[f'Event loss blur {blur_level} improvement'] = initial_losses_event_blur[blur_level] - loss_event_blur

                                self.experiment.log(dict_log)

                    # use event loss as criteria because it is always available
                    if loss_event < current_min_loss_event:
                        current_min_loss_event = loss_event
                        candidate_cam_tensor = camera_tensor.clone().detach()

                    self.visualizer.vis_event(
                        idx, cam_iter, gt_depth, gt_color, gt_event, gt_event_lores, pred_event, 
                        gts_event_list, preds_event_list, 
                        camera_tensor, self.c, self.decoders)
    
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
                pre_gt_color = gt_color
                self.gt_event_integrate = gt_event_integrate.clone() # share with mapper
                print("integrated GT events updated!")
                gt_event_integrate = torch.zeros_like(gt_event)