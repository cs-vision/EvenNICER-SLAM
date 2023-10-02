import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera, quad2rotation,get_tensor_from_camera_in_pytorch)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer

# PoseGrid
from src.pose_grid import PoseGrid_decoder
import transforms3d.euler as txe


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
        
        #wandb
        self.slam = slam
        self.experiment = slam.experiment

        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, 
                                     experiment=self.experiment, # wandb
                                     device=self.device,
                                     stage = 'tracker')
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

        # TODO: PoseGrid (refer to self.c init) and decoder init
        self.pose_decoder = PoseGrid_decoder(self.cfg)
        self.load_pretrained(self.cfg)
        self.posegrid_init(self.cfg)
        self.hidden_dim = 32 # length of pose encoding for each DoF is 32, size of B is (32, 32)
    
        # NOTE : Depth 
        self.use_color_in_depth = False

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
        # self.zeropose_encoding = torch.from_numpy(np.load(cfg['PoseGrid']['zeropose_encoding']))
        self.zeropose_encoding = torch.from_numpy(np.load(cfg['PoseGrid']['startpose_encoding']))

    def posegrid_init(self, cfg):
        self.encoding_interval = cfg['PoseGrid']['encoding_interval']
        self.encoding_dim = 199 # 32*6 + 7, 32 each DoF, 7 for translation & quaternion prime
        self.n_encoding = int(np.ceil(self.n_img / self.encoding_interval) + 1) # make sure the first and the last frame has its encoding
        self.posegrid = torch.cat([self.zeropose_encoding, torch.zeros(7)]).view(self.encoding_dim, 1).repeat(1, self.n_encoding) # (self.encoding_dim, self.n_encoding)
    
    # The new methods below are modified from https://github.com/AlvinZhuyx/camera_pose_representation
    def rotate_by_B(self, base_enc, residual, B):
        M = self.get_M(B, residual)
        new_enc = self.motion_model(M, base_enc)
        return new_enc

    def get_M(self, B, a):
        B_re = torch.unsqueeze(B, 0)
        a_re = a.view(-1, 1, 1)
        M = torch.unsqueeze(torch.eye(self.hidden_dim), 0) + B_re * a_re + torch.matmul(B_re, B_re) * (a_re ** 2) / 2
        return M

    def motion_model(self, M, base_enc):
        new_enc = torch.matmul(M, torch.unsqueeze(base_enc, -1))
        new_enc = new_enc.view(-1, self.hidden_dim)
        return new_enc


    def run(self):
        device = self.device

        cfg = self.cfg
        self.fix_decoder = cfg['PoseGrid']['fix_decoder']
        self.min_locs = self.bound[:, 0].to(device)
        self.boxing_scales = torch.from_numpy(np.array(cfg['PoseGrid']['boxing_scales'])).to(device)
        
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

            # indices for the encodings
            _ = idx/self.encoding_interval
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

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)

            if idx == 0 or self.gt_camera:
                c2w = gt_c2w

            else:
                gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                t = torch.tensor(idx).unsqueeze(0).to(device)
                t1, t2 = idx_enc_prev_slam.to(device), idx_enc_next_slam.to(device)
                enc1, enc2 = self.posegrid[:, idx_enc_prev].clone().to(device), self.posegrid[:, idx_enc_next].clone().to(device)
                pose_enc1, pose_enc2 = enc1[None, :-7], enc2[None, :-7]
                prime1, prime2 = enc1[None, -7:], enc2[None, -7:]
                with torch.no_grad():
                    pred_trans, pred_quat = self.pose_decoder.forward(t, t1, t2, pose_enc1, pose_enc2, prime1, prime2)
                    print(pred_trans.device, self.boxing_scales, self.min_locs)
                    pred_trans_unboxed = pred_trans / self.boxing_scales + self.min_locs
                    pred_rot = quad2rotation(pred_quat)
                    pred_c2w = torch.cat([pred_rot, pred_trans_unboxed[..., None]], dim=-1).squeeze()
                    pred_c2w_homo = torch.cat([pred_c2w, torch.tensor([[0,0,0,1]], device=device)], dim=0)
                    camera_tensor = get_tensor_from_camera_in_pytorch(pred_c2w_homo)

                initial_loss_camera_tensor = torch.abs(
                    gt_camera_tensor.to(device)- camera_tensor).mean().item()

                # specify trainable variables
                trans_enc1, trans_enc2 = pose_enc1[:96], pose_enc2[:96]
                quat_enc1, quat_enc2 = pose_enc1[96:], pose_enc2[96:]
                trans_prime1, trans_prime2 = prime1[:3], prime2[:3]
                quat_prime1, quat_prime2 = prime1[3:], prime2[3:]
                # pose_enc1_grad, pose_enc2_grad = Variable(pose_enc1, requires_grad=True), Variable(pose_enc2, requires_grad=True)
                trans_enc1_grad, trans_enc2_grad = Variable(trans_enc1, requires_grad=True), Variable(trans_enc2, requires_grad=True)
                quat_enc1_grad, quat_enc2_grad = Variable(quat_enc1, requires_grad=True), Variable(quat_enc2, requires_grad=True)
                # prime1_grad, prime2_grad = Variable(prime1, requires_grad=True), Variable(prime2, requires_grad=True)
                trans_prime1_grad, trans_prime2_grad = Variable(trans_prime1, requires_grad=True), Variable(trans_prime2, requires_grad=True)
                quat_prime1_grad, quat_prime2_grad = Variable(quat_prime1, requires_grad=True), Variable(quat_prime2, requires_grad=True)
                # pose_enc_para_list = [pose_enc1_grad, pose_enc2_grad]
                trans_enc_para_list = [trans_enc1_grad, trans_enc2_grad]
                quat_enc_para_list = [quat_enc1_grad, quat_enc2_grad]
                # prime_para_list = [prime1_grad, prime2_grad]
                trans_prime_para_list = [trans_prime1_grad, trans_prime2_grad]
                quat_prime_para_list = [quat_prime1_grad, quat_prime2_grad]
                decoder_para_list = []
                if not self.fix_decoder:
                    decoder_para_list += list(self.pose_decoder.parameters())
                
                # set up optimizer
                optimizer = torch.optim.Adam([{'params': decoder_para_list, 'lr': 0},
                                            #   {'params': pose_enc_para_list, 'lr': 0},
                                              {'params': trans_enc_para_list, 'lr': 0},
                                              {'params': quat_enc_para_list, 'lr': 0},
                                            #   {'params': prime_para_list, 'lr': 0}])
                                              {'params': trans_prime_para_list, 'lr': 0}, 
                                              {'params': quat_prime_para_list, 'lr': 0}])
                # TODO: consider wrapping the optimizer with a LR scheduler as in mapper
                optimizer.param_groups[0]['lr'] = cfg['PoseGrid']['decoder_lr']
                # optimizer.param_groups[1]['lr'] = cfg['PoseGrid']['pose_enc_lr']
                # optimizer.param_groups[2]['lr'] = cfg['PoseGrid']['prime_lr']
                optimizer.param_groups[1]['lr'] = cfg['PoseGrid']['trans_enc_lr']
                optimizer.param_groups[2]['lr'] = cfg['PoseGrid']['quat_enc_lr']
                # optimizer.param_groups[3]['lr'] = cfg['PoseGrid']['prime_lr']
                optimizer.param_groups[3]['lr'] = cfg['PoseGrid']['trans_prime_lr']
                optimizer.param_groups[4]['lr'] = cfg['PoseGrid']['quat_prime_lr']

                candidate_cam_tensor = None
                current_min_loss = 10000000000.
                for cam_iter in range(self.num_cam_iters):

                    # optimizer.zero_grad()

                    # cat back together
                    pose_enc1_grad, pose_enc2_grad = torch.cat([trans_enc1_grad, quat_enc1_grad]), torch.cat([trans_enc2_grad, quat_enc2_grad])
                    prime1_grad, prime2_grad = torch.cat([trans_prime1_grad, quat_prime1_grad]), torch.cat([trans_prime2_grad, quat_prime2_grad])

                    # estimated_correct_cam_trans = self.transNet.forward(idx_tensor).unsqueeze(0)
                    # estimated_new_cam_quad = self.quatsNet.forward(idx_tensor).unsqueeze(0)
                    # estimated_correct_cam_rots = quad2rotation(estimated_new_cam_quad)
                    # estimated_correct_new_cam_c2w = torch.concat([estimated_correct_cam_rots, estimated_correct_cam_trans[...,None] ],dim=-1).squeeze()
                    # bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(torch.float32).to(device)
                    # estimated_correct_new_cam_c2w_homogeneous = torch.cat([estimated_correct_new_cam_c2w, bottom], dim=0)
                    # compose_pose = torch.matmul(estimated_new_cam_c2w, estimated_correct_new_cam_c2w_homogeneous)[:3, :]
                    # camera_tensor = get_tensor_from_camera_in_pytorch(compose_pose)
                    pred_trans, pred_quat = self.pose_decoder.forward(t, t1, t2, pose_enc1_grad, pose_enc2_grad, prime1_grad, prime2_grad)
                    pred_trans_unboxed = pred_trans / self.boxing_scales + self.min_locs
                    pred_rot = quad2rotation(pred_quat)
                    pred_c2w = torch.cat([pred_rot, pred_trans_unboxed[..., None]], dim=-1).squeeze()
                    pred_c2w_homo = torch.cat([pred_c2w, torch.tensor([[0,0,0,1]], device=device)], dim=0)
                    camera_tensor = get_tensor_from_camera_in_pytorch(pred_c2w_homo)

                    # Here learn gt pose directly to test if PoseGrid successfully fits it
                    # loss = self.optimize_cam_in_batch(camera_tensor, gt_color, gt_depth, self.tracking_pixels, self.optim_quats_init, self.optim_trans_init)
                    loss = torch.abs(gt_camera_tensor.to(device)- camera_tensor).mean()
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    if cam_iter == 0:
                        initial_loss = loss

                    # loss_camera_tensor = torch.abs(
                    #     gt_camera_tensor.to(device)-camera_tensor.to(device)).mean().item()
                    loss_camera_tensor = loss.item()
                    
                    fixed_camera_tensor = camera_tensor.clone().detach()
                    if camera_tensor[0] < 0:
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
                                'gt quat w' : gt_camera_tensor[0],
                                'pred quat x' : camera_tensor[1], 
                                'gt quat x' : gt_camera_tensor[1],
                                'pred quat y' : camera_tensor[2], 
                                'gt quat y' : gt_camera_tensor[2],
                                'pred quat z' : camera_tensor[3], 
                                'gt quat z' : gt_camera_tensor[3],
                                'pred trans x' : camera_tensor[4],
                                'gt trans x' : gt_camera_tensor[4],
                                'pred trans y' : camera_tensor[5],
                                'gt trans y' : gt_camera_tensor[5],
                                'pred trans z' : camera_tensor[6],
                                'gt trans z' : gt_camera_tensor[6]
                                }
                            self.experiment.log(dict_log)
        
                    if loss < current_min_loss:
                        candidate_idx = cam_iter
                        current_min_loss = loss
                        candidate_cam_tensor = camera_tensor.to(device)
                        candidate_loss_camera_tensor = loss_camera_tensor

                        # store pred_trans & pred_quat
                        candidate_pred_trans, candidate_pred_quat = pred_trans, pred_quat

                        # if not self.use_last:
                        # # save the best parameters
                        #     candidate_transNet_para = self.transNet.state_dict()
                        #     candidate_quatsNet_para = self.quatsNet.state_dict()
                        if not self.fix_decoder:
                            candidate_decoder_para = self.PoseGrid_decoder.state_dict()
                        candidate_trans_enc_para_list = trans_enc_para_list
                        candidate_quat_enc_para_list = quat_enc_para_list
                        candidate_trans_prime_para_list = trans_prime_para_list
                        candidate_quat_prime_para_list = quat_prime_para_list

                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                    [1, 4])).type(torch.float32).to(self.device)
                print("candidate_cam_tensor:", candidate_cam_tensor)

                # if self.use_last:
                #     c2w = get_camera_from_tensor(
                #         camera_tensor.clone().detach())
                #     c2w = torch.cat([c2w, bottom], dim=0)
                # else:
                c2w = get_camera_from_tensor(
                    candidate_cam_tensor.clone().detach())
                c2w = torch.cat([c2w, bottom], dim=0)
            
                # self.transNet.load_state_dict(candidate_transNet_para)
                # self.quatsNet.load_state_dict(candidate_quatsNet_para)
                # del candidate_transNet_para
                # del candidate_quatsNet_para
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

            self.estimate_c2w_list[idx] = c2w.clone().cpu()
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
            pre_c2w = c2w.clone()
            self.idx[0] = idx

            # TODO: constant speed assumption after optimizing grid point encoding
            if self.const_speed_assumption and idx_enc_prev == idx_enc_next:
                with torch.no_grad():
                    pose_enc_now = self.posegrid[:-7, idx_enc_next]
                    prime_now = self.posegrid[-7:, idx_enc_next]
                    delta = prime_now * self.encoding_interval
                    delta_trans = delta[:3]
                    delta_quat = delta[3:]

                    # trans_now = candidate_pred_trans # this translation is bounded by the 4x4x4 box
                    quat_now = F.normalize(candidate_pred_quat, p=2, dim=-1)
                    quat_next = quat_now + delta_quat
                    quat_next = F.normalize(quat_next, p=2, dim=-1)

                    # NOTE: zero rotation (0, 0, 0) is mapped to (pi, pi/2, pi) for encoding, the following real world Euler angles are bounded by (+-pi, +-pi/2, +-pi)
                    theta_now, phi_now, gamma_now = txe.quat2euler(quat_now)
                    theta_next, phi_next, gamma_next = txe.quat2euler(quat_next)
                    delta_euler = torch.tensor([theta_next-theta_now, phi_next-phi_now, gamma_next-gamma_now])

                    # get a good initialization of the next encodings using pretrained B's
                    delta_transeuler = torch.cat([delta_trans, delta_euler], dim=-1)
                    B_list = [self.B_x, self.B_y, self.B_z, self.B_theta, self.B_phi, self.B_gamma]
                    pose_enc_new = []
                    n_dof = 6
                    for dof in range(n_dof):
                        pose_enc_dof = pose_enc_now[dof*self.hidden_dim:(dof+1)*self.hidden_dim]
                        pose_enc_new.append(self.rotate_by_B(pose_enc_dof, delta_transeuler[dof], B_list[dof]))
                    pose_enc_new = torch.cat(pose_enc_new, dim=-1)
                    prime_new = prime_now
                    enc_new = torch.cat([pose_enc_new, prime_new], dim=-1)
                    self.posegrid[:, idx_enc_next+1] = enc_new.clone()

            if self.low_gpu_mem:
                torch.cuda.empty_cache()
