import copy
import argparse
import cv2
import os
import time
import random

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.datasets import get_dataset
from src.common import (get_rays_from_uv, get_camera_from_tensor, get_tensor_from_camera)
from collections import Counter

import torch.nn.functional as F
import matplotlib.pyplot as plt

from src import config
from src.EvenNICER_SLAM import EvenNICER_SLAM
import time

def rgb_to_luma(rgb, esim=True):
    r = 0.299
    g = 0.587
    b = 0.114
    factors = torch.Tensor([r, g, b]).to("cuda:0")
    luma = torch.sum(rgb * factors[None, :], axis=-1)
    return luma[..., None]  # (N_evs, 1)

def lin_log(color, linlog_thres=20):
    lin_slope = np.log(linlog_thres) / linlog_thres
      # Peform linear-map for smaller thres, and log-mapping for above thresh
    lin_log_rgb = torch.where(color < linlog_thres, lin_slope * color, torch.log(color))
    return lin_log_rgb

def inverse_lin_log(lin_log_rgb, linlog_thres=20):
    lin_slope = np.log(linlog_thres) / linlog_thres
    inverse_lin_log_rgb = torch.where(
        lin_log_rgb < lin_slope * linlog_thres,
        lin_log_rgb / lin_slope,
        torch.exp(lin_log_rgb)
    )
    return inverse_lin_log_rgb

def rescale_tensor_image(gt_depth, gt_color, scale=4):
    H = gt_color.size(0)
    W = gt_color.size(1)
    H = H // scale
    W = W // scale
    print(gt_depth.shape)
    print(gt_color.shape)
    gt_depth = F.interpolate(gt_depth.unsqueeze(0).unsqueeze(0), (H, W)).squeeze()
    gt_color = F.interpolate(gt_color.permute(2, 0, 1).unsqueeze(0), (H, W)).squeeze().permute(1, 2, 0)
    gt_depth_np = gt_depth.cpu().numpy()
    gt_color_np = gt_color.cpu().numpy()
    print(gt_depth.shape)
    fig, axs = plt.subplots(1, 2)
    max_depth = np.max(gt_depth_np)
    axs[0].imshow(gt_depth_np, cmap="plasma", vmin=0, vmax=max_depth)
    axs[0].set_title('Input Depth')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    gt_color_np = np.clip(gt_color_np, 0, 1)
    axs[1].imshow(gt_color_np, cmap="plasma")
    axs[1].set_title('Input Color')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    plt.savefig('output.jpg', bbox_inches='tight', pad_inches=0.2)
    plt.clf()


def asynchronous_event_sampling_optimize(pre_gt_color,gt_event, gt_depth, camera_tensor):
        # 1. load events.txt per frame → change class "Replica_event(Replica)" in datasets.py
        # 2. count the number of events per pixel 
        # timestamp, x, y, polarity → x, y, +, - , 絶対値
        # 3. + と - の絶対値が一定値以上なら発火する→ event_lossの計算
        H = pre_gt_color.size(0) # y 
        new_H = H // 4
        W = pre_gt_color.size(1) # x
        new_W = W // 4
        gt_depth = F.interpolate(gt_depth.unsqueeze(0).unsqueeze(0), (new_H, new_W)).squeeze()
        pre_gt_color = F.interpolate(pre_gt_color.permute(2, 0, 1).unsqueeze(0), (new_H, new_W)).squeeze().permute(1, 2, 0)

        #pre_gt_color_array = pre_gt_color.cpu().numpy()
        gt_event_array = gt_event.cpu().numpy()
        #pre_gt_color_array = cv2.resize(pre_gt_color, (new_H, new_W))
        loss_events = 0
        
        events_array = np.zeros((new_H, new_W))
        count = 0
        for event in gt_event_array:
            i = int(event[0]) # x
            j = int(event[1]) # y
            events_array[j][i] += event[3]
            max_events = np.max(np.abs(events_array))
            if max_events >= 10 :
              #print(events_array[j][i])
              # print(i, j)
              #print(pre_gt_color.shape)
              pre_gt_color_pixel = pre_gt_color[j, i]
              #print(pre_gt_color_pixel.shape)
              #print(events_array[j, i])
              events_array[j][i] = 0
              # camera_tensor = # PoseNetから求める
              #ray_o, ray_d = get_rays_from_uv(i, j, camera_tensor, new_H, new_W, fx, fy, cx, cy, device) # TODO:rescale        
              #ret = self.renderer.render_batch_ray(
                   #self.c, self.decoders, ray_d, ray_o,  self.device, stage='color',  gt_depth=gt_depth[j, i])
                # _, event_uncertainty, rendered_color = ret
                # rendered_gray = self.rgb_to_luma(rendered_color, esim=True)

    
              pre_gray = rgb_to_luma(pre_gt_color_pixel, esim=True)
              if pre_gray > 0.5:
                  count += 1

              pre_loggray = lin_log(pre_gray*255, linlog_thres=20)

              C_thres = 0.1
              loggray_add_events = pre_loggray + events_array[j, i]*C_thres
              inverse_loggray = inverse_lin_log(loggray_add_events)

                
                # loss_event = torch.abs(inverse_loggray - rendered_gray*255).sum()
                # balancer = self.cfg['event']['balancer'] # coefficient to balance event loss and rgbd loss
                # loss_event = loss_event * balancer
                # loss_events += loss_event
        return count  # per frame


def main():
    parser = argparse.ArgumentParser(
        description='Arguments for running EvenNICER-SLAM.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--event_folder', type=str,
                        help='event input folder, this have higher priority, can overwrite the one in config file')
    
    args = parser.parse_args()
    cfg = config.load_config('/scratch_net/biwidl215/myamaguchi/EvenNICER-SLAM/configs/Replica/room0_test_asynchronous.yaml')
    # slam = EvenNICER_SLAM(cfg, args)
    frame_reader = get_dataset(
            cfg, args, scale=1)
    pbar = DataLoader(frame_reader, batch_size=1, shuffle=False, num_workers=0) 

    # TODO : change frame_loader() and remove gt_mask
    # TODO : framewise → asynchronous
    for idx, gt_color, gt_depth, gt_event, gt_c2w, evs_dict_xy, pos_evs_dict_xy, neg_evs_dict_xy in pbar:
      idx = idx[0]
      gt_depth = gt_depth[0]
      gt_color = gt_color[0]
      gt_event = gt_event[0]
      gt_c2w = gt_c2w[0]
      #evs_dict_xy = evs_dict_xy
    
      if idx == 2:
        gt_event_array = gt_event.cpu().numpy()
        #evs_dict = {key: np.array(value) for key, value in evs_dict_xy.items()}
        # pos_dict_xy = Counter(pos_evs_dict_xy)
        # neg_dict_xy = Counter(neg_evs_dict_xy)
        # merged_counter = pos_dict_xy + neg_dict_xy
        #evs_dict_xy = dict(merged_counter)
        xys_mtNevs = list(evs_dict_xy.keys())
        pos_xys_mtNevs = list(pos_evs_dict_xy.keys())
        neg_xys_mtNevs = list(neg_evs_dict_xy.keys())

        sampled_xys = random.sample(xys_mtNevs, k=10)
        # pos_sampled_xys = list(set(pos_xys_mtNevs) & set(sampled_xys))
        # neg_sampled_xys = list(set(neg_xys_mtNevs) & set(sampled_xys))

        num_pos_evs_at_xy = np.asarray([len(pos_evs_dict_xy.get(xy, [])) for xy in sampled_xys])
        num_neg_evs_at_xy = np.asarray([len(neg_evs_dict_xy.get(xy, [])) for xy in sampled_xys])
        evs_at_xy = num_pos_evs_at_xy*0.1 - num_neg_evs_at_xy*0.1
        evs_at_xy = torch.tensor(evs_at_xy).unsqueeze(1)
        print(evs_at_xy)

        loss = evs_at_xy
        zeros_tensor = torch.zeros_like(loss, dtype=torch.double)
        print(torch.where((loss < 0.5) & (-0.5 < loss), zeros_tensor, loss))

        #num_pos_evs_at_xy = np.asarray([len(pos_evs_dict_xy[xy]) for xy in pos_sampled_xys])
        #print(num_pos_evs_at_xy)
        #sampled_indices = random.sample(range(len(xys_mtNevs)), k=10)
        #sampled_xys = torch.tensor([xys_mtNevs[i] for i in sampled_indices])
        #num_neg_evs_at_xy = np.asarray([len(neg_evs_dict_xy[xy]) for xy in neg_sampled_xys])
        #print(num_neg_evs_at_xy)
        
        #print(evs_dict[sampled_xys[0]].shape)
        #print(evs_dict[0])
        #sampled_items = torch.tensor([items[i] for i in sampled_indices])
        #print(sampled_items)
        #sampled_indices = random.sample(range(len(xys_mtNevs)), k=10)
        #sampled_xys = torch.tensor([xys_mtNevs[i] for i in sampled_indices])

        print(sampled_xys)
        tensor = torch.tensor(sampled_xys).view(10, -1)
        print(tensor)
        i_tensor = tensor[:, 0].long()
        j_tensor = tensor[:, 1].long()
        gt_color = gt_color[j_tensor, i_tensor]
        print(gt_color)

      
        #sampled_num_evs_at_xy = [num_evs_at_xy[i] for i in sampled_indices]
        #print(sampled_num_evs_at_xy)
        xy_numEvs_Idx = np.concatenate((num_evs_at_xy[:, None], np.append(0, np.cumsum(num_evs_at_xy)[:-1])[:, None]), axis=1)
        cumnum_evs_at_xy = np.cumsum(num_evs_at_xy)
        idx_no_successor = cumnum_evs_at_xy - 1
        #print(idx_no_successor)
        #print(evs_dict_xy[(240.0, 38.0)])
        # for event in gt_event_array:
        #     i = int(event[0])
        #     j = int(event[1])
        #     event_value = float(event[2])
        #     time = torch.tensor(event_value*120, dtype=torch.double).unsqueeze(0)
        # print(time)
        # if idx % 5 == 1:
        #    gt_integrate = gt_event
        # else:
            # gt_integrate = torch.cat((gt_integrate, gt_event), dim = 0)
        #print(gt_integrate.shape)
      
      if idx == 13:
          break
    
    
      #pre_gt_color = gt_color
    #   if idx > 0:
    #      count = asynchronous_event_sampling_optimize(pre_gt_color, gt_event, gt_depth, gt_camera_tensor)
    #      print(count)
    #      # NOTE : 1frameで大体100~400くらいのpixelでevents>=10
    #      #print(events_array.shape)
    #      break
      # end_time = time.time()
      # execution_time = end_time - start_time
      # print(f"実行時間: {execution_time}秒")
      
      
  
if __name__ == '__main__':
  main()