import os
import time
from datetime import datetime
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.common import get_camera_from_tensor
from src import config
from src.utils.datasets import get_dataset
# wandb
import wandb


def rgb_to_luma(rgb, esim=True):
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

    factors = torch.Tensor([r, g, b]).cpu()  # (3)
    luma = torch.sum(rgb * factors[None, :], axis=-1)  # (N_evs, 3) * (1, 3) => (N_evs)
    return luma[..., None]  # (N_evs, 1)

def lin_log(color, linlog_thres=20):
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

def inverse_lin_log(lin_log_rgb, linlog_thres=20):
    lin_slope = np.log(linlog_thres) / linlog_thres

    # Perform inverse linear mapping for values below linlog_thres
    inverse_lin_log_rgb = torch.where(
        lin_log_rgb < lin_slope * linlog_thres,
        lin_log_rgb / lin_slope,
        torch.exp(lin_log_rgb)
    )
    return inverse_lin_log_rgb

def main():
    # setup_seed(20)

    parser = argparse.ArgumentParser(
        description='Arguments for running EvenNICER-SLAM.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--event_folder', type=str,
                        help='event input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    args = parser.parse_args()

    cfg = config.load_config(
        args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')
    scale = cfg['scale']

    scene_name = cfg['data']['input_folder'].split('/')[-1]
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    experiment = wandb.init(config=cfg, project='EvenNICER-SLAM', group=scene_name, 
                                     name=f'{dt_string}', 
                                     settings=wandb.Settings(code_dir="."), dir=cfg['wandb_dir'],
                                     tags=[scene_name], 
                                     resume='allow', anonymous='must')
    
    frame_reader = get_dataset(cfg, args, scale, device = 'cpu')
    n_img = len(frame_reader)

    idx, gt_color, gt_depth, __, gt_mask, gt_c2w = frame_reader[0] 
    gt_gray = rgb_to_luma(gt_color, esim=True)
    gt_gray_np = (gt_gray*255).cpu().numpy()
    gt_loggray = lin_log(gt_gray*255, linlog_thres=20)
    gt_loggray_np = gt_loggray.cpu().numpy()
    experiment.log({
         'Gray': {'GT Gray' : wandb.Image(gt_gray_np)},
         'Gray_log' : {'GT LogGray' : wandb.Image(gt_loggray_np)},
    })

    idx, next_gt_color, _, gt_event, _, _ = frame_reader[1] 
    next_gt_gray = rgb_to_luma(next_gt_color, esim=True)
    next_gt_gray_np = (next_gt_gray*255).cpu().numpy()

    gt_pos = torch.unsqueeze(gt_event[:, :, 0] , dim = 2)
    gt_neg = torch.unsqueeze(gt_event[:, :, 1] , dim = 2)
    C_thres = 0.1  

    gt_loggray_events = gt_loggray -  gt_pos * C_thres + gt_neg * C_thres
    gt_inverse_loggray = inverse_lin_log(gt_loggray_events)
    gt_inverse_loggray_np = gt_inverse_loggray.cpu().numpy().clip(0, 255)

    gt_residual_np = np.abs(gt_inverse_loggray_np - next_gt_gray_np)

    gt_event_lores_np = gt_event.cpu().numpy()
    gt_event_lores_img = (np.concatenate([gt_event_lores_np, np.zeros_like(gt_event_lores_np[:, :, 0][:, :, None])], axis=-1)).clip(0, 1)
    experiment.log({
        'Gray' : {'GT Gray' : wandb.Image(next_gt_gray_np)},
        'Rendered Gray': {'Renderd Gray' : wandb.Image(gt_inverse_loggray_np)},
        'Event' : {'Event': wandb.Image(gt_event_lores_img)},
        'Residual' : {'Residual': wandb.Image(gt_residual_np)}
    })


if __name__ == '__main__':
    main()