import copy
import argparse
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.datasets import get_dataset

from src import config
from src.EvenNICER_SLAM import EvenNICER_SLAM

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
    cfg = config.load_config('configs/Replica/room0.yaml')
    # slam = EvenNICER_SLAM(cfg, args)
    frame_reader = get_dataset(
            cfg, args, scale=1)
    pbar = DataLoader(frame_reader, batch_size=1, shuffle=False, num_workers=0) 

    # TODO : change frame_loader() and remove gt_mask
    # TODO : framewise → asynchronous
    for idx, gt_color, gt_depth, gt_event, gt_c2w in pbar:
      idx = idx[0]
      gt_depth = gt_depth[0]
      gt_color = gt_color[0]
      gt_event = gt_event[0]
      gt_c2w = gt_c2w[0]

      gt_event 
      
  
if __name__ == '__main__':
  main()