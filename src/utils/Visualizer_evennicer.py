import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.common import get_camera_from_tensor

# wandb
import wandb


class Visualizer(object):
    """
    Visualize intermediate results, render out depth, color and depth uncertainty images.
    It can be called per iteration, which is good for debugging (to see how each tracking/mapping iteration performs).

    """

    def __init__(self, freq, inside_freq, vis_dir, renderer, verbose, 
                 experiment, # wandb
                 device='cuda:0', 
                 stage='tracker'):
        self.freq = freq
        self.device = device
        self.vis_dir = vis_dir
        self.verbose = verbose
        self.renderer = renderer
        self.inside_freq = inside_freq
        os.makedirs(f'{vis_dir}', exist_ok=True)

        # wandb
        self.experiment = experiment
        self.stage = stage

    def vis_event(self, idx, iter, gt_depth, gt_color, gt_event, gt_event_lores, pred_event, 
                  gts_event_list, preds_event_list, 
                #   gt_mask, gt_mask_lores, pred_mask, 
                  c2w_or_camera_tensor, c,
                  decoders):
        """
        Visualization of depth, color, event images and save to file.
        """
        with torch.no_grad():
            if (idx % self.freq == 0) and (iter % self.inside_freq == 0):
                gt_depth_np = gt_depth.cpu().numpy()
                gt_color_np = gt_color.cpu().numpy()

                # gt_event_np = gt_event.cpu().numpy()
                gt_event_lores_np = gt_event_lores.cpu().numpy()
                pred_event_np = pred_event.cpu().numpy()

                # gt_event_img = (np.concatenate([gt_event_np, np.zeros_like(gt_event_np[:, :, 0][:, :, None])], axis=-1) * 50).clip(0, 255).astype(np.uint8)
                gt_event_lores_img = (np.concatenate([gt_event_lores_np, np.zeros_like(gt_event_lores_np[:, :, 0][:, :, None])], axis=-1) * 50).clip(0, 255).astype(np.uint8)
                pred_event_img = (np.concatenate([pred_event_np, np.zeros_like(pred_event_np[:, :, 0][:, :, None])], axis=-1) * 50).clip(0, 255).astype(np.uint8)

                # blurred event images
                gts_event_list_img = []
                preds_event_list_img = []
                for gt_event_blur, pred_event_blur in zip(gts_event_list, preds_event_list):
                    gt_event_blur_np = gt_event_blur.cpu().numpy()
                    pred_event_blur_np = pred_event_blur.cpu().numpy()
                    gts_event_list_img.append((np.concatenate([gt_event_blur_np, np.zeros_like(gt_event_blur_np[:, :, 0][:, :, None])], axis=-1) * 50).clip(0, 255).astype(np.uint8))
                    preds_event_list_img.append((np.concatenate([pred_event_blur_np, np.zeros_like(pred_event_blur_np[:, :, 0][:, :, None])], axis=-1) * 50).clip(0, 255).astype(np.uint8))

                # gt_mask_lores_np = gt_mask_lores.squeeze().cpu().numpy()
                # pred_mask_np = pred_mask.squeeze().cpu().numpy()
                # gt_mask_lores_img = gt_mask_lores_np.clip(0, 1).astype(np.float32)
                # pred_mask_img = pred_mask_np.clip(0, 1).astype(np.float32)

                if len(c2w_or_camera_tensor.shape) == 1:
                    bottom = torch.from_numpy(
                        np.array([0, 0, 0, 1.]).reshape([1, 4])).type(
                            torch.float32).to(self.device)
                    c2w = get_camera_from_tensor(
                        c2w_or_camera_tensor.clone().detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                else:
                    c2w = c2w_or_camera_tensor

                depth, uncertainty, color = self.renderer.render_img(
                    c,
                    decoders,
                    c2w,
                    self.device,
                    stage='color',
                    gt_depth=gt_depth)
                depth_np = depth.detach().cpu().numpy()
                color_np = color.detach().cpu().numpy()
                depth_residual = np.abs(gt_depth_np - depth_np)
                depth_residual[gt_depth_np == 0.0] = 0.0
                color_residual = np.abs(gt_color_np - color_np)
                color_residual[gt_depth_np == 0.0] = 0.0

                event_residual = np.abs(gt_event_lores_img - pred_event_img)

                # mask_residual = np.abs(gt_mask_lores_img - pred_mask_img)

                fig, axs = plt.subplots(3, 3)
                fig.tight_layout()
                max_depth = np.max(gt_depth_np)
                axs[0, 0].imshow(gt_depth_np, cmap="plasma",
                                 vmin=0, vmax=max_depth)
                axs[0, 0].set_title('Input Depth')
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                axs[0, 1].imshow(depth_np, cmap="plasma",
                                 vmin=0, vmax=max_depth)
                axs[0, 1].set_title('Generated Depth')
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])
                axs[0, 2].imshow(depth_residual, cmap="plasma",
                                 vmin=0, vmax=max_depth)
                axs[0, 2].set_title('Depth Residual')
                axs[0, 2].set_xticks([])
                axs[0, 2].set_yticks([])
                gt_color_np = np.clip(gt_color_np, 0, 1)
                color_np = np.clip(color_np, 0, 1)
                color_residual = np.clip(color_residual, 0, 1)
                axs[1, 0].imshow(gt_color_np, cmap="plasma")
                axs[1, 0].set_title('Input RGB')
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                axs[1, 1].imshow(color_np, cmap="plasma")
                axs[1, 1].set_title('Generated RGB')
                axs[1, 1].set_xticks([])
                axs[1, 1].set_yticks([])
                axs[1, 2].imshow(color_residual, cmap="plasma")
                axs[1, 2].set_title('RGB Residual')
                axs[1, 2].set_xticks([])
                axs[1, 2].set_yticks([])

                # axs[2, 0].imshow(gt_event_img)
                # axs[2, 0].set_title('GT Event')
                # axs[2, 0].set_xticks([])
                # axs[2, 0].set_yticks([])
                # axs[2, 1].imshow(gt_event_lores_img)
                # axs[2, 1].set_title('Lo-Res GT Event')
                # axs[2, 1].set_xticks([])
                # axs[2, 1].set_yticks([])
                # axs[2, 2].imshow(pred_event_img)
                # axs[2, 2].set_title('Generated Event')
                # axs[2, 2].set_xticks([])
                # axs[2, 2].set_yticks([])
                axs[2, 0].imshow(gt_event_lores_img)
                axs[2, 0].set_title('Lo-Res GT Event')
                axs[2, 0].set_xticks([])
                axs[2, 0].set_yticks([])
                axs[2, 1].imshow(pred_event_img)
                axs[2, 1].set_title('Generated Event')
                axs[2, 1].set_xticks([])
                axs[2, 1].set_yticks([])
                axs[2, 2].imshow(event_residual)
                axs[2, 2].set_title('Event Residual')
                axs[2, 2].set_xticks([])
                axs[2, 2].set_yticks([])

                # axs[3, 0].imshow(gt_mask_lores_img)
                # axs[3, 0].set_title('Lo-Res GT Mask')
                # axs[3, 0].set_xticks([])
                # axs[3, 0].set_yticks([])
                # axs[3, 1].imshow(pred_mask_img)
                # axs[3, 1].set_title('Generated Mask')
                # axs[3, 1].set_xticks([])
                # axs[3, 1].set_yticks([])
                # axs[3, 2].imshow(mask_residual)
                # axs[3, 2].set_title('Mask Residual')
                # axs[3, 2].set_xticks([])
                # axs[3, 2].set_yticks([])

                plt.subplots_adjust(wspace=0, hspace=0)
                plt.savefig(
                    f'{self.vis_dir}/{idx:05d}_{iter:04d}.jpg', bbox_inches='tight', pad_inches=0.2)
                plt.clf()

                if self.verbose:
                    print(
                        f'Saved rendering visualization of color/depth/event image at {self.vis_dir}/{idx:05d}_{iter:04d}.jpg')
                        # f'Saved rendering visualization of color/depth/event/mask image at {self.vis_dir}/{idx:05d}_{iter:04d}.jpg')

                    # wandb image logging
                    event_dict = {
                        f'Lo-Res GT Event ({self.stage})': wandb.Image(gt_event_lores_img),
                        f'Rendered Event ({self.stage})': wandb.Image(pred_event_img),
                        # f'Event Residual ({self.stage})': wandb.Image(event_residual),
                    }
                    for blur_level, (gt_event_blur, pred_event_blur) in enumerate(zip(gts_event_list_img, preds_event_list_img)):
                        event_dict[f'GT Event Blurred {blur_level+1} ({self.stage})'] = wandb.Image(gt_event_blur)
                        event_dict[f'Rendered Event {blur_level+1} ({self.stage})'] = wandb.Image(pred_event_blur)
                        # event_dict[f'Event Residual {blur_level+1} ({self.stage})'] = wandb.Image(np.abs(gt_event_blur - pred_event_blur))

                    self.experiment.log({
                        'Depth': {
                            'GT Depth': wandb.Image(gt_depth_np / max_depth),
                            f'Rendered Depth ({self.stage})': wandb.Image((depth_np / max_depth).clip(0, 1)),
                            f'Depth Residual ({self.stage})': wandb.Image((depth_residual / max_depth).clip(0, 1)),
                        },
                        'RGB': {
                            'GT RGB': wandb.Image(gt_color_np),
                            f'Rendered RGB ({self.stage})': wandb.Image(color_np),
                            f'RGB Residual ({self.stage})': wandb.Image(color_residual),
                        },
                        # 'Event': {
                        #     'Lo-Res GT Event': wandb.Image(gt_event_lores_img),
                        #     'Rendered Event': wandb.Image(pred_event_img),
                        #     'Event Residual': wandb.Image(event_residual),
                        # },
                        'Event': event_dict, 
                        # 'Mask': {
                        #     'Lo-Res GT Mask': wandb.Image(gt_mask_lores_img),
                        #     'Rendered Mask': wandb.Image(pred_mask_img),
                        #     'Mask Residual': wandb.Image(mask_residual),
                        # },
                        'Frame': idx # what if multiple visualizations for the same frame?
                    })

    def vis(self, idx, iter, gt_depth, gt_color, c2w_or_camera_tensor, c,
            decoders):
        """
        Visualization of depth, color images and save to file.

        Args:
            idx (int): current frame index.
            iter (int): the iteration number.
            gt_depth (tensor): ground truth depth image of the current frame.
            gt_color (tensor): ground truth color image of the current frame.
            c2w_or_camera_tensor (tensor): camera pose, represented in 
                camera to world matrix or quaternion and translation tensor.
            c (dicts): feature grids.
            decoders (nn.module): decoders.
        """
        with torch.no_grad():
            if (idx % self.freq == 0) and (iter % self.inside_freq == 0):
                gt_depth_np = gt_depth.cpu().numpy()
                gt_color_np = gt_color.cpu().numpy()
                if len(c2w_or_camera_tensor.shape) == 1:
                    bottom = torch.from_numpy(
                        np.array([0, 0, 0, 1.]).reshape([1, 4])).type(
                            torch.float32).to(self.device)
                    c2w = get_camera_from_tensor(
                        c2w_or_camera_tensor.clone().detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                else:
                    c2w = c2w_or_camera_tensor

                depth, uncertainty, color = self.renderer.render_img(
                    c,
                    decoders,
                    c2w,
                    self.device,
                    stage='color',
                    gt_depth=gt_depth)
                depth_np = depth.detach().cpu().numpy()
                color_np = color.detach().cpu().numpy()
                depth_residual = np.abs(gt_depth_np - depth_np)
                depth_residual[gt_depth_np == 0.0] = 0.0
                color_residual = np.abs(gt_color_np - color_np)
                color_residual[gt_depth_np == 0.0] = 0.0

                fig, axs = plt.subplots(2, 3)
                fig.tight_layout()
                max_depth = np.max(gt_depth_np)
                axs[0, 0].imshow(gt_depth_np, cmap="plasma",
                                 vmin=0, vmax=max_depth)
                axs[0, 0].set_title('Input Depth')
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                axs[0, 1].imshow(depth_np, cmap="plasma",
                                 vmin=0, vmax=max_depth)
                axs[0, 1].set_title('Generated Depth')
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])
                axs[0, 2].imshow(depth_residual, cmap="plasma",
                                 vmin=0, vmax=max_depth)
                axs[0, 2].set_title('Depth Residual')
                axs[0, 2].set_xticks([])
                axs[0, 2].set_yticks([])
                gt_color_np = np.clip(gt_color_np, 0, 1)
                color_np = np.clip(color_np, 0, 1)
                color_residual = np.clip(color_residual, 0, 1)
                axs[1, 0].imshow(gt_color_np, cmap="plasma")
                axs[1, 0].set_title('Input RGB')
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                axs[1, 1].imshow(color_np, cmap="plasma")
                axs[1, 1].set_title('Generated RGB')
                axs[1, 1].set_xticks([])
                axs[1, 1].set_yticks([])
                axs[1, 2].imshow(color_residual, cmap="plasma")
                axs[1, 2].set_title('RGB Residual')
                axs[1, 2].set_xticks([])
                axs[1, 2].set_yticks([])
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.savefig(
                    f'{self.vis_dir}/{idx:05d}_{iter:04d}.jpg', bbox_inches='tight', pad_inches=0.2)
                plt.clf()

                if self.verbose:
                    print(
                        f'Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}_{iter:04d}.jpg')

                    # wandb image logging
                    self.experiment.log({
                        'Depth': {
                            'GT Depth': wandb.Image(gt_depth_np / max_depth),
                            f'Rendered Depth ({self.stage})': wandb.Image((depth_np / max_depth).clip(0, 1)),
                            f'Depth Residual ({self.stage})': wandb.Image((depth_residual / max_depth).clip(0, 1)),
                        },
                        'RGB': {
                            'GT RGB': wandb.Image(gt_color_np),
                            f'Rendered RGB ({self.stage})': wandb.Image(color_np),
                            f'RGB Residual ({self.stage})': wandb.Image(color_residual),
                        },
                        'Frame': idx
                    })
