import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from event_net import UNet, UNet_event, UNet_2heads

def preprocess(pil_img, scale, is_event, is_pil=True):
    if is_pil:
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_event else Image.BICUBIC)
        pil_img = pil_img.resize((newW, newH), resample=Image.BILINEAR if is_event else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)
    else:
        # if not PIL image, no scaling
        img_ndarray = pil_img

    if not is_event:
        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        img_ndarray = img_ndarray / 255
    # take only the G, B channels of events
    # the correspondence between channels and event polarities is quite messy right now. Better fix it...
    else:
        img_ndarray = img_ndarray.transpose((2, 0, 1))[1:] # [-, +]

    return img_ndarray

def predict_event(net,
                  img1,
                  img2,
                  device,
                  scale_factor=1,
                  out_threshold=0.5,
                  return_numpy=True):
    net.eval()
    img1 = preprocess(img1, scale_factor, is_event=False, is_pil=True)
    img2 = preprocess(img2, scale_factor, is_event=False, is_pil=True)
    img_pair = torch.from_numpy(np.concatenate([img1, img2], axis=0))
    img_pair = img_pair.unsqueeze(0)

    img_pair = img_pair.to(device=device, dtype=torch.float32)

    events_pred, masks_pred = net(img_pair)
    mask_binary = (masks_pred[:, 1][:, None, :, :] > out_threshold)[0] * 1
    events_pred_roi = (events_pred * mask_binary)[0]

    if return_numpy:
        full_mask = mask_binary.detach().cpu().squeeze().numpy()
        full_events = events_pred_roi.detach().cpu().squeeze().numpy().transpose((1, 2, 0))
    else:
        full_mask = mask_binary
        full_events = events_pred_roi.squeeze().permute(1, 2, 0)

    return full_events, full_mask

def inference_event(net,
                    img1,
                    img2,
                    device,
                    scale_factor=1,
                    out_threshold=0.5):
    net.eval()
    img1 = img1.permute(2, 0, 1)
    img2 = img2.permute(2, 0, 1)
    assert img1.shape == img2.shape, 'The sizes of the two input images are not the same!'
    if scale_factor != 1.0:
        c, h, w = img1.shape
        h_new, w_new = int(scale_factor * h), int(scale_factor * w)
        assert h_new > 0 and w_new > 0, 'Scale is too small, resized images would have no pixels'
        # transform = transforms.Resize((h_new, w_new), interpolation=transforms.InterpolationMode.BILINEAR)
        transform = transforms.Resize((h_new, w_new), interpolation=transforms.InterpolationMode.NEAREST)
        img1 = transform(img1)
        img2 = transform(img2)
    img_pair = torch.cat((img1, img2), dim=0)
    img_pair = img_pair.unsqueeze(0)
    img_pair = img_pair.to(device=device, dtype=torch.float32)

    events_pred, masks_pred = net(img_pair)
    mask_prob = masks_pred[:, 1][:, None, :, :]
    # mask_binary = (mask_prob > out_threshold)[0] * 1
    # events_pred_roi = (events_pred * mask_binary)[0]
    events_pred_roi = (events_pred * mask_prob)[0]

    # full_mask = mask_binary
    full_mask = masks_pred
    full_events = events_pred_roi.squeeze().permute(1, 2, 0)

    return full_events, full_mask

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input1', '-img1', metavar='INPUT1', nargs='+', help='Filenames of input images 1', required=True)
    parser.add_argument('--input2', '-img2', metavar='INPUT2', nargs='+', help='Filenames of input images 2', required=True)
    parser.add_argument('--event', '-e', metavar='EVENT', nargs='+', help='Filenames of output event images')
    parser.add_argument('--binary', '-b', metavar='BINARY', nargs='+', help='Filenames of output binary mask images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=True, help='Use bilinear upsampling')

    return parser.parse_args()

def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))

def event_to_image(event: np.ndarray):
    event_img = np.concatenate([(event * 50).clip(0, 255), np.zeros(event[:, :, 0][:, :, None].shape)], axis=-1).astype(np.uint8)
    return Image.fromarray(event_img)


if __name__ == '__main__':
    args = get_args()
    in_file1 = args.input1[0]
    in_file2 = args.input2[0]
    event_file = os.path.splitext(in_file1)[0] + '_' + os.path.splitext(in_file2)[0] + '_event.png'
    binary_file = os.path.splitext(in_file1)[0] + '_' + os.path.splitext(in_file2)[0] + '_binary.png'

    net = UNet_2heads(n_channels=6, n_classes1=2, n_classes2=2, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    logging.info(f'\nPredicting ...')
    img1 = Image.open(in_file1)
    img2 = Image.open(in_file2)

    print('prediction started!')

    img1 = np.asarray(img1).copy()
    img2 = np.asarray(img2).copy()
    img1 = torch.from_numpy(img1)
    img2 = torch.from_numpy(img2)
    event, mask = inference_event(net=net,
                                  img1=img1,
                                  img2=img2,
                                  scale_factor=args.scale,
                                  out_threshold=args.mask_threshold,
                                  device=device)
    event = event.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy().squeeze()
    print('prediction ended!')

    event_img = event_to_image(event)
    binary_img = mask_to_image(mask)
    event_img.save('event_prediction.png')
    logging.info(f'Event saved to {event_file}')
    binary_img.save('binary_mask_prediction.png')
    logging.info(f'Binary mask saved to {binary_file}')
