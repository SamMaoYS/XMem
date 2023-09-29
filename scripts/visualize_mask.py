import os
import argparse
import json
# import pandas as pd
# from lzstring import LZString
# from pycocotools import mask as mask_utils
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
# from decord import VideoReader
# from decord import cpu
from natsort import natsorted
from tqdm.auto import tqdm
import pandas as pd

def getIoU(gt_mask, pred_mask): 
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union

def getMidDist(gt_mask, pred_mask):

    gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    try:
        gt_bigc = max(gt_contours, key = cv2.contourArea)
        pred_bigc = max(pred_contours, key = cv2.contourArea)
    except:
        print('no contour')
        return -1

    gt_mid = gt_bigc.mean(axis=0)[0]
    pred_mid = pred_bigc.mean(axis=0)[0]

    return np.linalg.norm(gt_mid - pred_mid)

def getMidDistNorm(gt_mask, pred_mask):
    H, W = gt_mask.shape[:2]
    mdist = getMidDist(gt_mask, pred_mask)
    return mdist / np.sqrt(H**2 + W**2)

def getMidBinning(gt_mask, pred_mask, bin_size=5):

    H, W = gt_mask.shape
    gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    try:
        gt_bigc = max(gt_contours, key = cv2.contourArea)
        pred_bigc = max(pred_contours, key = cv2.contourArea)
    except:
        print('no contour')
        return -1

    gt_mid = gt_bigc.mean(axis=0)[0]
    pred_mid = pred_bigc.mean(axis=0)[0]

    # TODO: confirm x, y correspond to widht and height
    gt_x, gt_y = gt_mid.round()
    pred_x, pred_y = pred_mid.round()
    
    gt_bin_x, gt_bin_y = gt_x // bin_size, gt_y // bin_size
    pred_bin_x, pred_bin_y = pred_x // bin_size, pred_y // bin_size

    return (gt_bin_x == pred_bin_x) and (gt_bin_y == pred_bin_y)

def blend_mask(input_img, binary_mask, alpha=0.5):
    if input_img.ndim==2:
        return input_img
    
    mask_image = np.zeros(input_img.shape,np.uint8)
    mask_image[:,:,1] = 255
    mask_image = mask_image*np.repeat(binary_mask[:,:,np.newaxis],3,axis=2)

    blend_image = input_img[:,:,:]
    pos_idx = binary_mask>0
    for ind in range(input_img.ndim):
        ch_img1 = input_img[:,:,ind]
        ch_img2 = mask_image[:,:,ind]
        ch_img3 = blend_image[:,:,ind]
        ch_img3[pos_idx] = alpha*ch_img1[pos_idx] + (1-alpha)*ch_img2[pos_idx]
        blend_image[:,:,ind] = ch_img3
    return blend_image

def show_img(img):
    plt.figure(facecolor="white", figsize=(30, 10), dpi=100)
    plt.grid("off")
    plt.axis("off")
    plt.imshow(img)
    plt.show()

def save_img(img, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    Image.fromarray(img).save(output)

def main(args):
    with open(args.split_json, 'r') as fp:
        split = json.load(fp)
    
    if split is None:
        print('No split found')
        return
    if args.compute_stats:
        takes = split[args.split]
        df_list = []
        for take_id in tqdm(takes):
            df_i = process_take(take_id[0], args.input, args.pred, args.output, args.split, args.visualize)
            if df_i is not None:
                df_list.append(df_i)
        df = pd.concat(df_list, ignore_index=True)
        df.to_csv(os.path.join(args.output, f'{args.split}.csv'), index=False)
    else:
        df = pd.read_csv(os.path.join(args.output, f'{args.split}.csv'))

    take_mean_iou = df.groupby(['take'])['iou'].mean()
    object_mean_iou = df.groupby(['object'])['iou'].mean()
    mean_iou = df['iou'].mean()
    # mean_mid_dist = df[df['mid_dist'] != -1]['mid_dist'].mean()
    # mean_mid_dist_norm = df[df['mid_dist_norm'] != -1]['mid_dist_norm'].mean()
    # mean_mid_binning = df[df['mid_binning'] != -1]['mid_binning'].mean()

    print(f'Take Mean IoU: {take_mean_iou}')
    print(f'Object Mean IoU: {object_mean_iou}')
    print(f'Mean IoU: {mean_iou}')
    # print(f'Mean mid_dist: {mean_mid_dist}')
    # print(f'Mean mid_dist_norm: {mean_mid_dist_norm}')
    # print(f'Mean mid_binning: {mean_mid_binning}')


def process_take(take_id, input, pred, output, split, visualize=False):
    cameras = ['cam01', 'cam02', 'cam03', 'cam04']
    df_list = []
    for cam in cameras:
        if not os.path.isdir(os.path.join(input, split, take_id, cam)):
            continue
        input_cam_root = os.path.join(input, split, take_id, cam)
        objects = os.listdir(input_cam_root)
        for object_name in objects:
            rgb_root = os.path.join(input_cam_root, object_name, 'rgb')
            rgb_frames = natsorted(os.listdir(rgb_root))
            pred_mask_root = os.path.join(pred, take_id, cam, object_name)
            if not os.path.isdir(pred_mask_root):
                # import pdb; pdb.set_trace()
                continue
            pred_mask_frames = natsorted(os.listdir(pred_mask_root))
            gt_mask_root = os.path.join(input_cam_root, object_name, 'mask')
            gt_mask_frames = natsorted(os.listdir(gt_mask_root))
            frame_ids = np.intersect1d(rgb_frames, pred_mask_frames)
            count = 0
            for frame_id in tqdm(frame_ids):
                rgb = np.array(Image.open(os.path.join(rgb_root, frame_id)))
                mask = np.array(Image.open(os.path.join(pred_mask_root, frame_id))) / 255
                gt_mask = np.array(Image.open(os.path.join(gt_mask_root, frame_id))) / 255
                if np.sum(gt_mask == 1) == 0:
                    continue
                count += 1

                inner = np.logical_and(mask == 1, gt_mask == 1)
                outer = np.logical_or(mask == 1, gt_mask == 1)
                iou = np.sum(inner) / np.sum(outer)
                # mid_dist = getMidDist(gt_mask, mask)
                # mid_dist_norm = getMidDistNorm(gt_mask, mask)
                # mid_binning = getMidBinning(gt_mask, mask)
                # row = pd.DataFrame([[take_id, cam, object_name, frame_id, iou, mid_dist, mid_dist_norm, mid_binning]], columns=['take', 'camera', 'object', 'frame', 'iou', 'mid_dist', 'mid_dist_norm', 'mid_binning'])
                row = pd.DataFrame([[take_id, cam, object_name, frame_id, iou]], columns=['take', 'camera', 'object', 'frame', 'iou'])
                df_list.append(row)

                if visualize and count % 5 == 0:
                    blended_img = blend_mask(rgb, mask.astype(np.uint8), alpha=0.5)
                    # show_img(blended_img)
                    output_path = os.path.join(output, split, take_id, cam, object_name)
                    os.makedirs(output_path, exist_ok=True)
                    save_img(blended_img, os.path.join(output_path, frame_id))

    df = pd.concat(df_list, ignore_index=True) if len(df_list) > 0 else None
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='EgoExo take data root', default='../data/correspondence/dataset')
    parser.add_argument('--split_json', help='EgoExo take data root', default='../data/correspondence/split.json')
    parser.add_argument('--split', help='EgoExo take data root', default='val')
    parser.add_argument('--pred', help='EgoExo take data root', default='../output/E23_val/Annotations')
    parser.add_argument('--visualize', action='store_true', help='EgoExo take data root')
    parser.add_argument('--compute_stats', action='store_true', help='EgoExo take data root')
    parser.add_argument('--output', help='Output data root', default='../output/E23_val/visualization')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)

    main(args)