import os
from os import path, replace

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from natsort import natsorted
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed


class VOSDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, egoexo_root, ego_cam_name, max_jump, is_bl, subset=None, num_frames=3, max_num_obj=1, finetune=False, augmentation=False):
        self.takes = sorted(os.listdir(egoexo_root))
        self.egoexo_root = egoexo_root
        self.ego_cam_name = ego_cam_name
        self.frame_folder = 'rgb'
        self.mask_folder = 'mask'
        self.max_jump = max_jump
        self.is_bl = is_bl
        self.num_frames = num_frames
        self.max_num_obj = max_num_obj
        self.augmentation = augmentation

        self.videos = []
        self.frames = {}

        exo_cam_names = ['cam01', 'cam02', 'cam03', 'cam04']

        for exo_cam_name in exo_cam_names:
            # Pre-filtering
            for take in self.takes:
                take_dir = path.join(egoexo_root, take)
                ego_dir = path.join(take_dir, ego_cam_name)
                exo_dir = path.join(take_dir, exo_cam_name)
                if not os.path.isdir(exo_dir) or not os.path.isdir(ego_dir):
                    continue
                ego_objects = natsorted(os.listdir(ego_dir))
                exo_objects = natsorted(os.listdir(exo_dir))
                objects = np.intersect1d(ego_objects, exo_objects)
                for obj in objects:
                    if subset is not None:
                        if obj not in subset:
                            continue
                    ego_frames = natsorted(os.listdir(path.join(ego_dir, obj, self.mask_folder)))
                    exo_frames = natsorted(os.listdir(path.join(exo_dir, obj, self.mask_folder)))
                    frames = np.intersect1d(ego_frames, exo_frames)
                    if len(frames) < num_frames:
                        continue
                    vid = path.join(take, exo_cam_name, obj)
                    # self.frames[vid] = [None] * (len(frames) * 2)
                    self.frames[vid] = [None] * (len(frames))
                    # self.frames[vid][0] = path.join(ego_dir, obj, self.frame_folder, frames[0])
                    for i, f in enumerate(frames):
                        self.frames[vid][i] = path.join(exo_dir, obj, self.frame_folder, f)

                    self.videos.append(vid)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(self.takes), path.join(egoexo_root, take, exo_cam_name)))

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune or self.is_bl else 15, shear=0 if finetune or self.is_bl else 10, interpolation=InterpolationMode.BILINEAR, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune or self.is_bl else 15, shear=0 if finetune or self.is_bl else 10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        if self.is_bl:
            # Use a different cropping scheme for the blender dataset because the image size is different
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.BILINEAR)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.NEAREST)
            ])
        else:
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.BILINEAR)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
            ])

        # Final transform without randomness
        self.final_gt_transform = transforms.Compose([
            transforms.Resize((384, 384)),
        ])

        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((384, 384)),
            im_normalization,
        ])

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video

        frames = self.frames[video]

        trials = 0
        while trials < 5:
            info['frames'] = [] # Appended with actual frames

            num_frames = self.num_frames
            length = len(frames)
            this_max_jump = min(len(frames), self.max_jump)

            # iterative sampling
            frames_idx = [np.random.randint(length)]
            acceptable_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1))).difference(set(frames_idx))
            while(len(frames_idx) < num_frames):
                idx = np.random.choice(list(acceptable_set))
                frames_idx.append(idx)
                new_set = set(range(max(0, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1)))
                acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))

            frames_idx = natsorted(frames_idx)
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_objects = []
            for f_idx in frames_idx:
                rgb_name = frames[f_idx]
                gt_name = frames[f_idx].replace(self.frame_folder, self.mask_folder)
                info['frames'].append(rgb_name)

                this_im = Image.open(rgb_name).convert('RGB')
                if self.augmentation:
                    reseed(sequence_seed)
                    this_im = self.all_im_dual_transform(this_im)
                    this_im = self.all_im_lone_transform(this_im)
                    reseed(sequence_seed)
                this_gt = Image.open(gt_name).convert('P')
                if self.augmentation:
                    this_gt = self.all_gt_dual_transform(this_gt)

                    pairwise_seed = np.random.randint(2147483647)
                    reseed(pairwise_seed)
                    this_im = self.pair_im_dual_transform(this_im)
                    this_im = self.pair_im_lone_transform(this_im)
                    reseed(pairwise_seed)
                    this_gt = self.pair_gt_dual_transform(this_gt)
                else:
                    this_gt = self.final_gt_transform(this_gt)
                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels!=0]

            if self.is_bl:
                # Find large enough labels
                good_lables = []
                for l in labels:
                    pixel_sum = (masks[0]==l).sum()
                    if pixel_sum > 10*10:
                        # OK if the object is always this small
                        # Not OK if it is actually much bigger
                        if pixel_sum > 30*30:
                            good_lables.append(l)
                        elif max((masks[1]==l).sum(), (masks[2]==l).sum()) < 20*20:
                            good_lables.append(l)
                labels = np.array(good_lables, dtype=np.uint8)
            
            if len(labels) == 0:
                target_objects = []
                trials += 1
            else:
                target_objects = labels.tolist()
                break

        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)

        info['num_objects'] = max(1, len(target_objects))

        masks = np.stack(masks, 0)

        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.num_frames, 384, 384), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, 384, 384), dtype=np.int64)
        for i, l in enumerate(target_objects):
            this_mask = (masks==l)
            cls_gt[this_mask] = i+1
            first_frame_gt[0,i] = (this_mask[0])
        cls_gt = np.expand_dims(cls_gt, 1)

        # 1 if object exist, 0 otherwise
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        selector = torch.FloatTensor(selector)

        data = {
            'rgb': images,
            'first_frame_gt': first_frame_gt,
            'cls_gt': cls_gt,
            'selector': selector,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos)