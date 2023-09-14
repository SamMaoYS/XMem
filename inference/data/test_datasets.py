import os
from os import path
import json
import numpy as np
from natsort import natsorted
from inference.data.video_reader import VideoReader, EgoExoVideoReader


class LongTestDataset:
    def __init__(self, data_root, size=-1):
        self.image_dir = path.join(data_root, 'JPEGImages')
        self.mask_dir = path.join(data_root, 'Annotations')
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                to_save = [
                    name[:-4] for name in os.listdir(path.join(self.mask_dir, video))
                ],
                size=self.size,
            )

    def __len__(self):
        return len(self.vid_list)


class DAVISTestDataset:
    def __init__(self, data_root, imset='2017/val.txt', size=-1):
        if size != 480:
            self.image_dir = path.join(data_root, 'JPEGImages', 'Full-Resolution')
            self.mask_dir = path.join(data_root, 'Annotations', 'Full-Resolution')
            if not path.exists(self.image_dir):
                print(f'{self.image_dir} not found. Look at other options.')
                self.image_dir = path.join(data_root, 'JPEGImages', '1080p')
                self.mask_dir = path.join(data_root, 'Annotations', '1080p')
            assert path.exists(self.image_dir), 'path not found'
        else:
            self.image_dir = path.join(data_root, 'JPEGImages', '480p')
            self.mask_dir = path.join(data_root, 'Annotations', '480p')
        self.size_dir = path.join(data_root, 'JPEGImages', '480p')
        self.size = size

        with open(path.join(data_root, 'ImageSets', imset)) as f:
            self.vid_list = sorted([line.strip() for line in f])

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
                size_dir=path.join(self.size_dir, video),
            )

    def __len__(self):
        return len(self.vid_list)


class YouTubeVOSTestDataset:
    def __init__(self, data_root, split, size=480):
        self.image_dir = path.join(data_root, 'all_frames', split+'_all_frames', 'JPEGImages')
        self.mask_dir = path.join(data_root, split, 'Annotations')
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))
        self.req_frame_list = {}

        with open(path.join(data_root, split, 'meta.json')) as f:
            # read meta.json to know which frame is required for evaluation
            meta = json.load(f)['videos']

            for vid in self.vid_list:
                req_frames = []
                objects = meta[vid]['objects']
                for value in objects.values():
                    req_frames.extend(value['frames'])

                req_frames = list(set(req_frames))
                self.req_frame_list[vid] = req_frames

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
                to_save=self.req_frame_list[video], 
                use_all_mask=True
            )

    def __len__(self):
        return len(self.vid_list)

class EgoExoTestDataset:
    def __init__(self, data_root, split, size=480, ego_cam_name='aria01_214-1', num_frames=8):
        self.data_root = data_root
        self.req_frame_list = {}
        self.vid_list = []
        self.ego_cam_name = ego_cam_name
        self.frame_folder = 'rgb'
        self.mask_folder = 'mask'
        self.split_root = path.join(data_root, split)
        self.takes = natsorted(os.listdir(self.split_root))
        exo_cam_names = ['cam01', 'cam02', 'cam03', 'cam04']
        for exo_cam_name in exo_cam_names:
            # Pre-filtering
            for take in self.takes:
                take_dir = path.join(self.split_root, take)
                ego_dir = path.join(take_dir, ego_cam_name)
                exo_dir = path.join(take_dir, exo_cam_name)
                if not os.path.isdir(exo_dir) or not os.path.isdir(ego_dir):
                    continue
                ego_objects = natsorted(os.listdir(ego_dir))
                exo_objects = natsorted(os.listdir(exo_dir))
                objects = np.intersect1d(ego_objects, exo_objects)
                for obj in objects:
                    ego_frames = natsorted(os.listdir(path.join(ego_dir, obj, self.mask_folder)))
                    exo_frames = natsorted(os.listdir(path.join(exo_dir, obj, self.mask_folder)))
                    frames = np.intersect1d(ego_frames, exo_frames)
                    if len(frames) < num_frames:
                        continue
                    vid = path.join(take, exo_cam_name, obj)
                    # self.req_frame_list[vid] = [None] * (len(frames) * 2)
                    self.req_frame_list[vid] = [None] * len(frames)
                    # self.frames[vid][0] = path.join(ego_dir, obj, self.frame_folder, frames[0])
                    for i, f in enumerate(frames):
                        # self.req_frame_list[vid][2*i] = path.join(ego_dir, obj, self.frame_folder, f)
                        # self.req_frame_list[vid][2*i+1] = path.join(exo_dir, obj, self.frame_folder, f)
                        self.req_frame_list[vid][i] = path.join(exo_dir, obj, self.frame_folder, f)

                    self.vid_list.append(vid)
        self.size = size

    def get_datasets(self):
        for video in self.vid_list:
            tmp = video.split('/')
            take = tmp[0]
            obj = tmp[-1]
            yield EgoExoVideoReader(video, 
                path.join(self.split_root, take, self.ego_cam_name, obj, self.frame_folder),
                path.join(self.split_root, video, self.frame_folder),
                size=self.size,
                to_save=self.req_frame_list[video], 
                use_all_mask=False
            )

    def __len__(self):
        return len(self.vid_list)
