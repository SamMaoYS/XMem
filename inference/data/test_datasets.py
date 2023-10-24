import os
from os import path
import json
import numpy as np
from natsort import natsorted
from inference.data.video_reader import VideoReader, EgoExoVideoReader


class LongTestDataset:
    def __init__(self, data_root, size=-1):
        self.image_dir = path.join(data_root, "JPEGImages")
        self.mask_dir = path.join(data_root, "Annotations")
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(
                video,
                path.join(self.image_dir, video),
                path.join(self.mask_dir, video),
                to_save=[
                    name[:-4] for name in os.listdir(path.join(self.mask_dir, video))
                ],
                size=self.size,
            )

    def __len__(self):
        return len(self.vid_list)


class DAVISTestDataset:
    def __init__(self, data_root, imset="2017/val.txt", size=-1):
        if size != 480:
            self.image_dir = path.join(data_root, "JPEGImages", "Full-Resolution")
            self.mask_dir = path.join(data_root, "Annotations", "Full-Resolution")
            if not path.exists(self.image_dir):
                print(f"{self.image_dir} not found. Look at other options.")
                self.image_dir = path.join(data_root, "JPEGImages", "1080p")
                self.mask_dir = path.join(data_root, "Annotations", "1080p")
            assert path.exists(self.image_dir), "path not found"
        else:
            self.image_dir = path.join(data_root, "JPEGImages", "480p")
            self.mask_dir = path.join(data_root, "Annotations", "480p")
        self.size_dir = path.join(data_root, "JPEGImages", "480p")
        self.size = size

        with open(path.join(data_root, "ImageSets", imset)) as f:
            self.vid_list = sorted([line.strip() for line in f])

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(
                video,
                path.join(self.image_dir, video),
                path.join(self.mask_dir, video),
                size=self.size,
                size_dir=path.join(self.size_dir, video),
            )

    def __len__(self):
        return len(self.vid_list)


class YouTubeVOSTestDataset:
    def __init__(self, data_root, split, size=480):
        self.image_dir = path.join(
            data_root, "all_frames", split + "_all_frames", "JPEGImages"
        )
        self.mask_dir = path.join(data_root, split, "Annotations")
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))
        self.req_frame_list = {}

        with open(path.join(data_root, split, "meta.json")) as f:
            # read meta.json to know which frame is required for evaluation
            meta = json.load(f)["videos"]

            for vid in self.vid_list:
                req_frames = []
                objects = meta[vid]["objects"]
                for value in objects.values():
                    req_frames.extend(value["frames"])

                req_frames = list(set(req_frames))
                self.req_frame_list[vid] = req_frames

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(
                video,
                path.join(self.image_dir, video),
                path.join(self.mask_dir, video),
                size=self.size,
                to_save=self.req_frame_list[video],
                use_all_mask=True,
            )

    def __len__(self):
        return len(self.vid_list)


class EgoExoTestDataset:
    def __init__(
        self, data_root, split, size=480, ego_cam_name="aria01_214-1", num_frames=8
    ):
        self.data_root = data_root
        self.req_frame_list = {}
        self.vid_list = []
        takes = natsorted(os.listdir(self.data_root))

        splits_path = os.path.join(self.data_root, "split.json")
        with open(splits_path, "r") as fp:
            split_data = json.load(fp)
        data_split = split_data[split]
        self.takes = [take_id for take_id in data_split if take_id in takes]

        for take_id in self.takes:
            annotation_path = os.path.join(self.data_root, take_id, "annotation.json")
            with open(annotation_path, "r") as fp:
                annotation = json.load(fp)
            masks = annotation["masks"]

            for object_name, cams in masks.items():
                for cam_name in list(cams.keys()):
                    if "aria" in cam_name:
                        ego_cam_name = cam_name
                if cams.get(ego_cam_name) is None:
                    continue

                ego_frames = list(cams[ego_cam_name].keys())
                for cam_name, cam_data in cams.items():
                    if not os.path.isdir(
                        os.path.join(self.data_root, take_id, cam_name)
                    ):
                        continue
                    exo_frames = list(cam_data.keys())
                    if cam_name == ego_cam_name:
                        continue

                    frames = np.intersect1d(ego_frames, exo_frames)
                    if len(frames) < num_frames:
                        continue

                    vid = path.join(take_id, ego_cam_name, cam_name, object_name)
                    self.req_frame_list[vid] = [None] * len(frames)
                    for i, f in enumerate(frames):
                        self.req_frame_list[vid][i] = path.join(
                            ego_cam_name, object_name, f
                        )
                    self.vid_list.append(vid)
        self.size = size

    def get_datasets(self):
        for video in self.vid_list:
            tmp = video.split("/")
            take = tmp[0]
            ego_cam_name = tmp[1]
            exo_cam_name = tmp[2]
            obj = tmp[-1]
            yield EgoExoVideoReader(
                os.path.join(take, exo_cam_name, obj),
                path.join(self.data_root, take, exo_cam_name, obj),
                path.join(self.data_root, take, ego_cam_name, obj),
                size=self.size,
                to_save=self.req_frame_list[video],
                use_all_mask=False,
            )

    def __len__(self):
        return len(self.vid_list)
