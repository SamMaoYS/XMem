import json
from tqdm import tqdm
from pycocotools import mask as mask_utils
import numpy as np


def remove_pad(img, orig_size):
    cur_H, cur_W = img.shape[:2]
    orig_H, orig_W = orig_size
    if orig_W > orig_H:
        ratio = 1.0 / orig_W * cur_W
    else:
        ratio = 1.0 / orig_H * cur_H
    new_H, new_W = int(orig_H * ratio), int(orig_W * ratio)
    if new_W > new_H:
        diff_H = (cur_H - new_H) // 2
        img = img[diff_H:-diff_H]
    else:
        diff_W = (cur_W - new_W) // 2
        img = img[:, diff_W:-diff_W]
    return img


def check_pred_format():
    from copy import deepcopy

    ROOT_DIR = "../SegSwap/meta_relations_data_final/"
    PRED_DIR = "../SegSwap/train/checkpoints/egoexo_480x480_final_weighted_nowarping_dice_mlp_hardneg/eval_ep199_test"

    annotations_file = f"{ROOT_DIR}/relations_objects_latest.json"
    with open(annotations_file, "r") as fp:
        gt_annotations = json.load(fp)["annotations"]

    # load split
    with open(f"{ROOT_DIR}/split.json", "r") as fp:
        splits = json.load(fp)

    videos = splits["test"]

    annotations = {"version": "xx", "challenge": "xx", "results": {}}
    for vid in tqdm.tqdm(videos):
        with open(f"{PRED_DIR}/{vid}/pred_annotations.json", "r") as fp:
            vid_anno = json.load(fp)

        correct_anno = deepcopy(vid_anno)
        for obj in vid_anno["masks"]:
            for cam in vid_anno["masks"][obj]:
                # exo_cam = cam.split('_')[-1]
                for frame_idx in vid_anno["masks"][obj][cam]:
                    mask = mask_utils.decode(
                        vid_anno["masks"][obj][cam][frame_idx]["pred_mask"]
                    )

                    if (
                        frame_idx
                        in gt_annotations[vid]["object_masks"][obj][cam.split("_")[-1]][
                            "annotation"
                        ]
                    ):
                        width = gt_annotations[vid]["object_masks"][obj][
                            cam.split("_")[-1]
                        ]["annotation"][frame_idx]["width"]
                        height = gt_annotations[vid]["object_masks"][obj][
                            cam.split("_")[-1]
                        ]["annotation"][frame_idx]["height"]
                        # gt_mask = gt_annotations[vid]['object_masks'][obj][cam]
                        mask = remove_pad(mask, orig_size=(height, width))

                    encoded_mask = mask_utils.encode(np.asfortranarray(mask))
                    encoded_mask["counts"] = encoded_mask["counts"].decode("ascii")
                    vid_anno["masks"][obj][cam][frame_idx]["pred_mask"] = encoded_mask

                # vid_anno['masks'][obj][cam]['4350']['pred_mask']
                #
                # del correct_anno['masks'][obj][cam]
                correct_anno["masks"][obj][cam] = vid_anno["masks"][obj][cam]

        annotations["results"][vid] = correct_anno

    with open(f"final_results_new.json", "w") as fp:
        json.dump(annotations, fp)
