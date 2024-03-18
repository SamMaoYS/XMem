import json
from tqdm import tqdm
from pycocotools import mask as mask_utils
import numpy as np


def check_pred_format():
    from copy import deepcopy

    ROOT_DIR = "/localscratch/yma50/XMem/data/correspondence"
    PRED_DIR = "/localscratch/yma50/results/egoexo_segswap_test_new/coco"

    annotations_file = f"{ROOT_DIR}/relations_objects_latest.json"
    with open(annotations_file, "r") as fp:
        gt_annotations = json.load(fp)["annotations"]

    # load split
    with open(f"{ROOT_DIR}/split.json", "r") as fp:
        splits = json.load(fp)

    videos = splits["test"]

    annotations = {"version": "xx", "challenge": "xx", "results": {}}
    for vid in tqdm(videos):
        with open(f"{PRED_DIR}/{vid}/annotations.json", "r") as fp:
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
                        # mask = utils.remove_pad(mask, orig_size=(height, width))

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
