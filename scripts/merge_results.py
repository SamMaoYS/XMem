import os
import json
from tqdm import tqdm
from pycocotools import mask as mask_utils
import numpy as np
import argparse


def check_pred_format(PRED_DIR):
    from copy import deepcopy

    ROOT_DIR = "/localscratch/yma50/XMem/data/correspondence"

    annotations_file = f"{ROOT_DIR}/relations_objects_latest.json"
    with open(annotations_file, "r") as fp:
        gt_annotations = json.load(fp)["annotations"]

    # load split
    with open(f"{ROOT_DIR}/split.json", "r") as fp:
        splits = json.load(fp)

    videos = splits["test"]

    annotations = {"version": "xx", "challenge": "xx", "results": {}}
    for vid in tqdm(videos):
        with open(f"{PRED_DIR}/{vid}/annotation.json", "r") as fp:
            vid_anno = json.load(fp)

        correct_anno = deepcopy(vid_anno)
        for obj in vid_anno["masks"]:
            for cam in vid_anno["masks"][obj]:
                # exo_cam = cam.split('_')[-1]
                for frame_idx in vid_anno["masks"][obj][cam]:
                    mask = mask_utils.decode(vid_anno["masks"][obj][cam][frame_idx])

                    # if (
                    #     frame_idx
                    #     in gt_annotations[vid]["object_masks"][obj][cam.split("_")[-1]][
                    #         "annotation"
                    #     ]
                    # ):
                    #     width = gt_annotations[vid]["object_masks"][obj][
                    #         cam.split("_")[-1]
                    #     ]["annotation"][frame_idx]["width"]
                    #     height = gt_annotations[vid]["object_masks"][obj][
                    #         cam.split("_")[-1]
                    #     ]["annotation"][frame_idx]["height"]
                    # gt_mask = gt_annotations[vid]['object_masks'][obj][cam]
                    # mask = utils.remove_pad(mask, orig_size=(height, width))

                    encoded_mask = mask_utils.encode(np.asfortranarray(mask))
                    encoded_mask["counts"] = encoded_mask["counts"].decode("ascii")
                    vid_anno["masks"][obj][cam][frame_idx] = encoded_mask

                # vid_anno['masks'][obj][cam]['4350']['pred_mask']
                #
                # del correct_anno['masks'][obj][cam]
                correct_anno["masks"][obj][cam] = vid_anno["masks"][obj][cam]

        annotations["results"][vid] = correct_anno

    output_dir = os.path.dirname(PRED_DIR)
    with open(f"{output_dir}/gt_results.json", "w") as fp:
        json.dump(annotations, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        help="The predicted path",
        required=True,
    )
    args = parser.parse_args()
    check_pred_format(args.pred_dir)
