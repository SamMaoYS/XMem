import argparse
import json
import tqdm
import cv2
import os
import numpy as np
from pycocotools import mask as mask_utils
import random

EVALMODE = "test"


def blend_mask(input_img, binary_mask, alpha=0.5):
    if input_img.ndim == 2:
        return input_img
    mask_image = np.zeros(input_img.shape, np.uint8)
    mask_image[:, :, 1] = 255
    mask_image = mask_image * np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
    blend_image = input_img[:, :, :].copy()
    pos_idx = binary_mask > 0
    for ind in range(input_img.ndim):
        ch_img1 = input_img[:, :, ind]
        ch_img2 = mask_image[:, :, ind]
        ch_img3 = blend_image[:, :, ind]
        ch_img3[pos_idx] = alpha * ch_img1[pos_idx] + (1 - alpha) * ch_img2[pos_idx]
        blend_image[:, :, ind] = ch_img3
    return blend_image


def upsample_mask(mask, frame):
    H, W = frame.shape[:2]
    mH, mW = mask.shape[:2]

    if W > H:
        ratio = mW / W
        h = H * ratio
        diff = int((mH - h) // 2)
        if diff == 0:
            mask = mask
        else:
            mask = mask[diff:-diff]
    else:
        ratio = mH / H
        w = W * ratio
        diff = int((mW - w) // 2)
        if diff == 0:
            mask = mask
        else:
            mask = mask[:, diff:-diff]

    mask = cv2.resize(mask, (W, H))
    return mask


def downsample(mask, frame):
    H, W = frame.shape[:2]
    mH, mW = mask.shape[:2]

    mask = cv2.resize(mask, (W, H))
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        help="the correspondence dataset path",
        required=True,
    )
    parser.add_argument(
        "--inference_path",
        help="The predicted path",
        required=True,
    )
    parser.add_argument(
        "--out_path", help="Output path to save the predictions", required=True
    )
    parser.add_argument(
        "--show_gt",
        help="if true, visualize anotations instead of predictions",
        action="store_true",
    )
    parser.add_argument("--reverse", help="if evaluating exo->ego", action="store_true")

    args = parser.parse_args()

    split_file = f"{args.datapath}/split.json"
    with open(split_file, "r") as f:
        split = json.load(f)
        test_ids = split[EVALMODE]

    random.seed(0)
    random.shuffle(test_ids)

    for take_id in tqdm.tqdm(test_ids[:30]):
        if take_id not in [
            "ac259bd8-f9a1-4456-99b3-610f80351c06",
            "e8cf53f3-a9e3-45b6-a313-10765ae183e2",
        ]:
            continue
        print(f"Processing take {take_id}")
        # Load the GT annotations
        gt_file = f"{args.datapath}/{take_id}/annotation.json"
        with open(gt_file, "r") as f:
            gt = json.load(f)
        # Load the predictions
        pred_file = f"{args.inference_path}/{take_id}/pred_annotations.json"
        with open(pred_file, "r") as f:
            pred = json.load(f)

        # breakpoint()
        obj = random.choice(list(pred["masks"].keys()))
        if obj not in ["basketball", "steel plate_0"]:
            continue
        while len(list(pred["masks"][obj].keys())) <= 0:
            obj = random.choice(list(pred["masks"].keys()))
        CAM = random.choice(list(pred["masks"][obj].keys()))
        if args.reverse:
            query_cam = "_".join(CAM.split("_")[:1])
            target_cam = "_".join(CAM.split("_")[1:])
        else:
            query_cam = "_".join(CAM.split("_")[:-1])
            target_cam = CAM.split("_")[-1]

        for frame_idx in gt["masks"][obj][query_cam].keys():
            idx = max(int(frame_idx) // 30 + 1, 0)
            frame = cv2.imread(f"{args.datapath}/{take_id}/{target_cam}/{idx:06d}.jpg")
            mask = mask_utils.decode(pred["masks"][obj][CAM][frame_idx]["pred_mask"])
            # breakpoint()
            try:
                mask = upsample_mask(mask, frame)
                out = blend_mask(frame, mask)
            except:
                breakpoint()

            os.makedirs(f"{args.out_path}/{take_id}_{target_cam}_{obj}", exist_ok=True)
            cv2.imwrite(
                f"{args.out_path}/{take_id}_{target_cam}_{obj}/{frame_idx}.jpg", out
            )

            # gt
            if args.show_gt:
                # target gt
                if frame_idx in gt["masks"][obj][target_cam]:
                    gt_mask = mask_utils.decode(gt["masks"][obj][target_cam][frame_idx])
                else:
                    gt_mask = np.zeros_like(frame)[..., 0]

                # gt_mask = upsample_mask(gt_mask, frame)
                gt_mask = downsample(gt_mask, frame)
                out = blend_mask(frame, gt_mask)

                os.makedirs(
                    f"{args.out_path}/{take_id}_{target_cam}_{obj}_gt", exist_ok=True
                )
                cv2.imwrite(
                    f"{args.out_path}/{take_id}_{target_cam}_{obj}_gt/{frame_idx}.jpg",
                    out,
                )

                # query gt
                frame = cv2.imread(
                    f"{args.datapath}/{take_id}/{query_cam}/{idx:06d}.jpg"
                )
                if frame_idx in gt["masks"][obj][query_cam]:
                    gt_mask = mask_utils.decode(gt["masks"][obj][query_cam][frame_idx])
                else:
                    gt_mask = np.zeros_like(frame)[..., 0]

                # gt_mask = upsample_mask(gt_mask, frame)
                gt_mask = downsample(gt_mask, frame)
                out = blend_mask(frame, gt_mask)

                os.makedirs(
                    f"{args.out_path}/{take_id}_{target_cam}_{obj}_gt_query",
                    exist_ok=True,
                )
                cv2.imwrite(
                    f"{args.out_path}/{take_id}_{target_cam}_{obj}_gt_query/{frame_idx}.jpg",
                    out,
                )
