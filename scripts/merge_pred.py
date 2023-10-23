import os
import argparse
import json

from tqdm.auto import tqdm


def main(args):
    split_data = None
    splits_path = args.split_json
    with open(splits_path, "r") as fp:
        split_data = json.load(fp)

    if split_data is None:
        print("No split found")
        return

    takes = [take_id for take_id in split_data[args.split]]
    for take_id in tqdm(takes):
        result = process_take(take_id, args.input, args.pred)

        if not os.path.isdir(os.path.join(args.pred, take_id)):
            result = {}
            os.makedirs(os.path.join(args.pred, take_id))

        with open(
            os.path.join(args.pred, take_id, "pred_annotations.json"), "w+"
        ) as fp:
            json.dump(result, fp)


def process_take(take_id, input, pred):
    annotation_path = os.path.join(input, take_id, "annotation.json")
    with open(annotation_path, "r") as fp:
        annotation = json.load(fp)
    masks = annotation["masks"]

    pred_masks = {}
    empty = True
    for object_name, cams in masks.items():
        pred_masks[object_name] = {}
        for cam_name, cam_data in cams.items():
            if not os.path.isdir(os.path.join(input, take_id, cam_name)):
                continue

            pred_masks[object_name][cam_name] = {}

            frames = list(cam_data.keys())
            for f_name in frames:
                f_str = "{:06d}".format(int(int(f_name) / 30 + 1))

                pred_mask_path = os.path.join(
                    pred, take_id, cam_name, object_name, f_str + ".json"
                )
                if not os.path.isfile(pred_mask_path):
                    continue

                with open(pred_mask_path, "r") as fp:
                    pred_mask_data = json.load(fp)
                pred_masks[object_name][cam_name][f_name] = pred_mask_data
                empty = False
    return {"masks": pred_masks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="EgoExo take data root",
        default="../data/correspondence",
    )
    parser.add_argument(
        "--split_json",
        help="EgoExo take data root",
        default="../data/correspondence/split.json",
    )
    parser.add_argument("--split", help="EgoExo take data root", default="val")
    parser.add_argument(
        "--pred", help="EgoExo take data root", default="../output/E23_val/Annotations"
    )
    args = parser.parse_args()

    main(args)
