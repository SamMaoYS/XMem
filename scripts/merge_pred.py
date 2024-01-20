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
        if not os.path.isdir(os.path.join(args.input, take_id)):
            continue

        result = process_take(take_id, args.input, args.pred)

        with open(os.path.join(args.pred, take_id, "annotations.json"), "w+") as fp:
            json.dump(result, fp)


def get_folders(path):
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def process_take(take_id, input, pred):
    annotation_path = os.path.join(input, take_id, "annotation.json")
    with open(annotation_path, "r") as fp:
        annotation = json.load(fp)
    subsample_idx = annotation["subsample_idx"]

    pred_masks = {}
    cam_names = get_folders(os.path.join(pred, take_id))
    for cams_str in cam_names:
        for object_name in get_folders(os.path.join(pred, take_id, cams_str)):
            pred_masks[object_name] = {}
            pred_masks[object_name][cams_str] = {}
            for f_name in subsample_idx:
                f_str = "{:06d}".format(int(int(f_name) / 30 + 1))

                pred_mask_path = os.path.join(
                    pred, take_id, cams_str, object_name, f_str + ".json"
                )
                with open(pred_mask_path, "r") as fp:
                    pred_mask_data = json.load(fp)
                pred_masks[object_name][cams_str][f_name] = pred_mask_data
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
