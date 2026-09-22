#!/usr/bin/env python3

import os
import sys
import cv2
import json
import random
import argparse
import numpy as np

sys.path.append("../../")

from dataset.dataset import DTTDDataset
from utils.visualizer import visualize


def parse():
    parser = argparse.ArgumentParser(
        description="Visualize GT poses only. No model loading or inference."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="dataset root directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./eval_results",
        help="output directory",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="save GT visualization images",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="number of random test samples; 0 = all",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    return parser.parse_args()


def load_image(root, prefix, frame_id):
    path = os.path.join(root, prefix, f"{frame_id}_color.jpg")
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read RGB image: {path}")
    return img


def load_meta(root, prefix, frame_id):
    path = os.path.join(root, prefix, f"{frame_id}_meta.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Meta file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def validate_pose(meta, frame_id):
    if "objects" not in meta:
        raise ValueError(f"{frame_id}: meta.json has no 'objects'")
    if "object_poses" not in meta:
        raise ValueError(f"{frame_id}: meta.json has no 'object_poses'")

    objects = meta["objects"]
    object_poses = meta["object_poses"]

    if not isinstance(objects, list):
        raise ValueError(f"{frame_id}: 'objects' is not a list")
    if not isinstance(object_poses, dict):
        raise ValueError(f"{frame_id}: 'object_poses' is not a dict")

    return objects, object_poses


def main():
    opt = parse()

    dataset_root = os.path.abspath(opt.dataset_root)
    output_dir = os.path.abspath(opt.output)

    if opt.visualize:
        visualize_dir = os.path.join(output_dir, "visualize")
        os.makedirs(visualize_dir, exist_ok=True)

    print("=" * 70)
    print("GT POSE VISUALIZATION")
    print("=" * 70)
    print(f"Dataset root    : {dataset_root}")
    print(f"Output          : {output_dir}")
    print("Model loading   : DISABLED")
    print("Model inference : DISABLED")
    print("GT pose only    : ENABLED")
    print(f"Random samples  : {opt.num_samples}")
    print(f"Random seed     : {opt.seed}")
    print("=" * 70)

    # Dataset configuration is taken from dataset_root.
    dataset = DTTDDataset(
        root=dataset_root,
        mode="test",
        add_noise=False,
        config_path=os.path.join(dataset_root, "dataset_config"),
    )

    all_testlist = dataset.all_data_dirs

    # Randomly select test frames.
    random.seed(opt.seed)

    if opt.num_samples <= 0 or len(all_testlist) <= opt.num_samples:
        testlist = all_testlist
    else:
        testlist = random.sample(all_testlist, opt.num_samples)

    print(f"Total test frames : {len(all_testlist)}")
    print(f"Selected frames   : {len(testlist)}")
    print()

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
        (123, 10, 265),
        (245, 163, 101),
        (100, 100, 178),
    ]

    valid_frames = 0
    failed_frames = 0

    for i, frame_id in enumerate(testlist):
        try:
            print(
                f"\rProcessing {i + 1}/{len(testlist)} "
                f"frame={frame_id}",
                end="",
                flush=True,
            )

            img = load_image(
                dataset_root,
                dataset.prefix,
                frame_id,
            )

            meta = load_meta(
                dataset_root,
                dataset.prefix,
                frame_id,
            )

            objects, object_poses = validate_pose(
                meta,
                frame_id,
            )

            if opt.visualize:
                vis_img = img.copy()
                intrinsics = np.asarray(
                    meta["intrinsic"],
                    dtype=np.float32,
                )

                for idx, itemid in enumerate(objects):
                    itemid = int(itemid)
                    pose_key = str(itemid)

                    if pose_key not in object_poses:
                        print(
                            f"\n[ERROR] frame={frame_id}: "
                            f"object_id={itemid} not in object_poses"
                        )
                        continue

                    pose = np.asarray(
                        object_poses[pose_key],
                        dtype=np.float32,
                    )

                    if pose.shape != (4, 4):
                        print(
                            f"\n[ERROR] frame={frame_id}: "
                            f"object_id={itemid}, "
                            f"pose shape={pose.shape}, expected (4, 4)"
                        )
                        continue

                    if itemid not in dataset.model_points:
                        print(
                            f"\n[ERROR] frame={frame_id}: "
                            f"model points not found for object_id={itemid}"
                        )
                        continue

                    R = pose[:3, :3]
                    T = pose[:3, 3].reshape(1, 3)
                    model_pts = np.asarray(
                        dataset.model_points[itemid],
                        dtype=np.float32,
                    )

                    vis_img = visualize(
                        img=vis_img,
                        model_pts=model_pts,
                        R=R,
                        T=T,
                        intrinsics=intrinsics,
                        color=colors[idx % len(colors)],
                    )

                output_path = os.path.join(
                    visualize_dir,
                    f"{i:04d}.png",
                )
                cv2.imwrite(output_path, vis_img)

            valid_frames += 1

        except Exception as e:
            failed_frames += 1
            print(f"\n[ERROR] frame={frame_id}: {e}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total test frames : {len(all_testlist)}")
    print(f"Selected frames   : {len(testlist)}")
    print(f"Valid frames      : {valid_frames}")
    print(f"Failed            : {failed_frames}")

    if opt.visualize:
        print(f"Visualization     : {visualize_dir}")

    print("=" * 70)


if __name__ == "__main__":
    main()