#!/usr/bin/env python3
import argparse
import random
from pathlib import Path


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--converted_root",
        type=str,
        required=True,
        help="transfered dataset path",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="train ratio when splitting frames, should be between 0 and 1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    args = parser.parse_args()

    root = Path(args.converted_root)
    data_root = root / "data"

    split_ratio = float(args.split_ratio)

    if not (0.0 < split_ratio < 1.0):
        raise ValueError("split_ratio must be between 0 and 1")

    all_frames = []

    # 收集所有 frame
    for scene_dir in sorted(data_root.iterdir()):
        if not scene_dir.is_dir():
            continue

        scene_name = scene_dir.name
        inner_dir = scene_dir / "data"

        if not inner_dir.exists():
            continue

        for img_path in sorted(inner_dir.glob("*_color.jpg")):
            stem = img_path.name.replace("_color.jpg", "")
            global_id = f"{scene_name}/data/{stem}"
            all_frames.append(global_id)

    if len(all_frames) == 0:
        raise RuntimeError("empty frame")

    # shuffle and split by frames
    random.seed(args.seed)
    random.shuffle(all_frames)

    # split by frames
    split_idx = int(len(all_frames) * split_ratio)

    train_lines = all_frames[:split_idx]
    test_lines = all_frames[split_idx:]

    # export  dataset_config
    config_dir = root / "dataset_config"
    ensure_dir(config_dir)

    train_txt = config_dir / "train_data_list.txt"
    test_txt = config_dir / "test_data_list.txt"

    with open(train_txt, "w") as f:
        f.write("\n".join(train_lines))

    with open(test_txt, "w") as f:
        f.write("\n".join(test_lines))

    print(f"Total frames : {len(all_frames)}")
    print(f"Train frames : {len(train_lines)}")
    print(f"Test frames  : {len(test_lines)}")
    print(f"Split ratio  : {split_ratio}")
    print(f"Seed         : {args.seed}")

    print(f"\nSaved:")
    print(train_txt)
    print(test_txt)


if __name__ == "__main__":
    main()