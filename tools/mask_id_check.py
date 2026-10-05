#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np


def inspect_folder(root: Path, folder_name: str):
    mask_dir = root / folder_name

    print()
    print("=" * 70)
    print(folder_name)
    print("=" * 70)

    if not mask_dir.exists():
        print(f"[ERROR] Directory does not exist:")
        print(f"  {mask_dir}")
        return

    files = sorted(mask_dir.glob("*.png"))

    if not files:
        print("[ERROR] No PNG files found.")
        return

    total_counts = {}
    failed = 0
    total_files = 0

    for path in files:
        mask = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED,
        )

        if mask is None:
            failed += 1
            continue

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        values, counts = np.unique(
            mask,
            return_counts=True,
        )

        for value, count in zip(values, counts):
            value = int(value)
            count = int(count)

            total_counts[value] = (
                total_counts.get(value, 0)
                + count
            )

        total_files += 1

    print(f"Directory       : {mask_dir}")
    print(f"PNG files       : {len(files)}")
    print(f"Successfully read: {total_files}")
    print(f"Failed          : {failed}")

    print()
    print("All pixel values:")
    for value in sorted(total_counts):
        print(
            f"  {value:6d} -> "
            f"{total_counts[value]:12d} pixels"
        )

    nonzero = {
        value: count
        for value, count in total_counts.items()
        if value > 0
    }

    print()
    print("Nonzero values:")
    print(
        " ",
        sorted(nonzero.keys())
    )

    print()
    print("Nonzero pixel counts:")
    for value in sorted(nonzero):
        print(
            f"  {value:6d} -> "
            f"{nonzero[value]:12d} pixels"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Summarize mask pixel values."
    )

    parser.add_argument(
        "dataset_path",
        type=str,
        help="Dataset folder",
    )

    args = parser.parse_args()

    root = Path(args.dataset_path)

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist:\n{root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Not a directory:\n{root}"
        )

    print()
    print("=" * 70)
    print("MASK SUMMARY")
    print("=" * 70)
    print(f"Dataset: {root}")

    inspect_folder(
        root,
        "instance_masks",
    )

    inspect_folder(
        root,
        "part_instance_masks",
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
