#!/usr/bin/env python3
"""Merge two datasets with identical dataset_config/objectids.csv."""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================
# Utilities
# ============================================================

def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# objectids.csv
# ============================================================

def get_objectids_path(dataset_dir: Path) -> Path:
    return dataset_dir / "dataset_config" / "objectids.csv"


def read_objectids(dataset_dir: Path) -> bytes:
    path = get_objectids_path(dataset_dir)
    if not path.exists():
        raise RuntimeError(f"Missing objectids.csv:\n{path}")
    return path.read_bytes()


def check_objectids(dataset_a: Path, dataset_b: Path) -> None:
    obj_a = read_objectids(dataset_a)
    obj_b = read_objectids(dataset_b)

    if obj_a != obj_b:
        raise RuntimeError(
            "\n[ERROR] objectids.csv are NOT identical!\n"
            f"Dataset A: {get_objectids_path(dataset_a)}\n"
            f"Dataset B: {get_objectids_path(dataset_b)}\n"
        )

    print("[OK] objectids.csv are identical.")


# ============================================================
# Get scenes
# ============================================================

def get_scenes(dataset_dir: Path) -> List[Path]:
    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        raise RuntimeError(f"Missing data directory:\n{data_dir}")

    scenes = [p for p in data_dir.iterdir() if p.is_dir()]

    def sort_key(p: Path) -> Tuple[int, str | int]:
        try:
            return (0, int(p.name))
        except ValueError:
            return (1, p.name)

    scenes.sort(key=sort_key)
    return scenes


# ============================================================
# Copy objects
# ============================================================

def copy_objects(src_dataset: Path, output_dataset: Path) -> None:
    src = src_dataset / "objects"
    dst = output_dataset / "objects"

    if not src.exists():
        print(f"[WARNING] objects directory does not exist:\n{src}")
        return

    if dst.exists():
        print(f"[INFO] objects already exists, skipping:\n{dst}")
        return

    print(f"[INFO] Copy objects:\n    {src}\n -> {dst}")
    shutil.copytree(src, dst)


# ============================================================
# Copy objectids.csv
# ============================================================

def copy_objectids(src_dataset: Path, output_dataset: Path) -> None:
    src = get_objectids_path(src_dataset)
    dst_dir = output_dataset / "dataset_config"
    dst = dst_dir / "objectids.csv"

    ensure_dir(dst_dir)
    shutil.copy2(src, dst)
    print(f"[INFO] Copied objectids.csv:\n    {src}\n -> {dst}")


# ============================================================
# Build scene mapping
# ============================================================

def build_scene_mapping(
    scenes_a: List[Path],
    scenes_b: List[Path],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    mapping_a = {}
    mapping_b = {}
    next_id = 0

    # Dataset A
    for scene in scenes_a:
        mapping_a[scene.name] = f"{next_id:06d}"
        next_id += 1

    # Dataset B
    for scene in scenes_b:
        mapping_b[scene.name] = f"{next_id:06d}"
        next_id += 1

    return mapping_a, mapping_b


# ============================================================
# Copy scenes
# ============================================================

def copy_scenes(
    scenes: List[Path],
    mapping: Dict[str, str],
    output_data_dir: Path,
    dataset_name: str,
) -> None:
    for scene in scenes:
        new_name = mapping[scene.name]
        dst = output_data_dir / new_name

        if dst.exists():
            raise RuntimeError(f"Destination scene already exists:\n{dst}")

        print(f"[{dataset_name}] {scene.name} -> {new_name}")
        shutil.copytree(scene, dst)


# ============================================================
# Convert train/test list
# ============================================================

def convert_list(
    src_dataset: Path,
    mapping: Dict[str, str],
    list_name: str,
) -> List[str]:
    src = src_dataset / "dataset_config" / list_name

    if not src.exists():
        print(f"[INFO] {list_name} does not exist in:\n    {src_dataset}")
        return []

    lines = read_lines(src)
    output_lines = []

    for line in lines:
        parts = line.split("/")
        if not parts:
            continue

        old_scene = parts[0]
        if old_scene not in mapping:
            raise RuntimeError(
                f"\n[ERROR] Cannot find scene mapping\n"
                f"Dataset : {src_dataset}\n"
                f"List    : {list_name}\n"
                f"Line    : {line}\n"
                f"Scene   : {old_scene}"
            )

        parts[0] = mapping[old_scene]
        output_lines.append("/".join(parts))

    return output_lines


# ============================================================
# Write list
# ============================================================

def write_list(path: Path, lines: List[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ============================================================
# Merge
# ============================================================

def merge_datasets(
    dataset_a_path: str,
    dataset_b_path: str,
    output_dataset_path: str,
) -> None:
    dataset_a = Path(dataset_a_path)
    dataset_b = Path(dataset_b_path)
    output_dataset = Path(output_dataset_path)

    print("\n==============================================")
    print("MERGE DATASETS")
    print("==============================================")
    print(f"Dataset A : {dataset_a}")
    print(f"Dataset B : {dataset_b}")
    print(f"Output    : {output_dataset}")
    print("==============================================\n")

    # Check input
    if not dataset_a.exists():
        raise RuntimeError(f"Dataset A does not exist:\n{dataset_a}")

    if not dataset_b.exists():
        raise RuntimeError(f"Dataset B does not exist:\n{dataset_b}")

    if output_dataset.exists():
        raise RuntimeError(
            f"Output directory already exists:\n{output_dataset}\n\n"
            "Please remove it first or choose another output."
        )

    # Check objectids
    check_objectids(dataset_a, dataset_b)

    # Get scenes
    scenes_a = get_scenes(dataset_a)
    scenes_b = get_scenes(dataset_b)

    print(f"[INFO] Dataset A scenes: {len(scenes_a)}")
    print(f"[INFO] Dataset B scenes: {len(scenes_b)}")

    # Create output directories
    ensure_dir(output_dataset)
    output_data_dir = output_dataset / "data"
    output_config_dir = output_dataset / "dataset_config"
    ensure_dir(output_data_dir)
    ensure_dir(output_config_dir)

    # Scene mapping
    mapping_a, mapping_b = build_scene_mapping(scenes_a, scenes_b)

    # Copy Dataset A
    print("\n[INFO] Copy Dataset A...")
    copy_scenes(scenes_a, mapping_a, output_data_dir, "A")

    # Copy Dataset B
    print("\n[INFO] Copy Dataset B...")
    copy_scenes(scenes_b, mapping_b, output_data_dir, "B")

    # Copy objects & objectids.csv
    print()
    copy_objects(dataset_a, output_dataset)
    print()
    copy_objectids(dataset_a, output_dataset)

    # Merge train/test lists
    for list_name in ["train_data_list.txt", "test_data_list.txt"]:
        print(f"\n[INFO] Merge {list_name}")

        lines_a = convert_list(dataset_a, mapping_a, list_name)
        lines_b = convert_list(dataset_b, mapping_b, list_name)
        merged_lines = lines_a + lines_b

        output_path = output_config_dir / list_name
        write_list(output_path, merged_lines)

        print(f"    Dataset A : {len(lines_a)}")
        print(f"    Dataset B : {len(lines_b)}")
        print(f"    Total     : {len(merged_lines)}")

    # Summary
    total_scenes = len(scenes_a) + len(scenes_b)
    print("\n==============================================")
    print("MERGE COMPLETE")
    print("==============================================")
    print(f"Dataset A scenes : {len(scenes_a)}")
    print(f"Dataset B scenes : {len(scenes_b)}")
    print(f"Total scenes     : {total_scenes}")
    print(f"Output           : {output_dataset}")
    print("==============================================")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge two datasets with identical dataset_config/objectids.csv"
    )

    parser.add_argument("--dataset-a", required=True, help="First dataset")
    parser.add_argument("--dataset-b", required=True, help="Second dataset")
    parser.add_argument("--output", required=True, help="Merged dataset output directory")

    args = parser.parse_args()

    merge_datasets(
        dataset_a_path=args.dataset_a,
        dataset_b_path=args.dataset_b,
        output_dataset_path=args.output,
    )


if __name__ == "__main__":
    main()