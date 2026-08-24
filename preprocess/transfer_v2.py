#!/usr/bin/env python3

import os

# ============================================================
# Enable OpenEXR support
# Must be set before importing cv2
# ============================================================

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import json
from pathlib import Path

from tqdm import tqdm

import cv2
import numpy as np


# ============================================================
# Utility
# ============================================================

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def find_files(folder: Path, extensions):
    files = {}

    if not folder.exists():
        return files

    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue

        if p.suffix.lower() in extensions:
            files[p.stem] = p

    return files


def is_dataset_folder(path: Path):
    """
    Check whether a folder directly contains one dataset.

    Required:
        rgb/
        depth/
        instance_masks/
        pose/
    """

    required_dirs = [
        path / "rgb",
        path / "depth",
        path / "instance_masks",
        path / "pose",
    ]

    return all(p.exists() and p.is_dir() for p in required_dirs)


# ============================================================
# Depth
# ============================================================

def depth_exr_to_uint16_mm(depth):
    depth = np.asarray(depth)

    if depth.ndim != 2:
        raise ValueError(
            f"Expected single-channel depth, "
            f"got shape={depth.shape}"
        )

    if not np.issubdtype(depth.dtype, np.floating):
        raise ValueError(
            f"Expected floating-point EXR depth, "
            f"got dtype={depth.dtype}"
        )

    depth = depth.astype(np.float32)

    depth = np.nan_to_num(
        depth,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    depth_mm = depth * 1000.0

    depth_u16 = np.clip(
        np.round(depth_mm),
        0,
        65535,
    ).astype(np.uint16)

    return depth_u16


def load_depth_exr(path: Path):
    depth = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if depth is None:
        raise RuntimeError(
            f"Failed to read EXR depth:\n"
            f"{path}\n"
            f"OpenCV OpenEXR support may not be enabled."
        )

    return depth


# ============================================================
# Camera
# ============================================================

def get_intrinsic_from_json(pose_data):
    camera = pose_data.get("camera", {})
    intrinsics = camera.get("intrinsics", {})

    matrix = intrinsics.get("matrix", None)

    if matrix is None:
        raise ValueError(
            "pose JSON does not contain "
            "camera.intrinsics.matrix"
        )

    cam_K = np.array(
        matrix,
        dtype=np.float32,
    )

    if cam_K.shape != (3, 3):
        raise ValueError(
            f"Invalid intrinsic matrix shape: "
            f"{cam_K.shape}"
        )

    return cam_K


def get_object_pose(pose_data, instance_id):
    objects = pose_data.get("objects", [])

    if not objects:
        raise ValueError(
            "pose JSON contains no objects"
        )

    for obj in objects:

        obj_instance_id = int(
            obj.get("instance_id", -1)
        )

        if obj_instance_id != instance_id:
            continue

        if "T_camera_object_cv" not in obj:
            raise ValueError(
                f"Object instance_id="
                f"{instance_id} does not contain "
                f"T_camera_object_cv"
            )

        pose = np.array(
            obj["T_camera_object_cv"],
            dtype=np.float32,
        )

        if pose.shape != (4, 4):
            raise ValueError(
                f"Invalid pose shape: "
                f"{pose.shape}"
            )

        return (
            obj_instance_id,
            pose,
        )

    raise ValueError(
        f"instance_id={instance_id} "
        f"not found in pose JSON"
    )


# ============================================================
# Convert Single Frame
# ============================================================

def convert_frame(
    stem,
    rgb_path,
    depth_path,
    mask_path,
    pose_path,
    out_scene_dir,
    instance_id,
    output_stem,
):

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb = cv2.imread(
        str(rgb_path),
        cv2.IMREAD_COLOR,
    )

    if rgb is None:
        raise RuntimeError(
            f"Failed to read RGB:\n"
            f"{rgb_path}"
        )

    rgb_h, rgb_w = rgb.shape[:2]

    # --------------------------------------------------------
    # Depth
    # --------------------------------------------------------

    depth_raw = load_depth_exr(
        depth_path
    )

    depth_u16 = depth_exr_to_uint16_mm(
        depth_raw
    )

    depth_h, depth_w = depth_u16.shape[:2]

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_UNCHANGED,
    )

    if mask is None:
        raise RuntimeError(
            f"Failed to read mask:\n"
            f"{mask_path}"
        )

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask = mask.astype(np.uint16)

    mask_h, mask_w = mask.shape[:2]

    # --------------------------------------------------------
    # Size validation
    # --------------------------------------------------------

    expected_size = (
        rgb_h,
        rgb_w,
    )

    if (depth_h, depth_w) != expected_size:
        raise ValueError(
            f"RGB / Depth size mismatch:\n"
            f"RGB   = {(rgb_h, rgb_w)}\n"
            f"Depth = {(depth_h, depth_w)}"
        )

    if (mask_h, mask_w) != expected_size:
        raise ValueError(
            f"RGB / Mask size mismatch:\n"
            f"RGB  = {(rgb_h, rgb_w)}\n"
            f"Mask = {(mask_h, mask_w)}"
        )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    label = np.zeros_like(
        mask,
        dtype=np.uint16,
    )

    label[mask > 0] = instance_id

    object_pixels = np.count_nonzero(
        mask > 0
    )

    if object_pixels == 0:
        raise ValueError(
            f"Mask contains no object pixels:\n"
            f"{mask_path}"
        )

    # --------------------------------------------------------
    # Pose
    # --------------------------------------------------------

    pose_data = load_json(
        pose_path
    )

    cam_K = get_intrinsic_from_json(
        pose_data
    )

    actual_instance_id, object_pose = get_object_pose(
        pose_data,
        instance_id,
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    color_out = (
        out_scene_dir
        / f"{output_stem}_color.jpg"
    )

    depth_out = (
        out_scene_dir
        / f"{output_stem}_depth.png"
    )

    label_out = (
        out_scene_dir
        / f"{output_stem}_label.png"
    )

    meta_out = (
        out_scene_dir
        / f"{output_stem}_meta.json"
    )

    # --------------------------------------------------------
    # Write RGB
    # --------------------------------------------------------

    if not cv2.imwrite(
        str(color_out),
        rgb,
    ):
        raise RuntimeError(
            f"Failed to write:\n"
            f"{color_out}"
        )

    # --------------------------------------------------------
    # Write Depth
    # --------------------------------------------------------

    if not cv2.imwrite(
        str(depth_out),
        depth_u16,
    ):
        raise RuntimeError(
            f"Failed to write:\n"
            f"{depth_out}"
        )

    # --------------------------------------------------------
    # Write Label
    # --------------------------------------------------------

    if not cv2.imwrite(
        str(label_out),
        label,
    ):
        raise RuntimeError(
            f"Failed to write:\n"
            f"{label_out}"
        )

    # --------------------------------------------------------
    # Write Meta
    # --------------------------------------------------------

    meta = {
        "objects": [
            int(actual_instance_id)
        ],
        "object_poses": {
            str(actual_instance_id):
                object_pose.tolist()
        },
        "intrinsic": cam_K.tolist(),
        "distortion": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    }

    with open(
        meta_out,
        "w",
    ) as f:
        json.dump(
            meta,
            f,
            indent=2,
        )

    return True


# ============================================================
# Convert One Dataset
# ============================================================

def convert_single_dataset(
    input_root,
    out_scene_dir,
    instance_id,
    dataset_name,
):

    input_root = Path(input_root)

    rgb_dir = input_root / "rgb"
    depth_dir = input_root / "depth"
    mask_dir = input_root / "instance_masks"
    pose_dir = input_root / "pose"

    # --------------------------------------------------------
    # Validate directories
    # --------------------------------------------------------

    for directory in [
        rgb_dir,
        depth_dir,
        mask_dir,
        pose_dir,
    ]:

        if not directory.exists():
            raise FileNotFoundError(
                f"Required directory does not exist:\n"
                f"{directory}"
            )

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    rgb_files = find_files(
        rgb_dir,
        {
            ".jpg",
            ".jpeg",
            ".png",
        },
    )

    depth_files = find_files(
        depth_dir,
        {
            ".exr",
        },
    )

    mask_files = find_files(
        mask_dir,
        {
            ".png",
        },
    )

    pose_files = find_files(
        pose_dir,
        {
            ".json",
        },
    )

    converted = 0
    skipped = 0

    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

    for stem in tqdm(
        sorted(rgb_files.keys()),
        desc=f"[{dataset_name}]",
        leave=False,
    ):

        rgb_path = rgb_files[stem]
        depth_path = depth_files.get(stem)
        mask_path = mask_files.get(stem)
        pose_path = pose_files.get(stem)

        missing = []

        if depth_path is None:
            missing.append("depth")

        if mask_path is None:
            missing.append("mask")

        if pose_path is None:
            missing.append("pose")

        if missing:
            print()
            print(
                f"[SKIP] "
                f"{dataset_name}/{stem}: "
                f"missing {', '.join(missing)}"
            )

            skipped += 1
            continue

        # ----------------------------------------------------
        # Add dataset name to avoid filename collision
        # ----------------------------------------------------

        if dataset_name:
            output_stem = (
                f"{dataset_name}_{stem}"
            )
        else:
            output_stem = stem

        try:

            convert_frame(
                stem=stem,
                rgb_path=rgb_path,
                depth_path=depth_path,
                mask_path=mask_path,
                pose_path=pose_path,
                out_scene_dir=out_scene_dir,
                instance_id=instance_id,
                output_stem=output_stem,
            )

            converted += 1

        except Exception as e:

            print()
            print(
                f"[ERROR] "
                f"{dataset_name}/{stem}: {e}"
            )

            skipped += 1

    return {
        "dataset": dataset_name,
        "rgb": len(rgb_files),
        "converted": converted,
        "skipped": skipped,
    }


# ============================================================
# Find Dataset Folders
# ============================================================

def find_dataset_folders(input_root):

    input_root = Path(input_root)

    # --------------------------------------------------------
    # Case 1:
    #
    # input_root itself is a dataset
    #
    # input_root/
    #   rgb/
    #   depth/
    #   instance_masks/
    #   pose/
    # --------------------------------------------------------

    if is_dataset_folder(input_root):
        return [input_root]

    # --------------------------------------------------------
    # Case 2:
    #
    # input_root contains multiple datasets
    #
    # input_root/
    #   dataset_001/
    #   dataset_002/
    #   dataset_003/
    # --------------------------------------------------------

    dataset_dirs = []

    for p in sorted(input_root.iterdir()):

        if not p.is_dir():
            continue

        if is_dataset_folder(p):
            dataset_dirs.append(p)

    return dataset_dirs


# ============================================================
# Convert Dataset Collection
# ============================================================

def convert_dataset(
    input_root,
    output_root,
    scene_name,
    instance_id,
):

    input_root = Path(input_root)
    output_root = Path(output_root)

    if not input_root.exists():
        raise FileNotFoundError(
            f"Input path does not exist:\n"
            f"{input_root}"
        )

    # --------------------------------------------------------
    # Output:
    #
    # output/data/scene_name/data/
    # --------------------------------------------------------

    out_scene_dir = (
        output_root
        / "data"
        / scene_name
        / "data"
    )

    ensure_dir(
        out_scene_dir
    )

    # --------------------------------------------------------
    # Find all datasets
    # --------------------------------------------------------

    dataset_dirs = find_dataset_folders(
        input_root
    )

    if not dataset_dirs:

        raise RuntimeError(
            f"No valid dataset folders found under:\n"
            f"{input_root}\n\n"
            f"Expected either:\n\n"
            f"1. A single dataset:\n"
            f"   rgb/\n"
            f"   depth/\n"
            f"   instance_masks/\n"
            f"   pose/\n\n"
            f"2. Or multiple datasets:\n"
            f"   dataset_001/\n"
            f"     rgb/\n"
            f"     depth/\n"
            f"     instance_masks/\n"
            f"     pose/\n"
        )

    print()
    print("=" * 70)
    print("Dataset Conversion")
    print("=" * 70)
    print(
        f"Input      : {input_root}"
    )
    print(
        f"Datasets   : {len(dataset_dirs)}"
    )
    print(
        f"Output     : {out_scene_dir}"
    )
    print("=" * 70)

    total_rgb = 0
    total_converted = 0
    total_skipped = 0

    results = []

    # --------------------------------------------------------
    # Process all datasets
    # --------------------------------------------------------

    for dataset_dir in dataset_dirs:

        # ----------------------------------------------------
        # For single dataset mode, do NOT add its parent name
        # to keep original filenames unchanged.
        #
        # For multiple dataset mode, use folder name to
        # prevent filename collisions.
        # ----------------------------------------------------

        if len(dataset_dirs) == 1:
            dataset_name = ""
        else:
            dataset_name = dataset_dir.name

        display_name = (
            dataset_dir.name
            if dataset_name
            else dataset_dir.name
        )

        print()
        print(
            f"[DATASET] {display_name}"
        )

        result = convert_single_dataset(
            input_root=dataset_dir,
            out_scene_dir=out_scene_dir,
            instance_id=instance_id,
            dataset_name=dataset_name,
        )

        results.append(result)

        total_rgb += result["rgb"]
        total_converted += result["converted"]
        total_skipped += result["skipped"]

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Conversion finished")
    print("=" * 70)

    for result in results:

        print(
            f"{result['dataset'] or '[single dataset]':30s} "
            f"RGB={result['rgb']:6d} "
            f"Converted={result['converted']:6d} "
            f"Skipped={result['skipped']:6d}"
        )

    print("-" * 70)

    print(
        f"{'TOTAL':30s} "
        f"RGB={total_rgb:6d} "
        f"Converted={total_converted:6d} "
        f"Skipped={total_skipped:6d}"
    )

    print()
    print(
        f"Output: {out_scene_dir}"
    )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    # ========================================================
    # KEEP ORIGINAL ARGUMENT NAMES
    # ========================================================

    parser.add_argument(
        "--bop_data_path",
        type=str,
        required=True,
        help=(
            "Input dataset path. "
            "Can be either a single dataset folder "
            "or a root folder containing multiple dataset folders. "
            "Each dataset folder must contain "
            "rgb/, depth/, instance_masks/, pose/"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--obj_id",
        type=int,
        default=0,
        help=(
            "Compatibility with original "
            "BOP converter. "
            "0 = use instance_id 1. "
            "Otherwise use specified instance_id."
        ),
    )

    parser.add_argument(
        "--scene_name",
        type=str,
        default="000000",
    )

    args = parser.parse_args()

    # ========================================================
    # Instance ID
    # ========================================================

    if args.obj_id == 0:
        instance_id = 1
    else:
        instance_id = args.obj_id

    # ========================================================
    # Convert
    # ========================================================

    convert_dataset(
        input_root=Path(
            args.bop_data_path
        ),
        output_root=Path(
            args.output_dir
        ),
        scene_name=args.scene_name,
        instance_id=instance_id,
    )


if __name__ == "__main__":
    main()