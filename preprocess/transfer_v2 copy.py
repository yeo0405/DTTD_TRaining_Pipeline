#!/usr/bin/env python3

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "0")

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
        part_instance_masks/
        pose/
    """

    required_dirs = [
        path / "rgb",
        path / "depth",
        path / "instance_masks",
        path / "part_instance_masks",
        path / "pose",
    ]

    return all(
        p.exists() and p.is_dir()
        for p in required_dirs
    )


# ============================================================
# Mask
# ============================================================

def load_mask(path: Path):
    """
    Load instance mask.

    Expected:
        0 = background
        1, 2, 3, ... = instance IDs

    Returns:
        uint16 single-channel mask
    """

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if mask is None:
        raise RuntimeError(
            f"Failed to read mask:\n"
            f"{path}"
        )

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return mask.astype(np.uint16)


def get_mask_area(mask, instance_id=None):
    """
    Count visible object pixels.

    If instance_id is None:
        count all object pixels.

    Otherwise:
        count pixels belonging to that instance.
    """

    if instance_id is None:
        return int(
            np.count_nonzero(mask > 0)
        )

    return int(
        np.count_nonzero(
            mask == instance_id
        )
    )


# ============================================================
# Depth
# ============================================================

def depth_png_to_uint16_mm(depth):
    """
    Load PNG depth.

    PNG depth is expected to already be uint16
    with units of millimeters.
    """

    depth = np.asarray(depth)

    if depth.ndim != 2:
        raise ValueError(
            f"Expected single-channel PNG depth, "
            f"got shape={depth.shape}"
        )

    if depth.dtype != np.uint16:
        raise ValueError(
            f"Expected uint16 PNG depth in millimeters, "
            f"got dtype={depth.dtype}"
        )

    return depth


def load_depth(path: Path):
    """
    Load PNG depth.

    PNG:
        uint16 millimeters
        -> unchanged

    Returns:
        uint16 depth in millimeters
    """

    suffix = path.suffix.lower()

    if suffix != ".png":
        raise ValueError(
            f"Unsupported depth format: "
            f"{path.suffix}\n"
            f"Supported format: .png"
        )

    depth = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if depth is None:
        raise RuntimeError(
            f"Failed to read depth:\n"
            f"{path}"
        )

    return depth_png_to_uint16_mm(depth)


# ============================================================
# Camera
# ============================================================

def get_intrinsic_from_json(pose_data):

    camera = pose_data.get(
        "camera",
        {}
    )

    intrinsics = camera.get(
        "intrinsics",
        {}
    )

    matrix = intrinsics.get(
        "matrix",
        None
    )

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


# ============================================================
# Pose
# ============================================================

def get_all_object_poses(pose_data):
    """
    Read all objects from the original pose JSON.

    Returns:
        {
            instance_id: 4x4 pose
        }
    """

    objects = pose_data.get(
        "objects",
        []
    )

    if not objects:
        raise ValueError(
            "pose JSON contains no objects"
        )

    object_poses = {}

    for obj in objects:

        if "instance_id" not in obj:
            raise ValueError(
                "Object does not contain instance_id"
            )

        instance_id = int(
            obj["instance_id"]
        )

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
                f"Invalid pose shape for "
                f"instance_id={instance_id}: "
                f"{pose.shape}"
            )

        if not np.all(
            np.isfinite(pose)
        ):
            raise ValueError(
                f"Pose contains NaN/Inf for "
                f"instance_id={instance_id}"
            )

        if instance_id in object_poses:
            raise ValueError(
                f"Duplicate instance_id="
                f"{instance_id} in pose JSON"
            )

        object_poses[
            instance_id
        ] = pose

    return object_poses


# ============================================================
# Convert Single Frame
# ============================================================

def convert_frame(
    stem,
    rgb_path,
    depth_path,
    instance_mask_path,
    part_instance_mask_path,
    pose_path,
    out_scene_dir,
    requested_instance_id,
    instance_mask_ids,
    part_instance_mask_ids,
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

    depth_u16 = load_depth(
        depth_path
    )

    depth_h, depth_w = depth_u16.shape[:2]

    # --------------------------------------------------------
    # Instance Mask
    # --------------------------------------------------------

    instance_mask = load_mask(
        instance_mask_path
    )

    instance_mask_h, instance_mask_w = (
        instance_mask.shape[:2]
    )

    # --------------------------------------------------------
    # Part Instance Mask
    # --------------------------------------------------------

    part_instance_mask = load_mask(
        part_instance_mask_path
    )

    part_mask_h, part_mask_w = (
        part_instance_mask.shape[:2]
    )

    # --------------------------------------------------------
    # Size validation
    # --------------------------------------------------------

    expected_size = (
        rgb_h,
        rgb_w,
    )

    if (
        depth_h,
        depth_w,
    ) != expected_size:

        raise ValueError(
            f"RGB / Depth size mismatch:\n"
            f"RGB   = {(rgb_h, rgb_w)}\n"
            f"Depth = {(depth_h, depth_w)}"
        )

    if (
        instance_mask_h,
        instance_mask_w,
    ) != expected_size:

        raise ValueError(
            f"RGB / Instance Mask size mismatch:\n"
            f"RGB  = {(rgb_h, rgb_w)}\n"
            f"Mask = {(instance_mask_h, instance_mask_w)}"
        )

    if (
        part_mask_h,
        part_mask_w,
    ) != expected_size:

        raise ValueError(
            f"RGB / Part Instance Mask size mismatch:\n"
            f"RGB       = {(rgb_h, rgb_w)}\n"
            f"Part Mask = {(part_mask_h, part_mask_w)}"
        )

    # --------------------------------------------------------
    # Pose JSON
    # --------------------------------------------------------

    pose_data = load_json(
        pose_path
    )

    cam_K = get_intrinsic_from_json(
        pose_data
    )

    all_object_poses = get_all_object_poses(
        pose_data
    )

    pose_instance_ids = sorted(
        all_object_poses.keys()
    )

    # --------------------------------------------------------
    # Determine requested instances
    # --------------------------------------------------------

    if requested_instance_id == 0:

        selected_instance_ids = (
            pose_instance_ids
        )

    else:

        if (
            requested_instance_id
            not in all_object_poses
        ):

            raise ValueError(
                f"instance_id="
                f"{requested_instance_id} "
                f"not found in pose JSON. "
                f"Available instances: "
                f"{pose_instance_ids}"
            )

        selected_instance_ids = [
            requested_instance_id
        ]

    # --------------------------------------------------------
    # Build output label
    #
    # Only VISIBLE selected objects are included.
    #
    # object 1:
    #     instance_masks == 1
    #
    # object 2:
    #     part_instance_masks > 0
    #
    # If an object is completely invisible:
    #     simply ignore it.
    #
    # If ALL selected objects are invisible:
    #     skip the frame.
    # --------------------------------------------------------

    label = np.zeros(
        (rgb_h, rgb_w),
        dtype=np.uint16,
    )

    visible_instance_ids = []

    for instance_id in selected_instance_ids:

        # ----------------------------------------------------
        # Determine mask source
        # ----------------------------------------------------

        if instance_id in instance_mask_ids:

            object_mask = (
                instance_mask == instance_id
            )

            mask_source = "instance_masks"

        elif instance_id in part_instance_mask_ids:

            # ------------------------------------------------
            # part_instance_masks:
            #
            # 0 = background
            # >0 = this object's parts
            #
            # All nonzero pixels belong to this object.
            # ------------------------------------------------

            object_mask = (
                part_instance_mask > 0
            )

            mask_source = "part_instance_masks"

        else:

            raise ValueError(
                f"instance_id={instance_id} "
                f"has no mask source configured. "
                f"instance_mask_ids="
                f"{instance_mask_ids}, "
                f"part_instance_mask_ids="
                f"{part_instance_mask_ids}"
            )

        # ----------------------------------------------------
        # Check visibility
        # ----------------------------------------------------

        instance_area = int(
            np.count_nonzero(
                object_mask
            )
        )

        # ----------------------------------------------------
        # Object is NOT visible
        #
        # Do not skip the frame.
        # Simply ignore this object.
        # ----------------------------------------------------

        if instance_area == 0:

            print(
                f"[INFO] "
                f"{stem}: "
                f"instance_id={instance_id} "
                f"not visible "
                f"({mask_source}), "
                f"skip object only"
            )

            continue

        # ----------------------------------------------------
        # Object is visible
        # ----------------------------------------------------

        # Check overlap with already assigned object.
        overlap = (
            object_mask
            & (label > 0)
        )

        if np.any(overlap):

            overlap_pixels = int(
                np.count_nonzero(overlap)
            )

            raise ValueError(
                f"Mask overlap detected for "
                f"instance_id={instance_id}: "
                f"{overlap_pixels} pixels"
            )

        label[
            object_mask
        ] = instance_id

        visible_instance_ids.append(
            instance_id
        )

    # --------------------------------------------------------
    # No visible objects
    #
    # Only skip when ALL selected objects are invisible.
    # --------------------------------------------------------

    if not visible_instance_ids:

        raise ValueError(
            f"No selected objects are visible "
            f"in this frame. "
            f"selected_instance_ids="
            f"{selected_instance_ids}"
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
    # Build Meta
    #
    # IMPORTANT:
    # Only visible objects are written.
    # --------------------------------------------------------

    object_poses = {}

    for instance_id in visible_instance_ids:

        object_poses[
            str(instance_id)
        ] = all_object_poses[
            instance_id
        ].tolist()

    meta = {
        "objects": [
            int(instance_id)
            for instance_id in visible_instance_ids
        ],
        "object_poses": object_poses,
        "intrinsic": cam_K.tolist(),
        "distortion": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    }

    # --------------------------------------------------------
    # Write Meta
    # --------------------------------------------------------

    with open(
        meta_out,
        "w",
    ) as f:

        json.dump(
            meta,
            f,
            indent=2,
        )

    return {
        "status": "converted",
        "object_ids": visible_instance_ids,
        "invisible_object_ids": [
            instance_id
            for instance_id in selected_instance_ids
            if instance_id
            not in visible_instance_ids
        ],
    }


# ============================================================
# Parse Instance ID List
# ============================================================

def parse_instance_ids(value):
    """
    Parse:
        "1,2,3"
        "1"
        ""
    """

    if value is None:
        return []

    value = value.strip()

    if not value:
        return []

    result = []

    for item in value.split(","):

        item = item.strip()

        if not item:
            continue

        instance_id = int(item)

        if instance_id <= 0:
            raise ValueError(
                f"Invalid instance ID: {instance_id}"
            )

        if instance_id not in result:
            result.append(instance_id)

    return result


# ============================================================
# Convert One Dataset
# ============================================================

def convert_single_dataset(
    input_root,
    out_scene_dir,
    requested_instance_id,
    instance_mask_ids,
    part_instance_mask_ids,
    dataset_name,
):

    input_root = Path(
        input_root
    )

    rgb_dir = (
        input_root / "rgb"
    )

    depth_dir = (
        input_root / "depth"
    )

    instance_mask_dir = (
        input_root / "instance_masks"
    )

    part_instance_mask_dir = (
        input_root / "part_instance_masks"
    )

    pose_dir = (
        input_root / "pose"
    )

    # --------------------------------------------------------
    # Validate directories
    # --------------------------------------------------------

    for directory in [
        rgb_dir,
        depth_dir,
        instance_mask_dir,
        part_instance_mask_dir,
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
            ".png",
        },
    )

    instance_mask_files = find_files(
        instance_mask_dir,
        {
            ".png",
        },
    )

    part_instance_mask_files = find_files(
        part_instance_mask_dir,
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

    invisible_object_counter = {}

    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

    for stem in tqdm(
        sorted(
            rgb_files.keys()
        ),
        desc=f"[{dataset_name}]",
        leave=False,
    ):

        rgb_path = rgb_files[
            stem
        ]

        depth_path = depth_files.get(
            stem
        )

        instance_mask_path = (
            instance_mask_files.get(
                stem
            )
        )

        part_instance_mask_path = (
            part_instance_mask_files.get(
                stem
            )
        )

        pose_path = pose_files.get(
            stem
        )

        missing = []

        if depth_path is None:
            missing.append(
                "depth"
            )

        if instance_mask_path is None:
            missing.append(
                "instance_masks"
            )

        if part_instance_mask_path is None:
            missing.append(
                "part_instance_masks"
            )

        if pose_path is None:
            missing.append(
                "pose"
            )

        if missing:

            print()

            print(
                f"[SKIP] "
                f"{dataset_name}/{stem}: "
                f"missing "
                f"{', '.join(missing)}"
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

            result = convert_frame(
                stem=stem,
                rgb_path=rgb_path,
                depth_path=depth_path,
                instance_mask_path=(
                    instance_mask_path
                ),
                part_instance_mask_path=(
                    part_instance_mask_path
                ),
                pose_path=pose_path,
                out_scene_dir=out_scene_dir,
                requested_instance_id=(
                    requested_instance_id
                ),
                instance_mask_ids=(
                    instance_mask_ids
                ),
                part_instance_mask_ids=(
                    part_instance_mask_ids
                ),
                output_stem=output_stem,
            )

            if (
                result["status"]
                == "converted"
            ):

                converted += 1

                object_ids = tuple(
                    result["object_ids"]
                )

                for instance_id in result[
                    "invisible_object_ids"
                ]:

                    invisible_object_counter[
                        instance_id
                    ] = (
                        invisible_object_counter.get(
                            instance_id,
                            0
                        )
                        + 1
                    )

            else:

                skipped += 1

        except Exception as e:

            print()

            print(
                f"[ERROR] "
                f"{dataset_name}/{stem}: "
                f"{e}"
            )

            skipped += 1

    return {
        "dataset": dataset_name,
        "rgb": len(rgb_files),
        "converted": converted,
        "skipped": skipped,
        "invisible_objects": (
            invisible_object_counter
        ),
    }


# ============================================================
# Find Dataset Folders
# ============================================================

def find_dataset_folders(
    input_root
):

    input_root = Path(
        input_root
    )

    # --------------------------------------------------------
    # Case 1:
    #
    # input_root itself is a dataset
    # --------------------------------------------------------

    if is_dataset_folder(
        input_root
    ):

        return [
            input_root
        ]

    # --------------------------------------------------------
    # Case 2:
    #
    # input_root contains multiple datasets
    # --------------------------------------------------------

    dataset_dirs = []

    for p in sorted(
        input_root.iterdir()
    ):

        if not p.is_dir():
            continue

        if is_dataset_folder(
            p
        ):

            dataset_dirs.append(
                p
            )

    return dataset_dirs


# ============================================================
# Convert Dataset Collection
# ============================================================

def convert_dataset(
    input_root,
    output_root,
    scene_name,
    requested_instance_id,
    instance_mask_ids,
    part_instance_mask_ids,
):

    input_root = Path(
        input_root
    )

    output_root = Path(
        output_root
    )

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

    dataset_dirs = (
        find_dataset_folders(
            input_root
        )
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
            f"   part_instance_masks/\n"
            f"   pose/\n\n"
            f"2. Or multiple datasets:\n"
            f"   dataset_001/\n"
            f"     rgb/\n"
            f"     depth/\n"
            f"     instance_masks/\n"
            f"     part_instance_masks/\n"
            f"     pose/\n"
        )

    print()

    print("=" * 70)
    print("Dataset Conversion")
    print("=" * 70)

    print(
        f"Input               : "
        f"{input_root}"
    )

    print(
        f"Datasets            : "
        f"{len(dataset_dirs)}"
    )

    print(
        f"Instance selection  : "
        f"{'ALL' if requested_instance_id == 0 else requested_instance_id}"
    )

    print(
        f"Instance masks      : "
        f"{instance_mask_ids}"
    )

    print(
        f"Part instance masks : "
        f"{part_instance_mask_ids}"
    )

    print(
        f"Output              : "
        f"{out_scene_dir}"
    )

    print("=" * 70)

    total_rgb = 0
    total_converted = 0
    total_skipped = 0

    total_invisible_objects = {}

    results = []

    # --------------------------------------------------------
    # Process all datasets
    # --------------------------------------------------------

    for dataset_dir in dataset_dirs:

        if len(dataset_dirs) == 1:

            dataset_name = ""

        else:

            dataset_name = (
                dataset_dir.name
            )

        display_name = (
            dataset_dir.name
        )

        print()

        print(
            f"[DATASET] "
            f"{display_name}"
        )

        result = (
            convert_single_dataset(
                input_root=dataset_dir,
                out_scene_dir=out_scene_dir,
                requested_instance_id=(
                    requested_instance_id
                ),
                instance_mask_ids=(
                    instance_mask_ids
                ),
                part_instance_mask_ids=(
                    part_instance_mask_ids
                ),
                dataset_name=dataset_name,
            )
        )

        results.append(
            result
        )

        total_rgb += result[
            "rgb"
        ]

        total_converted += result[
            "converted"
        ]

        total_skipped += result[
            "skipped"
        ]

        for (
            instance_id,
            count,
        ) in result[
            "invisible_objects"
        ].items():

            total_invisible_objects[
                instance_id
            ] = (
                total_invisible_objects.get(
                    instance_id,
                    0
                )
                + count
            )

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

        if result[
            "invisible_objects"
        ]:

            print(
                "    Invisible object frames:"
            )

            for (
                instance_id,
                count,
            ) in sorted(
                result[
                    "invisible_objects"
                ].items()
            ):

                print(
                    f"      "
                    f"instance_id={instance_id}: "
                    f"{count} frame(s)"
                )

    print("-" * 70)

    print(
        f"{'TOTAL':30s} "
        f"RGB={total_rgb:6d} "
        f"Converted={total_converted:6d} "
        f"Skipped={total_skipped:6d}"
    )

    if total_invisible_objects:

        print()
        print(
            "Total invisible object frames:"
        )

        for (
            instance_id,
            count,
        ) in sorted(
            total_invisible_objects.items()
        ):

            print(
                f"  instance_id={instance_id}: "
                f"{count} frame(s)"
            )

    print()

    print(
        f"Output: "
        f"{out_scene_dir}"
    )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bop_data_path",
        type=str,
        required=True,
        help=(
            "Input dataset path. "
            "Can be either a single dataset folder "
            "or a root folder containing multiple dataset folders."
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
            "Instance selection. "
            "0 = convert ALL instances. "
            "Otherwise convert only the specified instance_id."
        ),
    )

    parser.add_argument(
        "--scene_name",
        type=str,
        default="000000",
    )

    parser.add_argument(
        "--instance_mask_ids",
        type=str,
        default="",
        help=(
            "Instance IDs obtained from instance_masks. "
            "Example: 1,3"
        ),
    )

    parser.add_argument(
        "--part_instance_mask_ids",
        type=str,
        default="",
        help=(
            "Instance IDs obtained from part_instance_masks. "
            "Example: 2"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Parse mask source configuration
    # ========================================================

    instance_mask_ids = parse_instance_ids(
        args.instance_mask_ids
    )

    part_instance_mask_ids = parse_instance_ids(
        args.part_instance_mask_ids
    )

    # --------------------------------------------------------
    # Same object cannot use both mask sources
    # --------------------------------------------------------

    overlap = set(
        instance_mask_ids
    ) & set(
        part_instance_mask_ids
    )

    if overlap:

        raise ValueError(
            f"Instance ID(s) appear in both "
            f"--instance_mask_ids and "
            f"--part_instance_mask_ids: "
            f"{sorted(overlap)}"
        )

    # --------------------------------------------------------
    # At least one source must be configured
    # --------------------------------------------------------

    if not instance_mask_ids and not part_instance_mask_ids:

        raise ValueError(
            "No mask source configured.\n"
            "Use for example:\n"
            "--instance_mask_ids 1 "
            "--part_instance_mask_ids 2"
        )

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
        requested_instance_id=(
            args.obj_id
        ),
        instance_mask_ids=(
            instance_mask_ids
        ),
        part_instance_mask_ids=(
            part_instance_mask_ids
        ),
    )


if __name__ == "__main__":
    main()
