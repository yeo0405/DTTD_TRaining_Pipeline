#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from tqdm import tqdm

import cv2
import numpy as np


# ============================================================
# Configuration
# ============================================================

DEFAULT_MIN_VISIBLE_RATIO = 0.0


# ============================================================
# Utility
# ============================================================

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def find_scene_dirs(bop_root: Path):
    scene_dirs = []

    for cam_json in sorted(
        bop_root.rglob("scene_camera.json")
    ):
        scene_dir = cam_json.parent

        gt_json = scene_dir / "scene_gt.json"
        rgb_dir = scene_dir / "rgb"

        if gt_json.exists() and rgb_dir.exists():
            scene_dirs.append(scene_dir)

    return scene_dirs


def resolve_frame_stem(
    rgb_dir: Path,
    frame_id: int,
):
    candidates = [
        f"{frame_id:06d}",
        f"{frame_id:05d}",
        str(frame_id),
    ]

    exts = [
        ".png",
        ".jpg",
        ".jpeg",
    ]

    for stem in candidates:

        for ext in exts:

            p = rgb_dir / f"{stem}{ext}"

            if p.exists():
                return stem, p

    return None, None


def resolve_image_path(
    folder: Path,
    stem: str,
):
    exts = [
        ".png",
        ".jpg",
        ".jpeg",
    ]

    for ext in exts:

        p = folder / f"{stem}{ext}"

        if p.exists():
            return p

    return None


def resolve_mask_path(
    mask_dir: Path,
    stem: str,
    orig_idx: int,
):
    """
    Resolve BOP mask / mask_visib filename.

    Common BOP formats:

        000001_000000.png
        000001_0.png
    """

    if not mask_dir.exists():
        return None

    candidates = [
        mask_dir / f"{stem}_{orig_idx:06d}.png",
        mask_dir / f"{stem}_{orig_idx}.png",
    ]

    for p in candidates:

        if p.exists():
            return p

    return None


# ============================================================
# Mask
# ============================================================

def load_mask(path: Path):
    """
    Load a BOP mask.

    Returns:
        Single-channel numpy array.
    """

    mask = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if mask is None:
        return None

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return mask


def get_mask_area(mask):
    """
    Count pixels belonging to the object.
    """

    return int(
        np.count_nonzero(mask > 0)
    )


# ============================================================
# Depth
# ============================================================

def depth_to_uint16(
    depth_img,
    depth_scale=None,
):
    depth = np.asarray(depth_img)

    # --------------------------------------------------------
    # BOP depth:
    #
    # real_depth = raw_depth * depth_scale
    # --------------------------------------------------------

    if (
        depth_scale is not None
        and depth_scale > 0
    ):

        depth_mm = (
            depth.astype(np.float32)
            * depth_scale
        )

        return np.clip(
            np.round(depth_mm),
            0,
            65535,
        ).astype(np.uint16)

    # --------------------------------------------------------
    # Fallback for non-BOP data
    # --------------------------------------------------------

    if np.issubdtype(
        depth.dtype,
        np.floating,
    ):

        max_val = (
            float(np.nanmax(depth))
            if depth.size
            else 0.0
        )

        # Meter input
        if max_val <= 100.0:
            depth_mm = depth * 1000.0
        else:
            depth_mm = depth

        return np.clip(
            np.round(depth_mm),
            0,
            65535,
        ).astype(np.uint16)

    # --------------------------------------------------------
    # Already uint16 mm
    # --------------------------------------------------------

    return depth.astype(np.uint16)


# ============================================================
# Pose
# ============================================================

def make_pose_4x4(obj_entry):

    r = np.array(
        obj_entry["cam_R_m2c"],
        dtype=np.float32,
    ).reshape(3, 3)

    t = (
        np.array(
            obj_entry["cam_t_m2c"],
            dtype=np.float32,
        ).reshape(3)
        / 1000.0
    )

    pose = np.eye(
        4,
        dtype=np.float32,
    )

    pose[:3, :3] = r
    pose[:3, 3] = t

    return pose.tolist()


# ============================================================
# Convert Scene
# ============================================================

def convert_scene(
    scene_dir,
    out_root,
    target_obj_id,
    min_visible_ratio,
    desc="",
):

    scene_name = scene_dir.name

    rgb_dir = scene_dir / "rgb"
    depth_dir = scene_dir / "depth"

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # BOP provides two different masks:
    #
    # mask/
    #     Full object silhouette
    #
    # mask_visib/
    #     Currently visible part
    # --------------------------------------------------------

    mask_dir = scene_dir / "mask"
    mask_visib_dir = scene_dir / "mask_visib"

    scene_camera = load_json(
        scene_dir / "scene_camera.json"
    )

    scene_gt = load_json(
        scene_dir / "scene_gt.json"
    )

    out_scene_dir = (
        out_root
        / "data"
        / scene_name
        / "data"
    )

    ensure_dir(
        out_scene_dir
    )

    # ========================================================
    # Statistics
    # ========================================================

    converted = 0
    filtered = 0
    skipped = 0

    # ========================================================
    # Convert frames
    # ========================================================

    for frame_key in tqdm(
        sorted(
            scene_gt.keys(),
            key=lambda x: int(x),
        ),
        desc=desc,
        leave=False,
    ):

        frame_id = int(frame_key)

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        stem, rgb_path = resolve_frame_stem(
            rgb_dir,
            frame_id,
        )

        if rgb_path is None:

            print(
                f"[SKIP] "
                f"{scene_name}/{frame_id}: "
                f"RGB not found"
            )

            skipped += 1
            continue

        # ----------------------------------------------------
        # Depth
        # ----------------------------------------------------

        depth_path = resolve_image_path(
            depth_dir,
            stem,
        )

        if depth_path is None:

            print(
                f"[SKIP] "
                f"{scene_name}/{stem}: "
                f"depth not found"
            )

            skipped += 1
            continue

        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------

        cam_info = scene_camera.get(
            frame_key,
            scene_camera.get(
                str(frame_id),
                {},
            ),
        )

        depth_scale = cam_info.get(
            "depth_scale",
            None,
        )

        cam_K = np.array(
            cam_info.get(
                "cam_K",
                np.eye(3).reshape(-1).tolist(),
            ),
            dtype=np.float32,
        ).reshape(3, 3)

        cam_dist = cam_info.get(
            "cam_dist",
            [
                0,
                0,
                0,
                0,
                0,
            ],
        )

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        rgb = cv2.imread(
            str(rgb_path),
            cv2.IMREAD_COLOR,
        )

        if rgb is None:

            print(
                f"[SKIP] "
                f"{scene_name}/{stem}: "
                f"failed to read RGB"
            )

            skipped += 1
            continue

        # ----------------------------------------------------
        # Depth
        # ----------------------------------------------------

        depth_raw = cv2.imread(
            str(depth_path),
            cv2.IMREAD_UNCHANGED,
        )

        if depth_raw is None:

            print(
                f"[SKIP] "
                f"{scene_name}/{stem}: "
                f"failed to read depth"
            )

            skipped += 1
            continue

        depth_u16 = depth_to_uint16(
            depth_raw,
            depth_scale=depth_scale,
        )

        # ====================================================
        # Objects
        # ====================================================

        objs = scene_gt[frame_key]

        selected_objects = []

        for orig_idx, entry in enumerate(objs):

            obj_id = int(
                entry["obj_id"]
            )

            # ------------------------------------------------
            # Object ID filter
            # ------------------------------------------------

            if (
                target_obj_id != 0
                and obj_id != target_obj_id
            ):
                continue

            selected_objects.append(
                (
                    orig_idx,
                    entry,
                )
            )

        if not selected_objects:

            skipped += 1
            continue

        # ----------------------------------------------------
        # Keep only one instance for each object ID
        # ----------------------------------------------------

        filtered_objects = []

        seen = set()

        for orig_idx, entry in selected_objects:

            obj_id = int(
                entry["obj_id"]
            )

            if obj_id in seen:
                continue

            seen.add(obj_id)

            filtered_objects.append(
                (
                    orig_idx,
                    entry,
                )
            )

        # ====================================================
        # Build label
        # ====================================================

        label = np.zeros(
            depth_u16.shape[:2],
            dtype=np.uint16,
        )

        object_poses = {}
        object_ids = []

        valid_objects = 0

        # ====================================================
        # Process each object
        # ====================================================

        for orig_idx, entry in filtered_objects:

            obj_id = int(
                entry["obj_id"]
            )

            # ------------------------------------------------
            # Full object mask
            #
            # mask/
            # ------------------------------------------------

            full_mask_path = resolve_mask_path(
                mask_dir,
                stem,
                orig_idx,
            )

            if (
                full_mask_path is None
                or not full_mask_path.exists()
            ):

                print(
                    f"[SKIP OBJECT] "
                    f"{scene_name}/{stem}: "
                    f"full mask not found, "
                    f"orig_idx={orig_idx}, "
                    f"obj_id={obj_id}"
                )

                continue

            # ------------------------------------------------
            # Visible object mask
            #
            # mask_visib/
            # ------------------------------------------------

            visible_mask_path = resolve_mask_path(
                mask_visib_dir,
                stem,
                orig_idx,
            )

            if (
                visible_mask_path is None
                or not visible_mask_path.exists()
            ):

                print(
                    f"[SKIP OBJECT] "
                    f"{scene_name}/{stem}: "
                    f"visible mask not found, "
                    f"orig_idx={orig_idx}, "
                    f"obj_id={obj_id}"
                )

                continue

            # ------------------------------------------------
            # Load full mask
            # ------------------------------------------------

            full_mask = load_mask(
                full_mask_path
            )

            if full_mask is None:

                print(
                    f"[SKIP OBJECT] "
                    f"{scene_name}/{stem}: "
                    f"failed to read full mask, "
                    f"obj_id={obj_id}"
                )

                continue

            # ------------------------------------------------
            # Load visible mask
            # ------------------------------------------------

            visible_mask = load_mask(
                visible_mask_path
            )

            if visible_mask is None:

                print(
                    f"[SKIP OBJECT] "
                    f"{scene_name}/{stem}: "
                    f"failed to read visible mask, "
                    f"obj_id={obj_id}"
                )

                continue

            # ------------------------------------------------
            # Mask size validation
            # ------------------------------------------------

            expected_shape = (
                depth_u16.shape[:2]
            )

            if (
                full_mask.shape[:2]
                != expected_shape
            ):

                print(
                    f"[SKIP OBJECT] "
                    f"{scene_name}/{stem}: "
                    f"full mask size mismatch, "
                    f"obj_id={obj_id}, "
                    f"mask={full_mask.shape[:2]}, "
                    f"depth={expected_shape}"
                )

                continue

            if (
                visible_mask.shape[:2]
                != expected_shape
            ):

                print(
                    f"[SKIP OBJECT] "
                    f"{scene_name}/{stem}: "
                    f"visible mask size mismatch, "
                    f"obj_id={obj_id}, "
                    f"mask_visib="
                    f"{visible_mask.shape[:2]}, "
                    f"depth={expected_shape}"
                )

                continue

            # =================================================
            # Calculate areas
            # =================================================

            full_area = get_mask_area(
                full_mask
            )

            visible_area = get_mask_area(
                visible_mask
            )

            if full_area <= 0:

                print(
                    f"[SKIP OBJECT] "
                    f"{scene_name}/{stem}: "
                    f"empty full mask, "
                    f"obj_id={obj_id}"
                )

                continue

            if visible_area <= 0:

                print(
                    f"[FILTER] "
                    f"{scene_name}/{stem} "
                    f"obj_id={obj_id}: "
                    f"visible_area=0 / "
                    f"full_area={full_area} "
                    f"(0.00%)"
                )

                filtered += 1
                continue

            # =================================================
            # Visible ratio
            #
            # visible mask / full mask
            # =================================================

            visible_ratio = (
                visible_area
                / full_area
            )

            # =================================================
            # Filter
            #
            # < 50% -> remove
            # >= 50% -> keep
            # =================================================

            if (
                visible_ratio
                < min_visible_ratio
            ):

                print(
                    f"[FILTER] "
                    f"{scene_name}/{stem} "
                    f"obj_id={obj_id}: "
                    f"visible_area="
                    f"{visible_area} / "
                    f"full_area="
                    f"{full_area} "
                    f"("
                    f"{visible_ratio * 100.0:.2f}%"
                    f") < "
                    f"{min_visible_ratio * 100.0:.2f}%"
                )

                filtered += 1
                continue

            # =================================================
            # Keep object
            # =================================================

            label[
                visible_mask > 0
            ] = obj_id

            object_ids.append(
                obj_id
            )

            object_poses[
                str(obj_id)
            ] = make_pose_4x4(
                entry
            )

            valid_objects += 1

        # ====================================================
        # If no object survived
        # ====================================================

        if valid_objects == 0:

            skipped += 1
            continue

        # ====================================================
        # Output paths
        # ====================================================

        color_out = (
            out_scene_dir
            / f"{stem}_color.jpg"
        )

        depth_out = (
            out_scene_dir
            / f"{stem}_depth.png"
        )

        label_out = (
            out_scene_dir
            / f"{stem}_label.png"
        )

        meta_out = (
            out_scene_dir
            / f"{stem}_meta.json"
        )

        # ====================================================
        # Write RGB
        # ====================================================

        if not cv2.imwrite(
            str(color_out),
            rgb,
        ):

            print(
                f"[ERROR] Failed to write "
                f"{color_out}"
            )

            skipped += 1
            continue

        # ====================================================
        # Write Depth
        # ====================================================

        if not cv2.imwrite(
            str(depth_out),
            depth_u16,
        ):

            print(
                f"[ERROR] Failed to write "
                f"{depth_out}"
            )

            skipped += 1
            continue

        # ====================================================
        # Write Label
        # ====================================================

        if not cv2.imwrite(
            str(label_out),
            label,
        ):

            print(
                f"[ERROR] Failed to write "
                f"{label_out}"
            )

            skipped += 1
            continue

        # ====================================================
        # Write Meta
        # ====================================================

        meta = {
            "objects": object_ids,
            "object_poses": object_poses,
            "intrinsic": cam_K.tolist(),
            "distortion": cam_dist,
        }

        with open(
            meta_out,
            "w",
        ) as f:

            json.dump(
                meta,
                f,
            )

        converted += 1

    # ========================================================
    # Scene result
    # ========================================================

    return {
        "scene": scene_name,
        "converted": converted,
        "filtered": filtered,
        "skipped": skipped,
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bop_data_path",
        type=str,
        required=True,
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
            "0=keep all objects, "
            "otherwise keep only specified obj_id"
        ),
    )

    parser.add_argument(
        "--min_visible_ratio",
        type=float,
        default=DEFAULT_MIN_VISIBLE_RATIO,
        help=(
            "Minimum visible object ratio required "
            "to keep an object. "
            "Default: 0.5 (50%%). "
            "Ratio is calculated as "
            "mask_visib area / mask area."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Validate
    # ========================================================

    if not (
        0.0
        <= args.min_visible_ratio
        <= 1.0
    ):

        parser.error(
            "--min_visible_ratio must be "
            "between 0.0 and 1.0"
        )

    # ========================================================
    # Paths
    # ========================================================

    bop_root = Path(
        args.bop_data_path
    )

    out_root = Path(
        args.output_dir
    )

    if not bop_root.exists():

        raise FileNotFoundError(
            f"BOP data path does not exist:\n"
            f"{bop_root}"
        )

    ensure_dir(
        out_root / "data"
    )

    target_obj_id = args.obj_id

    # ========================================================
    # Find scenes
    # ========================================================

    scene_dirs = sorted(
        find_scene_dirs(bop_root),
        key=lambda p: p.name,
    )

    if not scene_dirs:

        raise RuntimeError(
            f"No valid BOP scenes found under:\n"
            f"{bop_root}"
        )

    # ========================================================
    # Header
    # ========================================================

    print()
    print("=" * 70)
    print("BOP Dataset Conversion")
    print("=" * 70)
    print(
        f"Input              : {bop_root}"
    )
    print(
        f"Scenes             : {len(scene_dirs)}"
    )
    print(
        f"Target object ID   : {target_obj_id}"
    )
    print(
        f"Min visible ratio  : "
        f"{args.min_visible_ratio * 100.0:.2f}%"
    )
    print(
        f"Ratio calculation  : "
        f"mask_visib / mask"
    )
    print(
        f"Output             : {out_root}"
    )
    print("=" * 70)

    # ========================================================
    # Convert scenes
    # ========================================================

    results = []

    for scene_dir in tqdm(
        scene_dirs,
        desc="convert scenes",
    ):

        result = convert_scene(
            scene_dir=scene_dir,
            out_root=out_root,
            target_obj_id=target_obj_id,
            min_visible_ratio=(
                args.min_visible_ratio
            ),
            desc=scene_dir.name,
        )

        results.append(
            result
        )

    # ========================================================
    # Summary
    # ========================================================

    total_converted = sum(
        r["converted"]
        for r in results
    )

    total_filtered = sum(
        r["filtered"]
        for r in results
    )

    total_skipped = sum(
        r["skipped"]
        for r in results
    )

    print()
    print("=" * 70)
    print("Conversion finished")
    print("=" * 70)

    for result in results:

        print(
            f"{result['scene']:30s} "
            f"Converted={result['converted']:6d} "
            f"Filtered={result['filtered']:6d} "
            f"Skipped={result['skipped']:6d}"
        )

    print("-" * 70)

    print(
        f"{'TOTAL':30s} "
        f"Converted={total_converted:6d} "
        f"Filtered={total_filtered:6d} "
        f"Skipped={total_skipped:6d}"
    )

    print()
    print(
        f"Output: {out_root}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()