#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


DEFAULT_OBJ_ID = 0
DEFAULT_DEPTH_UNIT = "mm"
DEFAULT_K = np.array([
    [1070.7898,    0, 962.30939],
    [   0,     1070.5270, 550.0],
    [   0,        0,       1.0]
], dtype=np.float32)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def resolve_image(folder: Path, stem: str):
    for ext in IMAGE_EXTENSIONS:
        path = folder / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def load_pose(path: Path):
    try:
        values = np.loadtxt(str(path), dtype=np.float32)
    except Exception as e:
        print(f"[ERROR] Failed to load pose: {path}: {e}")
        return None

    values = np.asarray(values, dtype=np.float32)

    if values.size != 16:
        print(
            f"[ERROR] Invalid pose size: {path} "
            f"(expected 16 values, got {values.size})"
        )
        return None

    pose = values.reshape(4, 4)

    if not np.all(np.isfinite(pose)):
        print(f"[ERROR] Pose contains NaN/Inf: {path}")
        return None

    return pose


def load_multi_pose(path: Path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load pose JSON: {path}: {e}")
        return None

    poses = data.get("objects")

    if poses is None:
        poses = data.get("object_poses")

    if not isinstance(poses, dict) or not poses:
        print(
            f"[ERROR] Invalid multi-object pose file: {path}"
        )
        return None

    result = {}

    for obj_id, value in poses.items():
        try:
            obj_id = int(obj_id)
            pose = np.asarray(value, dtype=np.float32)
        except Exception:
            print(
                f"[ERROR] Invalid object pose: "
                f"{path}, object={obj_id}"
            )
            return None

        if obj_id <= 0:
            print(
                f"[ERROR] Invalid object ID: "
                f"{path}, object={obj_id}"
            )
            return None

        if pose.size != 16:
            print(
                f"[ERROR] Invalid pose size: "
                f"{path}, object={obj_id}, "
                f"expected 16 values, got {pose.size}"
            )
            return None

        pose = pose.reshape(4, 4)

        if not np.all(np.isfinite(pose)):
            print(
                f"[ERROR] Pose contains NaN/Inf: "
                f"{path}, object={obj_id}"
            )
            return None

        result[obj_id] = pose

    return result


def load_mask(path: Path):
    mask = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if mask is None:
        return None

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return mask


def load_depth(path: Path):
    depth = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if depth is None:
        return None

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    return depth


def depth_to_uint16(depth, depth_unit):
    depth = np.asarray(depth)

    if depth_unit == "mm":
        if depth.dtype == np.uint16:
            return depth

        return np.clip(
            np.round(depth),
            0,
            65535,
        ).astype(np.uint16)

    if depth_unit == "m":
        depth_mm = depth.astype(np.float32) * 1000.0

        return np.clip(
            np.round(depth_mm),
            0,
            65535,
        ).astype(np.uint16)

    raise ValueError(
        f"Unsupported depth unit: {depth_unit}"
    )


def pose_to_list(pose):
    return np.asarray(
        pose,
        dtype=np.float32,
    ).tolist()


def find_old_pose(pose_dir: Path, stem: str):
    txt = pose_dir / f"{stem}.txt"

    if txt.exists():
        return txt

    pose = pose_dir / f"{stem}.pose"

    if pose.exists():
        return pose

    return None


def convert_old_frame(
    stem,
    rgb,
    depth,
    mask,
    pose_path,
    obj_id,
):
    pose = load_pose(pose_path)

    if pose is None:
        return None

    if mask.shape[:2] != rgb.shape[:2]:
        print(
            f"[SKIP] {stem}: "
            f"mask size {mask.shape[:2]} != "
            f"RGB size {rgb.shape[:2]}"
        )
        return None

    mask_binary = np.where(
        mask > 0,
        obj_id,
        0,
    ).astype(np.uint16)

    if np.count_nonzero(mask_binary) == 0:
        print(f"[SKIP] {stem}: empty mask")
        return None

    meta = {
        "objects": [obj_id],
        "object_poses": {
            str(obj_id): pose_to_list(pose),
        },
    }

    return mask_binary, meta


def convert_new_frame(
    stem,
    rgb,
    mask,
    pose_path,
):
    poses = load_multi_pose(pose_path)

    if poses is None:
        return None

    if mask.shape[:2] != rgb.shape[:2]:
        print(
            f"[SKIP] {stem}: "
            f"mask size {mask.shape[:2]} != "
            f"RGB size {rgb.shape[:2]}"
        )
        return None

    if mask.dtype.kind not in "iu":
        print(
            f"[SKIP] {stem}: "
            f"multi-object mask must be integer type"
        )
        return None

    label = mask.astype(np.uint16)

    mask_ids = set(
        int(v)
        for v in np.unique(label)
        if int(v) > 0
    )

    pose_ids = set(poses.keys())

    missing_pose = sorted(mask_ids - pose_ids)
    missing_mask = sorted(pose_ids - mask_ids)

    if missing_pose:
        print(
            f"[SKIP] {stem}: "
            f"mask objects without pose: {missing_pose}"
        )
        return None

    if missing_mask:
        print(
            f"[WARNING] {stem}: "
            f"pose objects without visible mask: {missing_mask}"
        )

    objects = sorted(mask_ids)

    if not objects:
        print(f"[SKIP] {stem}: empty multi-object mask")
        return None

    meta = {
        "objects": objects,
        "object_poses": {
            str(obj_id): pose_to_list(poses[obj_id])
            for obj_id in objects
        },
    }

    return label, meta


def convert_dataset(
    input_root: Path,
    output_root: Path,
    obj_id: int,
    K: np.ndarray,
    distortion,
    depth_unit: str,
):
    rgb_dir = input_root / "rgb"
    depth_dir = input_root / "depth"
    mask_dir = input_root / "masks"
    pose_dir = input_root / "pose"

    for path in (
        rgb_dir,
        depth_dir,
        mask_dir,
        pose_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required directory does not exist: {path}"
            )

    multi_object = obj_id == 0
    scene_name = input_root.name

    out_scene_dir = (
        output_root
        / "data"
        / scene_name
        / "data"
    )

    ensure_dir(out_scene_dir)

    rgb_files = sorted(
        [
            p for p in rgb_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.stem,
    )

    converted = 0
    skipped = 0

    for rgb_path in tqdm(
        rgb_files,
        desc="convert",
    ):
        stem = rgb_path.stem

        depth_path = resolve_image(
            depth_dir,
            stem,
        )

        mask_path = resolve_image(
            mask_dir,
            stem,
        )

        if multi_object:
            pose_path = pose_dir / f"{stem}.json"
        else:
            pose_path = find_old_pose(
                pose_dir,
                stem,
            )

        if depth_path is None:
            print(
                f"[SKIP] {stem}: depth not found"
            )
            skipped += 1
            continue

        if mask_path is None:
            print(
                f"[SKIP] {stem}: mask not found"
            )
            skipped += 1
            continue

        if pose_path is None or not pose_path.exists():
            print(
                f"[SKIP] {stem}: pose not found"
            )
            skipped += 1
            continue

        rgb = cv2.imread(
            str(rgb_path),
            cv2.IMREAD_COLOR,
        )

        depth = load_depth(depth_path)
        mask = load_mask(mask_path)

        if rgb is None:
            print(
                f"[SKIP] {stem}: failed to read RGB"
            )
            skipped += 1
            continue

        if depth is None:
            print(
                f"[SKIP] {stem}: failed to read depth"
            )
            skipped += 1
            continue

        if mask is None:
            print(
                f"[SKIP] {stem}: failed to read mask"
            )
            skipped += 1
            continue

        depth = depth_to_uint16(
            depth,
            depth_unit,
        )

        if depth.shape[:2] != rgb.shape[:2]:
            print(
                f"[SKIP] {stem}: "
                f"depth size {depth.shape[:2]} != "
                f"RGB size {rgb.shape[:2]}"
            )
            skipped += 1
            continue

        if multi_object:
            result = convert_new_frame(
                stem,
                rgb,
                mask,
                pose_path,
            )
        else:
            result = convert_old_frame(
                stem,
                rgb,
                depth,
                mask,
                pose_path,
                obj_id,
            )

        if result is None:
            skipped += 1
            continue

        label, meta = result

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

        if not cv2.imwrite(
            str(color_out),
            rgb,
        ):
            print(
                f"[ERROR] Failed to write {color_out}"
            )
            skipped += 1
            continue

        if not cv2.imwrite(
            str(depth_out),
            depth,
        ):
            print(
                f"[ERROR] Failed to write {depth_out}"
            )
            skipped += 1
            continue

        if not cv2.imwrite(
            str(label_out),
            label,
        ):
            print(
                f"[ERROR] Failed to write {label_out}"
            )
            skipped += 1
            continue

        meta["intrinsic"] = K.tolist()
        meta["distortion"] = distortion

        with open(
            meta_out,
            "w",
        ) as f:
            json.dump(
                meta,
                f,
            )

        converted += 1

    return {
        "scene": scene_name,
        "converted": converted,
        "skipped": skipped,
        "multi_object": multi_object,
    }


def load_intrinsic(path: Path):
    if not path:
        return DEFAULT_K.copy()

    if not path.exists():
        raise FileNotFoundError(
            f"Intrinsic file does not exist: {path}"
        )

    values = np.loadtxt(
        str(path),
        dtype=np.float32,
    )

    values = np.asarray(values)

    if values.size != 9:
        raise ValueError(
            f"Intrinsic file must contain 9 values: {path}"
        )

    return values.reshape(3, 3)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="FoundationPose dataset directory",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Target training dataset directory",
    )

    parser.add_argument(
        "--obj_id",
        type=int,
        default=DEFAULT_OBJ_ID,
        help="0 = multi-object JSON mode; >0 = legacy single-object mode",
    )

    parser.add_argument(
        "--depth_unit",
        choices=("mm", "m"),
        default=DEFAULT_DEPTH_UNIT,
        help="Input depth unit. Default: mm",
    )

    parser.add_argument(
        "--intrinsic",
        type=str,
        default=None,
        help=(
            "Optional 3x3 intrinsic matrix text file. "
            "Default: built-in camera K"
        ),
    )

    args = parser.parse_args()

    input_root = Path(
        args.input_dir
    )

    output_root = Path(
        args.output_dir
    )

    if not input_root.exists():
        raise FileNotFoundError(
            f"Input directory does not exist:\n"
            f"{input_root}"
        )

    if args.obj_id < 0:
        raise ValueError(
            "--obj_id must be 0 or greater"
        )

    K = load_intrinsic(
        Path(args.intrinsic)
        if args.intrinsic
        else None
    )

    distortion = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    multi_object = args.obj_id == 0

    print()
    print("=" * 70)
    print("FoundationPose Dataset Conversion")
    print("=" * 70)
    print(f"Input              : {input_root}")
    print(f"Output             : {output_root}")

    if multi_object:
        print("Mode               : Multi-object")
        print("Pose format        : JSON")
        print("Object ID          : Read from label/pose")
    else:
        print("Mode               : Legacy single-object")
        print(f"Object ID          : {args.obj_id}")
        print("Pose format        : TXT/POSE")

    print(f"Depth unit         : {args.depth_unit}")
    print("Pose               : FoundationPose object-to-camera")
    print("Translation        : meters")
    print(f"Intrinsic          :\n{K}")
    print("=" * 70)

    result = convert_dataset(
        input_root=input_root,
        output_root=output_root,
        obj_id=args.obj_id,
        K=K,
        distortion=distortion,
        depth_unit=args.depth_unit,
    )

    print()
    print("=" * 70)
    print("Conversion finished")
    print("=" * 70)
    print(
        f"Scene              : {result['scene']}"
    )
    print(
        f"Mode               : "
        f"{'Multi-object' if result['multi_object'] else 'Legacy'}"
    )
    print(
        f"Converted          : {result['converted']}"
    )
    print(
        f"Skipped            : {result['skipped']}"
    )
    print(
        f"Output             : "
        f"{output_root / 'data' / result['scene'] / 'data'}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()