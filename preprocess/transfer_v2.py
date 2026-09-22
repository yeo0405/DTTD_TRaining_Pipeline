#!/usr/bin/env python3

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "0")

import argparse
import json
from pathlib import Path

from tqdm import tqdm
import cv2
import numpy as np


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def find_files(folder: Path, extensions):
    if not folder.is_dir():
        return {}
    return {
        p.stem: p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in extensions
    }


def is_dataset_folder(path: Path):
    return all(
        (path / name).is_dir()
        for name in ("rgb", "depth", "pose")
    )


def load_mask(path: Path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Failed to read mask:\n{path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint16)


def load_depth(path: Path):
    if path.suffix.lower() != ".png":
        raise ValueError(
            f"Unsupported depth format: {path.suffix}\n"
            f"Supported format: .png"
        )

    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read depth:\n{path}")

    if depth.ndim != 2:
        raise ValueError(
            f"Expected single-channel PNG depth, got shape={depth.shape}"
        )

    if depth.dtype != np.uint16:
        raise ValueError(
            f"Expected uint16 PNG depth in millimeters, got dtype={depth.dtype}"
        )

    return depth


def get_intrinsic_from_json(pose_data):
    matrix = pose_data.get("camera", {}).get("intrinsics", {}).get("matrix")

    if matrix is None:
        raise ValueError("pose JSON does not contain camera.intrinsics.matrix")

    cam_K = np.asarray(matrix, dtype=np.float32)

    if cam_K.shape != (3, 3):
        raise ValueError(f"Invalid intrinsic matrix shape: {cam_K.shape}")

    return cam_K


def get_all_object_poses(pose_data):
    objects = pose_data.get("objects", [])

    if not objects:
        raise ValueError("pose JSON contains no objects")

    object_poses = {}

    for obj in objects:
        if "instance_id" not in obj:
            raise ValueError("Object does not contain instance_id")

        instance_id = int(obj["instance_id"])

        if "T_camera_object_cv" not in obj:
            raise ValueError(
                f"Object instance_id={instance_id} does not contain "
                f"T_camera_object_cv"
            )

        pose = np.asarray(obj["T_camera_object_cv"], dtype=np.float32)

        if pose.shape != (4, 4):
            raise ValueError(
                f"Invalid pose shape for instance_id={instance_id}: {pose.shape}"
            )

        if not np.all(np.isfinite(pose)):
            raise ValueError(
                f"Pose contains NaN/Inf for instance_id={instance_id}"
            )

        if instance_id in object_poses:
            raise ValueError(
                f"Duplicate instance_id={instance_id} in pose JSON"
            )

        object_poses[instance_id] = pose

    return object_poses


def get_selected_instance_ids(
    requested_instance_id,
    pose_instance_ids,
    instance_mask_ids,
    part_instance_mask_ids,
):
    configured_ids = set(instance_mask_ids) | set(part_instance_mask_ids)

    if requested_instance_id == 0:
        return [
            instance_id
            for instance_id in pose_instance_ids
            if instance_id in configured_ids
        ]

    if requested_instance_id not in pose_instance_ids:
        raise ValueError(
            f"instance_id={requested_instance_id} not found in pose JSON. "
            f"Available instances: {pose_instance_ids}"
        )

    if requested_instance_id not in configured_ids:
        raise ValueError(
            f"instance_id={requested_instance_id} has no mask source configured. "
            f"instance_mask_ids={instance_mask_ids}, "
            f"part_instance_mask_ids={part_instance_mask_ids}"
        )

    return [requested_instance_id]


def convert_frame(
    stem,
    rgb_path,
    depth_path,
    instance_mask_path,
    part_instance_mask_path,
    pose_path,
    out_scene_dir,
    selected_instance_ids,
    instance_mask_ids,
    part_instance_mask_ids,
    output_stem,
):
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)

    if rgb is None:
        raise RuntimeError(f"Failed to read RGB:\n{rgb_path}")

    rgb_h, rgb_w = rgb.shape[:2]
    expected_size = (rgb_h, rgb_w)

    depth_u16 = load_depth(depth_path)

    if depth_u16.shape[:2] != expected_size:
        raise ValueError(
            f"RGB / Depth size mismatch: "
            f"RGB={expected_size}, Depth={depth_u16.shape[:2]}"
        )

    instance_mask = None
    if instance_mask_path is not None:
        instance_mask = load_mask(instance_mask_path)

        if instance_mask.shape[:2] != expected_size:
            raise ValueError(
                f"RGB / Instance Mask size mismatch: "
                f"RGB={expected_size}, Mask={instance_mask.shape[:2]}"
            )

    part_instance_mask = None
    if part_instance_mask_path is not None:
        part_instance_mask = load_mask(part_instance_mask_path)

        if part_instance_mask.shape[:2] != expected_size:
            raise ValueError(
                f"RGB / Part Instance Mask size mismatch: "
                f"RGB={expected_size}, Part Mask={part_instance_mask.shape[:2]}"
            )

    pose_data = load_json(pose_path)
    cam_K = get_intrinsic_from_json(pose_data)
    all_object_poses = get_all_object_poses(pose_data)

    label = np.zeros((rgb_h, rgb_w), dtype=np.uint16)
    visible_instance_ids = []
    invisible_instance_ids = []

    for instance_id in selected_instance_ids:
        if instance_id in instance_mask_ids:
            mask_source = "instance_masks"

            if instance_mask is None:
                print(
                    f"[INFO] {stem}: instance_id={instance_id} "
                    f"instance_masks file missing, skip object only"
                )
                invisible_instance_ids.append(instance_id)
                continue

            object_mask = instance_mask == instance_id

        elif instance_id in part_instance_mask_ids:
            mask_source = "part_instance_masks"

            if part_instance_mask is None:
                print(
                    f"[INFO] {stem}: instance_id={instance_id} "
                    f"part_instance_masks file missing, skip object only"
                )
                invisible_instance_ids.append(instance_id)
                continue

            object_mask = part_instance_mask > 0

        else:
            print(
                f"[INFO] {stem}: instance_id={instance_id} "
                f"has no usable mask source, skip object only"
            )
            invisible_instance_ids.append(instance_id)
            continue

        instance_area = int(np.count_nonzero(object_mask))

        if instance_area == 0:
            print(
                f"[INFO] {stem}: instance_id={instance_id} "
                f"not visible ({mask_source}), skip object only"
            )
            invisible_instance_ids.append(instance_id)
            continue

        overlap = object_mask & (label > 0)

        if np.any(overlap):
            overlap_pixels = int(np.count_nonzero(overlap))
            raise ValueError(
                f"Mask overlap detected for instance_id={instance_id}: "
                f"{overlap_pixels} pixels"
            )

        label[object_mask] = instance_id
        visible_instance_ids.append(instance_id)

    if not visible_instance_ids:
        raise ValueError(
            f"No selected objects are visible or have valid masks. "
            f"selected_instance_ids={selected_instance_ids}"
        )

    color_out = out_scene_dir / f"{output_stem}_color.jpg"
    depth_out = out_scene_dir / f"{output_stem}_depth.png"
    label_out = out_scene_dir / f"{output_stem}_label.png"
    meta_out = out_scene_dir / f"{output_stem}_meta.json"

    if not cv2.imwrite(str(color_out), rgb):
        raise RuntimeError(f"Failed to write:\n{color_out}")

    if not cv2.imwrite(str(depth_out), depth_u16):
        raise RuntimeError(f"Failed to write:\n{depth_out}")

    if not cv2.imwrite(str(label_out), label):
        raise RuntimeError(f"Failed to write:\n{label_out}")

    object_poses = {
        str(instance_id): all_object_poses[instance_id].tolist()
        for instance_id in visible_instance_ids
    }

    meta = {
        "objects": [int(instance_id) for instance_id in visible_instance_ids],
        "object_poses": object_poses,
        "intrinsic": cam_K.tolist(),
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    }

    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "status": "converted",
        "object_ids": visible_instance_ids,
        "invisible_object_ids": invisible_instance_ids,
    }


def parse_instance_ids(value):
    if not value:
        return []

    result = []

    for item in value.split(","):
        item = item.strip()

        if not item:
            continue

        instance_id = int(item)

        if instance_id <= 0:
            raise ValueError(f"Invalid instance ID: {instance_id}")

        if instance_id not in result:
            result.append(instance_id)

    return result


def convert_single_dataset(
    input_root,
    out_scene_dir,
    requested_instance_id,
    instance_mask_ids,
    part_instance_mask_ids,
    dataset_name,
):
    input_root = Path(input_root)

    rgb_dir = input_root / "rgb"
    depth_dir = input_root / "depth"
    instance_mask_dir = input_root / "instance_masks"
    part_instance_mask_dir = input_root / "part_instance_masks"
    pose_dir = input_root / "pose"

    for directory in (rgb_dir, depth_dir, pose_dir):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Required directory does not exist:\n{directory}"
            )

    has_instance_masks = instance_mask_dir.is_dir()
    has_part_instance_masks = part_instance_mask_dir.is_dir()

    rgb_files = find_files(rgb_dir, {".jpg", ".jpeg", ".png"})
    depth_files = find_files(depth_dir, {".png"})
    pose_files = find_files(pose_dir, {".json"})

    instance_mask_files = (
        find_files(instance_mask_dir, {".png"})
        if has_instance_masks else {}
    )

    part_instance_mask_files = (
        find_files(part_instance_mask_dir, {".png"})
        if has_part_instance_masks else {}
    )

    converted = 0
    skipped = 0
    invisible_object_counter = {}

    for stem in tqdm(
        sorted(rgb_files.keys()),
        desc=f"[{dataset_name or 'dataset'}]",
        leave=False,
    ):
        rgb_path = rgb_files[stem]
        depth_path = depth_files.get(stem)
        pose_path = pose_files.get(stem)

        if depth_path is None:
            print(f"\n[SKIP] {dataset_name}/{stem}: missing depth")
            skipped += 1
            continue

        if pose_path is None:
            print(f"\n[SKIP] {dataset_name}/{stem}: missing pose")
            skipped += 1
            continue

        try:
            pose_data = load_json(pose_path)
            pose_instance_ids = sorted(
                get_all_object_poses(pose_data).keys()
            )

            selected_instance_ids = get_selected_instance_ids(
                requested_instance_id,
                pose_instance_ids,
                instance_mask_ids,
                part_instance_mask_ids,
            )

            if not selected_instance_ids:
                print(
                    f"\n[SKIP] {dataset_name}/{stem}: "
                    f"no configured objects available"
                )
                skipped += 1
                continue

            needs_instance_mask = any(
                instance_id in instance_mask_ids
                for instance_id in selected_instance_ids
            )

            needs_part_instance_mask = any(
                instance_id in part_instance_mask_ids
                for instance_id in selected_instance_ids
            )

            instance_mask_path = (
                instance_mask_files.get(stem)
                if needs_instance_mask
                else None
            )

            part_instance_mask_path = (
                part_instance_mask_files.get(stem)
                if needs_part_instance_mask
                else None
            )

            if needs_instance_mask and not has_instance_masks:
                print(
                    f"\n[INFO] {dataset_name}/{stem}: "
                    f"instance_masks directory missing, "
                    f"instance object(s) will be skipped"
                )

            if needs_part_instance_mask and not has_part_instance_masks:
                print(
                    f"\n[INFO] {dataset_name}/{stem}: "
                    f"part_instance_masks directory missing, "
                    f"part object(s) will be skipped"
                )

            output_stem = (
                f"{dataset_name}_{stem}"
                if dataset_name
                else stem
            )

            result = convert_frame(
                stem=stem,
                rgb_path=rgb_path,
                depth_path=depth_path,
                instance_mask_path=instance_mask_path,
                part_instance_mask_path=part_instance_mask_path,
                pose_path=pose_path,
                out_scene_dir=out_scene_dir,
                selected_instance_ids=selected_instance_ids,
                instance_mask_ids=instance_mask_ids,
                part_instance_mask_ids=part_instance_mask_ids,
                output_stem=output_stem,
            )

            converted += 1

            for instance_id in result["invisible_object_ids"]:
                invisible_object_counter[instance_id] = (
                    invisible_object_counter.get(instance_id, 0) + 1
                )

        except Exception as e:
            print(f"\n[SKIP] {dataset_name}/{stem}: {e}")
            skipped += 1

    return {
        "dataset": dataset_name,
        "rgb": len(rgb_files),
        "converted": converted,
        "skipped": skipped,
        "invisible_objects": invisible_object_counter,
    }


def find_dataset_folders(input_root):
    input_root = Path(input_root)

    if is_dataset_folder(input_root):
        return [input_root]

    if not input_root.is_dir():
        return []

    return [
        p
        for p in sorted(input_root.iterdir())
        if p.is_dir() and is_dataset_folder(p)
    ]


def convert_dataset(
    input_root,
    output_root,
    scene_name,
    requested_instance_id,
    instance_mask_ids,
    part_instance_mask_ids,
):
    input_root = Path(input_root)
    output_root = Path(output_root)

    if not input_root.exists():
        raise FileNotFoundError(
            f"Input path does not exist:\n{input_root}"
        )

    out_scene_dir = output_root / "data" / scene_name / "data"
    ensure_dir(out_scene_dir)

    dataset_dirs = find_dataset_folders(input_root)

    if not dataset_dirs:
        raise RuntimeError(
            f"No valid dataset folders found under:\n{input_root}\n\n"
            f"Each dataset must contain:\n"
            f"rgb/\n"
            f"depth/\n"
            f"pose/\n"
            f"instance_masks/ and part_instance_masks/ are optional."
        )

    print()
    print("=" * 70)
    print("Dataset Conversion")
    print("=" * 70)
    print(f"Input               : {input_root}")
    print(f"Datasets            : {len(dataset_dirs)}")
    print(
        f"Instance selection  : "
        f"{'ALL CONFIGURED' if requested_instance_id == 0 else requested_instance_id}"
    )
    print(f"Instance masks      : {instance_mask_ids}")
    print(f"Part instance masks : {part_instance_mask_ids}")
    print(f"Output              : {out_scene_dir}")
    print("=" * 70)

    total_rgb = 0
    total_converted = 0
    total_skipped = 0
    total_invisible_objects = {}
    results = []

    for dataset_dir in dataset_dirs:
        dataset_name = (
            ""
            if len(dataset_dirs) == 1
            else dataset_dir.name
        )

        print(f"\n[DATASET] {dataset_dir.name}")

        result = convert_single_dataset(
            input_root=dataset_dir,
            out_scene_dir=out_scene_dir,
            requested_instance_id=requested_instance_id,
            instance_mask_ids=instance_mask_ids,
            part_instance_mask_ids=part_instance_mask_ids,
            dataset_name=dataset_name,
        )

        results.append(result)

        total_rgb += result["rgb"]
        total_converted += result["converted"]
        total_skipped += result["skipped"]

        for instance_id, count in result["invisible_objects"].items():
            total_invisible_objects[instance_id] = (
                total_invisible_objects.get(instance_id, 0) + count
            )

    print()
    print("=" * 70)
    print("Conversion finished")
    print("=" * 70)

    for result in results:
        name = result["dataset"] or "[single dataset]"
        print(
            f"{name:30s} "
            f"RGB={result['rgb']:6d} "
            f"Converted={result['converted']:6d} "
            f"Skipped={result['skipped']:6d}"
        )

        if result["invisible_objects"]:
            print("    Invisible object frames:")
            for instance_id, count in sorted(
                result["invisible_objects"].items()
            ):
                print(
                    f"      instance_id={instance_id}: "
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
        print("Total invisible object frames:")
        for instance_id, count in sorted(
            total_invisible_objects.items()
        ):
            print(
                f"  instance_id={instance_id}: "
                f"{count} frame(s)"
            )

    print()
    print(f"Output: {out_scene_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bop_data_path",
        type=str,
        required=True,
        help="Single dataset folder or root containing multiple datasets.",
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
        help="0 = all configured instances; otherwise convert only the specified instance_id.",
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
        help="Instance IDs obtained from instance_masks. Example: 1,3",
    )
    parser.add_argument(
        "--part_instance_mask_ids",
        type=str,
        default="",
        help="Instance IDs obtained from part_instance_masks. Example: 2",
    )

    args = parser.parse_args()

    instance_mask_ids = parse_instance_ids(args.instance_mask_ids)
    part_instance_mask_ids = parse_instance_ids(args.part_instance_mask_ids)

    overlap = set(instance_mask_ids) & set(part_instance_mask_ids)

    if overlap:
        raise ValueError(
            f"Instance ID(s) appear in both mask sources: {sorted(overlap)}"
        )

    if not instance_mask_ids and not part_instance_mask_ids:
        raise ValueError(
            "No mask source configured. Use for example:\n"
            "--instance_mask_ids 1 --part_instance_mask_ids 2"
        )

    convert_dataset(
        input_root=Path(args.bop_data_path),
        output_root=Path(args.output_dir),
        scene_name=args.scene_name,
        requested_instance_id=args.obj_id,
        instance_mask_ids=instance_mask_ids,
        part_instance_mask_ids=part_instance_mask_ids,
    )


if __name__ == "__main__":
    main()
