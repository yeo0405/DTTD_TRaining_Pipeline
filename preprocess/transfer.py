#!/usr/bin/env python3
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


def find_scene_dirs(bop_root: Path):
    scene_dirs = []
    for cam_json in sorted(bop_root.rglob("scene_camera.json")):
        scene_dir = cam_json.parent
        gt_json = scene_dir / "scene_gt.json"
        rgb_dir = scene_dir / "rgb"
        if gt_json.exists() and rgb_dir.exists():
            scene_dirs.append(scene_dir)
    return scene_dirs


def resolve_frame_stem(rgb_dir: Path, frame_id: int):
    candidates = [
        f"{frame_id:06d}",
        f"{frame_id:05d}",
        str(frame_id),
    ]
    exts = [".png", ".jpg", ".jpeg"]
    for stem in candidates:
        for ext in exts:
            p = rgb_dir / f"{stem}{ext}"
            if p.exists():
                return stem, p
    return None, None


def resolve_image_path(folder: Path, stem: str):
    exts = [".png", ".jpg", ".jpeg"]
    for ext in exts:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def depth_to_uint16(depth_img, depth_scale=None):
    depth = np.asarray(depth_img)

    # BOP depth:
    # real_depth = raw_depth * depth_scale
    if depth_scale is not None and depth_scale > 0:
        depth_mm = depth.astype(np.float32) * depth_scale

        return np.clip(
            np.round(depth_mm),
            0,
            65535
        ).astype(np.uint16)


    # fallback for non-BOP data
    if np.issubdtype(depth.dtype, np.floating):

        max_val = float(np.nanmax(depth)) if depth.size else 0.0

        # meter input
        if max_val <= 100.0:
            depth_mm = depth * 1000.0
        else:
            depth_mm = depth

        return np.clip(
            np.round(depth_mm),
            0,
            65535
        ).astype(np.uint16)


    # already uint16 mm
    return depth.astype(np.uint16)


def make_pose_4x4(obj_entry):
    r = np.array(obj_entry["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
    t = np.array(obj_entry["cam_t_m2c"], dtype=np.float32).reshape(3) / 1000.0
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = r
    pose[:3, 3] = t
    return pose.tolist()


def convert_scene(scene_dir, out_root, target_obj_id, desc=""):
    scene_name = scene_dir.name
    rgb_dir = scene_dir / "rgb"
    depth_dir = scene_dir / "depth"
    mask_dir = scene_dir / "mask_visib"

    scene_camera = load_json(scene_dir / "scene_camera.json")
    scene_gt = load_json(scene_dir / "scene_gt.json")

    out_scene_dir = out_root / "data" / scene_name / "data"
    ensure_dir(out_scene_dir)

    kept_lines = []

    for frame_key in tqdm(sorted(scene_gt.keys(), key=lambda x: int(x)), desc=desc, leave=False):
        frame_id = int(frame_key)
        stem, rgb_path = resolve_frame_stem(rgb_dir, frame_id)
        if rgb_path is None:
            continue

        depth_path = resolve_image_path(depth_dir, stem)
        if depth_path is None:
            continue

        cam_info = scene_camera.get(frame_key, scene_camera.get(str(frame_id), {}))
        depth_scale = cam_info.get("depth_scale", None)
        cam_K = np.array(
            cam_info.get("cam_K", np.eye(3).reshape(-1).tolist()),
            dtype=np.float32,
        ).reshape(3, 3)
        cam_dist = cam_info.get("cam_dist", [0, 0, 0, 0, 0])

        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            continue

        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            continue
        depth_u16 = depth_to_uint16(depth_raw, depth_scale=depth_scale)

        objs = scene_gt[frame_key]
        filtered = []
        seen = set()

        for idx, entry in enumerate(objs):
            obj_id = int(entry["obj_id"])

            # 0 = keep all objects
            if target_obj_id != 0 and obj_id != target_obj_id:
                continue

            # keep only one instance for each object id
            if obj_id in seen:
                continue

            seen.add(obj_id)
            filtered.append((idx, entry))

        if not filtered:
            continue

        label = np.zeros(depth_u16.shape[:2], dtype=np.uint16)
        object_poses = {}
        object_ids = []

        for orig_idx, entry in filtered:
            obj_id = int(entry["obj_id"])
            object_ids.append(obj_id)
            object_poses[str(obj_id)] = make_pose_4x4(entry)

            mask_path = None
            if mask_dir.exists():
                mask_path = mask_dir / f"{stem}_{orig_idx:06d}.png"
                if not mask_path.exists():
                    mask_path = mask_dir / f"{stem}_{orig_idx}.png"

            if mask_path is None or not mask_path.exists():
                print(f"[Warning] mask not found for {stem}, orig_idx={orig_idx}, obj_id={obj_id}")
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue

            if mask.ndim == 3:
                mask = mask[:, :, 0]
            label[mask > 0] = obj_id

        if len(object_ids) == 0:
            continue

        color_out = out_scene_dir / f"{stem}_color.jpg"
        depth_out = out_scene_dir / f"{stem}_depth.png"
        label_out = out_scene_dir / f"{stem}_label.png"
        meta_out = out_scene_dir / f"{stem}_meta.json"

        cv2.imwrite(str(color_out), rgb)
        cv2.imwrite(str(depth_out), depth_u16)
        cv2.imwrite(str(label_out), label)

        meta = {
            "objects": object_ids,
            "object_poses": object_poses,
            "intrinsic": cam_K.tolist(),
            "distortion": cam_dist,
        }

        with open(meta_out, "w") as f:
            json.dump(meta, f)

        kept_lines.append(f"{scene_name}/data/{stem}")

    return kept_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bop_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--obj_id",
        type=int,
        default=0,
        help="0=keep all objects, otherwise keep only specified obj_id",
    )
    args = parser.parse_args()

    bop_root = Path(args.bop_data_path)
    out_root = Path(args.output_dir)
    ensure_dir(out_root / "data")

    target_obj_id = args.obj_id

    scene_dirs = sorted(find_scene_dirs(bop_root), key=lambda p: p.name)

    all_lines = []
    for scene_dir in tqdm(scene_dirs, desc="convert scenes"):
        all_lines.extend(
            convert_scene(
                scene_dir,
                out_root,
                target_obj_id,
                desc=scene_dir.name,
            )
        )

    print(f"converted frames: {len(all_lines)}")
    print(f"output: {out_root}")


if __name__ == "__main__":
    main()