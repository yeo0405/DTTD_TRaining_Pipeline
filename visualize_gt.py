#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_ply_vertices(path):
    vertices = []
    with open(path, "r") as f:
        header = True
        num_vertices = None
        for line in f:
            line = line.strip()
            if header:
                if line.startswith("element vertex"):
                    num_vertices = int(line.split()[-1])
                elif line == "end_header":
                    header = False
                continue
            if num_vertices is not None and len(vertices) < num_vertices:
                parts = line.split()
                if len(parts) >= 3:
                    vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
    vertices = np.asarray(vertices, dtype=np.float32)
    if len(vertices) == 0:
        raise RuntimeError(f"No vertices found in {path}")
    return vertices


def transform_points(points, pose):
    R = pose[:3, :3]
    t = pose[:3, 3]
    return points @ R.T + t


def project_points(points_camera, K):
    z = points_camera[:, 2]
    valid = z > 1e-6
    uv = np.zeros((len(points_camera), 2), dtype=np.float32)
    x = points_camera[:, 0]
    y = points_camera[:, 1]
    uv[valid, 0] = K[0, 0] * x[valid] / z[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * y[valid] / z[valid] + K[1, 2]
    return uv, valid


def draw_axis(image, pose, K, axis_length=0.05):
    R = pose[:3, :3]
    t = pose[:3, 3]
    origin = t
    axis_points = np.array(
        [
            origin,
            origin + R[:, 0] * axis_length,
            origin + R[:, 1] * axis_length,
            origin + R[:, 2] * axis_length,
        ]
    )
    uv, valid = project_points(axis_points, K)
    if not np.all(valid):
        return
    p0 = tuple(np.round(uv[0]).astype(int))
    px = tuple(np.round(uv[1]).astype(int))
    py = tuple(np.round(uv[2]).astype(int))
    pz = tuple(np.round(uv[3]).astype(int))

    cv2.line(image, p0, px, (0, 0, 255), 3)
    cv2.line(image, p0, py, (0, 255, 0), 3)
    cv2.line(image, p0, pz, (255, 0, 0), 3)
    cv2.circle(image, p0, 5, (255, 255, 255), -1)


def draw_model_projection(image, model_points, pose, K):
    pts_camera = transform_points(model_points, pose)
    uv, valid = project_points(pts_camera, K)
    h, w = image.shape[:2]
    for p, v in zip(uv, valid):
        if not v:
            continue
        x = int(round(p[0]))
        y = int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), 1, (0, 255, 255), -1)


def get_bbox_corners(points):
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    x0, y0, z0 = min_xyz
    x1, y1, z1 = max_xyz
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    return corners


def draw_bbox(image, model_points, pose, K):
    corners = get_bbox_corners(model_points)
    corners_camera = transform_points(corners, pose)
    uv, valid = project_points(corners_camera, K)
    if not np.all(valid):
        return
    uv = np.round(uv).astype(int)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for a, b in edges:
        cv2.line(image, tuple(uv[a]), tuple(uv[b]), (0, 255, 255), 2)


def draw_mask_overlay(image, label, obj_id):
    mask = label == obj_id
    overlay = image.copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    return overlay


def draw_pose_info(image, obj_id, pose):
    t = pose[:3, 3]
    text1 = f"OBJ {obj_id}"
    text2 = f"T = " f"{t[0]:.4f}, " f"{t[1]:.4f}, " f"{t[2]:.4f} m"
    cv2.putText(image, text1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(image, text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def process_frame(data_dir, stem, output_dir, model_dir=None):
    color_path = data_dir / f"{stem}_color.jpg"
    depth_path = data_dir / f"{stem}_depth.png"
    label_path = data_dir / f"{stem}_label.png"
    meta_path = data_dir / f"{stem}_meta.json"
    if not color_path.exists():
        print(f"[SKIP] RGB missing: {color_path}")
        return
    if not label_path.exists():
        print(f"[SKIP] label missing: {label_path}")
        return
    if not meta_path.exists():
        print(f"[SKIP] meta missing: {meta_path}")
        return

    image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    meta = load_json(meta_path)
    if image is None:
        print(f"[SKIP] failed to read RGB")
        return
    if label is None:
        print(f"[SKIP] failed to read label")
        return

    K = np.asarray(meta["intrinsic"], dtype=np.float32)
    objects = meta["objects"]
    object_poses = meta["object_poses"]

    print()
    print("=" * 70)
    print(f"Frame: {stem}")
    print(f"Image: {image.shape}")
    print(f"Objects: {objects}")
    print("K:")
    print(K)

    for obj_id in objects:
        obj_id = int(obj_id)
        if str(obj_id) not in object_poses:
            print(f"[WARNING] " f"pose missing for object {obj_id}")
            continue
        pose = np.asarray(object_poses[str(obj_id)], dtype=np.float32)
        print()
        print(f"Object {obj_id}")
        print("GT Pose:")
        print(pose)

        mask = label == obj_id
        ys, xs = np.where(mask)

        if len(xs) == 0:
            print(f"[WARNING] " f"No mask pixels for object {obj_id}")
        else:
            print(f"Mask pixels: {len(xs)}")
            print(f"Mask bbox: " f"x={xs.min()}:{xs.max()} " f"y={ys.min()}:{ys.max()}")

        vis = image.copy()
        # mask
        vis = draw_mask_overlay(vis, label, obj_id)
        if model_dir is not None:
            model_path = model_dir / f"{obj_id}.ply"
            if model_path.exists():
                try:
                    model_points = load_ply_vertices(model_path)
                    # projected model
                    draw_model_projection(vis, model_points, pose, K)
                    # bbox
                    draw_bbox(vis, model_points, pose, K)
                except Exception as e:
                    print(f"[WARNING] " f"Model visualization failed: {e}")
            else:
                print(f"[WARNING] " f"Model not found: {model_path}")

        draw_axis(vis, pose, K, axis_length=0.05)
        draw_pose_info(vis, obj_id, pose)
        obj_output = output_dir / f"{stem}_obj{obj_id}_gt.png"
        cv2.imwrite(str(obj_output), vis)
        print(f"[SAVE] {obj_output}")

    label_vis = np.zeros_like(image)
    for obj_id in objects:
        obj_id = int(obj_id)
        mask = label == obj_id
        label_vis[mask] = (0, 255, 0)
    label_output = output_dir / f"{stem}_mask.png"
    cv2.imwrite(str(label_output), label_vis)
    side = np.concatenate([image, label_vis], axis=1)
    side_output = output_dir / f"{stem}_compare.png"
    cv2.imwrite(str(side_output), side)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="DTTD dataset root")
    parser.add_argument(
        "--output", default="./gt_visualization", help="output visualization directory"
    )
    parser.add_argument(
        "--model_dir", default=None, help="directory containing object PLY files"
    )
    parser.add_argument(
        "--frame", default=None, help="specific frame stem, e.g. 000001"
    )
    parser.add_argument("--max_frames", type=int, default=20)
    args = parser.parse_args()
    dataset_root = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dirs = sorted(dataset_root.rglob("*_color.jpg"))
    print(f"[INFO] Found {len(data_dirs)} frames")
    count = 0
    for color_file in data_dirs:
        if count >= args.max_frames:
            break
        stem = color_file.name.replace("_color.jpg", "")
        if args.frame is not None:
            if stem != args.frame:
                continue
        data_dir = color_file.parent
        process_frame(
            data_dir=data_dir,
            stem=stem,
            output_dir=output_dir,
            model_dir=(Path(args.model_dir) if args.model_dir else None),
        )
        count += 1
        if args.frame is not None:
            break

    print()
    print("=" * 70)
    print(f"[DONE] Visualized {count} frames")
    print(f"[DONE] Output: {output_dir}")


if __name__ == "__main__":
    main()
