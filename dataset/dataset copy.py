import json
import os
import random

import numpy as np
import numpy.ma as ma
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

Borderlist = [-1] + list(range(40, 1960, 40))


class DTTDDataset(data.Dataset):
    def __init__(
        self,
        root,
        mode,
        refine=False,
        add_noise=True,
        config_path="/home/yeo/Downloads/POSE/dataset/dataset_config",
    ):
        self.Trainlist = os.path.join(config_path, "train_data_list.txt")
        self.Testlist = os.path.join(config_path, "test_data_list.txt")
        self.Classlist = os.path.join(config_path, "objectids.csv")

        self.mode = mode
        self.root = root
        self.refine = refine
        self.add_noise = add_noise

        self._init_config()
        self._init_data_info()

        self.model_points = self.load_points(self.root, self.classes)
        self._validate_model_points()

    def __len__(self):
        return len(self.all_data_dirs)

    def __getitem__(self, index):
        base_path = os.path.join(
            self.root,
            self.prefix,
            str(self.all_data_dirs[index]),
        )

        rgb_path = base_path + "_color.jpg"
        depth_path = base_path + "_depth.png"
        label_path = base_path + "_label.png"
        meta_path = base_path + "_meta.json"

        img = Image.open(rgb_path)
        depth = np.array(Image.open(depth_path), dtype=np.uint16)
        label = np.array(Image.open(label_path))

        with open(meta_path, "r") as f:
            meta = json.load(f)

        img_np = np.array(img)
        img_h, img_w = label.shape

        if img_np.shape[0] != img_h or img_np.shape[1] != img_w:
            raise RuntimeError(
                f"RGB / Label resolution mismatch:\n"
                f"RGB   : {img_np.shape}\n"
                f"Label : {label.shape}"
            )

        if depth.shape != label.shape:
            raise RuntimeError(
                f"Depth / Label resolution mismatch:\n"
                f"Depth : {depth.shape}\n"
                f"Label : {label.shape}"
            )

        cam = np.asarray(meta["intrinsic"], dtype=np.float32)
        cam_fx = cam[0, 0]
        cam_fy = cam[1, 1]
        cam_cx = cam[0, 2]
        cam_cy = cam[1, 2]

        xmap = np.broadcast_to(
            np.arange(img_h, dtype=np.float32)[:, None],
            (img_h, img_w),
        )
        ymap = np.broadcast_to(
            np.arange(img_w, dtype=np.float32)[None, :],
            (img_h, img_w),
        )

        if self.add_noise:
            img = self.trancolor(img)

        img = np.array(img)
        objs = np.asarray(meta["objects"], dtype=np.int32).flatten()

        if len(objs) == 0:
            return self._get_negative_sample(img, label, depth, meta)

        obj_indices = list(range(len(objs)))
        random.shuffle(obj_indices)
        obj_found = False

        for obj_idx in obj_indices:
            obj_id = int(objs[obj_idx])
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj_id))

            mask = (
                mask_label * mask_depth
                if self.use_labelmask
                else mask_depth
            )

            valid_px = int((mask_label * mask_depth).sum())
            if valid_px > self.minimum_px_num:
                obj_found = True
                break

        if not obj_found:
            return self._get_negative_sample(img, label, depth, meta)

        obj_id = int(objs[obj_idx])

        if obj_id not in self.model_points:
            raise RuntimeError(
                f"Model points not found for object ID {obj_id}.\n"
                f"Available model point IDs: {sorted(self.model_points.keys())}\n"
                f"Sample: {self.all_data_dirs[index]}"
            )

        pose = np.asarray(
            meta["object_poses"][str(obj_id)],
            dtype=np.float32,
        )
        R_gt = pose[0:3, 0:3]
        T_gt = pose[0:3, 3:4].T

        model_points = self.model_points[obj_id]
        required_points = (
            self.pt_num_mesh_large
            if self.refine
            else self.pt_num_mesh_small
        )

        if len(model_points) < required_points:
            raise RuntimeError(
                f"Not enough model points for object {obj_id}.\n"
                f"Available : {len(model_points)}\n"
                f"Required  : {required_points}\n"
                f"Sample    : {self.all_data_dirs[index]}"
            )

        model_sample_list = list(range(len(model_points)))
        model_sample_list = sorted(
            random.sample(model_sample_list, required_points)
        )

        sampled_model_pt = np.asarray(
            model_points[model_sample_list, :],
            dtype=np.float32,
        )
        sampled_model_pt_world = np.add(
            np.dot(sampled_model_pt, R_gt.T),
            T_gt,
        )

        rmin, rmax, cmin, cmax = get_discrete_width_bbox(
            mask_label,
            Borderlist,
            img_w,
            img_h,
        )

        sample2D = (
            mask[rmin:rmax, cmin:cmax]
            .flatten()
            .nonzero()[0]
        )

        if len(sample2D) >= self.sample_2d_pt_num:
            sample2D = np.asarray(
                sorted(
                    np.random.choice(
                        sample2D,
                        self.sample_2d_pt_num,
                        replace=False,
                    )
                ),
                dtype=np.int64,
            )
        elif len(sample2D) == 0:
            sample2D = np.zeros(self.sample_2d_pt_num, dtype=np.int64)
        else:
            sample2D = np.pad(
                sample2D,
                (0, self.sample_2d_pt_num - len(sample2D)),
                mode="wrap",
            ).astype(np.int64)

        img_crop = np.transpose(
            img[:, :, :3],
            (2, 0, 1),
        )[:, rmin:rmax, cmin:cmax]

        depth_crop = (
            depth[rmin:rmax, cmin:cmax]
            .flatten()[sample2D][:, None]
            .astype(np.float32)
        )
        xmap_crop = (
            xmap[rmin:rmax, cmin:cmax]
            .flatten()[sample2D][:, None]
            .astype(np.float32)
        )
        ymap_crop = (
            ymap[rmin:rmax, cmin:cmax]
            .flatten()[sample2D][:, None]
            .astype(np.float32)
        )

        cam_scale = 1000.0
        pz = depth_crop / cam_scale
        px = (ymap_crop - cam_cx) * pz / cam_fx
        py = (xmap_crop - cam_cy) * pz / cam_fy

        point_cloud = np.concatenate((px, py, pz), axis=1)

        if self.add_noise:
            add_noise_t = np.array(
                [
                    random.uniform(-self.noise_trans, self.noise_trans)
                    for _ in range(3)
                ],
                dtype=np.float32,
            )
            point_cloud = np.add(point_cloud, add_noise_t)
            sampled_model_pt_world = np.add(
                sampled_model_pt_world,
                add_noise_t,
            )

        return {
            "img": torch.from_numpy(img),
            "label": torch.from_numpy(label),
            "depth": torch.from_numpy(depth.astype(np.float32)),
            "point_cloud": torch.from_numpy(point_cloud.astype(np.float32)),
            "sample_2d": torch.from_numpy(sample2D.astype(np.int64)),
            "img_crop": self.norm(
                torch.from_numpy(img_crop.astype(np.float32))
            ),
            "sampled_model_pt_camera": torch.from_numpy(
                sampled_model_pt_world.astype(np.float32)
            ),
            "sampled_model_pt": torch.from_numpy(
                sampled_model_pt.astype(np.float32)
            ),
            "obj_id": torch.LongTensor([obj_id - 1]),
            "R": torch.from_numpy(R_gt.astype(np.float32)),
            "T": torch.from_numpy(T_gt.astype(np.float32)),
        }

    def _get_negative_sample(self, img, label, depth, meta):
        img_h, img_w = label.shape
        rmin, rmax = 0, min(img_h, 400)
        cmin, cmax = 0, min(img_w, 400)

        img_crop = np.transpose(
            np.asarray(img)[:, :, :3],
            (2, 0, 1),
        )[:, rmin:rmax, cmin:cmax]

        return {
            "img": torch.from_numpy(np.asarray(img)),
            "label": torch.from_numpy(label),
            "depth": torch.from_numpy(depth.astype(np.float32)),
            "point_cloud": torch.zeros(
                (self.sample_2d_pt_num, 3),
                dtype=torch.float32,
            ),
            "sample_2d": torch.zeros(
                self.sample_2d_pt_num,
                dtype=torch.long,
            ),
            "img_crop": self.norm(
                torch.from_numpy(img_crop.astype(np.float32))
            ),
            "sampled_model_pt_camera": torch.zeros(
                (self.get_model_point_num(), 3),
                dtype=torch.float32,
            ),
            "sampled_model_pt": torch.zeros(
                (self.get_model_point_num(), 3),
                dtype=torch.float32,
            ),
            "obj_id": torch.LongTensor([-1]),
            "R": torch.eye(3, dtype=torch.float32),
            "T": torch.zeros((1, 3), dtype=torch.float32),
            "is_negative_sample": True,
        }

    def _init_config(self):
        self.prefix = "data"
        self.use_labelmask = True
        self.debug_mode = False

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.noise_trans = 0.02
        self.minimum_px_num = 50
        self.symmetry_obj_cls = []
        self.sample_2d_pt_num = 1000
        self.pt_num_mesh_small = 500
        self.pt_num_mesh_large = 2600
        self.num_obj = 20

    def _init_data_info(self):
        if self.mode == "train":
            self.data_list_path = self.Trainlist
        elif self.mode == "test":
            self.data_list_path = self.Testlist
        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

        self.class_path = self.Classlist

        with open(self.data_list_path, "r") as f:
            self.all_data_dirs = [
                line.strip() for line in f if line.strip()
            ]

        self.real_data_dirs = [
            d for d in self.all_data_dirs if d.startswith("scene")
        ]
        self.syn_data_dirs = [
            d for d in self.all_data_dirs if d.startswith("synthetic/data")
        ]

        self.classes = pd.read_csv(self.class_path, index_col="id")
        self.symmetry_obj_cls = []

        if "symmetry" in self.classes.columns:
            for idx, row in self.classes.iterrows():
                if int(row["symmetry"]) == 1:
                    self.symmetry_obj_cls.append(int(idx) - 1)

        print("[Dataset] Symmetric objects:", self.symmetry_obj_cls)

    def load_points(self, root, classes):
        model_points = {}

        print()
        print("=" * 70)
        print("Loading model points")
        print("=" * 70)

        for idx, cls in classes.iterrows():
            obj_id = int(idx)
            cls_filepath = os.path.join(
                root,
                "objects",
                str(cls["name"]),
                "points.xyz",
            )

            if os.path.isfile(cls_filepath):
                try:
                    points = np.loadtxt(cls_filepath, dtype=np.float32)
                    points = np.asarray(points)

                    if points.ndim == 1:
                        points = points.reshape(1, -1)

                    if points.ndim != 2 or points.shape[1] != 3:
                        print(
                            f"[ERROR] Object {obj_id:3d} | "
                            f"Invalid points shape {points.shape} | "
                            f"{cls_filepath}"
                        )
                        continue

                    model_points[obj_id] = points
                    print(
                        f"[OK] Object {obj_id:3d} | "
                        f"{len(points):6d} points | "
                        f"{cls_filepath}"
                    )
                except Exception as e:
                    print(
                        f"[ERROR] Object {obj_id:3d} | Failed to load | "
                        f"{cls_filepath}\n        {e}"
                    )
            else:
                print(f"[MISSING] Object {obj_id:3d} | {cls_filepath}")

        print(
            f"Loaded model points: "
            f"{len(model_points)} / {len(classes)} objects"
        )
        print("=" * 70)
        print()

        return model_points

    def _validate_model_points(self):
        dataset_object_ids = set()

        for data_dir in self.all_data_dirs:
            meta_path = os.path.join(
                self.root,
                self.prefix,
                f"{data_dir}_meta.json",
            )

            if not os.path.isfile(meta_path):
                continue

            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                for obj_id in meta.get("objects", []):
                    dataset_object_ids.add(int(obj_id))
            except Exception as e:
                print(
                    f"[WARNING] Failed to read metadata: {meta_path}\n"
                    f"          {e}"
                )

        model_point_ids = set(int(k) for k in self.model_points.keys())
        missing_ids = sorted(dataset_object_ids - model_point_ids)

        print()
        print("=" * 70)
        print("MODEL POINT VALIDATION")
        print("=" * 70)
        print("Dataset object IDs :", sorted(dataset_object_ids))
        print("Model point IDs    :", sorted(model_point_ids))

        if missing_ids:
            print("[ERROR] Missing model points for object IDs:", missing_ids)
            for obj_id in missing_ids:
                if obj_id in self.classes.index:
                    name = self.classes.loc[obj_id, "name"]
                    expected_path = os.path.join(
                        self.root,
                        "objects",
                        str(name),
                        "points.xyz",
                    )
                    print(f"  Object {obj_id}: {expected_path}")

            raise RuntimeError(
                "Dataset contains objects without corresponding model points."
            )

        print("[OK] All dataset objects have model points.")
        print("=" * 70)
        print()

    def get_object_num(self):
        return self.num_obj

    def get_sym_list(self):
        return self.symmetry_obj_cls

    def get_model_point_num(self):
        if self.refine:
            return self.pt_num_mesh_large
        return self.pt_num_mesh_small

    def get_2d_sample_num(self):
        return self.sample_2d_pt_num


def get_discrete_width_bbox(label, border_list, img_w, img_h):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)

    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        return (0, min(img_h, 1), 0, min(img_w, 1))

    rmin, rmax = row_indices[[0, -1]]
    cmin, cmax = col_indices[[0, -1]]

    rmax += 1
    cmax += 1

    r_b = border_list[binary_search(border_list, rmax - rmin)]
    c_b = border_list[binary_search(border_list, cmax - cmin)]

    center = [
        int((rmin + rmax) / 2),
        int((cmin + cmax) / 2),
    ]

    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)

    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt

    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt

    if rmax > img_h:
        delt = rmax - img_h
        rmax = img_h
        rmin -= delt

    if cmax > img_w:
        delt = cmax - img_w
        cmax = img_w
        cmin -= delt

    rmin = max(0, rmin)
    cmin = max(0, cmin)
    rmax = min(img_h, rmax)
    cmax = min(img_w, cmax)

    return rmin, rmax, cmin, cmax


def discretize_bbox(rmin, rmax, cmin, cmax, border_list, img_w, img_h):
    rmax += 1
    cmax += 1

    r_b = border_list[binary_search(border_list, rmax - rmin)]
    c_b = border_list[binary_search(border_list, cmax - cmin)]

    center = [
        int((rmin + rmax) / 2),
        int((cmin + cmax) / 2),
    ]

    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)

    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt

    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt

    if rmax > img_h:
        delt = rmax - img_h
        rmax = img_h
        rmin -= delt

    if cmax > img_w:
        delt = cmax - img_w
        cmax = img_w
        cmin -= delt

    rmin = max(0, rmin)
    cmin = max(0, cmin)
    rmax = min(img_h, rmax)
    cmax = min(img_w, cmax)

    return rmin, rmax, cmin, cmax


def binary_search(sorted_list, target):
    left = 0
    right = len(sorted_list) - 1

    while left != right:
        mid = (left + right) >> 1
        if sorted_list[mid] > target:
            right = mid
        elif sorted_list[mid] < target:
            left = mid + 1
        else:
            return mid

    return left


if __name__ == "__main__":
    dataset = DTTDDataset(
        root="./DTTD_IPhone_Dataset/root",
        mode="test",
        config_path="./dataset_config",
    )

    print()
    print("=" * 70)
    print("DATASET TEST")
    print("=" * 70)

    print("Dataset length:", len(dataset))
    print("Prefix:", dataset.prefix)
    print("Model point IDs:", sorted(dataset.model_points.keys()))

    dt = dataset[0]

    print()
    print("Returned keys:")

    for key, value in dt.items():
        if torch.is_tensor(value):
            print(f"  {key:30s} {tuple(value.shape)} {value.dtype}")
        else:
            print(f"  {key:30s} {type(value)}")

    print()
    print("Dataset test finished.")