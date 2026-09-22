#!/usr/bin/env python3

import os
import sys
import cv2
import json
import argparse
from datetime import datetime

import numpy as np
import numpy.ma as ma
from PIL import Image
import scipy.io as scio
import yaml

import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable

sys.path.append("../../")

from dataset.dataset import DTTDDataset, Borderlist
from model.posefusion import PoseNet
from utils.log import Logger
from utils.file import get_checkpoint
from utils.visualizer import visualize
from utils.transformations import quaternion_matrix
from utils.image import get_discrete_width_bbox


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_root',
        type=str,
        default='../../dataset/cartridge/DTTD_Cartridge_Dataset/root',
        help='dataset root dir'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./eval_results',
        help='output directory to save results'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='',
        help='path to resume model file'
    )
    parser.add_argument(
        '--result',
        type=str,
        default='eval_results',
        help='Directory to save results'
    )
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--filter', action='store_true')

    parser.add_argument(
        '--base_latent',
        type=int,
        default=256,
        help='base latent dim for unimodal encoder'
    )
    parser.add_argument(
        '--embed_dim',
        type=int,
        default=512,
        help='embedding dim for transformer encoder'
    )
    parser.add_argument(
        '--fusion_block_num',
        type=int,
        default=1,
        help='number of fusion block'
    )
    parser.add_argument(
        '--layer_num_m',
        type=int,
        default=4,
        help='layer num for modality fusion per block'
    )
    parser.add_argument(
        '--layer_num_p',
        type=int,
        default=2,
        help='layer num for point-to-point fusion per block'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='number of pose hypotheses used for aggregation'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='maximum number of randomly sampled test frames; 0 means all'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='random sampling seed'
    )

    return parser.parse_args()


def rotation_error_deg(R_pred, R_gt):
    cos_theta = (np.trace(R_pred @ R_gt.T) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def translation_error_mm(T_pred, T_gt):
    return float(
        np.linalg.norm(
            T_pred.reshape(-1) - T_gt.reshape(-1)
        ) * 1000.0
    )


def process_data(
    itemid,
    img,
    depth,
    label,
    meta,
    border_list,
    num_points=1000,
    norm=None
):
    img_h, img_w, _ = img.shape

    xmap = np.array([
        [j for i in range(img_w)]
        for j in range(img_h)
    ])

    ymap = np.array([
        [i for i in range(img_w)]
        for j in range(img_h)
    ])

    mask_depth = ma.getmaskarray(
        ma.masked_not_equal(depth, 0)
    )

    mask_label = ma.getmaskarray(
        ma.masked_equal(label, itemid)
    )

    mask = mask_label * mask_depth

    rmin, rmax, cmin, cmax = get_discrete_width_bbox(
        mask_label,
        border_list,
        img_w,
        img_h
    )

    sample = (
        mask[rmin:rmax, cmin:cmax]
        .flatten()
        .nonzero()[0]
    )

    if len(sample) >= num_points:
        sample = np.array(
            sorted(
                np.random.choice(
                    sample,
                    num_points
                )
            )
        )
    elif len(sample) == 0:
        sample = np.pad(
            sample,
            (0, num_points - len(sample)),
            'constant'
        )
    else:
        sample = np.pad(
            sample,
            (0, num_points - len(sample)),
            'wrap'
        )

    img_crop = np.transpose(
        img[:, :, :3],
        (2, 0, 1)
    )[:, rmin:rmax, cmin:cmax]

    depth_crop = (
        depth[rmin:rmax, cmin:cmax]
        .flatten()[sample][:, np.newaxis]
        .astype(np.float32)
    )

    xmap_crop = (
        xmap[rmin:rmax, cmin:cmax]
        .flatten()[sample][:, np.newaxis]
        .astype(np.float32)
    )

    ymap_crop = (
        ymap[rmin:rmax, cmin:cmax]
        .flatten()[sample][:, np.newaxis]
        .astype(np.float32)
    )

    cam = np.array(meta['intrinsic'])

    cam_cx = cam[0][2]
    cam_cy = cam[1][2]
    cam_fx = cam[0][0]
    cam_fy = cam[1][1]

    pz = depth_crop / 1000.0

    px = (
        (ymap_crop - cam_cx)
        * pz
        / cam_fx
    )

    py = (
        (xmap_crop - cam_cy)
        * pz
        / cam_fy
    )

    point_cloud = np.concatenate(
        (px, py, pz),
        axis=1
    )

    if norm is None:
        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    return {
        "cloud": torch.from_numpy(
            point_cloud.astype(np.float32)
        ),
        "sample": torch.LongTensor(
            sample.astype(np.int32)
        ),
        "img_crop": norm(
            torch.from_numpy(
                img_crop.astype(np.float32)
            )
        ),
        "index": torch.LongTensor(
            [itemid - 1]
        )
    }


def aggregate_topk_pose(
    pred_r,
    pred_t,
    pred_c,
    points,
    top_k=50
):
    pred_r = pred_r / (
        torch.norm(
            pred_r,
            dim=2,
            keepdim=True
        ) + 1e-8
    )

    pred_c = pred_c.view(-1)
    pred_t = pred_t.view(-1, 3)
    points = points.view(-1, 3)
    pred_r = pred_r[0]

    actual_top_k = min(
        top_k,
        pred_c.numel()
    )

    top_conf, top_idx = torch.topk(
        pred_c,
        actual_top_k,
        largest=True,
        sorted=True
    )

    weights = torch.softmax(
        top_conf,
        dim=0
    )

    candidate_trans = points + pred_t

    top_trans = candidate_trans[
        top_idx
    ]

    trans = (
        top_trans
        * weights.unsqueeze(1)
    ).sum(dim=0)

    top_quat = pred_r[
        top_idx
    ]

    reference_quat = top_quat[0]

    dots = torch.sum(
        top_quat
        * reference_quat.unsqueeze(0),
        dim=1
    )

    signs = torch.where(
        dots < 0,
        -torch.ones_like(dots),
        torch.ones_like(dots)
    )

    aligned_quat = (
        top_quat
        * signs.unsqueeze(1)
    )

    quat = (
        aligned_quat
        * weights.unsqueeze(1)
    ).sum(dim=0)

    quat_norm = torch.norm(quat)

    if quat_norm < 1e-8:
        raise RuntimeError(
            "Quaternion averaging produced near-zero quaternion."
        )

    quat = quat / quat_norm

    return (
        quat.detach().cpu().numpy(),
        trans.detach().cpu().numpy(),
        top_idx.detach().cpu().numpy(),
        top_conf.detach().cpu().numpy(),
        weights.detach().cpu().numpy()
    )


def save_ptcld(xyz, fn):
    with open(fn, "w") as f:
        for i in range(len(xyz)):
            f.write(
                f"v {xyz[i][0]} {xyz[i][1]} {xyz[i][2]} \n"
            )


def save_freq(estimator, x):
    estimator.get_freq_domain(x)


def save_att(
    estimator,
    img_crop,
    cloud,
    sample,
    fn_
):
    estimator.eval()

    attn1, attn2 = estimator.get_attention_map(
        img_crop,
        cloud,
        sample
    )

    hn1 = attn1.shape[0]
    hn2 = attn2.shape[0]

    draw_attn_map(
        fn_ + 'm_avg.png',
        attn1
    )

    for i in range(hn1):
        draw_attn_map(
            fn_ + f'm_{i}.png',
            attn1,
            i
        )

    draw_attn_map(
        fn_ + 'p_avg.png',
        attn2
    )

    for i in range(hn2):
        draw_attn_map(
            fn_ + f'p_{i}.png',
            attn2,
            i
        )


def draw_attn_map(
    fn,
    attn,
    head_idx=None
):
    if head_idx is not None:
        attn = attn[head_idx]
    else:
        attn = torch.mean(attn, dim=0)

    attn = np.asarray(
        attn.detach().cpu().numpy()
    )

    min_value = np.min(attn)
    max_value = np.max(attn)

    if max_value - min_value > 1e-12:
        attn = (
            255
            * (attn - min_value)
            / (max_value - min_value)
        )
    else:
        attn = np.zeros_like(attn)

    Image.fromarray(
        attn.astype(np.uint8)
    ).save(fn)


def numpy_to_python(obj):
    if isinstance(obj, dict):
        return {
            str(k): numpy_to_python(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            numpy_to_python(v)
            for v in obj
        ]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    return obj


def save_yaml(yaml_path, data):
    data = numpy_to_python(data)

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False
        )


def eval():
    opt = parse()

    dataset = DTTDDataset(
        opt.dataset_root,
        "test",
        add_noise=False,
        config_path=os.path.join(opt.dataset_root, "dataset_config"),
    )

    os.makedirs(
        opt.output,
        exist_ok=True
    )

    mats_dir = os.path.join(
        opt.output,
        "mats"
    )

    os.makedirs(
        mats_dir,
        exist_ok=True
    )

    if opt.visualize:
        visualize_dir = os.path.join(
            opt.output,
            "visualize"
        )

        os.makedirs(
            visualize_dir,
            exist_ok=True
        )

        color_list = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 255, 255),
            (123, 10, 265),
            (245, 163, 101),
            (100, 100, 178)
        ]

    estimator = PoseNet(
        num_points=dataset.sample_2d_pt_num,
        num_obj=dataset.num_obj,
        base_latent=opt.base_latent,
        embedding_dim=opt.embed_dim,
        fusion_block_num=opt.fusion_block_num,
        layer_num_m=opt.layer_num_m,
        layer_num_p=opt.layer_num_p,
        filter_enhance=opt.filter
    )

    estimator.cuda()

    opt.model = get_checkpoint(
        opt.model
    )

    estimator.load_state_dict(
        torch.load(
            opt.model,
            map_location=torch.device('cuda')
        ),
        strict=True
    )

    estimator.eval()

    logger = Logger(
        os.path.join(
            opt.output,
            "log.txt"
        )
    )

    logger.log(
        "evaluation of DTTDDataset..."
    )

    logger.log(
        "eval date and time: "
        + datetime.now().strftime(
            '%d/%m/%Y %H:%M:%S'
        )
    )

    logger.log(
        "mode: use groudtruth label"
    )

    logger.log(
        f"dataset root: {opt.dataset_root}"
    )

    logger.log(
        f"load model: {opt.model}"
    )

    logger.log(
        f"output directory: {opt.output}"
    )

    logger.log(
        f"if visualize: {opt.visualize}"
    )

    logger.log(
        f"Top-K aggregation: {opt.top_k}"
    )

    testlist = dataset.all_data_dirs

    logger.log(
        f"len of evaluation data list: {len(testlist)}"
    )

    if opt.num_samples <= 0 or opt.num_samples >= len(testlist):
        selected_indices = np.arange(len(testlist))
    else:
        rng = np.random.default_rng(opt.seed)
        selected_indices = rng.choice(
            len(testlist),
            size=opt.num_samples,
            replace=False
        )
        selected_indices.sort()

    selected_testlist = [
        testlist[i]
        for i in selected_indices
    ]

    logger.log(
        f"selected evaluation data list: {len(selected_testlist)}"
    )

    if opt.seed is not None:
        logger.log(
            f"random seed: {opt.seed}"
        )

    per_object_errors = []
    per_frame_trans_errors = []
    per_frame_rot_errors = []

    yaml_results = {
        "evaluation": {
            "date": datetime.now().strftime(
                '%Y-%m-%d %H:%M:%S'
            ),
            "model": opt.model,
            "dataset_root": opt.dataset_root,
            "total_dataset_frames": len(testlist),
            "num_frames": len(selected_testlist),
            "num_samples": opt.num_samples,
            "seed": opt.seed,
            "top_k": opt.top_k,
            "top_k_method": {
                "translation": (
                    "Top-K confidence weighted average "
                    "of point + pred_t"
                ),
                "rotation": (
                    "Top-K quaternion weighted average "
                    "with sign alignment"
                )
            }
        },
        "frames": []
    }

    for i, frame_name in enumerate(selected_testlist):
        img_path = os.path.join(
            opt.dataset_root,
            dataset.prefix,
            f"{frame_name}_color.jpg"
        )

        img = np.array(
            Image.open(img_path)
        )

        depth_path = os.path.join(
            opt.dataset_root,
            dataset.prefix,
            f"{frame_name}_depth.png"
        )

        depth = np.array(
            Image.open(depth_path),
            dtype=np.uint16
        )

        label_path = os.path.join(
            opt.dataset_root,
            dataset.prefix,
            f"{frame_name}_label.png"
        )

        label = np.array(
            Image.open(label_path)
        )

        meta_path = os.path.join(
            opt.dataset_root,
            dataset.prefix,
            f"{frame_name}_meta.json"
        )

        with open(
            meta_path,
            "r"
        ) as f:
            meta = json.load(f)

        lst = np.array(
            meta['objects']
        ).flatten().astype(
            np.int32
        )

        results = []
        prediction_success = []

        frame_trans_errors = []
        frame_rot_errors = []

        yaml_frame = {
            "frame": int(i),
            "dataset_index": int(selected_indices[i]),
            "frame_name": str(frame_name),
            "objects": []
        }

        for idx in range(len(lst)):
            itemid = int(lst[idx])

            try:
                data = process_data(
                    itemid,
                    img,
                    depth,
                    label,
                    meta,
                    Borderlist,
                    num_points=dataset.sample_2d_pt_num,
                    norm=dataset.norm
                )

                cloud = Variable(
                    data["cloud"]
                ).cuda()

                sample = Variable(
                    data["sample"]
                ).cuda()

                img_crop = Variable(
                    data["img_crop"]
                ).cuda()

                index = Variable(
                    data["index"]
                ).cuda()

                cloud = cloud.view(
                    1,
                    dataset.sample_2d_pt_num,
                    3
                )

                img_crop = img_crop.view(
                    1,
                    3,
                    img_crop.size()[1],
                    img_crop.size()[2]
                )

                with torch.no_grad():
                    with torch.cuda.amp.autocast(
                        enabled=False
                    ):
                        pred_r, pred_t, pred_c, _, pt_recon, _ = estimator(
                            img_crop,
                            cloud,
                            sample,
                            index
                        )

                if opt.debug:
                    os.makedirs(
                        "debug",
                        exist_ok=True
                    )

                    save_att(
                        estimator,
                        img_crop,
                        cloud,
                        sample,
                        f'debug/{i}_{idx}_'
                    )

                    save_freq(
                        estimator,
                        cloud
                    )

                (
                    pred_quat,
                    pred_xyz,
                    top_idx,
                    top_conf,
                    top_weights
                ) = aggregate_topk_pose(
                    pred_r,
                    pred_t,
                    pred_c,
                    cloud,
                    opt.top_k
                )

                prediction_success.append(True)

                pred_concat = np.append(
                    pred_quat,
                    pred_xyz
                )

                results.append(
                    pred_concat.tolist()
                )

                gt_pose = np.array(
                    meta['object_poses'][str(itemid)],
                    dtype=np.float64
                )

                R_gt = gt_pose[0:3, 0:3]
                T_gt = gt_pose[0:3, 3]

                gt_quat = np.array(
                    rotation_matrix_to_quaternion(R_gt),
                    dtype=np.float64
                )

                R_pred = quaternion_matrix(
                    pred_quat
                )[0:3, 0:3]

                T_pred = np.array(
                    pred_xyz,
                    dtype=np.float64
                )

                rot_err_deg = rotation_error_deg(
                    R_pred,
                    R_gt
                )

                trans_err_mm = translation_error_mm(
                    T_pred,
                    T_gt
                )

                frame_rot_errors.append(
                    rot_err_deg
                )

                frame_trans_errors.append(
                    trans_err_mm
                )

                per_frame_rot_errors.append(
                    rot_err_deg
                )

                per_frame_trans_errors.append(
                    trans_err_mm
                )

                per_object_errors.append({
                    "frame": int(i),
                    "dataset_index": int(
                        selected_indices[i]
                    ),
                    "frame_name": str(frame_name),
                    "obj_id": int(itemid),
                    "rot_err_deg": float(
                        rot_err_deg
                    ),
                    "trans_err_mm": float(
                        trans_err_mm
                    )
                })

                yaml_frame["objects"].append({
                    "obj_id": int(itemid),
                    "status": "success",
                    "gt": {
                        "xyz": [
                            float(x)
                            for x in T_gt
                        ],
                        "quaternion": [
                            float(x)
                            for x in gt_quat
                        ]
                    },
                    "predict": {
                        "xyz": [
                            float(x)
                            for x in T_pred
                        ],
                        "quaternion": [
                            float(x)
                            for x in pred_quat
                        ]
                    },
                    "error": {
                        "translation_mm": float(
                            trans_err_mm
                        ),
                        "rotation_deg": float(
                            rot_err_deg
                        )
                    },
                    "top_k": {
                        "k": int(
                            len(top_idx)
                        ),
                        "indices": [
                            int(x)
                            for x in top_idx
                        ],
                        "confidence": [
                            float(x)
                            for x in top_conf
                        ],
                        "weights": [
                            float(x)
                            for x in top_weights
                        ]
                    }
                })

                logger.log(
                    f"[ERR] No.{i} obj={itemid} "
                    f"rot={rot_err_deg:.4f} deg, "
                    f"trans={trans_err_mm:.4f} mm"
                )

            except Exception as e:
                prediction_success.append(False)

                logger.log(
                    f"Detector Lost {itemid} "
                    f"at No.{i} keyframe. "
                    f"Error message: {e}"
                )

                results.append(
                    [0.0 for _ in range(7)]
                )

                yaml_frame["objects"].append({
                    "obj_id": int(itemid),
                    "status": "failed",
                    "error_message": str(e)
                })

        scio.savemat(
            os.path.join(
                mats_dir,
                '%04d.mat' % i
            ),
            {'poses': results}
        )

        if frame_trans_errors:
            logger.log(
                f"[FRAME] No.{i} "
                f"mean rot="
                f"{np.mean(frame_rot_errors):.4f} deg, "
                f"mean trans="
                f"{np.mean(frame_trans_errors):.4f} mm"
            )

            yaml_frame["statistics"] = {
                "mean_rotation_deg": float(
                    np.mean(frame_rot_errors)
                ),
                "mean_translation_mm": float(
                    np.mean(frame_trans_errors)
                )
            }

        yaml_results["frames"].append(
            yaml_frame
        )

        if opt.visualize:
            try:
                vis_img = cv2.imread(
                    os.path.join(
                        opt.dataset_root,
                        dataset.prefix,
                        f"{frame_name}_color.jpg"
                    )
                )

                for idx in range(len(lst)):
                    if (
                        idx >= len(results)
                        or not prediction_success[idx]
                    ):
                        continue

                    itemid = int(lst[idx])

                    model_pts = np.array(
                        dataset.model_points[itemid]
                    )

                    quat = np.array(
                        results[idx][0:4]
                    )

                    T = np.array(
                        results[idx][4:7]
                    )

                    R = quaternion_matrix(
                        quat
                    )[0:3, 0:3]

                    T = T.reshape(
                        1,
                        3
                    )

                    color = color_list[
                        idx % len(color_list)
                    ]

                    vis_img = visualize(
                        img=vis_img,
                        model_pts=model_pts,
                        R=R,
                        T=T,
                        intrinsics=np.array(
                            meta['intrinsic']
                        ),
                        color=color
                    )

                cv2.imwrite(
                    os.path.join(
                        visualize_dir,
                        '%04d.png' % i
                    ),
                    vis_img
                )

            except Exception as e:
                logger.log(
                    f"Visualization fail at No.{i} "
                    f"keyframe. Error message: {e}"
                )

        logger.log(
            f"Finish No.{i} keyframe"
        )

    if per_frame_trans_errors:
        mean_rot = float(
            np.mean(per_frame_rot_errors)
        )

        std_rot = float(
            np.std(per_frame_rot_errors)
        )

        mean_trans = float(
            np.mean(per_frame_trans_errors)
        )

        std_trans = float(
            np.std(per_frame_trans_errors)
        )

        max_trans = float(
            np.max(per_frame_trans_errors)
        )

        max_rot = float(
            np.max(per_frame_rot_errors)
        )

        logger.log(
            "[SUMMARY] "
            f"objects={len(per_object_errors)}, "
            f"mean rot={mean_rot:.4f} deg, "
            f"std rot={std_rot:.4f} deg, "
            f"max rot={max_rot:.4f} deg, "
            f"mean trans={mean_trans:.4f} mm, "
            f"std trans={std_trans:.4f} mm, "
            f"max trans={max_trans:.4f} mm"
        )

        yaml_results["evaluation"]["summary"] = {
            "objects": int(
                len(per_object_errors)
            ),
            "mean_rotation_deg": mean_rot,
            "std_rotation_deg": std_rot,
            "max_rotation_deg": max_rot,
            "mean_translation_mm": mean_trans,
            "std_translation_mm": std_trans,
            "max_translation_mm": max_trans
        }

    else:
        logger.log(
            "[SUMMARY] No valid pose error was computed."
        )

        yaml_results["evaluation"]["summary"] = {
            "objects": 0,
            "message": "No valid pose error was computed."
        }

    yaml_path = os.path.join(
        opt.output,
        "pose_results.yaml"
    )

    save_yaml(
        yaml_path,
        yaml_results
    )

    logger.log(
        f"[YAML] Saved pose results to: {yaml_path}"
    )

    print("")
    print("=" * 70)
    print("Evaluation finished.")
    print("=" * 70)
    print(
        f"Total test frames : {len(testlist)}"
    )
    print(
        f"Sampled frames    : {len(selected_testlist)}"
    )
    print(
        f"Evaluated objects : {len(per_object_errors)}"
    )
    print(
        f"YAML result       : {yaml_path}"
    )
    print(
        f"MAT result        : {mats_dir}"
    )

    if opt.visualize:
        print(
            f"Visualization     : {visualize_dir}"
        )

    print("=" * 70)


def rotation_matrix_to_quaternion(R):
    R = np.asarray(
        R,
        dtype=np.float64
    )

    if R.shape != (3, 3):
        raise ValueError(
            f"Invalid rotation matrix shape: {R.shape}"
        )

    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(
            trace + 1.0
        ) * 2.0

        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s

    elif (
        R[0, 0] > R[1, 1]
        and R[0, 0] > R[2, 2]
    ):
        s = np.sqrt(
            1.0
            + R[0, 0]
            - R[1, 1]
            - R[2, 2]
        ) * 2.0

        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(
            1.0
            + R[1, 1]
            - R[0, 0]
            - R[2, 2]
        ) * 2.0

        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s

    else:
        s = np.sqrt(
            1.0
            + R[2, 2]
            - R[0, 0]
            - R[1, 1]
        ) * 2.0

        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array(
        [x, y, z, w],
        dtype=np.float64
    )

    norm = np.linalg.norm(q)

    if norm < 1e-12:
        raise RuntimeError(
            "Failed to convert rotation matrix to quaternion."
        )

    return q / norm


if __name__ == "__main__":
    eval()