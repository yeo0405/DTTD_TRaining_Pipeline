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
    parser.add_argument('--dataset_root', type=str, default = '../../dataset/cartridge/DTTD_Cartridge_Dataset/root', help='dataset root dir')
    parser.add_argument('--output', type=str, default = './eval_results', help='output directory to save results')
    parser.add_argument('--model', type=str, default = '',  help='path to resume model file')
    parser.add_argument('--result', type=str, default = 'eval_results',  help='Directory to save results')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--filter', action='store_true')
    parser.add_argument('--base_latent', type=int, default = 256, help='base latent dim for unimodal encoder')
    parser.add_argument('--embed_dim', type=int, default = 512, help='embedding dim for transformer encoder')
    parser.add_argument('--fusion_block_num', type=int, default = 1, help='number of fusion block')
    parser.add_argument('--layer_num_m', type=int, default = 4, help='layer num for modality fusion per block')
    parser.add_argument('--layer_num_p', type=int, default = 2, help='layer num for point-to-point fuison per block')
    return parser.parse_args()

def eval():
    opt = parse()
    # get meta parameters
    dataset = DTTDDataset(opt.dataset_root, "test", add_noise=False)
    # set dirs
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
        os.mkdir(os.path.join(opt.output, "mats"))
    # set visualize
    if opt.visualize:
        if not os.path.exists(os.path.join(opt.output, "visualize")):
            os.mkdir(os.path.join(opt.output, "visualize"))
        color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255), (123, 10, 265), (245, 163, 101), (100, 100, 178)]
    # set model
    estimator = PoseNet(num_points = dataset.sample_2d_pt_num, 
                        num_obj = dataset.num_obj,  # pt_num:1000, num_obj: 20 
                        base_latent=opt.base_latent, 
                        embedding_dim=opt.embed_dim, 
                        fusion_block_num=opt.fusion_block_num, 
                        layer_num_m=opt.layer_num_m, layer_num_p=opt.layer_num_p, 
                        filter_enhance=opt.filter) 
    estimator.cuda()
    opt.model = get_checkpoint(opt.model)
    estimator.load_state_dict(torch.load(opt.model, map_location=torch.device('cuda')), strict=True)
    estimator.eval()

    # set logger
    logger = Logger(os.path.join(opt.output, "log.txt"))
    logger.log("evaluation of DTTDDataset...")
    logger.log(f"eval date and time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    logger.log(f"mode: use groudtruth label")
    logger.log(f"dataset root: {opt.dataset_root}")
    logger.log(f"load model: {opt.model}")
    logger.log(f"output directory: {opt.output}")
    logger.log(f"if visualize: {opt.visualize}\n")
    # set data
    testlist = dataset.all_data_dirs
    logger.log(f"len of evaluation data list: {len(testlist)}")
    
    all_pose_errors = []
    all_rot_errors = []
    all_trans_errors = []

    def _rotation_error_deg(R_pred, R_gt):
        cos_theta = (np.trace(R_pred @ R_gt.T) - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def _translation_error_mm(T_pred, T_gt):
        return float(np.linalg.norm(T_pred.reshape(-1) - T_gt.reshape(-1)) * 1000.0)

    per_object_errors = []
    per_frame_trans_errors = []
    per_frame_rot_errors = []

    for i in range(len(testlist)):
        img = np.array(Image.open(os.path.join(opt.dataset_root, dataset.prefix, f"{testlist[i]}_color.jpg")))
        depth = np.array(Image.open(os.path.join(opt.dataset_root, dataset.prefix, f"{testlist[i]}_depth.png")), dtype=np.uint16)
        label = np.array(Image.open(os.path.join(opt.dataset_root, dataset.prefix, f"{testlist[i]}_label.png")))

        with open(os.path.join(opt.dataset_root, dataset.prefix, f"{testlist[i]}_meta.json"), "r") as f:
            meta = json.load(f)  # dict_keys(['objects', 'object_poses', 'intrinsic', 'distortion'])

        lst = np.array(meta['objects']).flatten().astype(np.int32)
        results = []
        frame_trans_errors = []
        frame_rot_errors = []

        for idx in range(len(lst)):
            itemid = lst[idx]
            try:
                data = process_data(itemid, img, depth, label, meta, Borderlist, num_points=dataset.sample_2d_pt_num, norm=dataset.norm)

                cloud = Variable(data["cloud"]).cuda()
                sample = Variable(data["sample"]).cuda()
                img_crop = Variable(data["img_crop"]).cuda()
                index = Variable(data["index"]).cuda()

                cloud = cloud.view(1, dataset.sample_2d_pt_num, 3)
                img_crop = img_crop.view(1, 3, img_crop.size()[1], img_crop.size()[2])

                pred_r, pred_t, pred_c, _, pt_recon, _ = estimator(img_crop, cloud, sample, index)

                if opt.debug:
                    save_att(estimator, img_crop, cloud, sample, f'debug/{i}_{idx}_')
                    save_freq(estimator, cloud)

                pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, dataset.sample_2d_pt_num, 1)
                pred_c = pred_c.view(1, dataset.sample_2d_pt_num)
                pred_t = pred_t.view(1 * dataset.sample_2d_pt_num, 1, 3)
                points = cloud.view(1 * dataset.sample_2d_pt_num, 1, 3)

                how_max, which_max = torch.max(pred_c, 1)

                pred_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                pred_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                pred_concat = np.append(pred_r, pred_t)
                results.append(pred_concat.tolist())

                R_gt = np.array(meta['object_poses'][str(itemid)])[0:3, 0:3]
                T_gt = np.array(meta['object_poses'][str(itemid)])[0:3, 3]

                R_pred = quaternion_matrix(pred_r)[0:3, 0:3]
                T_pred = pred_t

                rot_err_deg = _rotation_error_deg(R_pred, R_gt)
                trans_err_mm = _translation_error_mm(T_pred, T_gt)

                frame_rot_errors.append(rot_err_deg)
                frame_trans_errors.append(trans_err_mm)
                per_frame_rot_errors.append(rot_err_deg)
                per_frame_trans_errors.append(trans_err_mm)

                per_object_errors.append({
                    "frame": i,
                    "obj_id": int(itemid),
                    "rot_err_deg": rot_err_deg,
                    "trans_err_mm": trans_err_mm,
                })

                logger.log(
                    f"[ERR] No.{i} obj={itemid} "
                    f"rot={rot_err_deg:.4f} deg, trans={trans_err_mm:.4f} mm"
                )

            except Exception as e:
                logger.log(f"Detector Lost {itemid} at No.{i} keyframe. Error message: {e}")
                results.append([0.0 for _ in range(7)])

        scio.savemat(os.path.join(opt.output, "mats", '%04d.mat' % i), {'poses': results})

        if frame_trans_errors:
            logger.log(
                f"[FRAME] No.{i} "
                f"mean rot={np.mean(frame_rot_errors):.4f} deg, "
                f"mean trans={np.mean(frame_trans_errors):.4f} mm"
            )

        if opt.visualize:
            try:
                img = cv2.imread(os.path.join(opt.dataset_root, dataset.prefix, f"{testlist[i]}_color.jpg"))
                for idx in range(len(lst)):
                    itemid = lst[idx]
                    model_pts = np.array(dataset.model_points[itemid])
                    pose = np.array(meta['object_poses'][str(itemid)])
                    R = pose[:3,:3]
                    T = pose[:3,3].reshape(1,3)
                    img = visualize(
                        img=img,
                        model_pts=model_pts,
                        R=R,
                        T=T,
                        intrinsics=np.array(meta['intrinsic']),
                        color=color_list[idx],
                    )
                cv2.imwrite(os.path.join(opt.output, "visualize", '%04d.png' % i), img)
            except Exception as e:
                logger.log(f"Visualization fail at No.{i} keyframe. Error message: {e}")

        logger.log("Finish No.{0} keyframe".format(i))

    if per_frame_trans_errors:
        logger.log(
            "[SUMMARY] "
            f"objects={len(per_object_errors)}, "
            f"mean rot={np.mean(per_frame_rot_errors):.4f} deg, "
            f"std rot={np.std(per_frame_rot_errors):.4f} deg, "
            f"mean trans={np.mean(per_frame_trans_errors):.4f} mm, "
            f"std trans={np.std(per_frame_trans_errors):.4f} mm, "
            f"max trans={np.max(per_frame_trans_errors):.4f} mm"
        )
    else:
        logger.log("[SUMMARY] No valid pose error was computed.")
        
            
def process_data(itemid, img, depth, label, meta, border_list, num_points=1000, norm=None):
    img_h, img_w, _ = img.shape
    xmap = np.array([[j for i in range(img_w)] for j in range(img_h)])
    ymap = np.array([[i for i in range(img_w)] for j in range(img_h)])
    
    # mask valid depth (!= 0) and valid label (== itemid)
    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
    mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
    mask = mask_label * mask_depth
    
    # get object's discretized bounding box
    rmin, rmax, cmin, cmax = get_discrete_width_bbox(mask_label, border_list, img_w, img_h)
    
    # set sample points (2D) on depth/point cloud
    sample = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0] # non-zero positions on flattened mask, 1-D array
    if len(sample) >= num_points:
        sample = np.array(sorted(np.random.choice(sample, num_points))) # randomly choose pt_num points (idx)
    elif len(sample) == 0:
        sample = np.pad(sample, (0, num_points - len(sample)), 'constant')
    else:
        sample = np.pad(sample, (0, num_points - len(sample)), 'wrap')
        
    # crop image, depth and xy map with bbox, take the sample points
    img_crop = np.transpose(img[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
    # Image.fromarray(np.transpose(img_crop, (1,2,0))).convert('RGB').save(f'rgb{itemid}.png')
    depth_crop = depth[rmin:rmax, cmin:cmax].flatten()[sample][:, np.newaxis].astype(np.float32) # (pt_num, )
    # Image.fromarray(depth[rmin:rmax, cmin:cmax]).convert('RGB').save(f'depth{itemid}.png')
    xmap_crop = xmap[rmin:rmax, cmin:cmax].flatten()[sample][:, np.newaxis].astype(np.float32) # (pt_num, ) store y for sample points
    ymap_crop = ymap[rmin:rmax, cmin:cmax].flatten()[sample][:, np.newaxis].astype(np.float32) # (pt_num, ) store x for sample points
    
    # set camera focus and center
    cam = np.array(meta['intrinsic'])
    cam_cx, cam_cy, cam_fx, cam_fy =  cam[0][2], cam[1][2], cam[0][0], cam[1][1]
    
    # get point cloud [[px, py, pz], ...]
    cam_scale = 1000 # uint16 * 1000
    pz = depth_crop / cam_scale
    px = (ymap_crop - cam_cx) * pz / cam_fx
    py = (xmap_crop - cam_cy) * pz / cam_fy
    point_cloud = np.concatenate((px, py, pz), axis=1) # (pt_num, 3) store XYZ point cloud value for sample points
    
    # trans to torch tensor
    if norm == None:
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return {
        "cloud": torch.from_numpy(point_cloud.astype(np.float32)),
        "sample": torch.LongTensor(sample.astype(np.int32)),
        "img_crop": norm(torch.from_numpy(img_crop.astype(np.float32))),
        "index": torch.LongTensor([itemid - 1])
    }
    
def save_ptcld(xyz, fn):
    with open(fn, "w") as f:
        for i in range(len(xyz)):
            f.write(f"v {xyz[i][0]} {xyz[i][1]} {xyz[i][2]} \n")

def save_freq(estimator, x):
    estimator.get_freq_domain(x)

def save_att(estimator, img_crop, cloud, sample, fn_):
    estimator.eval()
    attn1, attn2 = estimator.get_attention_map(img_crop, cloud, sample)
    hn1 = attn1.shape[0]
    hn2 = attn2.shape[0]

    draw_attn_map(fn_+'m_avg.png', attn1)
    for i in range(hn1):
        fn = fn_+f'm_{i}'+'.png'
        draw_attn_map(fn, attn1, i)

    draw_attn_map(fn_+'p_avg.png', attn2)
    for i in range(hn2):
        fn = fn_+f'p_{i}'+'.png'
        draw_attn_map(fn, attn2, i)

def draw_attn_map(fn, attn, head_idx=None):
    if head_idx is not None:
        attn = attn[head_idx]
    else:
        attn = torch.mean(attn, dim=0)
    attn = np.asarray(attn.detach().cpu().numpy())
    attn = 255*(attn - np.min(attn))/(np.max(attn) - np.min(attn))
    image = Image.fromarray(attn)
    if image.mode == 'F':
        image = image.convert('RGB')
    image.save(fn)

if __name__ == "__main__":
    eval()