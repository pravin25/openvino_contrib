import argparse
import os
import sys
from PIL import Image
import numpy as np
import time
import json
import cv2
from collections import defaultdict

import heapq
from openvino import Core

from utils.data_utils import load_im, get_bbox, get_point_cloud_from_depth, get_resize_rgb_choose
from utils.draw_utils import draw_detections

import pycocotools.mask as cocomask
import trimesh

from eval_utils import (
    load_symmetries, compute_mssd, compute_mspd, compute_vsd,
    compute_ar_mssd, compute_ar_mspd, compute_ar_vsd,
    VSD_DELTA_MM,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, '..', 'Pose_Estimation_Model')
sys.path.append(os.path.join(ROOT_DIR, 'provider'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))

def get_parser():
    parser = argparse.ArgumentParser(description="[OpenVINO] BOP PEM Evaluation")
    # pem model
    parser.add_argument("--device", type=str, default="GPU", help="device to run on (CPU/GPU)")
    parser.add_argument("--config", type=str, default="config/base.yaml", help="path to PEM config YAML")
    parser.add_argument("--det_score_thresh", type=float, default=0.2, help="ISM detection score threshold")
    parser.add_argument("--batch_size", type=int, default=4, help="PEM inference batch size")
    parser.add_argument("--topk_ism_score", type=int, default=3, help="Top-K ISM scores fed into PEM")
    parser.add_argument("--skip_vsd", action="store_true", help="Skip VSD computation")
    # BOP
    parser.add_argument("--bop_dir", type=str, required=True, help="BOP dataset root")
    parser.add_argument("--ism_results_root", type=str, default=None, help="Root for ISM results (default: <bop_dir>/bop/ism_ov_gpu_fastsam_results)")
    parser.add_argument("--templates_root", type=str, default=None, help="Root for templates (default: <bop_dir>/eval_output)")
    parser.add_argument("--max_images", type=int, default=None, help="Limit to first N unique images")
    parser.add_argument("--obj_ids", type=int, nargs="+", default=None, help="Restrict to these object IDs")
    parser.add_argument("--max_objects", type=int, default=None, help="Limit to first N object IDs")
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True, help="Skip existing result JSONs")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--segmentor_model", type=str, default="fastsam", choices=["sam", "fastsam"], help="Segmentor tag for naming")
    return parser.parse_args()


def load_yaml_config(config_path):
    import yaml
    class Config:
        def __init__(self, data):
            for key, value in data.items():
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)
    with open(config_path, 'r') as f:
        return Config(yaml.safe_load(f))


def visualize(rgb, pred_rot, pred_trans, model_points, K, save_path):
    """
    Visualize the predicted pose by drawing the 3D model overlay on the RGB image.
    Args:
        rgb: np.ndarray, shape (H, W, 3), uint8, input RGB image
        pred_rot: np.ndarray, shape (N, 3, 3), predicted rotation matrices
        pred_trans: np.ndarray, shape (N, 3), predicted translations (mm)
        model_points: np.ndarray, shape (M, 3), 3D model points (mm)
        K: np.ndarray, shape (N, 3, 3), camera intrinsics
        save_path: str, path to save the visualization image
    Returns:
        concat: PIL.Image, side-by-side visualization image
    """
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)

    # concat side by side in PIL
    rgb = Image.fromarray(np.uint8(rgb))
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat


def get_templates_np(path, cfg):
    """
    Load multiple rendered templates for the CAD model from disk using numpy.
    Args:
        path: str, directory containing template files
        cfg: config object, must have n_template_view, img_size, n_sample_template_point
    Returns:
        all_tem: list[np.ndarray], each (1, 3, img_size, img_size)
        all_tem_pts: list[np.ndarray], each (1, n_sample_template_point, 3)
        all_tem_choose: list[np.ndarray], each (1, n_sample_template_point)
    """
    n_template_view = cfg.n_template_view
    all_tem = []
    all_tem_choose = []
    all_tem_pts = []

    total_nView = 42
    for v in range(n_template_view):
        i = int(total_nView / n_template_view * v)
        tem, tem_choose, tem_pts = _get_template_np(path, cfg, i)
        all_tem.append(np.expand_dims(tem, axis=0))  # (1, 3, img_size, img_size)
        all_tem_choose.append(np.expand_dims(tem_choose, axis=0))  # (1, n_sample_template_point)
        all_tem_pts.append(np.expand_dims(tem_pts, axis=0))  # (1, n_sample_template_point, 3)
    return all_tem, all_tem_pts, all_tem_choose

def _get_template_np(path, cfg, tem_index=1):
    """
    Load a single template (rendered view) for the CAD model using numpy.
    Args:
        path: str, directory containing template files
        cfg: config object, must have img_size, n_sample_template_point, rgb_mask_flag
        tem_index: int, template index
    Returns:
        rgb: np.ndarray, shape (3, img_size, img_size), normalized RGB image
        rgb_choose: np.ndarray, shape (n_sample_template_point,), selected pixel indices
        xyz: np.ndarray, shape (n_sample_template_point, 3), 3D points (meters)
    """
    rgb_path = os.path.join(path, 'rgb_'+str(tem_index)+'.png')
    mask_path = os.path.join(path, 'mask_'+str(tem_index)+'.png')
    xyz_path = os.path.join(path, 'xyz_'+str(tem_index)+'.npy')

    # Load data using numpy/cv2
    rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    xyz = np.load(xyz_path).astype(np.float32) / 1000.0  # Convert mm to meters
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8) == 255

    # Get bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    mask = mask[y1:y2, x1:x2]

    # Process RGB image
    rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]  # BGR to RGB
    if cfg.rgb_mask_flag:
        rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

    # Resize RGB image
    rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)

    # Normalize RGB (same as torchvision transforms)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    rgb = rgb.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

    # Process point cloud
    xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))

    # Sample points
    choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
    if len(choose) <= cfg.n_sample_template_point:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_template_point, replace=False)
    choose = choose[choose_idx]
    xyz = xyz[choose, :]

    # Calculate RGB choose indices
    h, w = y2 - y1, x2 - x1
    scale_h, scale_w = cfg.img_size / h, cfg.img_size / w

    choose_y = (choose // w).astype(np.float32) * scale_h
    choose_x = (choose % w).astype(np.float32) * scale_w
    rgb_choose = (choose_y * cfg.img_size + choose_x).astype(np.int32)

    return rgb, rgb_choose, xyz

def get_test_data_np(rgb_path, depth_path, cam_path, cad_path, seg_path, det_score_thresh, cfg, topk):
    """
    Prepare test data for pose estimation using numpy.
    Args:
        rgb_path: str, path to RGB image
        depth_path: str, path to depth image
        cam_path: str, path to camera intrinsics (json)
        cad_path: str, path to CAD model
        seg_path: str, path to segmentation results (json)
        det_score_thresh: float, detection score threshold
        cfg: config object, must have n_sample_observed_point, img_size, rgb_mask_flag
        topk: ism detection topk scores
    Returns:
        ret_dict: dict with keys:
            'pts': np.ndarray, (N, n_sample_observed_point, 3)
            'rgb': np.ndarray, (N, 3, img_size, img_size)
            'rgb_choose': np.ndarray, (N, n_sample_observed_point)
            'score': np.ndarray, (N,)
            'model': np.ndarray, (N, n_sample_model_point, 3)
            'K': np.ndarray, (N, 3, 3)
        whole_image: np.ndarray, (H, W, 3), original RGB image
        whole_pts: np.ndarray, (H*W, 3), full point cloud
        model_points: np.ndarray, (n_sample_model_point, 3)
        all_dets: list[dict], detection info
    """
    dets = []
    with open(seg_path) as f:
        dets_ = json.load(f) # keys: scene_id, image_id, category_id, bbox, score, segmentation

    if dets_:
        top_k_dets = heapq.nlargest(topk, dets_, key=lambda det: det['score'])
        for det in top_k_dets:
          if det['score'] > det_score_thresh:
              dets.append(det)

    del dets_

    cam_info = json.load(open(cam_path))
    K = np.array(cam_info['cam_K']).reshape(3, 3)

    whole_image = load_im(rgb_path).astype(np.uint8)
    if len(whole_image.shape)==2:
        whole_image = np.concatenate([whole_image[:,:,None], whole_image[:,:,None], whole_image[:,:,None]], axis=2)
    whole_depth = load_im(depth_path).astype(np.float32) * cam_info['depth_scale'] / 1000.0
    whole_pts = get_point_cloud_from_depth(whole_depth, K)

    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(cfg.n_sample_model_point).astype(np.float32) / 1000.0
    radius = np.max(np.linalg.norm(model_points, axis=1))

    all_rgb = []
    all_cloud = []
    all_rgb_choose = []
    all_score = []
    all_dets = []
    for inst in dets:
        seg = inst['segmentation']
        score = inst['score']

        # mask
        h,w = seg['size']
        try:
            rle = cocomask.frPyObjects(seg, h, w)
        except:
            rle = seg
        mask = cocomask.decode(rle)
        mask = np.logical_and(mask > 0, whole_depth > 0)
        if np.sum(mask) > 32:
            bbox = get_bbox(mask)
            y1, y2, x1, x2 = bbox
        else:
            continue
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        # pts
        cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
        center = np.mean(cloud, axis=0)
        tmp_cloud = cloud - center[None, :]
        flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
        if np.sum(flag) < 4:
            continue
        choose = choose[flag]
        cloud = cloud[flag]

        if len(choose) <= cfg.n_sample_observed_point:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), cfg.n_sample_observed_point, replace=False)
        choose = choose[choose_idx]
        cloud = cloud[choose_idx]

        # rgb
        rgb = whole_image.copy()[y1:y2, x1:x2, :][:,:,::-1]
        if cfg.rgb_mask_flag:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)

        # Normalize RGB (same as torchvision transforms)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        rgb = rgb.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], cfg.img_size)

        all_rgb.append(rgb.astype(np.float32))
        all_cloud.append(cloud.astype(np.float32))
        all_rgb_choose.append(rgb_choose.astype(np.int32))
        all_score.append(score)
        all_dets.append(inst)

    ret_dict = {}
    ret_dict['pts'] = np.stack(all_cloud)
    ret_dict['rgb'] = np.stack(all_rgb)
    ret_dict['rgb_choose'] = np.stack(all_rgb_choose)
    ret_dict['score'] = np.array(all_score, dtype=np.float32)

    ninstance = ret_dict['pts'].shape[0]
    ret_dict['model'] = np.repeat(model_points[np.newaxis, :, :], ninstance, axis=0)
    ret_dict['K'] = np.repeat(K[np.newaxis, :, :], ninstance, axis=0)
    return ret_dict, whole_image, whole_pts.reshape(-1, 3), model_points, all_dets


def run_pem_for_testcase(ov_fe_compiled_model, ov_pem_compiled_model,
                         rgb_path, depth_path, cam_path, cad_path, seg_path,
                         template_dir, cfg_test, det_score_thresh, topk_ism_score,
                         batch_size, obj_fe_cache, obj_id):
    """Run PEM inference for one testcase. Returns (best_R, best_t, best_score, ninstance, timing) or None."""
    timing = {}

    # Template FE (cached per object)
    t0 = time.time()
    if obj_id not in obj_fe_cache:
        all_tem, all_tem_pts, all_tem_choose = get_templates_np(template_dir, cfg_test)
        tem_rgb = np.concatenate(all_tem, axis=1)
        tem_pts = np.concatenate(all_tem_pts, axis=1)
        tem_choose = np.concatenate(all_tem_choose, axis=1)
        fe_results = ov_fe_compiled_model({
            "rgb_input": tem_rgb, "pts_input": tem_pts, "choose_input": tem_choose,
        })
        results_list = list(fe_results.values())
        obj_fe_cache[obj_id] = (results_list[0], results_list[1])
    tem_pts_out, tem_feat = obj_fe_cache[obj_id]
    timing["template_fe"] = time.time() - t0

    # Prepare test data from ISM detections
    t0 = time.time()
    try:
        input_data, img, whole_pts, model_points, detections = get_test_data_np(
            rgb_path, depth_path, cam_path, cad_path, seg_path,
            det_score_thresh, cfg_test, topk_ism_score,
        )
    except Exception as e:
        return None
    ninstance = input_data['pts'].shape[0]
    if ninstance == 0:
        return None
    input_data['dense_po'] = np.repeat(tem_pts_out, ninstance, axis=0)
    input_data['dense_fo'] = np.repeat(tem_feat, ninstance, axis=0)
    timing["data_prep"] = time.time() - t0

    # PEM inference (batched)
    t0 = time.time()
    all_R, all_t, all_score = [], [], []
    for start in range(0, ninstance, batch_size):
        end = min(start + batch_size, ninstance)
        batch_inputs = {
            "pts": input_data['pts'][start:end],
            "rgb": input_data['rgb'][start:end],
            "rgb_choose": input_data['rgb_choose'][start:end],
            "model": input_data['model'][start:end],
            "dense_po": input_data['dense_po'][start:end],
            "dense_fo": input_data['dense_fo'][start:end],
        }
        results = ov_pem_compiled_model(batch_inputs)
        results_output = list(results.values())
        all_R.append(results_output[0])
        all_t.append(results_output[1])
        all_score.append(results_output[2])

    pred_R = np.concatenate(all_R, axis=0)
    pred_t = np.concatenate(all_t, axis=0) * 1000  # m -> mm
    pred_pose_score = np.concatenate(all_score, axis=0)
    pose_scores = pred_pose_score * input_data['score']
    timing["pem_inference"] = time.time() - t0

    best_idx = int(np.argmax(pose_scores))
    return pred_R[best_idx], pred_t[best_idx], float(pose_scores[best_idx]), ninstance, timing


def main():
    args = get_parser()
    cfg = load_yaml_config(args.config)

    # -- Resolve BOP paths ------------------------------------------
    scene_dir = os.path.join(args.bop_dir, "test", "000002")
    scene_gt_path = os.path.join(scene_dir, "scene_gt.json")
    scene_cam_path = os.path.join(scene_dir, "scene_camera.json")
    models_dir = os.path.join(args.bop_dir, "models")
    models_info_path = os.path.join(models_dir, "models_info.json")
    ism_root = args.ism_results_root or os.path.join(args.bop_dir, "bop", "ism_ov_gpu_fastsam_results")
    templates_root = args.templates_root or os.path.join(args.bop_dir, "eval_output")

    device_tag = args.device.lower()
    seg_tag = args.segmentor_model
    output_dir = args.output_dir or os.path.join(
        args.bop_dir, "bop", f"pem_ov_{device_tag}_{seg_tag}_results")
    os.makedirs(output_dir, exist_ok=True)

    # -- Load BOP annotations ---------------------------------------
    with open(scene_gt_path) as f:
        gt_data = json.load(f)
    with open(scene_cam_path) as f:
        cam_data = json.load(f)
    with open(models_info_path) as f:
        models_info = json.load(f)

    # -- Determine target objects -----------------------------------
    all_obj_ids = sorted({e["obj_id"] for entries in gt_data.values() for e in entries})
    if args.obj_ids:
        target_obj_ids = [o for o in args.obj_ids if o in all_obj_ids]
    else:
        target_obj_ids = list(all_obj_ids)
    if args.max_objects:
        target_obj_ids = target_obj_ids[:args.max_objects]
    target_set = set(target_obj_ids)

    # -- Build testcases sorted by (obj_id, image_id) ---------------
    testcases = []
    for img_id_str in sorted(gt_data.keys(), key=int):
        img_id = int(img_id_str)
        for entry in gt_data[img_id_str]:
            if entry["obj_id"] in target_set:
                testcases.append({
                    "image_id": img_id, "obj_id": entry["obj_id"],
                    "gt_R": entry["cam_R_m2c"], "gt_t": entry["cam_t_m2c"],
                })
    if args.max_images is not None:
        keep = set(sorted({tc["image_id"] for tc in testcases})[:args.max_images])
        testcases = [tc for tc in testcases if tc["image_id"] in keep]
    testcases.sort(key=lambda tc: (tc["obj_id"], tc["image_id"]))
    n_total = len(testcases)

    print(f"\n{'='*65}")
    print(f"  BOP PEM Evaluation — OpenVINO Pipeline")
    print(f"{'='*65}")
    print(f"  BOP dir        : {args.bop_dir}")
    print(f"  Device         : {args.device}")
    print(f"  Segmentor tag  : {seg_tag}")
    print(f"  Objects        : {target_obj_ids}")
    print(f"  Testcases      : {n_total}")
    print(f"  Max images     : {args.max_images or 'all'}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  Top-K ISM      : {args.topk_ism_score}")
    print(f"  Skip VSD       : {args.skip_vsd}")
    print(f"  Output         : {output_dir}")
    print(f"  ISM results    : {ism_root}")
    print()

    if n_total == 0:
        print("No testcases to evaluate.")
        return

    # -- Init OpenVINO PEM models -----------------------------------
    print("Initializing OpenVINO PEM models ...")
    t_init_start = time.time()
    core = Core()
    ov_extension_lib_path = os.path.join(BASE_DIR, "model/ov_pointnet2_op/build/libopenvino_operation_extension.so")
    ov_gpu_kernel_path = os.path.join(BASE_DIR, "model/ov_pointnet2_op/pem_gpu_ops.xml")
    core.add_extension(ov_extension_lib_path)
    if "GPU" in args.device:
        core.set_property("GPU", {"INFERENCE_PRECISION_HINT": "f32"})
        core.set_property("GPU", {"CONFIG_FILE": ov_gpu_kernel_path})
    ov_fe_model_path = os.path.join(BASE_DIR, "model_save/ov_fe_model_cpu.xml")
    ov_pem_model_path = os.path.join(BASE_DIR, "model_save/ov_pem_model_cpu.xml")
    ov_fe_compiled = core.compile_model(core.read_model(ov_fe_model_path), args.device)
    ov_pem_compiled = core.compile_model(core.read_model(ov_pem_model_path), args.device)
    init_time = time.time() - t_init_start
    print(f"  Model init: {init_time:.1f}s\n")

    # -- Prepare per-object model data for AR -----------------------
    obj_model_data = {}
    renderer = None
    if not args.skip_vsd:
        try:
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            import pyrender
            renderer = pyrender.OffscreenRenderer(640, 480)
        except Exception as e:
            print(f"  [WARN] VSD disabled: {e}")
            args.skip_vsd = True

    for oid in target_obj_ids:
        cad_path = os.path.join(models_dir, f"obj_{oid:06d}.ply")
        mesh = trimesh.load_mesh(cad_path, process=False)
        pts_mm = np.array(mesh.vertices, dtype=np.float64)
        if str(oid) in models_info:
            diameter = float(models_info[str(oid)]["diameter"])
            syms = load_symmetries(models_info, oid)
        else:
            center = pts_mm.mean(axis=0)
            diameter = float(2.0 * np.max(np.linalg.norm(pts_mm - center, axis=1)))
            syms = [{"R": np.eye(3), "t": np.zeros(3)}]
        obj_data = {"pts": pts_mm, "diameter": diameter, "syms": syms}
        if renderer is not None:
            import pyrender
            obj_data["mesh_pyrender"] = pyrender.Mesh.from_trimesh(mesh)
        obj_model_data[oid] = obj_data

    # -- Caches & tracking ------------------------------------------
    obj_fe_cache = {}
    depth_cache = {}
    mssd_errors, mssd_diameters, mspd_errors, vsd_errors = [], [], [], []
    by_obj = defaultdict(lambda: {"mssd": [], "mspd": [], "vsd": [], "diameters": []})
    timings = []
    n_completed, n_skipped = 0, 0
    t_eval_start = time.time()
    cam_tmp_path = os.path.join(output_dir, "_tmp_cam.json")

    for tc_idx, tc in enumerate(testcases):
        img_id = tc["image_id"]
        obj_id = tc["obj_id"]
        result_path = os.path.join(output_dir, f"img{img_id:06d}_obj{obj_id:06d}.json")

        # -- skip_existing ------------------------------------------
        if args.skip_existing and os.path.exists(result_path):
            try:
                with open(result_path) as f:
                    existing = json.load(f)
                if "mssd_mm" in existing:
                    mssd_errors.append(existing["mssd_mm"])
                    mssd_diameters.append(existing["diameter_mm"])
                    by_obj[obj_id]["mssd"].append(existing["mssd_mm"])
                    by_obj[obj_id]["diameters"].append(existing["diameter_mm"])
                if "mspd_px" in existing:
                    mspd_errors.append(existing["mspd_px"])
                    by_obj[obj_id]["mspd"].append(existing["mspd_px"])
                if "vsd_per_tau" in existing:
                    vsd = {int(k): v for k, v in existing["vsd_per_tau"].items()}
                    vsd_errors.append(vsd)
                    by_obj[obj_id]["vsd"].append(vsd)
                timings.append(existing.get("timing_total", 0.0))
            except Exception:
                pass
            n_completed += 1
            n_skipped += 1
            if n_completed % 500 == 0:
                _print_progress(n_completed, n_total, mssd_errors, mssd_diameters, mspd_errors, vsd_errors, timings)
            continue

        t_tc_start = time.time()

        # -- Locate ISM detection + paths ---------------------------
        ism_seg_path = os.path.join(
            ism_root,
            f"img{img_id:06d}_obj{obj_id:06d}_detection_ism.json")
        if not os.path.isfile(ism_seg_path):
            print(f"  [SKIP] img={img_id} obj={obj_id}: no ISM at {ism_seg_path}")
            continue

        rgb_path = os.path.join(scene_dir, "rgb", f"{img_id:06d}.png")
        depth_path = os.path.join(scene_dir, "depth", f"{img_id:06d}.png")
        cad_path = os.path.join(models_dir, f"obj_{obj_id:06d}.ply")
        template_dir = os.path.join(templates_root, f"obj_{obj_id:06d}", "templates")
        if not os.path.isdir(template_dir):
            print(f"  [SKIP] img={img_id} obj={obj_id}: no templates")
            continue

        # Write per-image camera JSON for get_test_data_np
        cam_info = cam_data[str(img_id)]
        with open(cam_tmp_path, "w") as f:
            json.dump(cam_info, f)

        # -- Run PEM ------------------------------------------------
        result = run_pem_for_testcase(
            ov_fe_compiled, ov_pem_compiled,
            rgb_path, depth_path, cam_tmp_path, cad_path, ism_seg_path,
            template_dir, cfg.test_dataset, args.det_score_thresh,
            args.topk_ism_score, args.batch_size, obj_fe_cache, obj_id,
        )
        if result is None:
            print(f"  [SKIP] img={img_id} obj={obj_id}: PEM inference failed")
            continue
        best_R, best_t, best_score, ninstance, timing = result

        # -- Compute AR metrics -------------------------------------
        t0 = time.time()
        gt_R = np.array(tc["gt_R"]).reshape(3, 3)
        gt_t = np.array(tc["gt_t"]).reshape(3)
        K = np.array(cam_info["cam_K"]).reshape(3, 3)
        pts = obj_model_data[obj_id]["pts"]
        diameter = obj_model_data[obj_id]["diameter"]
        syms = obj_model_data[obj_id]["syms"]

        mssd = compute_mssd(best_R, best_t, gt_R, gt_t, pts, syms)
        mspd = compute_mspd(best_R, best_t, gt_R, gt_t, pts, K, syms)
        mssd_errors.append(mssd)
        mssd_diameters.append(diameter)
        mspd_errors.append(mspd)
        by_obj[obj_id]["mssd"].append(mssd)
        by_obj[obj_id]["mspd"].append(mspd)
        by_obj[obj_id]["diameters"].append(diameter)

        vsd_per_tau = None
        if not args.skip_vsd and renderer is not None:
            if img_id not in depth_cache:
                dp = os.path.join(scene_dir, "depth", f"{img_id:06d}.png")
                if os.path.exists(dp):
                    depth_raw = np.array(Image.open(dp)).astype(np.float64)
                    depth_cache[img_id] = depth_raw * cam_info.get("depth_scale", 1.0)
                else:
                    depth_cache[img_id] = None
            depth_mm = depth_cache[img_id]
            if depth_mm is not None:
                vsd_per_tau = compute_vsd(
                    best_R, best_t, gt_R, gt_t, depth_mm, K,
                    renderer, obj_model_data[obj_id]["mesh_pyrender"],
                    diameter, delta=VSD_DELTA_MM)
                vsd_errors.append(vsd_per_tau)
                by_obj[obj_id]["vsd"].append(vsd_per_tau)
        timing["eval"] = time.time() - t0
        total_time = time.time() - t_tc_start

        # -- Save result JSON ---------------------------------------
        tc_result = {
            "image_id": img_id, "obj_id": obj_id,
            "n_instances": ninstance, "best_score": best_score,
            "best_R": best_R.tolist(), "best_t": best_t.tolist(),
            "gt_R": gt_R.tolist(), "gt_t": gt_t.tolist(),
            "diameter_mm": diameter,
            "mssd_mm": mssd, "mssd_norm": mssd / diameter, "mspd_px": mspd,
            "timing_total": total_time, "timing": timing,
        }
        if vsd_per_tau is not None:
            tc_result["vsd_per_tau"] = vsd_per_tau
        with open(result_path, "w") as f:
            json.dump(tc_result, f, indent=2, default=str)

        timings.append(total_time)
        n_completed += 1
        if n_completed % 500 == 0:
            _print_progress(n_completed, n_total, mssd_errors, mssd_diameters, mspd_errors, vsd_errors, timings)

    # -- Cleanup ----------------------------------------------------
    if renderer:
        renderer.delete()
    if os.path.exists(cam_tmp_path):
        os.remove(cam_tmp_path)
    total_eval_time = time.time() - t_eval_start

    # -- Final summary ----------------------------------------------
    print(f"\n{'='*65}")
    print(f"  BOP PEM Evaluation — FINAL RESULTS")
    print(f"{'='*65}")
    print(f"  Completed      : {n_completed}/{n_total} (skipped: {n_skipped})")
    print(f"  Model init     : {init_time:.1f}s")
    if timings:
        print(f"  Avg time/tc    : {np.mean(timings):.2f}s")
        print(f"  Total eval time: {total_eval_time:.1f}s")

    print(f"\n  Per-Object AR:")
    for oid in sorted(by_obj.keys()):
        obj = by_obj[oid]
        n = len(obj["mssd"])
        if n == 0:
            continue
        obj_mssd = compute_ar_mssd(obj["mssd"], obj["diameters"]) * 100
        obj_mspd = compute_ar_mspd(obj["mspd"]) * 100
        line = f"    Object {oid:>2} ({n:>4} tc)  MSSD={obj_mssd:5.1f}%  MSPD={obj_mspd:5.1f}%"
        if obj["vsd"]:
            obj_vsd = compute_ar_vsd(obj["vsd"]) * 100
            line += f"  VSD={obj_vsd:5.1f}%  AR={(obj_vsd+obj_mssd+obj_mspd)/3:.1f}%"
        print(line)

    if mssd_errors:
        ar_mssd = compute_ar_mssd(mssd_errors, mssd_diameters) * 100
        ar_mspd = compute_ar_mspd(mspd_errors) * 100
        print(f"\n  {'='*55}")
        print(f"  PEM OVERALL ({n_completed} testcases)")
        print(f"    AR_MSSD : {ar_mssd:6.2f}%")
        print(f"    AR_MSPD : {ar_mspd:6.2f}%")
        if vsd_errors:
            ar_vsd = compute_ar_vsd(vsd_errors) * 100
            ar = (ar_vsd + ar_mssd + ar_mspd) / 3.0
            print(f"    AR_VSD  : {ar_vsd:6.2f}%")
            print(f"    AR      : {ar:6.2f}%")
        ref = 69.9 if seg_tag == "sam" else 66.7
        print(f"\n  Paper reference (SAM-6D {seg_tag.upper()}, LM-O): {ref}% AR")

    # -- Save summary JSON ------------------------------------------
    summary = {"n_total": n_total, "n_completed": n_completed, "n_skipped": n_skipped,
               "init_time_s": init_time, "total_eval_time_s": total_eval_time,
               "avg_inference_s": float(np.mean(timings)) if timings else 0.0}
    if mssd_errors:
        summary["AR_MSSD"] = compute_ar_mssd(mssd_errors, mssd_diameters) * 100
        summary["AR_MSPD"] = compute_ar_mspd(mspd_errors) * 100
        if vsd_errors:
            summary["AR_VSD"] = compute_ar_vsd(vsd_errors) * 100
            summary["AR"] = (summary["AR_VSD"] + summary["AR_MSSD"] + summary["AR_MSPD"]) / 3.0
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved to: {summary_path}")
    print(f"  Results dir: {output_dir}")


def _print_progress(n_done, n_total, mssd_errors, mssd_diameters, mspd_errors, vsd_errors, timings):
    ar_mssd = compute_ar_mssd(mssd_errors, mssd_diameters) * 100
    ar_mspd = compute_ar_mspd(mspd_errors) * 100
    avg_time = np.mean(timings) if timings else 0.0
    parts = [f"MSSD={ar_mssd:.1f}%", f"MSPD={ar_mspd:.1f}%"]
    if vsd_errors:
        parts.append(f"VSD={compute_ar_vsd(vsd_errors)*100:.1f}%")
    print(f"\n  --- Progress: {n_done}/{n_total} ({100*n_done/n_total:.1f}%) | AR: {' '.join(parts)} | Avg: {avg_time:.2f}s ---")


if __name__ == "__main__":
    main()
