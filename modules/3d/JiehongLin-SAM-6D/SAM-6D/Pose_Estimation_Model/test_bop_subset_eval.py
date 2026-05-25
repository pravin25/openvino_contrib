import argparse
import importlib
import json
import logging
import os
import os.path as osp
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "provider"))
sys.path.append(os.path.join(BASE_DIR, "utils"))
sys.path.append(os.path.join(BASE_DIR, "model"))
sys.path.append(os.path.join(BASE_DIR, "model", "pointnet2"))


DETECTION_PATHS = {
    "ycbv": "../Instance_Segmentation_Model/log/sam/result_ycbv.json",
    "tudl": "../Instance_Segmentation_Model/log/sam/result_tudl.json",
    "tless": "../Instance_Segmentation_Model/log/sam/result_tless.json",
    "lmo": "../Instance_Segmentation_Model/log/sam/result_lmo.json",
    "itodd": "../Instance_Segmentation_Model/log/sam/result_itodd.json",
    "icbin": "../Instance_Segmentation_Model/log/sam/result_icbin.json",
    "hb": "../Instance_Segmentation_Model/log/sam/result_hb.json",
}


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate PEM on a subset of BOP samples")
    parser.add_argument("--gpus", type=str, default="0", help="Index of gpu")
    parser.add_argument("--model", type=str, default="pose_estimation_model", help="Model module name")
    parser.add_argument("--config", type=str, default="config/base.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, default="lmo", help="Dataset name")
    parser.add_argument("--checkpoint_path", type=str, default="none", help="Path to checkpoint file")
    parser.add_argument("--iter", type=int, default=0, help="Checkpoint iter if checkpoint_path is none")
    parser.add_argument("--view", type=int, default=-1, help="Number of template views")
    parser.add_argument("--exp_id", type=int, default=0, help="Experiment id")

    parser.add_argument("--max_samples", type=int, default=10, help="Maximum number of image samples to evaluate")
    parser.add_argument("--detection_path", type=str, default="", help="Override detection json path")
    parser.add_argument("--output_name", type=str, default="subset_eval", help="Output folder suffix")

    # AR thresholds
    parser.add_argument("--mssd_thresholds", type=str, default="0.05,0.10,0.20,0.30,0.40,0.50",
                        help="Comma-separated MSSD thresholds as diameter ratio")
    parser.add_argument("--mspd_thresholds", type=str, default="5,10,20,30,40,50",
                        help="Comma-separated MSPD thresholds in pixels")
    parser.add_argument("--vsd_thresholds", type=str, default="0.05,0.10,0.20,0.30,0.40,0.50",
                        help="Comma-separated VSD thresholds as diameter ratio (for VSD approximation)")
    return parser


def parse_float_list(v: str) -> List[float]:
    return [float(x.strip()) for x in v.split(",") if x.strip()]


def init_cfg(args):
    try:
        import gorilla  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'gorilla'. Install PEM dependencies first: `sh dependencies.sh`"
        ) from exc

    exp_name = args.model + "_" + osp.splitext(args.config.split("/")[-1])[0] + "_id" + str(args.exp_id)
    log_dir = osp.join("log", exp_name)
    os.makedirs(log_dir, exist_ok=True)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.gpus = args.gpus
    cfg.model_name = args.model
    cfg.log_dir = log_dir
    cfg.checkpoint_path = args.checkpoint_path
    cfg.test_iter = args.iter
    cfg.dataset = args.dataset

    if args.view != -1:
        cfg.test_dataset.n_template_view = args.view

    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)
    return cfg


def project_points(K: np.ndarray, pts_mm: np.ndarray) -> np.ndarray:
    z = np.clip(pts_mm[:, 2], 1e-6, None)
    x = pts_mm[:, 0] / z
    y = pts_mm[:, 1] / z
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return np.stack([u, v], axis=1)


def transform_points(R_flat: np.ndarray, t_mm: np.ndarray, pts_mm: np.ndarray) -> np.ndarray:
    R = R_flat.reshape(3, 3)
    return (R @ pts_mm.T).T + t_mm.reshape(1, 3)


def compute_pose_errors(
    R_pred_flat: np.ndarray,
    t_pred_mm: np.ndarray,
    R_gt_flat: np.ndarray,
    t_gt_mm: np.ndarray,
    pts_mm: np.ndarray,
    K: np.ndarray,
) -> Tuple[float, float, float]:
    pred_pts = transform_points(R_pred_flat, t_pred_mm, pts_mm)
    gt_pts = transform_points(R_gt_flat, t_gt_mm, pts_mm)

    # Approximate MSSD: max surface distance without symmetry handling.
    mssd_mm = float(np.linalg.norm(pred_pts - gt_pts, axis=1).max())

    pred_uv = project_points(K, pred_pts)
    gt_uv = project_points(K, gt_pts)

    # Approximate MSPD: max projection distance without symmetry handling.
    mspd_px = float(np.linalg.norm(pred_uv - gt_uv, axis=1).max())

    # Approximate VSD surrogate: mean absolute depth disagreement on model correspondences.
    vsd_mm = float(np.mean(np.abs(pred_pts[:, 2] - gt_pts[:, 2])))

    return mssd_mm, mspd_px, vsd_mm


def ar_from_thresholds(value: float, thresholds: List[float]) -> float:
    if len(thresholds) == 0:
        return 0.0
    hits = [1.0 if value <= thr else 0.0 for thr in thresholds]
    return float(np.mean(hits))


def choose_detection_path(dataset_name: str, override_path: str) -> str:
    if override_path:
        return override_path
    if dataset_name not in DETECTION_PATHS:
        raise ValueError(f"No default detection path for dataset={dataset_name}")
    return DETECTION_PATHS[dataset_name]


def build_gt_cache(test_root: str) -> Dict[str, Dict[str, List[dict]]]:
    cache = {}
    test_dir = osp.join(test_root, "test")
    if not osp.isdir(test_dir):
        return cache
    for scene_id in sorted(os.listdir(test_dir)):
        scene_path = osp.join(test_dir, scene_id)
        if not osp.isdir(scene_path):
            continue
        gt_path = osp.join(scene_path, "scene_gt.json")
        cam_path = osp.join(scene_path, "scene_camera.json")
        if not osp.isfile(gt_path) or not osp.isfile(cam_path):
            continue
        with open(gt_path, "r") as f:
            scene_gt = json.load(f)
        with open(cam_path, "r") as f:
            scene_cam = json.load(f)
        cache[scene_id] = {"gt": scene_gt, "cam": scene_cam}
    return cache


def main():
    args = get_parser().parse_args()
    cfg = init_cfg(args)
    import gorilla  # pylint: disable=import-outside-toplevel

    random.seed(cfg.rd_seed)
    np.random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    mssd_thresh_ratio = parse_float_list(args.mssd_thresholds)
    mspd_thresh_px = parse_float_list(args.mspd_thresholds)
    vsd_thresh_ratio = parse_float_list(args.vsd_thresholds)

    detection_path = choose_detection_path(args.dataset, args.detection_path)
    if not osp.isfile(detection_path):
        raise FileNotFoundError(f"Detection file not found: {detection_path}")

    print("************************ Start Logging ************************")
    print(cfg)
    print(f"using gpu: {cfg.gpus}")
    print(f"detection_path: {detection_path}")

    print("creating model ...")
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Net(cfg.model)
    if len(cfg.gpus) > 1:
        model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
    model = model.cuda()

    if cfg.checkpoint_path == "none":
        checkpoint = os.path.join(cfg.log_dir, "checkpoint_iter" + str(cfg.test_iter).zfill(6) + ".pth")
    else:
        checkpoint = cfg.checkpoint_path
    gorilla.solver.load_checkpoint(model=model, filename=checkpoint)
    model.eval()

    dataset_module = importlib.import_module(cfg.test_dataset.name)
    dataset = dataset_module.BOPTestset(cfg.test_dataset, args.dataset, detection_path)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=cfg.test_dataloader.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=cfg.test_dataloader.pin_memory,
    )

    # Precompute object template features.
    all_tem, all_tem_pts, all_tem_choose = dataset.get_templates()
    with torch.no_grad():
        dense_po, dense_fo = model.feature_extraction.get_obj_feats(all_tem, all_tem_pts, all_tem_choose)

    test_root = osp.join(cfg.test_dataset.data_dir, args.dataset)
    gt_cache = build_gt_cache(test_root)
    has_gt = len(gt_cache) > 0
    print(f"GT available: {has_gt}")

    out_dir = osp.join(cfg.log_dir, f"{args.dataset}_{args.output_name}_iter{str(cfg.test_iter).zfill(6)}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = osp.join(out_dir, f"result_{args.dataset}.csv")
    per_sample_path = osp.join(out_dir, "per_sample_metrics.json")
    summary_path = osp.join(out_dir, "summary.json")

    lines = []
    per_sample = []

    total_images = min(args.max_samples, len(dataloader))
    total_instances = 0
    total_eval_instances = 0

    with tqdm(total=total_images, desc="Subset PEM eval") as pbar:
        for image_idx, data in enumerate(dataloader):
            if image_idx >= total_images:
                break

            torch.cuda.synchronize()
            start = time.time()

            for key in data:
                data[key] = data[key].cuda()

            n_instance = data["pts"].size(1)
            total_instances += int(n_instance)

            bs = cfg.test_dataloader.bs
            n_batch = int(np.ceil(n_instance / bs))

            pred_Rs = []
            pred_Ts = []
            pred_scores = []
            for j in range(n_batch):
                start_idx = j * bs
                end_idx = n_instance if j == n_batch - 1 else (j + 1) * bs
                obj = data["obj"][0][start_idx:end_idx].reshape(-1)

                inputs = {
                    "pts": data["pts"][0][start_idx:end_idx].contiguous(),
                    "rgb": data["rgb"][0][start_idx:end_idx].contiguous(),
                    "rgb_choose": data["rgb_choose"][0][start_idx:end_idx].contiguous(),
                    "model": data["model"][0][start_idx:end_idx].contiguous(),
                    "dense_po": dense_po[obj].contiguous(),
                    "dense_fo": dense_fo[obj].contiguous(),
                }
                with torch.no_grad():
                    end_points = model(inputs)

                pred_Rs.append(end_points["pred_R"])
                pred_Ts.append(end_points["pred_t"])
                pred_scores.append(end_points["pred_pose_score"])

            pred_Rs = torch.cat(pred_Rs, dim=0).reshape(-1, 9).detach().cpu().numpy()
            pred_Ts = torch.cat(pred_Ts, dim=0).detach().cpu().numpy() * 1000.0
            pred_scores = torch.cat(pred_scores, dim=0) * data["score"][0, :, 0]
            pred_scores = pred_scores.detach().cpu().numpy()

            scene_id = int(data["scene_id"].item())
            img_id = int(data["img_id"].item())
            image_time = time.time() - start + float(data["seg_time"].item())

            # Save PEM predictions in the same csv format as test_bop.py
            for k in range(n_instance):
                line = ",".join((
                    str(scene_id),
                    str(img_id),
                    str(data["obj_id"][0][k].item()),
                    str(pred_scores[k]),
                    " ".join((str(v) for v in pred_Rs[k])),
                    " ".join((str(v) for v in pred_Ts[k])),
                    f"{image_time}\n",
                ))
                lines.append(line)

            # Evaluate against GT when available.
            scene_key = f"{scene_id:06d}"
            img_key = str(img_id)
            if has_gt and scene_key in gt_cache and img_key in gt_cache[scene_key]["gt"]:
                gt_list = gt_cache[scene_key]["gt"][img_key]
                K = np.array(gt_cache[scene_key]["cam"][img_key]["cam_K"]).reshape(3, 3)

                # Greedy one-to-one matching by object and minimum MSSD approximation.
                used_gt = set()
                for k in range(n_instance):
                    obj_id = int(data["obj_id"][0][k].item())
                    if obj_id not in dataset.obj_idxs:
                        continue

                    obj_idx = dataset.obj_idxs[obj_id]
                    diameter_mm = float(dataset.objects[obj_idx].diameter * 1000.0)
                    model_pts_mm = dataset.objects[obj_idx].model_points * 1000.0

                    cand = []
                    for gi, gt in enumerate(gt_list):
                        if gi in used_gt:
                            continue
                        if int(gt["obj_id"]) != obj_id:
                            continue
                        R_gt = np.array(gt["cam_R_m2c"], dtype=np.float64)
                        t_gt = np.array(gt["cam_t_m2c"], dtype=np.float64)
                        mssd_mm, mspd_px, vsd_mm = compute_pose_errors(
                            pred_Rs[k], pred_Ts[k], R_gt, t_gt, model_pts_mm, K
                        )
                        cand.append((mssd_mm, gi, mspd_px, vsd_mm, R_gt, t_gt))

                    if len(cand) == 0:
                        continue

                    cand.sort(key=lambda x: x[0])
                    best = cand[0]
                    used_gt.add(best[1])
                    mssd_mm = float(best[0])
                    mspd_px = float(best[2])
                    vsd_mm = float(best[3])

                    mssd_abs_thresh = [r * diameter_mm for r in mssd_thresh_ratio]
                    vsd_abs_thresh = [r * diameter_mm for r in vsd_thresh_ratio]

                    ar_mssd = ar_from_thresholds(mssd_mm, mssd_abs_thresh)
                    ar_mspd = ar_from_thresholds(mspd_px, mspd_thresh_px)
                    ar_vsd = ar_from_thresholds(vsd_mm, vsd_abs_thresh)
                    ar_mean = float(np.mean([ar_mssd, ar_mspd, ar_vsd]))

                    rec = {
                        "sample_idx": len(per_sample),
                        "image_sample_idx": image_idx,
                        "scene_id": scene_id,
                        "img_id": img_id,
                        "obj_id": obj_id,
                        "score": float(pred_scores[k]),
                        "diameter_mm": diameter_mm,
                        "mssd_mm": mssd_mm,
                        "mspd_px": mspd_px,
                        "vsd_mm_approx": vsd_mm,
                        "AR_MSSD": ar_mssd,
                        "AR_MSPD": ar_mspd,
                        "AR_VSD": ar_vsd,
                        "AR_mean": ar_mean,
                    }
                    per_sample.append(rec)
                    total_eval_instances += 1

                    print(
                        f"[sample {rec['sample_idx']:04d}] scene={scene_id:06d} img={img_id:06d} obj={obj_id:02d} "
                        f"MSSD={mssd_mm:.2f}mm MSPD={mspd_px:.2f}px VSD~={vsd_mm:.2f}mm "
                        f"AR(MSSD/MSPD/VSD/mean)=({ar_mssd:.3f}/{ar_mspd:.3f}/{ar_vsd:.3f}/{ar_mean:.3f})"
                    )

            pbar.update(1)

    with open(csv_path, "w+") as f:
        f.writelines(lines)

    # Aggregate summary.
    if total_eval_instances > 0:
        mean_ar_mssd = float(np.mean([x["AR_MSSD"] for x in per_sample]))
        mean_ar_mspd = float(np.mean([x["AR_MSPD"] for x in per_sample]))
        mean_ar_vsd = float(np.mean([x["AR_VSD"] for x in per_sample]))
        mean_ar = float(np.mean([x["AR_mean"] for x in per_sample]))
    else:
        mean_ar_mssd = None
        mean_ar_mspd = None
        mean_ar_vsd = None
        mean_ar = None

    summary = {
        "dataset": args.dataset,
        "max_samples": int(args.max_samples),
        "n_images_tested": int(total_images),
        "n_instances_total": int(total_instances),
        "n_instances_evaluated": int(total_eval_instances),
        "detection_path": detection_path,
        "gt_available": has_gt,
        "metric_note": {
            "MSSD": "Approximate MSSD without symmetry handling",
            "MSPD": "Approximate MSPD without symmetry handling",
            "VSD": "Approximate VSD surrogate from mean depth disagreement",
        },
        "thresholds": {
            "mssd_ratio": mssd_thresh_ratio,
            "mspd_px": mspd_thresh_px,
            "vsd_ratio": vsd_thresh_ratio,
        },
        "mean_AR_MSSD": mean_ar_mssd,
        "mean_AR_MSPD": mean_ar_mspd,
        "mean_AR_VSD": mean_ar_vsd,
        "mean_AR_overall": mean_ar,
        "csv_path": csv_path,
        "per_sample_path": per_sample_path,
    }

    with open(per_sample_path, "w") as f:
        json.dump(per_sample, f, indent=2)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ SUBSET EVAL SUMMARY ================")
    print(json.dumps(summary, indent=2))
    print(f"Saved predictions csv: {csv_path}")
    print(f"Saved per-sample metrics: {per_sample_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
