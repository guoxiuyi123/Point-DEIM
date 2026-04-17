"""
DEIM: DETR with Improved Matching for Fast Convergence     
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.  
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.  
"""


import os, sys     
import math
import json
import gc
import numpy as np
from typing import Iterable
from tqdm import tqdm
from tidecv import TIDE, datasets
from scipy.optimize import linear_sum_assignment

import torch    
import torch.amp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter  
from torch.cuda.amp.grad_scaler import GradScaler
 
from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator    
from ..misc import MetricLogger, MetricLogger_progress, SmoothedValue, dist_utils, plot_sample     
from ..misc.modality_utils import normalize_tensor_minmax_per_sample 
from ..logger_module import get_logger     
from ..extre_module.ops import Profile
from ..extre_module.utils import TQDM, RANK
# from ..extre_module.yolo_metrice import get_yolo_det_metrice, get_yolo_seg_metrice 
from ..deim.utils import coco_evaluator_per_class 
from .sample_adapter import (
    move_samples_to_device,
    select_model_input_for_model,
    select_plot_samples_for_logging,  
)    

from pycocoeval.yoloeval import get_yolo_det_metrice, get_yolo_seg_metrice 

CLEAR_MEMORY_STEP = 100    
TIME_DEBUG = False
RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"
logger = get_logger(__name__)

def _is_point_supervision_enabled(point_sup):
    return isinstance(point_sup, dict) and point_sup.get("enabled", False)


def _is_point_teacher_enabled(point_teacher):
    return isinstance(point_teacher, dict) and point_teacher.get("enabled", False)

def _is_point_teacher_dspe_enabled(point_teacher):
    if not isinstance(point_teacher, dict):
        return False
    dspe = point_teacher.get("DSPE", None) or point_teacher.get("dspe", None) or {}
    return isinstance(dspe, dict) and bool(dspe.get("enabled", False))


def _get_point_teacher_dspe_cfg(point_teacher, point_teacher_state=None):
    dspe_cfg = {}
    if isinstance(point_teacher, dict):
        dspe_cfg = point_teacher.get("DSPE", None) or point_teacher.get("dspe", None) or {}
        if isinstance(dspe_cfg, dict) and dspe_cfg:
            return dspe_cfg
    if isinstance(point_teacher_state, dict):
        dspe_cfg = point_teacher_state.get("dspe_cfg", None)
        if isinstance(dspe_cfg, dict):
            return dspe_cfg
    return {}


def _get_point_sup_density_cfg(point_sup):
    if not isinstance(point_sup, dict):
        return {}
    cfg = point_sup.get("DensityLimit", None) or point_sup.get("density_limit", None) or {}
    return cfg if isinstance(cfg, dict) else {}


def _get_point_sup_feature_growth_cfg(point_sup):
    if not isinstance(point_sup, dict):
        return {}
    cfg = point_sup.get("FeatureGrowth", None) or point_sup.get("feature_growth", None) or {}
    return cfg if isinstance(cfg, dict) else {}


def _density_knn_max_wh_px(points_norm: torch.Tensor, w_img: float, h_img: float, knn_k: int, margin: float, global_min_wh_px: float, global_max_wh_px: float):
    n = int(points_norm.shape[0])
    if n == 0:
        return torch.empty((0,), device=points_norm.device, dtype=torch.float32)
    if n == 1:
        return torch.full((1,), float(global_max_wh_px), device=points_norm.device, dtype=torch.float32)
    pts = points_norm.detach().float().clone()
    pts_px = torch.stack([pts[:, 0] * w_img, pts[:, 1] * h_img], dim=1)
    d = torch.cdist(pts_px, pts_px, p=2)
    d.fill_diagonal_(float("inf"))
    k = max(1, int(knn_k))
    kth = torch.topk(d, k, dim=1, largest=False).values[:, -1]
    max_wh = kth * float(margin)
    max_wh = max_wh.clamp(min=float(global_min_wh_px), max=float(global_max_wh_px))
    return max_wh


class _ScaleMemoryBank:
    def __init__(self, num_classes: int, init_mean_wh_px=(20.0, 20.0), init_std_wh_px=(20.0, 20.0), min_std_px=(0.35, 0.35)):
        self.num_classes = int(num_classes)
        mean = torch.as_tensor(init_mean_wh_px, dtype=torch.float32).view(1, 2)
        std = torch.as_tensor(init_std_wh_px, dtype=torch.float32).view(1, 2)
        self.mean_wh_px = mean.repeat(self.num_classes, 1).clone()
        self.std_wh_px = std.repeat(self.num_classes, 1).clone()
        self.min_std_px = torch.as_tensor(min_std_px, dtype=torch.float32).view(1, 2).repeat(self.num_classes, 1).clone()
        self.count = torch.zeros((self.num_classes,), dtype=torch.long)

    def update(self, labels: torch.Tensor, wh_px: torch.Tensor, scores: torch.Tensor = None, beta: float = 0.001, score_thresh: float = 0.0, min_wh_px: float = 1.0, max_wh_px: float = 1e6, max_area_ratio: float = None):
        if labels.numel() == 0:
            return
        labels = labels.detach().long().view(-1)
        wh_px = wh_px.detach().float().view(-1, 2)
        if scores is None:
            keep = torch.ones((labels.numel(),), dtype=torch.bool, device=labels.device)
        else:
            scores = scores.detach().float().view(-1)
            keep = scores >= float(score_thresh)
        if not torch.any(keep):
            return
        labels = labels[keep].cpu()
        wh_px = wh_px[keep].cpu()
        beta = float(beta)
        min_wh_px = float(min_wh_px)
        max_wh_px = float(max_wh_px)
        max_area_ratio = float(max_area_ratio) if max_area_ratio is not None else None
        for l, wh in zip(labels.tolist(), wh_px):
            if l < 0 or l >= self.num_classes:
                continue
            w = float(wh[0].item())
            h = float(wh[1].item())
            if not (min_wh_px <= w <= max_wh_px and min_wh_px <= h <= max_wh_px):
                continue
            if max_area_ratio is not None and max_area_ratio > 0:
                mean_wh = self.mean_wh_px[l]
                mean_area = float((mean_wh[0] * mean_wh[1]).item())
                cur_area = float(w * h)
                if mean_area > 1e-6 and (cur_area / mean_area) > max_area_ratio:
                    continue
            prev_mean = self.mean_wh_px[l].clone()
            new_mean = (1.0 - beta) * prev_mean + beta * torch.tensor([w, h], dtype=torch.float32)
            diff = torch.tensor([w, h], dtype=torch.float32) - new_mean
            prev_var = (self.std_wh_px[l].clone().clamp(min=0.0)) ** 2
            new_var = (1.0 - beta) * prev_var + beta * (diff ** 2)
            new_std = torch.sqrt(new_var).clamp(min=self.min_std_px[l])
            self.mean_wh_px[l] = new_mean
            self.std_wh_px[l] = new_std
            self.count[l] += 1

    def sample_wh_pool(self, label: int, k: int, explore_ratio: float, explore_wh_px, min_wh_px: float, max_wh_px: float, device=None):
        k = int(k)
        if k <= 0:
            return []
        label = int(label)
        if label < 0 or label >= self.num_classes:
            label = 0
        mean = self.mean_wh_px[label].clone()
        std = self.std_wh_px[label].clone()
        if device is None:
            device = torch.device("cpu")
        mean = mean.to(device)
        std = std.to(device)
        min_wh_px = float(min_wh_px)
        max_wh_px = float(max_wh_px)
        explore_ratio = float(explore_ratio)
        if isinstance(explore_wh_px, (list, tuple)) and len(explore_wh_px) > 0:
            explore_pool = []
            for it in explore_wh_px:
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    explore_pool.append((float(it[0]), float(it[1])))
                else:
                    s = float(it)
                    explore_pool.append((s, s))
        else:
            explore_pool = [(float(mean[0].item()), float(mean[1].item()))]
        pool = []
        for _ in range(k):
            if float(torch.rand((), device=device).item()) < explore_ratio:
                ww, hh = explore_pool[int(torch.randint(0, len(explore_pool), (1,), device=device).item())]
                w = float(ww)
                h = float(hh)
            else:
                samp = torch.normal(mean=mean, std=std)
                w = float(samp[0].item())
                h = float(samp[1].item())
            w = float(max(min_wh_px, min(max_wh_px, w)))
            h = float(max(min_wh_px, min(max_wh_px, h)))
            pool.append((w, h))
        return pool


def _safe_sigmoid_probs(logits: torch.Tensor) -> torch.Tensor:
    return logits.float().sigmoid()


def _safe_softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return logits.float().softmax(-1)


def _extract_hw_from_model_inputs(model_inputs):
    if torch.is_tensor(model_inputs):
        return int(model_inputs.shape[-2]), int(model_inputs.shape[-1])
    if isinstance(model_inputs, dict):
        for v in model_inputs.values():
            if torch.is_tensor(v):
                return int(v.shape[-2]), int(v.shape[-1])
    raise ValueError("Cannot infer H,W from model_inputs")


def _random_block_mask(model_inputs, mask_cfg):
    if not torch.is_tensor(model_inputs):
        return model_inputs
    if not isinstance(mask_cfg, dict) or not mask_cfg.get("enabled", False):
        return model_inputs
    prob = float(mask_cfg.get("prob", 0.0))
    if prob <= 0:
        return model_inputs
    num_blocks = int(mask_cfg.get("num_blocks", 1))
    min_ratio = float(mask_cfg.get("min_ratio", 0.05))
    max_ratio = float(mask_cfg.get("max_ratio", 0.2))
    fill = float(mask_cfg.get("fill", 0.0))
    if num_blocks <= 0 or max_ratio <= 0:
        return model_inputs
    b, c, h, w = model_inputs.shape
    out = model_inputs.clone()
    for bi in range(b):
        if float(torch.rand((), device=out.device).item()) > prob:
            continue
        for _ in range(num_blocks):
            rh = float(torch.empty((), device=out.device).uniform_(min_ratio, max_ratio).item())
            rw = float(torch.empty((), device=out.device).uniform_(min_ratio, max_ratio).item())
            hh = max(1, int(round(h * rh)))
            ww = max(1, int(round(w * rw)))
            y0 = int(torch.randint(0, max(1, h - hh + 1), (1,), device=out.device).item())
            x0 = int(torch.randint(0, max(1, w - ww + 1), (1,), device=out.device).item())
            out[bi, :, y0:y0 + hh, x0:x0 + ww] = fill
    return out


def _resolve_fixed_wh_px(fixed_box_wh_px, label: int):
    if isinstance(fixed_box_wh_px, (list, tuple)) and len(fixed_box_wh_px) == 2:
        return float(fixed_box_wh_px[0]), float(fixed_box_wh_px[1])
    if isinstance(fixed_box_wh_px, dict):
        key = str(int(label))
        if key in fixed_box_wh_px:
            v = fixed_box_wh_px[key]
            if isinstance(v, (list, tuple)) and len(v) == 2:
                return float(v[0]), float(v[1])
    return 20.0, 20.0


def _build_point_fixed_targets(targets, point_teacher, model_inputs=None, point_teacher_state=None, point_sup=None):
    if not isinstance(point_teacher, dict):
        return targets
    fixed_box_wh_px = point_teacher.get("fixed_box_wh_px", (20, 20))
    if model_inputs is not None:
        h_img, w_img = _extract_hw_from_model_inputs(model_inputs)
        w_img = float(max(1, w_img))
        h_img = float(max(1, h_img))
    else:
        w_img, h_img = 640.0, 640.0
    bag_cfg = point_teacher.get("Bag", None) or point_teacher.get("bag", None) or {}
    bag_enabled = isinstance(bag_cfg, dict) and bag_cfg.get("enabled", False)
    bag_size = int(bag_cfg.get("bag_size", 16)) if bag_enabled else 0
    aspect_ratios = bag_cfg.get("aspect_ratios", [1.0]) if bag_enabled else [1.0]
    wh_px_list = bag_cfg.get("wh_px", []) if bag_enabled else []
    jitter = float(bag_cfg.get("jitter", 0.0)) if bag_enabled else 0.0
    dspe_cfg = _get_point_teacher_dspe_cfg(point_teacher, point_teacher_state=point_teacher_state)
    dspe_enabled = isinstance(dspe_cfg, dict) and bool(dspe_cfg.get("enabled", False))
    scale_bank = None
    if dspe_enabled and isinstance(point_teacher_state, dict):
        scale_bank = point_teacher_state.get("scale_bank", None)
    explore_ratio = float(dspe_cfg.get("explore_ratio", 0.5)) if dspe_enabled else 0.0
    explore_wh_px = dspe_cfg.get("explore_wh_px", wh_px_list) if dspe_enabled else wh_px_list
    min_wh_px = float(dspe_cfg.get("pseudo_min_wh_px", 2.0)) if dspe_enabled else 1.0
    max_wh_px = float(dspe_cfg.get("pseudo_max_wh_px", 256.0)) if dspe_enabled else 1e6
    density_cfg = _get_point_sup_density_cfg(point_sup)
    density_enabled = bool(density_cfg.get("enabled", False))
    knn_k = int(density_cfg.get("knn_k", 1))
    margin_factor = float(density_cfg.get("density_margin_factor", 1.2))
    global_max_wh_px = float(density_cfg.get("global_max_wh_px", 256.0))
    global_min_wh_px = float(density_cfg.get("global_min_wh_px", 2.0))
    out_targets = []
    for t in targets:
        boxes = t.get("boxes", None)
        labels = t.get("labels", None)
        if not isinstance(boxes, torch.Tensor) or not isinstance(labels, torch.Tensor):
            out_targets.append(t)
            continue
        if boxes.numel() == 0:
            out_targets.append(dict(t))
            continue
        pts = boxes[..., :2].detach().float()
        labs = labels.detach().long()
        max_wh_px_per_point = None
        if density_enabled:
            max_wh_px_per_point = _density_knn_max_wh_px(
                pts, w_img=w_img, h_img=h_img, knn_k=knn_k, margin=margin_factor, global_min_wh_px=global_min_wh_px, global_max_wh_px=global_max_wh_px
            )
        new_boxes = []
        for i in range(int(labs.numel())):
            ww, hh = _resolve_fixed_wh_px(fixed_box_wh_px, int(labs[i].item()))
            if density_enabled and max_wh_px_per_point is not None:
                mwh = float(max_wh_px_per_point[i].item())
                ww = min(float(ww), mwh)
                hh = min(float(hh), mwh)
            new_boxes.append([float(pts[i, 0].item()), float(pts[i, 1].item()), float(ww) / w_img, float(hh) / h_img])
        t_new = dict(t)
        t_new["boxes"] = torch.as_tensor(new_boxes, device=boxes.device, dtype=torch.float32).clamp(min=0.0, max=1.0)
        if bag_enabled and bag_size > 0:
            n = int(labs.numel())
            m = int(bag_size)
            centers = pts[:, None, :].repeat(1, m, 1)
            cand_wh = torch.zeros((n, m, 2), device=boxes.device, dtype=torch.float32)
            if dspe_enabled and isinstance(scale_bank, _ScaleMemoryBank):
                wh_pool_per_i = []
                for i in range(n):
                    wh_pool_per_i.append(
                        scale_bank.sample_wh_pool(
                            label=int(labs[i].item()),
                            k=max(1, m),
                            explore_ratio=explore_ratio,
                            explore_wh_px=explore_wh_px,
                            min_wh_px=min_wh_px,
                            max_wh_px=max_wh_px,
                            device=boxes.device,
                        )
                    )
            elif wh_px_list:
                wh_pool = []
                for it in wh_px_list:
                    if isinstance(it, (list, tuple)) and len(it) == 2:
                        wh_pool.append((float(it[0]), float(it[1])))
                    else:
                        s = float(it)
                        wh_pool.append((s, s))
            else:
                wh_pool = [(float(t_new["boxes"][i, 2].item() * w_img), float(t_new["boxes"][i, 3].item() * h_img)) for i in range(n)]
            ar_pool = [float(x) for x in aspect_ratios] if aspect_ratios else [1.0]
            for i in range(n):
                base_w = float(t_new["boxes"][i, 2].item() * w_img)
                base_h = float(t_new["boxes"][i, 3].item() * h_img)
                cand_wh[i, 0, 0] = base_w
                cand_wh[i, 0, 1] = base_h
                for j in range(1, m):
                    if dspe_enabled and isinstance(scale_bank, _ScaleMemoryBank):
                        wi, hi = wh_pool_per_i[i][j]
                    else:
                        wi, hi = wh_pool[int(torch.randint(0, len(wh_pool), (1,), device=boxes.device).item())]
                    ar = ar_pool[int(torch.randint(0, len(ar_pool), (1,), device=boxes.device).item())]
                    ar = max(1e-6, float(ar))
                    s = math.sqrt(ar)
                    ww = wi * s
                    hh = hi / s
                    if jitter > 0:
                        fac = float(torch.empty((), device=boxes.device).uniform_(max(0.0, 1.0 - jitter), 1.0 + jitter).item())
                        ww *= fac
                        hh *= fac
                    cand_wh[i, j, 0] = max(1.0, float(ww))
                    cand_wh[i, j, 1] = max(1.0, float(hh))
            cand_wh_norm = cand_wh.clone()
            cand_wh_norm[:, :, 0] /= float(w_img)
            cand_wh_norm[:, :, 1] /= float(h_img)
            t_new["mil_boxes"] = torch.cat([centers, cand_wh_norm], dim=2).clamp(min=0.0, max=1.0)
            t_new["mil_scores"] = torch.zeros((n, m), device=boxes.device, dtype=torch.float32)
        out_targets.append(t_new)
    return out_targets


def _ensure_point_teacher_bag(targets, point_teacher, model_inputs=None, point_teacher_state=None, point_sup=None):
    if not isinstance(point_teacher, dict):
        return targets
    bag_cfg = point_teacher.get("Bag", None) or point_teacher.get("bag", None) or {}
    if not isinstance(bag_cfg, dict) or not bag_cfg.get("enabled", False):
        return targets
    bag_size = int(bag_cfg.get("bag_size", 16))
    if bag_size <= 0:
        return targets
    if model_inputs is not None:
        h_img, w_img = _extract_hw_from_model_inputs(model_inputs)
        w_img = float(max(1, w_img))
        h_img = float(max(1, h_img))
    else:
        w_img, h_img = 640.0, 640.0
    aspect_ratios = bag_cfg.get("aspect_ratios", [1.0])
    wh_px_list = bag_cfg.get("wh_px", [])
    jitter = float(bag_cfg.get("jitter", 0.0))
    dspe_cfg = _get_point_teacher_dspe_cfg(point_teacher, point_teacher_state=point_teacher_state)
    dspe_enabled = isinstance(dspe_cfg, dict) and bool(dspe_cfg.get("enabled", False))
    scale_bank = None
    if dspe_enabled and isinstance(point_teacher_state, dict):
        scale_bank = point_teacher_state.get("scale_bank", None)
    explore_ratio = float(dspe_cfg.get("explore_ratio", 0.5)) if dspe_enabled else 0.0
    explore_wh_px = dspe_cfg.get("explore_wh_px", wh_px_list) if dspe_enabled else wh_px_list
    min_wh_px = float(dspe_cfg.get("pseudo_min_wh_px", 2.0)) if dspe_enabled else 1.0
    max_wh_px = float(dspe_cfg.get("pseudo_max_wh_px", 256.0)) if dspe_enabled else 1e6
    density_cfg = _get_point_sup_density_cfg(point_sup)
    density_enabled = bool(density_cfg.get("enabled", False))
    knn_k = int(density_cfg.get("knn_k", 1))
    margin_factor = float(density_cfg.get("density_margin_factor", 1.2))
    global_max_wh_px = float(density_cfg.get("global_max_wh_px", 256.0))
    global_min_wh_px = float(density_cfg.get("global_min_wh_px", 2.0))
    if wh_px_list:
        wh_pool = []
        for it in wh_px_list:
            if isinstance(it, (list, tuple)) and len(it) == 2:
                wh_pool.append((float(it[0]), float(it[1])))
            else:
                s = float(it)
                wh_pool.append((s, s))
    else:
        wh_pool = None
    ar_pool = [float(x) for x in aspect_ratios] if aspect_ratios else [1.0]
    out = []
    for t in targets:
        boxes = t.get("boxes", None)
        labels = t.get("labels", None)
        if not isinstance(boxes, torch.Tensor) or not isinstance(labels, torch.Tensor) or boxes.numel() == 0:
            out.append(t)
            continue
        if isinstance(t.get("mil_boxes", None), torch.Tensor) and t["mil_boxes"].numel() > 0:
            out.append(t)
            continue
        pts = boxes[..., :2].detach().float()
        wh_base = boxes[..., 2:].detach().float()
        n = int(pts.shape[0])
        m = int(bag_size)
        max_wh_px_per_point = None
        if density_enabled:
            max_wh_px_per_point = _density_knn_max_wh_px(
                pts, w_img=w_img, h_img=h_img, knn_k=knn_k, margin=margin_factor, global_min_wh_px=global_min_wh_px, global_max_wh_px=global_max_wh_px
            )
        centers = pts[:, None, :].repeat(1, m, 1)
        cand_wh = torch.zeros((n, m, 2), device=boxes.device, dtype=torch.float32)
        cand_wh[:, 0, 0] = (wh_base[:, 0] * w_img).clamp(min=1.0)
        cand_wh[:, 0, 1] = (wh_base[:, 1] * h_img).clamp(min=1.0)
        for i in range(n):
            if dspe_enabled and isinstance(scale_bank, _ScaleMemoryBank):
                wh_pool = scale_bank.sample_wh_pool(
                    label=int(labels[i].item()),
                    k=max(1, m),
                    explore_ratio=explore_ratio,
                    explore_wh_px=explore_wh_px,
                    min_wh_px=min_wh_px,
                    max_wh_px=max_wh_px,
                    device=boxes.device,
                )
            for j in range(1, m):
                if dspe_enabled and isinstance(scale_bank, _ScaleMemoryBank):
                    wi, hi = wh_pool[j]
                elif wh_pool is None:
                    wi = float(cand_wh[i, 0, 0].item())
                    hi = float(cand_wh[i, 0, 1].item())
                else:
                    wi, hi = wh_pool[int(torch.randint(0, len(wh_pool), (1,), device=boxes.device).item())]
                ar = ar_pool[int(torch.randint(0, len(ar_pool), (1,), device=boxes.device).item())]
                ar = max(1e-6, float(ar))
                s = math.sqrt(ar)
                ww = wi * s
                hh = hi / s
                if jitter > 0:
                    fac = float(torch.empty((), device=boxes.device).uniform_(max(0.0, 1.0 - jitter), 1.0 + jitter).item())
                    ww *= fac
                    hh *= fac
                if density_enabled and max_wh_px_per_point is not None:
                    mwh = float(max_wh_px_per_point[i].item())
                    ww = min(float(ww), mwh)
                    hh = min(float(hh), mwh)
                cand_wh[i, j, 0] = max(1.0, float(ww))
                cand_wh[i, j, 1] = max(1.0, float(hh))
        cand_wh_norm = cand_wh.clone()
        cand_wh_norm[:, :, 0] /= float(w_img)
        cand_wh_norm[:, :, 1] /= float(h_img)
        t_new = dict(t)
        t_new["mil_boxes"] = torch.cat([centers, cand_wh_norm], dim=2).clamp(min=0.0, max=1.0)
        t_new["mil_scores"] = torch.zeros((n, m), device=boxes.device, dtype=torch.float32)
        out.append(t_new)
    return out


def _build_point_pseudo_targets(teacher_outputs, targets, point_sup, use_focal_loss: bool, model_inputs=None, writer: SummaryWriter = None, global_step: int = 0, point_teacher=None, point_teacher_state=None):
    pred_boxes = teacher_outputs["pred_boxes"].detach()
    pred_logits = teacher_outputs["pred_logits"].detach()
    w_point = float(point_sup.get("cost_point", 1.0))
    w_class = float(point_sup.get("cost_class", 2.0))
    score_thresh = float(point_sup.get("score_thresh", 0.2))
    max_l1_dist = float(point_sup.get("max_l1_dist", 1.0))
    snap_center = bool(point_sup.get("snap_center", True))
    keep_all_points = bool(point_sup.get("keep_all_points", True))
    probs = _safe_sigmoid_probs(pred_logits) if use_focal_loss else _safe_softmax_probs(pred_logits)
    pseudo_targets = []
    bs = pred_boxes.shape[0]
    total_gt_points = torch.zeros((), dtype=torch.float32, device=pred_boxes.device)
    total_pseudo = torch.zeros((), dtype=torch.float32, device=pred_boxes.device)
    h_img, w_img = _extract_hw_from_model_inputs(model_inputs) if model_inputs is not None else (640, 640)
    h_img = float(max(1, h_img))
    w_img = float(max(1, w_img))
    dspe_cfg = _get_point_teacher_dspe_cfg(point_teacher, point_teacher_state=point_teacher_state)
    update_beta = float(dspe_cfg.get("update_beta", 0.001))
    update_score_thresh = float(dspe_cfg.get("update_score_thresh", 0.2))
    update_max_area_ratio = dspe_cfg.get("update_max_area_ratio", None)
    pseudo_min_wh_px = float(dspe_cfg.get("pseudo_min_wh_px", 2.0))
    pseudo_max_wh_px = float(dspe_cfg.get("pseudo_max_wh_px", 256.0))
    scale_bank = None
    if isinstance(dspe_cfg, dict) and bool(dspe_cfg.get("enabled", False)) and isinstance(point_teacher_state, dict):
        scale_bank = point_teacher_state.get("scale_bank", None)
    density_cfg = _get_point_sup_density_cfg(point_sup)
    density_enabled = bool(density_cfg.get("enabled", False))
    knn_k = int(density_cfg.get("knn_k", 1))
    margin_factor = float(density_cfg.get("density_margin_factor", 1.2))
    global_max_wh_px = float(density_cfg.get("global_max_wh_px", 256.0))
    global_min_wh_px = float(density_cfg.get("global_min_wh_px", 2.0))
    fg_cfg = _get_point_sup_feature_growth_cfg(point_sup)
    fg_enabled = bool(fg_cfg.get("enabled", False))
    fg_tau = float(fg_cfg.get("tau", 2.0))
    for b in range(bs):
        t = targets[b]
        pts = t["boxes"][..., :2].detach().float()
        labs = t["labels"].detach().long()
        total_gt_points += float(labs.numel())
        max_wh_px_per_point = None
        if density_enabled and pts.numel() > 0:
            max_wh_px_per_point = _density_knn_max_wh_px(
                pts, w_img=w_img, h_img=h_img, knn_k=knn_k, margin=margin_factor, global_min_wh_px=global_min_wh_px, global_max_wh_px=global_max_wh_px
            )
        if pts.numel() == 0:
            t_new = dict(t)
            t_new["boxes"] = t["boxes"].new_zeros((0, 4)).float()
            t_new["labels"] = t["labels"].new_zeros((0,), dtype=torch.long)
            pseudo_targets.append(t_new)
            continue
        bboxes = pred_boxes[b].float()
        bboxes = torch.nan_to_num(bboxes, nan=0.5, posinf=0.5, neginf=0.5)
        bboxes[:, :2] = bboxes[:, :2].clamp(0.0, 1.0)
        bboxes[:, 2:] = bboxes[:, 2:].clamp(1e-6, 1.0)
        prob = probs[b]
        cls_score = prob[:, labs]
        cost_cls = -cls_score
        cost_pt = torch.cdist(bboxes[:, :2], pts, p=1)
        C = w_point * cost_pt + w_class * cost_cls
        C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=1e6)
        row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())
        row_ind = torch.as_tensor(row_ind, dtype=torch.long, device=C.device)
        col_ind = torch.as_tensor(col_ind, dtype=torch.long, device=C.device)
        selected_boxes = bboxes[row_ind]
        selected_labels = labs[col_ind]
        selected_pts = pts[col_ind]
        selected_scores = cls_score[row_ind, col_ind]
        selected_dists = cost_pt[row_ind, col_ind]
        keep = (selected_scores >= score_thresh) & (selected_dists <= max_l1_dist)
        if keep_all_points:
            keep = torch.ones_like(keep, dtype=torch.bool)
        elif not torch.any(keep):
            best = torch.argmax(selected_scores - selected_dists)
            keep = torch.zeros_like(keep, dtype=torch.bool)
            keep[best] = True
        kept_boxes = selected_boxes[keep]
        kept_labels = selected_labels[keep]
        kept_pts = selected_pts[keep]
        total_pseudo += float(kept_labels.numel())
        if snap_center and kept_boxes.numel() > 0:
            kept_boxes = kept_boxes.clone()
            kept_boxes[:, :2] = kept_pts
        if kept_boxes.numel() > 0 and density_enabled and max_wh_px_per_point is not None:
            point_idx = col_ind[keep]
            mwh = max_wh_px_per_point[point_idx].to(kept_boxes.device).float()
            wh_px = kept_boxes[:, 2:].detach().float()
            wh_px = torch.stack([wh_px[:, 0] * w_img, wh_px[:, 1] * h_img], dim=1)
            wh_px[:, 0] = torch.minimum(wh_px[:, 0], mwh)
            wh_px[:, 1] = torch.minimum(wh_px[:, 1], mwh)
            kept_boxes = kept_boxes.clone()
            kept_boxes[:, 2] = (wh_px[:, 0] / w_img).clamp(min=1e-6, max=1.0)
            kept_boxes[:, 3] = (wh_px[:, 1] / h_img).clamp(min=1e-6, max=1.0)
        if kept_boxes.numel() > 0 and fg_enabled and isinstance(scale_bank, _ScaleMemoryBank):
            mean = scale_bank.mean_wh_px[kept_labels.detach().cpu().clamp(min=0, max=scale_bank.num_classes - 1)].to(kept_boxes.device).float()
            std = scale_bank.std_wh_px[kept_labels.detach().cpu().clamp(min=0, max=scale_bank.num_classes - 1)].to(kept_boxes.device).float()
            lo = (mean - fg_tau * std).clamp(min=float(pseudo_min_wh_px))
            hi = (mean + fg_tau * std).clamp(max=float(pseudo_max_wh_px))
            wh_px = kept_boxes[:, 2:].detach().float()
            wh_px = torch.stack([wh_px[:, 0] * w_img, wh_px[:, 1] * h_img], dim=1)
            wh_px = torch.max(lo, torch.min(hi, wh_px))
            kept_boxes = kept_boxes.clone()
            kept_boxes[:, 2] = (wh_px[:, 0] / w_img).clamp(min=1e-6, max=1.0)
            kept_boxes[:, 3] = (wh_px[:, 1] / h_img).clamp(min=1e-6, max=1.0)
        t_new = dict(t)
        t_new["boxes"] = kept_boxes
        t_new["labels"] = kept_labels
        pseudo_targets.append(t_new)
        if isinstance(scale_bank, _ScaleMemoryBank) and kept_boxes.numel() > 0:
            wh_px = kept_boxes[:, 2:].detach().float()
            wh_px = torch.stack([wh_px[:, 0] * w_img, wh_px[:, 1] * h_img], dim=1)
            scale_bank.update(
                kept_labels.detach(),
                wh_px,
                scores=selected_scores[keep].detach() if selected_scores.numel() > 0 else None,
                beta=update_beta,
                score_thresh=update_score_thresh,
                min_wh_px=pseudo_min_wh_px,
                max_wh_px=pseudo_max_wh_px,
                max_area_ratio=update_max_area_ratio,
            )
    if writer is not None and dist_utils.is_main_process() and global_step % 200 == 0:
        writer.add_scalar("PointSup/gt_points_per_iter", float(total_gt_points.item()), global_step)
        writer.add_scalar("PointSup/pseudo_boxes_per_iter", float(total_pseudo.item()), global_step)
    if dist_utils.is_main_process() and global_step % 200 == 0:
        gt = float(total_gt_points.item())
        ps = float(total_pseudo.item())
        ratio = ps / max(1e-6, gt)
        logger.info(
            ORANGE
            + f"[PointTeacher] step={int(global_step)} gt_points={gt:.1f} pseudo_boxes={ps:.1f} ratio={ratio:.3f} "
            + f"(score_thresh={float(point_sup.get('score_thresh', 0.0))}, max_l1_dist={float(point_sup.get('max_l1_dist', 1.0))}, keep_all_points={bool(point_sup.get('keep_all_points', False))})"
            + RESET
        )
    return pseudo_targets

def _plot_training_modalities(samples, targets, data_loader, output_dir, epoch):    
    modality_samples = select_plot_samples_for_logging(samples, keys=("rgb", "npy"))
    is_multimodal_plot = len(modality_samples) > 1

    for modality, plot_samples in modality_samples:     
        if modality == "npy":    
            plot_samples = normalize_tensor_minmax_per_sample(plot_samples)
        if modality == "rgb" and not is_multimodal_plot:  
            suffix = ""
        else: 
            suffix = f"_{modality}"
        save_path = output_dir / f"train_batch_{epoch}{suffix}.png"  
        if data_loader.dataset.remap_mscoco_category:
            plot_sample((plot_samples, targets), data_loader.dataset.category2name, save_path, data_loader.dataset.label2category)     
        else:
            plot_sample((plot_samples, targets), data_loader.dataset.category2name, save_path)  

# 训练单个 epoch
# self_lr_scheduler: 是否使用自定义学习率调度器
# lr_scheduler: 学习率调度器实例  
# model: 训练的 PyTorch 模型    
# criterion: 损失计算函数
# data_loader: 训练数据加载器  
# optimizer: 优化器
# device: 训练设备（CPU 或 GPU）  
# epoch: 当前 epoch 计数   
# max_norm: 梯度裁剪的最大范数
# **kwargs: 其他参数，例如日志记录等
def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,  
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,   
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()  # 设置模型为训练模式
    criterion.train()  # 设置损失函数为训练模式    
     
    print_freq = kwargs.get('print_freq', 10)  # 日志打印频率
    writer: SummaryWriter = kwargs.get('writer', None)  # TensorBoard 记录器
    ema: ModelEMA = kwargs.get('ema', None)  # 指数移动平均模型
    scaler: GradScaler = kwargs.get('scaler', None)  # 混合精度训练的梯度缩放器   
    lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)  # 预热学习率调度器  
    plot_train_batch_freq = kwargs.get('plot_train_batch_freq', 12)
    output_dir = kwargs.get('output_dir', None)  
    epoches = kwargs.get('epoches', -1) # 总的训练次数    
    verbose_type = kwargs.get('verbose_type', 'origin') # 显示方式  
    point_sup = kwargs.get("point_sup", None)
    point_teacher = kwargs.get("point_teacher", None)
    point_teacher_state = kwargs.get("point_teacher_state", None)
    use_focal_loss = bool(kwargs.get("use_focal_loss", True))
    header = 'Epoch: {}/{}'.format(epoch, epoches)  # 训练过程的日志标题
     
    cur_iters = epoch * len(data_loader)  # 计算当前 epoch 的起始迭代数
    
    if verbose_type == 'origin':
        metric_logger = MetricLogger(delimiter="  ")  # 记录训练过程中的度量信息
    else:
        metric_logger = MetricLogger_progress(delimiter="  ")  # 记录训练过程中的度量信息
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 记录学习率变化
    pbar = enumerate(metric_logger.log_every(data_loader, print_freq if verbose_type == 'origin' else 1, header))

    dt = [
        Profile(device=device),   
        Profile(device=device),   
        Profile(device=device),
        Profile(device=device),  
        Profile(device=device)
    ]    
   
    for i, (samples, targets) in pbar:
        if i % CLEAR_MEMORY_STEP == 0: 
            if torch.cuda.is_available():   
                torch.cuda.empty_cache()

        if epoch % plot_train_batch_freq == 0 and i == 0:    
            _plot_training_modalities(samples, targets, data_loader, output_dir, epoch)   
        with dt[0]:
            samples = move_samples_to_device(samples, device, non_blocking=True)  # 将输入数据移动到指定设备 
            model_inputs = select_model_input_for_model(samples, model=model, key='rgb')
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]  # 目标数据也移动到设备
        
        global_step = epoch * len(data_loader) + i  # 计算全局训练步数
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))  # 训练元数据  

        # 使用混合精度训练   
        if scaler is not None:    
            with dt[1]:  
                with torch.autocast(device_type=str(device), cache_enabled=True):     
                    targets_for_train = targets
                    student_inputs = model_inputs
                    if _is_point_teacher_enabled(point_teacher):
                        student_inputs = _random_block_mask(model_inputs, point_teacher.get("RandomMask", None) if isinstance(point_teacher, dict) else None)
                        burn_in_steps = int(point_teacher.get("burn_in_steps", 0) or 0) if isinstance(point_teacher, dict) else 0
                        if burn_in_steps > 0 and int(global_step) < burn_in_steps:
                            targets_for_train = _build_point_fixed_targets(targets, point_teacher, model_inputs=model_inputs, point_teacher_state=point_teacher_state, point_sup=point_sup)
                            targets_for_train = _ensure_point_teacher_bag(targets_for_train, point_teacher, model_inputs=model_inputs, point_teacher_state=point_teacher_state, point_sup=point_sup)
                        elif _is_point_supervision_enabled(point_sup):
                            if ema is None:
                                raise RuntimeError("PointTeacher requires EMA teacher. Set use_ema=True.")
                            with torch.no_grad():
                                teacher_outputs = ema.module(model_inputs, targets=None)
                            point_sup_eff = point_sup
                            if isinstance(point_sup, dict) and isinstance(point_teacher, dict):
                                point_sup_eff = dict(point_sup)
                                for k in ("score_thresh", "max_l1_dist", "cost_point", "cost_class", "snap_center", "keep_all_points"):
                                    if k in point_teacher:
                                        point_sup_eff[k] = point_teacher[k]
                            targets_for_train = _build_point_pseudo_targets(
                                teacher_outputs,
                                targets,
                                point_sup=point_sup_eff,
                                use_focal_loss=use_focal_loss,
                                model_inputs=model_inputs,
                                writer=writer,
                                global_step=global_step,
                                point_teacher=point_teacher,
                                point_teacher_state=point_teacher_state,
                            )
                            targets_for_train = _ensure_point_teacher_bag(targets_for_train, point_teacher, model_inputs=model_inputs, point_teacher_state=point_teacher_state, point_sup=point_sup)
                    outputs = model(student_inputs, targets=targets_for_train)     
 
            # 处理异常情况，避免 NaN 或 Inf 影响训练 
            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                # print(outputs['pred_boxes'])  
                logger.warning(outputs['pred_boxes'])
                state = model.state_dict()    
                new_state = {}  
                for key, value in model.state_dict().items():
                    new_key = key.replace('module.', '')  # 兼容多 GPU 训练的情况
                    state[new_key] = value 
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")  # 保存异常模型状态   
     
            with dt[2]:
            # 计算损失
                with torch.autocast(device_type=str(device), enabled=False):
                    loss_dict = criterion(outputs, targets_for_train, **metas)
                loss = sum(loss_dict.values())  # 总损失  
            
            with dt[3]:    
                scaler.scale(loss).backward()  # 反向传播
     
                # 进行梯度裁剪（如果 max_norm > 0）
                if max_norm > 0: 
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                
                scaler.step(optimizer)  # 更新参数
                scaler.update()  # 更新梯度缩放因子
                optimizer.zero_grad()  # 清空梯度
        
        else:    
            with dt[1]:
                targets_for_train = targets
                student_inputs = model_inputs
                if _is_point_teacher_enabled(point_teacher):
                    student_inputs = _random_block_mask(model_inputs, point_teacher.get("RandomMask", None) if isinstance(point_teacher, dict) else None)
                    burn_in_steps = int(point_teacher.get("burn_in_steps", 0) or 0) if isinstance(point_teacher, dict) else 0
                    if burn_in_steps > 0 and int(global_step) < burn_in_steps:
                        targets_for_train = _build_point_fixed_targets(targets, point_teacher, model_inputs=model_inputs, point_teacher_state=point_teacher_state, point_sup=point_sup)
                        targets_for_train = _ensure_point_teacher_bag(targets_for_train, point_teacher, model_inputs=model_inputs, point_teacher_state=point_teacher_state, point_sup=point_sup)
                    elif _is_point_supervision_enabled(point_sup):
                        if ema is None:
                            raise RuntimeError("PointTeacher requires EMA teacher. Set use_ema=True.")
                        with torch.no_grad():
                            teacher_outputs = ema.module(model_inputs, targets=None)
                        point_sup_eff = point_sup
                        if isinstance(point_sup, dict) and isinstance(point_teacher, dict):
                            point_sup_eff = dict(point_sup)
                            for k in ("score_thresh", "max_l1_dist", "cost_point", "cost_class", "snap_center", "keep_all_points"):
                                if k in point_teacher:
                                    point_sup_eff[k] = point_teacher[k]
                        targets_for_train = _build_point_pseudo_targets(
                            teacher_outputs,
                            targets,
                            point_sup=point_sup_eff,
                            use_focal_loss=use_focal_loss,
                            model_inputs=model_inputs,
                            writer=writer,
                            global_step=global_step,
                            point_teacher=point_teacher,
                            point_teacher_state=point_teacher_state,
                        )
                        targets_for_train = _ensure_point_teacher_bag(targets_for_train, point_teacher, model_inputs=model_inputs, point_teacher_state=point_teacher_state, point_sup=point_sup)
                outputs = model(student_inputs, targets=targets_for_train)  # 前向传播     
            with dt[2]: 
                loss_dict = criterion(outputs, targets_for_train, **metas)  # 计算损失   
                loss: torch.Tensor = sum(loss_dict.values())  # 总损失
            with dt[3]:
                optimizer.zero_grad()  # 清空梯度    
                loss.backward()  # 反向传播   
     
                # 进行梯度裁剪
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()  # 更新参数
    
        with dt[4]:
            # 更新 EMA（指数移动平均）
            if ema is not None:  
                ema.update(model)

            # 更新学习率 
            if self_lr_scheduler:   
                optimizer = lr_scheduler.step(cur_iters + i, optimizer)   
            else:   
                if lr_warmup_scheduler is not None:
                    lr_warmup_scheduler.step()    

            # 计算损失并检查是否异常    
            loss_dict_reduced = dist_utils.reduce_dict(loss_dict)    
            loss_value = sum(loss_dict_reduced.values()) 
            if not math.isfinite(loss_value):
                # print("Loss is {}, stopping training".format(loss_value))
                # print(loss_dict_reduced) 
                logger.warning("Loss is {}, stopping training".format(loss_value))
                logger.info(loss_dict_reduced)    
                sys.exit(1)

            # 记录日志
            metric_logger.update(loss=loss_value, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])    
  
            # 记录到 TensorBoard  
            if writer and dist_utils.is_main_process() and global_step % 10 == 0:   
                writer.add_scalar('Loss/total', loss_value.item(), global_step)  
                for j, pg in enumerate(optimizer.param_groups):   
                    writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)    
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar(f'Loss/{k}', v.item(), global_step) 

    # 统计并打印训练结果
    metric_logger.synchronize_between_processes()    
    logger.info(f'Averaged stats:{metric_logger}')  
    if TIME_DEBUG:    
        time_data = [x.t / len(data_loader) for x in dt]
        # print(RED + f"Data_to_Device:{time_data[0]:.6f}s Inference:{time_data[1]:.6f}s Loss:{time_data[2]:.6f}s Weight_Update:{time_data[3]:.6f}s" + RESET)  
        logger.debug(RED + f"Data_to_Device:{time_data[0]:.6f}s Inference:{time_data[1]:.6f}s Loss:{time_data[2]:.6f}s Weight_Update:{time_data[3]:.6f}s" + RESET)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
   

@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device, test_only=False, output_dir=None, yolo_metrice=False, other_platform_model=None):     
    # 评估函数，禁用梯度计算以减少内存占用并提高推理速度
    if model is not None:  
        model.eval()
    criterion.eval()  
    coco_evaluator.cleanup()
  
    metric_logger = MetricLogger_progress(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))     
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())  
    # 获取 IoU 计算类型（如 'bbox' 或 'segm'）
    iou_types = coco_evaluator.iou_types     
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)   
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]     
 
    # 初始化时间记录器   
    dt = [
        Profile(device=device),
        Profile(device=device)   
    ]
     
    # 遍历数据集进行评估   
    coco_det_pred_json, coco_seg_pred_json = [], []    
    for samples, targets in metric_logger.log_every(data_loader, 1, header):     
        samples = move_samples_to_device(samples, device, non_blocking=True)  # 将样本数据移动到指定设备（如 GPU）  
        model_inputs = select_model_input_for_model(samples, model=model, key='rgb')     
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]  # 目标数据也移动到设备
 
        if model is not None:
            with dt[0]:
                outputs = model(model_inputs)  # 前向传播，获取模型输出
     
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)  # 获取原始目标尺寸  
 
            with dt[1]:
                results = postprocessor(outputs, orig_target_sizes, for_eval=True)  # 通过后处理器处理模型输出
        else:
            if 'onnx' in other_platform_model:  
                with dt[0]:
                    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)  # 获取原始目标尺寸
                    onnx_result = other_platform_model['onnx'].run(  
                        output_names=None,
                        input_feed={'images': model_inputs.cpu().detach().numpy(), "orig_target_sizes": orig_target_sizes.cpu().detach().numpy()}
                    )
 
                    results = []
                    if len(onnx_result) == 3:   
                        labels, boxes, scores = onnx_result 
                        for lab, box, sco in zip(labels, boxes, scores):     
                            result = dict(labels=torch.from_numpy(lab), boxes=torch.from_numpy(box), scores=torch.from_numpy(sco))
                            results.append(result)
                    elif len(onnx_result) == 4:     
                        labels, boxes, scores, masks = onnx_result
                        for lab, box, sco, mask in zip(labels, boxes, scores, masks):
                            result = dict(labels=torch.from_numpy(lab), boxes=torch.from_numpy(box), scores=torch.from_numpy(sco), masks=torch.from_numpy(mask))  
                            results.append(result)
                        
            elif 'engine' in other_platform_model:
                with dt[0]:
                    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)  # 获取原始目标尺寸   
                    output = other_platform_model['engine']({'images': model_inputs,    
                                                             'orig_target_sizes': orig_target_sizes.to(device)})
                    labels, boxes, scores, masks = output['labels'], output['boxes'], output['scores'], output.get('masks', None)
     
                    results = [] 
                    if masks is None:
                        for lab, box, sco in zip(labels, boxes, scores):
                            result = dict(labels=lab, boxes=box, scores=sco)
                            results.append(result)
                    else:    
                        for lab, box, sco, mask in zip(labels, boxes, scores, masks):
                            result = dict(labels=lab, boxes=box, scores=sco, masks=mask) 
                            results.append(result)
       
        res = {target['image_id'].item(): output for target, output in zip(targets, results)} # 将评估结果与图像 ID 关联
        if coco_evaluator is not None:     
            coco_evaluator.update(res) # 更新 COCO 评估器
            coco_det_pred_json.extend(list(coco_evaluator.coco_eval['bbox'].cocoDt.anns.values()))   
            if 'segm' in coco_evaluator.coco_eval:
                coco_seg_pred_json.extend(list(coco_evaluator.coco_eval['segm'].cocoDt.anns.values()))
     
    # gather the stats from all processes 在多进程环境下同步评估数据    
    metric_logger.synchronize_between_processes()    
    if coco_evaluator is not None:     
        coco_evaluator.synchronize_between_processes()

    # 统计耗时    
    if test_only:     
        if model is not None:
            speed = dict(zip(['inference', 'postprocess'], (x.t / len(data_loader.dataset) * 1e3 for x in dt)))     
            logger.info(GREEN + f'Test On BatchSize:{data_loader.batch_size}' + RESET)
            logger.info(GREEN + f"Speed: {speed['inference']:.4f}ms inference, {speed['postprocess']:.4f}ms postprocess per image" + RESET) 
            logger.info(GREEN + f"FPS(inference+postprocess): {1000 / (speed['inference'] + speed['postprocess']):.2f}" + RESET)
        else:    
            inference_speed = dt[0].t / len(data_loader.dataset) * 1e3     
            logger.info(GREEN + f'Test On BatchSize:{data_loader.batch_size}' + RESET) 
            logger.info(GREEN + f"Speed: {inference_speed:.4f}ms inference per image" + RESET)
            logger.info(GREEN + f"FPS(inference): {1000 / inference_speed:.2f}" + RESET)  
  
    if yolo_metrice:     
        get_yolo_det_metrice(logger, coco_evaluator, coco_det_pred_json, output_dir if test_only else None)   
        if 'segm' in coco_evaluator.coco_eval:   
            get_yolo_seg_metrice(logger, coco_evaluator, coco_seg_pred_json, save_vis=False)   
 
    # accumulate predictions from all images 累积并计算最终评估结果 
    if coco_evaluator is not None:  
        logger.info(RED + "------------------------ COCO Metrice Start ------------------------" + RESET)    
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        if test_only:
            logger.info(ORANGE + f"Saving coco pred[{output_dir / 'pred.json'}] json..." + RESET)
            with open(output_dir / 'pred_bbox.json', 'w') as f:
                json.dump(coco_det_pred_json, f)     
            # if 'segm' in coco_evaluator.coco_eval:
            #     with open(output_dir / 'pred_segm.json', 'w') as f: 
            #         json.dump(coco_seg_pred_json, f)   
            logger.info(ORANGE + "save success." + RESET) 
     
            for iouType in coco_evaluator.coco_gt:
                model_metrice_table = coco_evaluator_per_class(coco_evaluator, iouType)
                print(ORANGE, model_metrice_table, RESET)

            try:
                logger.info(RED + "------------------------ TIDE Metrice Start ------------------------" + RESET)
                tide = TIDE()   
                tide.evaluate_range(datasets.COCO(data_loader.dataset.ann_file), datasets.COCOResult(output_dir / 'pred_bbox.json'))
                tide.summarize()
                tide.plot(out_dir=output_dir / 'tide_result')  
            except Exception as e:
                logger.error(RED, 'TIDE failure... skip message:', e, RESET)     
                logger.warning('------------------------ TIDE指标生成报错可以不用管 ------------------------')
 
    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()  
        if 'segm' in iou_types:     
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
     
    return stats, coco_evaluator
  
def distill_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, teahcer_model: torch.nn.Module, student_featureExt, teacher_featureExt, 
                    criterion: torch.nn.Module, feature_distill_criterion, logical_distill_criterion,   
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()  # 设置模型为训练模式    
    teahcer_model.train() # 设置模型为训练模式
    criterion.train()  # 设置损失函数为训练模式    
     
    print_freq = kwargs.get('print_freq', 10)  # 日志打印频率    
    writer: SummaryWriter = kwargs.get('writer', None)  # TensorBoard 记录器
    ema: ModelEMA = kwargs.get('ema', None)  # 指数移动平均模型  
    scaler: GradScaler = kwargs.get('scaler', None)  # 混合精度训练的梯度缩放器
    lr_warmup_scheduler: Warmup = kwargs.get('lr_warmup_scheduler', None)  # 预热学习率调度器
    plot_train_batch_freq = kwargs.get('plot_train_batch_freq', 12)
    output_dir = kwargs.get('output_dir', None)
    epoches = kwargs.get('epoches', -1) # 总的训练次数     
    verbose_type = kwargs.get('verbose_type', 'origin') # 显示方式
    feature_loss_ratio = kwargs.get('feature_loss_ratio', 1.0)
    logical_loss_ratio = kwargs.get('logical_loss_ratio', 1.0)    
    distill_loss_decay = kwargs.get('distill_loss_decay', 'constant')
    header = 'Epoch: {}/{}'.format(epoch, epoches)  # 训练过程的日志标题
    
    cur_iters = epoch * len(data_loader)  # 计算当前 epoch 的起始迭代数
   
    if verbose_type == 'origin':   
        metric_logger = MetricLogger(delimiter="  ")  # 记录训练过程中的度量信息  
    else:     
        metric_logger = MetricLogger_progress(delimiter="  ")  # 记录训练过程中的度量信息
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 记录学习率变化
    pbar = enumerate(metric_logger.log_every(data_loader, print_freq if verbose_type == 'origin' else 1, header))

    dt = [
        Profile(device=device), 
        Profile(device=device), 
        Profile(device=device),
        Profile(device=device),
        Profile(device=device)
    ]

    for i, (samples, targets) in pbar: 
        if i % CLEAR_MEMORY_STEP == 0:  
            if torch.cuda.is_available():     
                torch.cuda.empty_cache()  
     
        # -------------- 蒸馏损失的调度因子  可视化文件在tools/visualization/distill_decay_visual.py内
        if distill_loss_decay == 'constant':  
            # 特点：蒸馏损失权重保持不变
            # 适用场景：希望蒸馏损失在整个训练过程中保持恒定影响  
            distill_decay = 1.0     
        elif distill_loss_decay == 'cosine':     
            # 特点：在每个epoch内进行余弦衰减，epoch间重置   
            # 衰减曲线：平滑的余弦曲线，先快后慢
            # 适用场景：希望在每个epoch内逐渐减少蒸馏损失的影响     
            eta_min, base_ratio, T_max = 0.01, 1.0, 10
            distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * i / T_max)) / 2    
        elif distill_loss_decay == 'linear':   
            # 特点：在每个epoch内进行线性衰减 
            # 衰减曲线：均匀的线性下降
            # 适用场景：希望蒸馏损失在epoch内均匀递减 
            distill_decay = ((1 - math.cos(i * math.pi / len(data_loader))) / 2) * (0.01 - 1) + 1     
        elif distill_loss_decay == 'cosine_epoch':
            # 特点：跨epoch的连续余弦衰减
            # 衰减曲线：整个训练过程的平滑余弦衰减
            # 适用场景：希望蒸馏损失在整个训练过程中平滑递减
            eta_min, base_ratio, T_max = 0.01, 1.0, 10
            distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * (cur_iters + i) / T_max)) / 2    
        elif distill_loss_decay == 'linear_epoch':
            # 特点：跨epoch的连续线性衰减
            # 衰减曲线：整个训练过程的均匀线性下降 
            # 适用场景：希望蒸馏损失在整个训练过程中均匀递减 
            distill_decay = ((1 - math.cos((cur_iters + i) * math.pi / (epoches * len(data_loader)))) / 2) * (0.01 - 1) + 1

        if epoch % plot_train_batch_freq == 0 and i == 0:    
            _plot_training_modalities(samples, targets, data_loader, output_dir, epoch)  
        with dt[0]:     
            samples = move_samples_to_device(samples, device, non_blocking=True)  # 将输入数据移动到指定设备    
            model_inputs = select_model_input_for_model(samples, model=model, key='rgb')
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]  # 目标数据也移动到设备
    
        global_step = epoch * len(data_loader) + i  # 计算全局训练步数  
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))  # 训练元数据
  
        if feature_distill_criterion: 
            student_featureExt.clear_features()    
            teacher_featureExt.clear_features()

        with dt[1]:
            outputs = model(model_inputs, targets=targets)  # 前向传播    
            if feature_distill_criterion or logical_distill_criterion:
                with torch.no_grad():  
                    teacher_outputs = teahcer_model(model_inputs, targets=targets)

        with dt[2]:
            loss_dict = criterion(outputs, targets, **metas)  # 计算损失
   
            if feature_distill_criterion:
                feature_distill_loss = feature_distill_criterion(student_featureExt.get_features_in_order(), teacher_featureExt.get_features_in_order()) * feature_loss_ratio * distill_decay     
                loss_dict['fea_loss'] = feature_distill_loss
            else:
                loss_dict['fea_loss'] = torch.zeros(1, device=device)  
 
            if logical_distill_criterion:     
                logical_distill_loss = logical_distill_criterion(outputs, teacher_outputs, targets) * logical_loss_ratio * distill_decay
                loss_dict['log_loss'] = logical_distill_loss
            else:
                loss_dict['log_loss'] = torch.zeros(1, device=device)

            loss: torch.Tensor = sum(loss_dict.values())  # 总损失   
   
        with dt[3]: 
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            
            # 进行梯度裁剪
            if max_norm > 0:  
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)   
            optimizer.step()  # 更新参数

        with dt[4]:
            # 更新 EMA（指数移动平均）
            if ema is not None:
                ema.update(model)    
    
            # 更新学习率   
            if self_lr_scheduler:
                optimizer = lr_scheduler.step(cur_iters + i, optimizer)
            else:
                if lr_warmup_scheduler is not None:     
                    lr_warmup_scheduler.step()
     
            # 计算损失并检查是否异常 
            loss_dict_reduced = dist_utils.reduce_dict(loss_dict)   
            loss_value = sum(loss_dict_reduced.values())    
            if not math.isfinite(loss_value):    
                print("Loss is {}, stopping training".format(loss_value))   
                print(loss_dict_reduced)    
                sys.exit(1)

            # 记录日志
            metric_logger.update(loss=loss_value, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])    
     
            # 记录到 TensorBoard   
            if writer and dist_utils.is_main_process() and global_step % 10 == 0:   
                writer.add_scalar('Loss/total', loss_value.item(), global_step)
                writer.add_scalar('Distill/Decay', distill_decay, global_step)
                for j, pg in enumerate(optimizer.param_groups):
                    writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)     
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar(f'Loss/{k}', v.item(), global_step) 
   
    # 统计并打印训练结果
    metric_logger.synchronize_between_processes()
    logger.info(f'Averaged stats:{metric_logger}')  
    if TIME_DEBUG:
        time_data = [x.t / len(data_loader) for x in dt]
        print(RED + f"Data_to_Device:{time_data[0]:.6f}s Inference:{time_data[1]:.6f}s Loss:{time_data[2]:.6f}s Weight_Update:{time_data[3]:.6f}s" + RESET)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}  
