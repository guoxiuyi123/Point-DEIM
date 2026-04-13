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
 
 
def _safe_sigmoid_probs(logits: torch.Tensor) -> torch.Tensor:
    return logits.float().sigmoid()
 
 
def _safe_softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return logits.float().softmax(-1)
 
 
def _get_teacher_layer_boxes(teacher_outputs):
    layers = []
    if isinstance(teacher_outputs, dict) and "pred_boxes" in teacher_outputs:
        layers.append(teacher_outputs["pred_boxes"])
    aux = teacher_outputs.get("aux_outputs", None) if isinstance(teacher_outputs, dict) else None
    if isinstance(aux, (list, tuple)):
        for o in aux:
            if isinstance(o, dict) and "pred_boxes" in o:
                layers.append(o["pred_boxes"])
    return layers
 
 
def _query_logwh_stability_px(teacher_outputs, base_size: float, batch_index: int, query_indices: torch.Tensor):
    layers = _get_teacher_layer_boxes(teacher_outputs)
    if len(layers) <= 1:
        return torch.zeros((query_indices.numel(),), device=query_indices.device, dtype=torch.float32)
    logs = []
    for boxes in layers:
        b = boxes.detach()[batch_index].float()
        wh = b[query_indices][:, 2:].clamp(min=1e-6)
        logwh_px = torch.log(torch.clamp(wh * float(base_size), min=1.0))
        logs.append(logwh_px)
    stack = torch.stack(logs, dim=0)
    mean = stack.mean(dim=0)
    dev = (stack - mean).abs().mean(dim=0)
    return dev.mean(dim=1)
 
 
def _extract_hw_from_model_inputs(model_inputs):
    if torch.is_tensor(model_inputs):
        return int(model_inputs.shape[-2]), int(model_inputs.shape[-1])
    if isinstance(model_inputs, dict):
        for v in model_inputs.values():
            if torch.is_tensor(v):
                return int(v.shape[-2]), int(v.shape[-1])
    raise ValueError("Cannot infer H,W from model_inputs")
 
 
class _ScaleMemoryBank:
    def __init__(self, num_classes: int, init_mean_wh_px=(12.0, 12.0), init_std_wh_px=(1.0, 1.0), min_std_px=(0.35, 0.35)):
        self.num_classes = int(num_classes)
        mean = torch.tensor(init_mean_wh_px, dtype=torch.float32).log()
        std = torch.tensor(init_std_wh_px, dtype=torch.float32)
        self.mu = mean.repeat(self.num_classes, 1).clone()
        self.var = (std**2).repeat(self.num_classes, 1).clone()
        self.min_std = torch.tensor(min_std_px, dtype=torch.float32)
        self.steps = 0
 
    def get_stats(self, device=None):
        if device is None:
            return self.mu, self.var
        return self.mu.to(device), self.var.to(device)
 
    def update_from_moments(self, mean_logwh: torch.Tensor, var_logwh: torch.Tensor, count: torch.Tensor, beta: float):
        beta = float(beta)
        if beta <= 0:
            return
        valid = count > 0
        if not torch.any(valid):
            return
        mean_logwh = mean_logwh.to(self.mu.device)
        var_logwh = var_logwh.to(self.var.device)
        valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
        self.mu[valid_idx] = (1 - beta) * self.mu[valid_idx] + beta * mean_logwh[valid_idx]
        self.var[valid_idx] = (1 - beta) * self.var[valid_idx] + beta * var_logwh[valid_idx]
        std = torch.sqrt(torch.clamp(self.var, min=1e-6))
        std = torch.maximum(std, self.min_std[None, :])
        self.var = std**2
        self.steps += 1
 
 
def _sample_logwh_from_bank(bank: _ScaleMemoryBank, class_id: int, k: int, explore_ratio: float, explore_wh_px, device):
    mu, var = bank.get_stats(device=device)
    cid = int(class_id)
    k = int(k)
    explore_ratio = float(explore_ratio)
    k_explore = int(round(k * explore_ratio))
    k_exploit = max(0, k - k_explore)
 
    parts = []
    if k_exploit > 0:
        eps = torch.randn((k_exploit, 2), device=device)
        std = torch.sqrt(torch.clamp(var[cid], min=1e-6))
        parts.append(mu[cid][None, :] + eps * std[None, :])
    if k_explore > 0:
        explore_wh_px = torch.as_tensor(explore_wh_px, device=device, dtype=torch.float32)
        if explore_wh_px.ndim == 1:
            explore_wh_px = explore_wh_px[:, None].repeat(1, 2)
        idx = torch.randint(0, explore_wh_px.shape[0], (k_explore,), device=device)
        parts.append(torch.log(torch.clamp(explore_wh_px[idx], min=1.0)))
    if not parts:
        return mu[cid][None, :].repeat(k, 1)
    return torch.cat(parts, dim=0)
 
 
def _build_point_pseudo_targets(teacher_outputs, targets, point_sup, use_focal_loss: bool, model_inputs=None, point_sup_state=None, writer: SummaryWriter = None, global_step: int = 0, epoch: int = 0):
    pred_boxes = teacher_outputs["pred_boxes"].detach()
    pred_logits = teacher_outputs["pred_logits"].detach()
    w_point = float(point_sup.get("cost_point", 1.0))
    w_class = float(point_sup.get("cost_class", 2.0))
    w_prior = float(point_sup.get("cost_prior", 0.0))
    score_thresh = float(point_sup.get("score_thresh", 0.2))
    max_l1_dist = float(point_sup.get("max_l1_dist", 1.0))
    snap_center = bool(point_sup.get("snap_center", True))
    keep_all_points = bool(point_sup.get("keep_all_points", True))
    dspe_cfg = point_sup.get("DSPE", None) or point_sup.get("dspe", None) or {}
    dspe_enabled = isinstance(dspe_cfg, dict) and dspe_cfg.get("enabled", False) and point_sup_state is not None and point_sup_state.get("scale_bank", None) is not None
    base_size = float(dspe_cfg.get("base_size", 640.0))
    bag_size = int(dspe_cfg.get("bag_size", 16))
    explore_ratio = float(dspe_cfg.get("explore_ratio", 0.25))
    explore_wh_px = dspe_cfg.get("explore_wh_px", [4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64])
    update_top_frac = float(dspe_cfg.get("update_top_frac", 0.10))
    update_beta = float(dspe_cfg.get("update_beta", 0.02))
    update_score_thresh = float(dspe_cfg.get("update_score_thresh", max(score_thresh, 0.3)))
    update_max_dist_px = float(dspe_cfg.get("update_max_dist_px", 32.0))
    large_box_area_ratio_thresh = float(dspe_cfg.get("large_box_area_ratio_thresh", 2.0))
    large_box_area_penalty = float(dspe_cfg.get("large_box_area_penalty", 5.0))
    update_max_area_ratio = float(dspe_cfg.get("update_max_area_ratio", 6.0))
    update_stab_thresh = float(dspe_cfg.get("update_stab_thresh", 1e9))
    pseudo_prior_mix = float(dspe_cfg.get("pseudo_prior_mix", 0.0))
    pseudo_prior_mix_schedule = dspe_cfg.get("pseudo_prior_mix_schedule", None)
    pseudo_min_wh_px = float(dspe_cfg.get("pseudo_min_wh_px", 2.0))
    pseudo_max_wh_px = float(dspe_cfg.get("pseudo_max_wh_px", 128.0))
    wh_select_k = int(dspe_cfg.get("wh_select_k", bag_size))
    bag_aspect_ratios = dspe_cfg.get("bag_aspect_ratios", [0.5, 1.0, 2.0])
    bag_wh_weight = float(dspe_cfg.get("bag_wh_weight", 1.0))
    bag_stab_weight = float(dspe_cfg.get("bag_stab_weight", 0.0))
 
    mix = pseudo_prior_mix
    if isinstance(pseudo_prior_mix_schedule, (list, tuple)) and pseudo_prior_mix_schedule:
        try:
            for item in pseudo_prior_mix_schedule:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                start_epoch, value = int(item[0]), float(item[1])
                if int(epoch) >= start_epoch:
                    mix = value
        except Exception:
            mix = pseudo_prior_mix
 
    if model_inputs is not None:
        h_img, w_img = _extract_hw_from_model_inputs(model_inputs)
    else:
        h_img, w_img = 0, 0
 
    probs = _safe_sigmoid_probs(pred_logits) if use_focal_loss else _safe_softmax_probs(pred_logits)
 
    pseudo_targets = []
    bs = pred_boxes.shape[0]
    total_gt_points = torch.zeros((), dtype=torch.float32, device=pred_boxes.device)
    total_pseudo = torch.zeros((), dtype=torch.float32, device=pred_boxes.device)
    sums = None
    sums_sq = None
    counts = None
    bank = point_sup_state.get("scale_bank", None) if dspe_enabled else None
    num_classes = int(dspe_cfg.get("num_classes", point_sup_state.get("num_classes", 0) or 0))
    if dspe_enabled and num_classes > 0:
        sums = torch.zeros((num_classes, 2), dtype=torch.float32, device=pred_boxes.device)
        sums_sq = torch.zeros((num_classes, 2), dtype=torch.float32, device=pred_boxes.device)
        counts = torch.zeros((num_classes,), dtype=torch.float32, device=pred_boxes.device)
        mu_logwh, _ = bank.get_stats(device=pred_boxes.device)
        prior_wh_px = torch.clamp(mu_logwh.exp(), min=1.0)
        prior_wh_px = torch.clamp(prior_wh_px, min=pseudo_min_wh_px, max=pseudo_max_wh_px)
        prior_wh_norm = prior_wh_px / base_size
        prior_area_px = (prior_wh_px[:, 0] * prior_wh_px[:, 1]).clamp(min=1.0)
    else:
        prior_wh_norm = None
        prior_area_px = None
 
    for b in range(bs):
        t = targets[b]
        pts = t["boxes"][..., :2].detach().float()
        labs = t["labels"].detach()
        total_gt_points += float(labs.numel())
 
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
        if dspe_enabled and w_prior > 0 and num_classes > 0:
            wh_q_px = torch.clamp(bboxes[:, 2:] * base_size, min=pseudo_min_wh_px, max=pseudo_max_wh_px)
            logwh_q_px = torch.log(torch.clamp(wh_q_px, min=1.0))
            prior_cost_cols = []
            area_q_px = (wh_q_px[:, 0] * wh_q_px[:, 1]).clamp(min=1.0)
            for lab in labs.tolist():
                mu_c = mu_logwh[int(lab)][None, :]
                prior_cost_col = torch.cdist(logwh_q_px, mu_c, p=1).squeeze(1)
                if prior_area_px is not None and large_box_area_penalty > 0:
                    ratio = area_q_px / prior_area_px[int(lab)]
                    extra = torch.relu(ratio - large_box_area_ratio_thresh) * large_box_area_penalty
                    prior_cost_col = prior_cost_col + extra
                prior_cost_cols.append(prior_cost_col)
            prior_cost = torch.stack(prior_cost_cols, dim=1)
            C = w_point * cost_pt + w_class * cost_cls + w_prior * prior_cost
        else:
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
        selected_stab = _query_logwh_stability_px(teacher_outputs, base_size=base_size, batch_index=b, query_indices=row_ind)
 
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
        kept_scores = selected_scores[keep]
        kept_dists = selected_dists[keep]
        kept_stab = selected_stab[keep]
        total_pseudo += float(kept_labels.numel())
 
        if snap_center and kept_boxes.numel() > 0:
            kept_boxes = kept_boxes.clone()
            kept_boxes[:, :2] = kept_pts

        density_limit_cfg = point_sup.get("DensityLimit", None) or point_sup.get("density_limit", None) or {}
        density_limit_enabled = not isinstance(density_limit_cfg, dict) or bool(density_limit_cfg.get("enabled", True))
        density_knn_k = int(density_limit_cfg.get("knn_k", 1)) if isinstance(density_limit_cfg, dict) else 1
        density_margin_factor = float(density_limit_cfg.get("density_margin_factor", 1.2)) if isinstance(density_limit_cfg, dict) else 1.2
        density_class_margin_factors = density_limit_cfg.get("class_margin_factors", None) if isinstance(density_limit_cfg, dict) else None
        density_global_max_wh_px = float(density_limit_cfg.get("global_max_wh_px", 256.0)) if isinstance(density_limit_cfg, dict) else 256.0
        density_global_min_wh_px = float(density_limit_cfg.get("global_min_wh_px", 6.0)) if isinstance(density_limit_cfg, dict) else 6.0
        density_use_input_hw = bool(density_limit_cfg.get("use_input_hw", True)) if isinstance(density_limit_cfg, dict) else True

        if density_limit_enabled and kept_boxes.numel() > 0:
            N = kept_pts.shape[0]
            px_w = float(w_img) if (density_use_input_hw and w_img > 0) else float(base_size)
            px_h = float(h_img) if (density_use_input_hw and h_img > 0) else float(base_size)
            if N > 1:
                pts_px = kept_pts.clone()
                pts_px[:, 0] *= px_w
                pts_px[:, 1] *= px_h

                dist_mat = torch.cdist(pts_px, pts_px, p=2)
                dist_mat.fill_diagonal_(float('inf'))
                k = max(1, min(int(density_knn_k), N - 1))
                knn_dists, _ = torch.topk(dist_mat, k=k, dim=1, largest=False)
                knn_dist_px = knn_dists[:, -1]

                margin_factor = torch.full_like(knn_dist_px, float(density_margin_factor))
                if isinstance(density_class_margin_factors, dict) and kept_labels.numel() == margin_factor.numel():
                    for k_cls, v_fac in density_class_margin_factors.items():
                        try:
                            cid = int(k_cls)
                            fac = float(v_fac)
                        except Exception:
                            continue
                        margin_factor = torch.where(kept_labels == cid, margin_factor.new_full((), fac), margin_factor)
                dynamic_max_wh_px = knn_dist_px.unsqueeze(1).repeat(1, 2) * margin_factor.unsqueeze(1)
                global_max_wh = torch.tensor([density_global_max_wh_px, density_global_max_wh_px], device=pts_px.device)
                dynamic_max_wh_px = torch.min(dynamic_max_wh_px, global_max_wh)
            else:
                dynamic_max_wh_px = torch.tensor([[density_global_max_wh_px, density_global_max_wh_px]], device=kept_pts.device)

            teacher_wh_px = kept_boxes[:, 2:].clone()
            teacher_wh_px[:, 0] *= px_w
            teacher_wh_px[:, 1] *= px_h

            pseudo_wh_px = torch.min(teacher_wh_px, dynamic_max_wh_px)
            global_min_wh = torch.tensor([density_global_min_wh_px, density_global_min_wh_px], device=kept_pts.device)
            pseudo_wh_px = torch.max(pseudo_wh_px, global_min_wh)

            pseudo_wh_norm = pseudo_wh_px.clone()
            pseudo_wh_norm[:, 0] /= px_w
            pseudo_wh_norm[:, 1] /= px_h

            kept_boxes[:, 2:] = pseudo_wh_norm

        if dspe_enabled and num_classes > 0 and kept_boxes.numel() > 0:
            dist_px = kept_dists * base_size
            upd_wh_px = torch.clamp(kept_boxes[:, 2:].clamp(min=1e-6) * base_size, min=pseudo_min_wh_px, max=pseudo_max_wh_px)
            upd_area_px = (upd_wh_px[:, 0] * upd_wh_px[:, 1]).clamp(min=1.0)
            if prior_area_px is not None:
                ratio = upd_area_px / prior_area_px[kept_labels.long()]
                ratio_ok = ratio <= update_max_area_ratio
            else:
                ratio_ok = torch.ones_like(upd_area_px, dtype=torch.bool)
            update_mask = (kept_scores >= update_score_thresh) & (dist_px <= update_max_dist_px) & ratio_ok & (kept_stab <= update_stab_thresh)
            if torch.any(update_mask):
                upd_lab = kept_labels[update_mask]
                upd_scores = kept_scores[update_mask]
                upd_wh = kept_boxes[update_mask][:, 2:].clamp(min=1e-6)
                wh_px = torch.clamp(upd_wh * base_size, min=pseudo_min_wh_px, max=pseudo_max_wh_px)
                z = torch.log(torch.clamp(wh_px, min=1.0))
                for c in upd_lab.unique().tolist():
                    m = upd_lab == c
                    zc = z[m]
                    sc = upd_scores[m]
                    if zc.numel() == 0:
                        continue
                    k_keep = max(1, int(round(float(sc.numel()) * update_top_frac)))
                    topk = torch.topk(sc, k=min(k_keep, sc.numel()), largest=True).indices
                    zt = zc[topk]
                    sums[c] += zt.sum(dim=0)
                    sums_sq[c] += (zt**2).sum(dim=0)
                    counts[c] += float(zt.shape[0])
 
        t_new = dict(t)
        t_new["boxes"] = kept_boxes
        t_new["labels"] = kept_labels
        pseudo_targets.append(t_new)
 
    if dspe_enabled and num_classes > 0 and sums is not None and counts is not None:
        if dist_utils.is_dist_available_and_initialized():
            torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(sums_sq, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_gt_points, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_pseudo, op=torch.distributed.ReduceOp.SUM)
        mean = torch.zeros_like(sums)
        var = torch.zeros_like(sums)
        nonzero = counts > 0
        if torch.any(nonzero):
            mean[nonzero] = sums[nonzero] / counts[nonzero, None]
            var[nonzero] = sums_sq[nonzero] / counts[nonzero, None] - mean[nonzero] ** 2
            var = torch.clamp(var, min=1e-6)
        bank.update_from_moments(mean.detach().cpu(), var.detach().cpu(), counts.detach().cpu(), beta=update_beta)
        if writer is not None and dist_utils.is_main_process() and global_step % int(dspe_cfg.get("log_interval", 200)) == 0:
            mu_cpu, var_cpu = bank.get_stats()
            std_cpu = torch.sqrt(torch.clamp(var_cpu, min=1e-6))
            for c in range(min(num_classes, 50)):
                writer.add_scalar(f"DSPE/mu_w_px_cls_{c}", float(mu_cpu[c, 0].exp().item()), global_step)
                writer.add_scalar(f"DSPE/mu_h_px_cls_{c}", float(mu_cpu[c, 1].exp().item()), global_step)
                writer.add_scalar(f"DSPE/std_w_cls_{c}", float(std_cpu[c, 0].item()), global_step)
                writer.add_scalar(f"DSPE/std_h_cls_{c}", float(std_cpu[c, 1].item()), global_step)
            writer.add_scalar("DSPE/total_updates", float(bank.steps), global_step)
            writer.add_scalar("PointSup/gt_points_per_iter", float(total_gt_points.item()), global_step)
            writer.add_scalar("PointSup/pseudo_boxes_per_iter", float(total_pseudo.item()), global_step)
 
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
    point_sup_state = kwargs.get("point_sup_state", None)
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
                    if _is_point_supervision_enabled(point_sup):
                        if ema is None:
                            raise RuntimeError("PointSupervision requires EMA teacher. Set use_ema=True.")
                        with torch.no_grad():
                            teacher_outputs = ema.module(model_inputs, targets=None)
                        targets_for_train = _build_point_pseudo_targets(
                            teacher_outputs,
                            targets,
                            point_sup=point_sup,
                            use_focal_loss=use_focal_loss,
                            model_inputs=model_inputs,
                            point_sup_state=point_sup_state,
                            writer=writer,
                            global_step=global_step,
                            epoch=epoch,
                        )
                    outputs = model(model_inputs, targets=targets_for_train)     
 
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
                if _is_point_supervision_enabled(point_sup):
                    if ema is None:
                        raise RuntimeError("PointSupervision requires EMA teacher. Set use_ema=True.")
                    with torch.no_grad():
                        teacher_outputs = ema.module(model_inputs, targets=None)
                    targets_for_train = _build_point_pseudo_targets(
                        teacher_outputs,
                        targets,
                        point_sup=point_sup,
                        use_focal_loss=use_focal_loss,
                        model_inputs=model_inputs,
                        point_sup_state=point_sup_state,
                        writer=writer,
                        global_step=global_step,
                        epoch=epoch,
                    )
                outputs = model(model_inputs, targets=targets_for_train)  # 前向传播     
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
