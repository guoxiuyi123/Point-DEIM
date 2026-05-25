from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class PseudoBoxConfig:
    prior_wh: float = 0.2
    min_wh: float = 0.02
    max_wh: float = 0.9
    ema: float = 0.9
    score_thresh: float = 0.3
    score_weight_power: float = 1.0
    score_topk: int = 0
    score_topk_begin_epoch: int = 0
    score_topk_end_epoch: int = -1
    update_end_epoch: int = -1
    aggregate_by_target: bool = False
    refine_points: bool = False
    refine_lamda: float = 0.5
    refine_begin_epoch: int = 0
    refine_end_epoch: int = -1
    center_radius: float = 0.08
    class_agnostic_warmup_epochs: int = 3
    require_point_inside: bool = False
    max_scale_up: float = 1000000000.0
    max_scale_down: float = 0.0


class PseudoBoxMemory:
    def __init__(self, cfg: PseudoBoxConfig):
        self.cfg = cfg
        self._mem: Dict[int, torch.Tensor] = {}
        self._pts_mem: Dict[int, torch.Tensor] = {}

    def get(self, sample_idx: int, points: torch.Tensor, device, dtype) -> torch.Tensor:
        n = int(points.shape[0])
        if n == 0:
            return torch.zeros((0, 4), device=device, dtype=dtype)

        pb = self._mem.get(sample_idx)
        if pb is None or int(pb.shape[0]) != n:
            pb = self._init_from_points(points)
            self._mem[sample_idx] = pb
            self._pts_mem[sample_idx] = points.detach().cpu().float()
        elif sample_idx not in self._pts_mem or int(self._pts_mem[sample_idx].shape[0]) != n:
            self._pts_mem[sample_idx] = points.detach().cpu().float()

        return pb.to(device=device, dtype=dtype)

    def get_points(self, sample_idx: int, points: torch.Tensor, device, dtype) -> torch.Tensor:
        n = int(points.shape[0])
        if n == 0:
            return torch.zeros((0, 2), device=device, dtype=dtype)

        p = self._pts_mem.get(sample_idx)
        if p is None or int(p.shape[0]) != n:
            p = points.detach().cpu().float()
            self._pts_mem[sample_idx] = p
        return p.to(device=device, dtype=dtype)

    def update(
        self,
        sample_idx: int,
        tgt_indices: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        points: torch.Tensor,
        epoch: int,
    ) -> Dict[str, float]:
        if tgt_indices.numel() == 0:
            return {
                "total": 0.0,
                "radius_ok": 0.0,
                "score_ok": 0.0,
                "inside_ok": 0.0,
                "wh_ok": 0.0,
                "ok": 0.0,
                "clip_up": 0.0,
                "clip_down": 0.0,
                "mean_max_scale_ok": 0.0,
                "mean_area_ratio_ok": 0.0,
                "score_topk_used": 0.0,
                "score_thresh_eff": float(self.cfg.score_thresh),
                "frozen": 0.0,
                "tgt_total": 0.0,
                "tgt_updated": 0.0,
            }

        pb = self._mem.get(sample_idx)
        if pb is None:
            pb = self._init_from_points(points)
            self._mem[sample_idx] = pb
            self._pts_mem[sample_idx] = points.detach().cpu().float()

        if int(pb.shape[0]) != int(points.shape[0]):
            pb = self._init_from_points(points)
            self._mem[sample_idx] = pb
            self._pts_mem[sample_idx] = points.detach().cpu().float()
        elif sample_idx not in self._pts_mem or int(self._pts_mem[sample_idx].shape[0]) != int(points.shape[0]):
            self._pts_mem[sample_idx] = points.detach().cpu().float()

        cfg = self.cfg

        device = pred_boxes.device
        dtype = pred_boxes.dtype
        tgt_indices = tgt_indices.to(device=device)
        tgt_pts = points.to(device=device, dtype=dtype)[tgt_indices]
        pred_centers = pred_boxes[:, :2]
        d = (pred_centers - tgt_pts).abs().sum(-1)

        radius_ok = d <= cfg.center_radius
        score_thresh_eff = float(cfg.score_thresh)
        score_ok = pred_scores >= score_thresh_eff
        score_topk_used = False
        score_topk = int(getattr(cfg, "score_topk", 0))
        inside_ok = torch.ones_like(score_ok, dtype=torch.bool)
        if bool(cfg.require_point_inside):
            half = pred_boxes[:, 2:] * 0.5
            lt = pred_centers - half
            rb = pred_centers + half
            inside_ok = (tgt_pts >= lt).all(dim=-1) & (tgt_pts <= rb).all(dim=-1)

        wh_ok = torch.ones_like(score_ok, dtype=torch.bool)
        scale_up = float(getattr(cfg, "max_scale_up", 1000000000.0))
        scale_down = float(getattr(cfg, "max_scale_down", 0.0))
        cur_wh = None
        pred_wh = None
        clip_up = torch.zeros_like(score_ok, dtype=torch.bool)
        clip_down = torch.zeros_like(score_ok, dtype=torch.bool)
        if scale_up < 100000000.0 or scale_down > 0.0:
            pb_dev = pb.to(device=device, dtype=dtype)
            cur_wh = pb_dev[tgt_indices][:, 2:].clamp(min=1e-6)
            pred_wh = pred_boxes[:, 2:].clamp(min=1e-6)
            if scale_up < 100000000.0:
                clip_up = (pred_wh > cur_wh * scale_up).any(dim=-1)
            if scale_down > 0.0:
                clip_down = (pred_wh < cur_wh * scale_down).any(dim=-1)
            wh_ok = (pred_wh <= cur_wh * scale_up).all(dim=-1) & (pred_wh >= cur_wh * scale_down).all(dim=-1)

        base_ok = radius_ok & inside_ok & wh_ok
        ok = base_ok & score_ok

        if score_topk > 0:
            b = int(getattr(cfg, "score_topk_begin_epoch", 0))
            e = int(getattr(cfg, "score_topk_end_epoch", -1))
            if int(epoch) >= b and (e < 0 or int(epoch) <= e):
                uniq, inv = torch.unique(tgt_indices, return_inverse=True)
                for gi in range(int(uniq.numel())):
                    m = inv == gi
                    if not bool(m.any()):
                        continue
                    if bool((ok[m]).any()):
                        continue
                    pool = base_ok & m
                    if not bool(pool.any()):
                        continue
                    k = min(int(score_topk), int(pool.sum().item()))
                    cand_scores = pred_scores.clone()
                    cand_scores[~pool] = -1e9
                    topk_idx = cand_scores.topk(k=k, largest=True).indices
                    ok[topk_idx] = True
                    score_topk_used = True

        total = int(tgt_indices.numel())
        n_radius = int(radius_ok.sum().item())
        n_score = int(score_ok.sum().item())
        n_inside = int(inside_ok.sum().item())
        n_wh = int(wh_ok.sum().item())
        n_ok = int(ok.sum().item())
        n_clip_up = int(clip_up.sum().item())
        n_clip_down = int(clip_down.sum().item())
        mean_max_scale_ok = 0.0
        mean_area_ratio_ok = 0.0
        if n_ok > 0 and cur_wh is not None and pred_wh is not None:
            wh_ratio = (pred_wh / cur_wh).clamp(min=1e-6)
            max_scale = wh_ratio.max(dim=-1).values
            area_ratio = (pred_wh[:, 0] * pred_wh[:, 1]) / (cur_wh[:, 0] * cur_wh[:, 1]).clamp(min=1e-6)
            mean_max_scale_ok = float(max_scale[ok].mean().item())
            mean_area_ratio_ok = float(area_ratio[ok].mean().item())

        update_end_epoch = int(getattr(cfg, "update_end_epoch", -1))
        if update_end_epoch >= 0 and int(epoch) > update_end_epoch:
            tgt_total = int(torch.unique(tgt_indices).numel())
            tgt_updated = int(torch.unique(tgt_indices[ok]).numel()) if n_ok > 0 else 0
            return {
                "total": float(total),
                "radius_ok": float(n_radius),
                "score_ok": float(n_score),
                "inside_ok": float(n_inside),
                "wh_ok": float(n_wh),
                "ok": float(n_ok),
                "clip_up": float(n_clip_up),
                "clip_down": float(n_clip_down),
                "mean_max_scale_ok": float(mean_max_scale_ok),
                "mean_area_ratio_ok": float(mean_area_ratio_ok),
                "score_topk_used": float(1.0 if score_topk_used else 0.0),
                "score_thresh_eff": float(score_thresh_eff),
                "frozen": 1.0,
                "tgt_total": float(tgt_total),
                "tgt_updated": float(tgt_updated),
            }

        if ok.sum().item() == 0:
            tgt_total = int(torch.unique(tgt_indices).numel())
            return {
                "total": float(total),
                "radius_ok": float(n_radius),
                "score_ok": float(n_score),
                "inside_ok": float(n_inside),
                "wh_ok": float(n_wh),
                "ok": float(n_ok),
                "clip_up": float(n_clip_up),
                "clip_down": float(n_clip_down),
                "mean_max_scale_ok": float(mean_max_scale_ok),
                "mean_area_ratio_ok": float(mean_area_ratio_ok),
                "score_topk_used": float(1.0 if score_topk_used else 0.0),
                "score_thresh_eff": float(score_thresh_eff),
                "frozen": 0.0,
                "tgt_total": float(tgt_total),
                "tgt_updated": 0.0,
            }

        ema = float(cfg.ema)
        new_boxes = pred_boxes.detach().cpu().float()

        pb = pb.float()
        idx_cpu = tgt_indices.detach().cpu().long()
        ok_cpu = ok.detach().cpu()

        if bool(getattr(cfg, "aggregate_by_target", False)):
            uniq = torch.unique(idx_cpu)
            updated = []
            power = float(getattr(cfg, "score_weight_power", 1.0))
            w_cpu = pred_scores.detach().cpu().float().clamp(min=0.0).pow(power)
            for u in uniq.tolist():
                m = (idx_cpu == int(u)) & ok_cpu
                if not bool(m.any()):
                    continue
                w = w_cpu[m]
                s = float(w.sum().item())
                if s <= 0:
                    continue
                box = (new_boxes[m] * w[:, None]).sum(dim=0) / s
                pb[int(u)] = pb[int(u)] * ema + box * (1.0 - ema)
                updated.append(int(u))

            if bool(getattr(cfg, "refine_points", False)) and len(updated) > 0:
                lamda = float(getattr(cfg, "refine_lamda", 0.5))
                b = int(getattr(cfg, "refine_begin_epoch", 0))
                e = int(getattr(cfg, "refine_end_epoch", -1))
                if int(epoch) >= b and (e < 0 or int(epoch) <= e):
                    p = self._pts_mem.get(sample_idx)
                    if p is None or int(p.shape[0]) != int(points.shape[0]):
                        p = points.detach().cpu().float()
                    for u in updated:
                        pc = pb[u, :2].clone()
                        p[u] = (1.0 - lamda) * pc + lamda * p[u]
                    self._pts_mem[sample_idx] = p
        else:
            pb[idx_cpu[ok_cpu]] = pb[idx_cpu[ok_cpu]] * ema + new_boxes[ok_cpu] * (1.0 - ema)

        pb[:, 2:].clamp_(min=cfg.min_wh, max=cfg.max_wh)
        pb[:, :2].clamp_(min=0.0, max=1.0)

        self._mem[sample_idx] = pb
        tgt_total = int(torch.unique(tgt_indices).numel())
        tgt_updated = int(torch.unique(tgt_indices[ok]).numel()) if n_ok > 0 else 0
        return {
            "total": float(total),
            "radius_ok": float(n_radius),
            "score_ok": float(n_score),
            "inside_ok": float(n_inside),
            "wh_ok": float(n_wh),
            "ok": float(n_ok),
            "clip_up": float(n_clip_up),
            "clip_down": float(n_clip_down),
            "mean_max_scale_ok": float(mean_max_scale_ok),
            "mean_area_ratio_ok": float(mean_area_ratio_ok),
            "score_topk_used": float(1.0 if score_topk_used else 0.0),
            "score_thresh_eff": float(score_thresh_eff),
            "frozen": 0.0,
            "tgt_total": float(tgt_total),
            "tgt_updated": float(tgt_updated),
        }

    def _init_from_points(self, points: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        pts = points.detach().cpu().float()
        n = int(pts.shape[0])
        wh = torch.full((n, 2), float(cfg.prior_wh), dtype=torch.float32)
        boxes = torch.cat([pts, wh], dim=-1)
        boxes[:, 2:].clamp_(min=cfg.min_wh, max=cfg.max_wh)
        boxes[:, :2].clamp_(min=0.0, max=1.0)
        return boxes
