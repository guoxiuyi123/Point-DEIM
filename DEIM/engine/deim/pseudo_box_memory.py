from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class PseudoBoxConfig:
    prior_wh: float = 0.2
    min_wh: float = 0.02
    max_wh: float = 0.9
    ema: float = 0.9
    score_thresh: float = 0.3
    center_radius: float = 0.08
    class_agnostic_warmup_epochs: int = 3
    require_point_inside: bool = False
    max_scale_up: float = 1000000000.0
    max_scale_down: float = 0.0


class PseudoBoxMemory:
    def __init__(self, cfg: PseudoBoxConfig):
        self.cfg = cfg
        self._mem: Dict[int, torch.Tensor] = {}

    def get(self, sample_idx: int, points: torch.Tensor, device, dtype) -> torch.Tensor:
        n = int(points.shape[0])
        if n == 0:
            return torch.zeros((0, 4), device=device, dtype=dtype)

        pb = self._mem.get(sample_idx)
        if pb is None or int(pb.shape[0]) != n:
            pb = self._init_from_points(points)
            self._mem[sample_idx] = pb

        return pb.to(device=device, dtype=dtype)

    def update(
        self,
        sample_idx: int,
        tgt_indices: torch.Tensor,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        points: torch.Tensor,
        epoch: int,
    ) -> None:
        if tgt_indices.numel() == 0:
            return

        pb = self._mem.get(sample_idx)
        if pb is None:
            pb = self._init_from_points(points)
            self._mem[sample_idx] = pb

        if int(pb.shape[0]) != int(points.shape[0]):
            pb = self._init_from_points(points)
            self._mem[sample_idx] = pb

        cfg = self.cfg

        device = pred_boxes.device
        dtype = pred_boxes.dtype
        tgt_indices = tgt_indices.to(device=device)
        tgt_pts = points.to(device=device, dtype=dtype)[tgt_indices]
        pred_centers = pred_boxes[:, :2]
        d = (pred_centers - tgt_pts).abs().sum(-1)

        radius_ok = d <= cfg.center_radius
        score_ok = pred_scores >= cfg.score_thresh
        inside_ok = torch.ones_like(score_ok, dtype=torch.bool)
        if bool(cfg.require_point_inside):
            half = pred_boxes[:, 2:] * 0.5
            lt = pred_centers - half
            rb = pred_centers + half
            inside_ok = (tgt_pts >= lt).all(dim=-1) & (tgt_pts <= rb).all(dim=-1)

        wh_ok = torch.ones_like(score_ok, dtype=torch.bool)
        scale_up = float(getattr(cfg, "max_scale_up", 1000000000.0))
        scale_down = float(getattr(cfg, "max_scale_down", 0.0))
        if scale_up < 100000000.0 or scale_down > 0.0:
            pb_dev = pb.to(device=device, dtype=dtype)
            cur_wh = pb_dev[tgt_indices][:, 2:].clamp(min=1e-6)
            pred_wh = pred_boxes[:, 2:].clamp(min=1e-6)
            wh_ok = (pred_wh <= cur_wh * scale_up).all(dim=-1) & (pred_wh >= cur_wh * scale_down).all(dim=-1)

        ok = radius_ok & score_ok & inside_ok & wh_ok
        if ok.sum().item() == 0:
            return

        ema = float(cfg.ema)
        new_boxes = pred_boxes.detach().cpu().float()

        pb = pb.float()
        idx_cpu = tgt_indices.detach().cpu().long()
        ok_cpu = ok.detach().cpu()

        pb[idx_cpu[ok_cpu]] = pb[idx_cpu[ok_cpu]] * ema + new_boxes[ok_cpu] * (1.0 - ema)

        pb[:, 2:].clamp_(min=cfg.min_wh, max=cfg.max_wh)
        pb[:, :2].clamp_(min=0.0, max=1.0)

        self._mem[sample_idx] = pb

    def _init_from_points(self, points: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        pts = points.detach().cpu().float()
        n = int(pts.shape[0])
        wh = torch.full((n, 2), float(cfg.prior_wh), dtype=torch.float32)
        boxes = torch.cat([pts, wh], dim=-1)
        boxes[:, 2:].clamp_(min=cfg.min_wh, max=cfg.max_wh)
        boxes[:, :2].clamp_(min=0.0, max=1.0)
        return boxes
