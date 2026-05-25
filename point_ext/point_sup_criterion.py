from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from engine.core import register
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from engine.deim.pseudo_box_memory import PseudoBoxConfig, PseudoBoxMemory


@register()
class PointSupDEIMCriterionV2(DEIMCriterion):
    __share__ = ["num_classes"]
    __inject__ = ["matcher"]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses,
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        mal_alpha=None,
        use_uni_set=True,
        no_weight_vfl_epoch=-1,
        ccm_params=None,
        density_recall_penalty=1.1,
        density_precision_penalty=1.3,
        mask_point_sample_ratio=8,
        update_topk: int = 0,
        update_use_aux_outputs: bool = False,
        update_use_enc_aux_outputs: bool = False,
        update_use_pre_outputs: bool = False,
        update_burnin_epochs: int = 0,
        reg_quality_weight: str = "none",
        reg_quality_power: float = 1.0,
        pseudo_box=None,
    ):
        super().__init__(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            alpha=alpha,
            gamma=gamma,
            num_classes=num_classes,
            reg_max=reg_max,
            boxes_weight_format=boxes_weight_format,
            share_matched_indices=share_matched_indices,
            mal_alpha=mal_alpha,
            use_uni_set=use_uni_set,
            no_weight_vfl_epoch=no_weight_vfl_epoch,
            ccm_params=ccm_params,
            density_recall_penalty=density_recall_penalty,
            density_precision_penalty=density_precision_penalty,
            mask_point_sample_ratio=mask_point_sample_ratio,
        )
        cfg = PseudoBoxConfig()
        if isinstance(pseudo_box, dict):
            for k, v in pseudo_box.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        self.pseudo_box_cfg = cfg
        self.pseudo_box_memory = PseudoBoxMemory(cfg)
        self.update_topk = int(update_topk)
        self.update_use_aux_outputs = bool(update_use_aux_outputs)
        self.update_use_enc_aux_outputs = bool(update_use_enc_aux_outputs)
        self.update_use_pre_outputs = bool(update_use_pre_outputs)
        self.update_burnin_epochs = int(update_burnin_epochs)
        self.reg_quality_weight = str(reg_quality_weight)
        self.reg_quality_power = float(reg_quality_power)

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        if self.reg_quality_weight != "cls_score":
            return super().loss_boxes(outputs, targets, indices, num_boxes, boxes_weight=boxes_weight)

        assert "pred_boxes" in outputs and "pred_logits" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        src_logits = outputs["pred_logits"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_labels = torch.cat([t["labels"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        if src_boxes.numel() == 0:
            z = src_boxes.sum()
            return {"loss_bbox": z, "loss_giou": z}

        if self.matcher.use_focal_loss:
            prob = src_logits.sigmoid()
            w = prob.gather(1, target_labels[:, None]).squeeze(1)
        else:
            prob = src_logits.softmax(-1)
            w = prob.gather(1, target_labels[:, None]).squeeze(1)
        w = w.clamp(min=0.0).pow(float(self.reg_quality_power)).detach()
        denom = w.sum().clamp(min=1.0)

        loss_bbox = torch.nn.functional.l1_loss(src_boxes, target_boxes, reduction="none").sum(dim=1)
        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        losses = {
            "loss_bbox": (loss_bbox * w).sum() / denom,
            "loss_giou": (loss_giou * w).sum() / denom,
        }
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        if loss == "point":
            return self.loss_points(outputs, targets, indices, num_boxes)
        return super().get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)

    def loss_points(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_points = torch.cat([t["points"][j] for t, (_, j) in zip(targets, indices)], dim=0)
        loss = F.smooth_l1_loss(src_boxes[:, :2], target_points, reduction="none").sum() / num_boxes
        return {"loss_point": loss}

    def forward(self, outputs, targets, **kwargs):
        device = next(iter(outputs.values())).device
        dtype = outputs["pred_boxes"].dtype if "pred_boxes" in outputs else torch.float32

        pseudo_targets = []
        for t in targets:
            points = t.get("points", None)
            if points is None:
                raise ValueError('PointSupDEIMCriterion requires targets[*]["points"]')

            idx_tensor = t.get("idx", None)
            if idx_tensor is None:
                raise ValueError('PointSupDEIMCriterion requires targets[*]["idx"]')
            sample_idx = int(idx_tensor.item()) if hasattr(idx_tensor, "item") else int(idx_tensor)

            pb = self.pseudo_box_memory.get(sample_idx, points=points, device=device, dtype=dtype)
            refined_points = (
                self.pseudo_box_memory.get_points(sample_idx, points=points, device=device, dtype=dtype)
                if hasattr(self.pseudo_box_memory, "get_points")
                else points
            )
            pt = dict(t)
            pt["boxes"] = pb
            pt["points"] = refined_points
            pseudo_targets.append(pt)

        losses = super().forward(outputs, pseudo_targets, **kwargs)

        epoch = int(kwargs.get("epoch", 0))
        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}
        num_queries_list = outputs.get("num_queries_list", None)
        use_aux = bool(self.update_use_aux_outputs) and "aux_outputs" in outputs and epoch >= int(self.update_burnin_epochs)
        use_enc_aux = bool(self.update_use_enc_aux_outputs) and "enc_aux_outputs" in outputs and epoch >= int(self.update_burnin_epochs)
        use_pre = bool(self.update_use_pre_outputs) and "pre_outputs" in outputs and epoch >= int(self.update_burnin_epochs)

        layers = [outputs_without_aux]
        if use_pre:
            layers.append(outputs["pre_outputs"])
        if use_aux:
            layers.extend(list(outputs.get("aux_outputs", [])))
        if use_enc_aux:
            layers.extend(list(outputs.get("enc_aux_outputs", [])))

        topk = int(self.update_topk)
        if topk > 0:
            all_indices = [
                self.matcher(layer, pseudo_targets, return_topk=topk, epoch=epoch, num_queries_list=num_queries_list)[
                    "indices_o2m"
                ]
                for layer in layers
            ]
        else:
            all_indices = [
                self.matcher(layer, pseudo_targets, epoch=epoch, num_queries_list=num_queries_list)["indices"] for layer in layers
            ]

        matched = 0
        upd_total = 0.0
        upd_radius_ok = 0.0
        upd_score_ok = 0.0
        upd_inside_ok = 0.0
        upd_wh_ok = 0.0
        upd_ok = 0.0
        upd_clip_up = 0.0
        upd_clip_down = 0.0
        upd_sum_mean_max_scale_ok = 0.0
        upd_sum_mean_area_ratio_ok = 0.0
        upd_score_topk_used = 0.0
        upd_sum_score_thresh_eff = 0.0
        upd_images = 0.0
        upd_frozen_images = 0.0
        upd_tgt_total = 0.0
        upd_tgt_updated = 0.0
        n_pts = int(sum(int(t["points"].shape[0]) for t in targets))
        for b in range(len(pseudo_targets)):
            idx_tensor = targets[b]["idx"]
            sample_idx = int(idx_tensor.item()) if hasattr(idx_tensor, "item") else int(idx_tensor)

            src_all = []
            tgt_all = []
            logits_all = []
            boxes_all = []

            for layer, idx_list in zip(layers, all_indices):
                src_idx, tgt_idx = idx_list[b]
                if tgt_idx.numel() == 0:
                    continue
                src_all.append(src_idx)
                tgt_all.append(tgt_idx)
                logits_all.append(layer["pred_logits"][b][src_idx])
                boxes_all.append(layer["pred_boxes"][b][src_idx])

            if len(tgt_all) == 0:
                continue

            src_idx = torch.cat(src_all, dim=0)
            tgt_idx = torch.cat(tgt_all, dim=0)
            logits = torch.cat(logits_all, dim=0)
            pred_boxes = torch.cat(boxes_all, dim=0)

            matched += int(tgt_idx.numel())
            labels = pseudo_targets[b]["labels"][tgt_idx]

            if self.matcher.use_focal_loss:
                prob = logits.sigmoid()
                if epoch < int(self.pseudo_box_cfg.class_agnostic_warmup_epochs):
                    scores = prob.max(dim=-1).values
                else:
                    scores = prob.gather(1, labels[:, None]).squeeze(1)
            else:
                prob = logits.softmax(-1)
                if epoch < int(self.pseudo_box_cfg.class_agnostic_warmup_epochs):
                    scores = prob.max(dim=-1).values
                else:
                    scores = prob.gather(1, labels[:, None]).squeeze(1)

            upd = self.pseudo_box_memory.update(
                sample_idx=sample_idx,
                tgt_indices=tgt_idx,
                pred_boxes=pred_boxes,
                pred_scores=scores,
                points=pseudo_targets[b]["points"],
                epoch=epoch,
            )
            upd_images += 1.0
            upd_total += float(upd["total"])
            upd_radius_ok += float(upd["radius_ok"])
            upd_score_ok += float(upd["score_ok"])
            upd_inside_ok += float(upd["inside_ok"])
            upd_wh_ok += float(upd["wh_ok"])
            upd_ok += float(upd["ok"])
            upd_clip_up += float(upd["clip_up"])
            upd_clip_down += float(upd["clip_down"])
            upd_sum_mean_max_scale_ok += float(upd["mean_max_scale_ok"]) * float(upd["ok"])
            upd_sum_mean_area_ratio_ok += float(upd["mean_area_ratio_ok"]) * float(upd["ok"])
            upd_score_topk_used += float(upd.get("score_topk_used", 0.0))
            upd_sum_score_thresh_eff += float(upd.get("score_thresh_eff", float(self.pseudo_box_memory.cfg.score_thresh))) * float(
                upd["total"]
            )
            upd_frozen_images += float(upd.get("frozen", 0.0))
            upd_tgt_total += float(upd.get("tgt_total", 0.0))
            upd_tgt_updated += float(upd.get("tgt_updated", 0.0))

        step = kwargs.get("step", None)
        if step is not None:
            try:
                s = int(step)
            except Exception:
                s = None
            if s is not None and s % 20 == 0:
                n_pts = int(sum(int(t["points"].shape[0]) for t in targets))
                ratio = float(matched) / float(max(1, n_pts))
                print(
                    f"[PointSup] epoch={epoch} step={s} matched={matched} points={n_pts} ratio={ratio:.3f} "
                    f"score_thresh={float(self.pseudo_box_memory.cfg.score_thresh):.3f}"
                )

        losses.update(
            {
                "point_matched": torch.as_tensor(float(matched), device=device),
                "point_num_points": torch.as_tensor(float(n_pts), device=device),
                "point_match_ratio": torch.as_tensor(float(matched) / float(max(1, n_pts)), device=device),
                "pseudo_score_thresh": torch.as_tensor(float(self.pseudo_box_memory.cfg.score_thresh), device=device),
                "pseudo_update_total": torch.as_tensor(float(upd_total), device=device),
                "pseudo_update_ok_ratio": torch.as_tensor(float(upd_ok) / float(max(1.0, upd_total)), device=device),
                "pseudo_update_radius_ok_ratio": torch.as_tensor(
                    float(upd_radius_ok) / float(max(1.0, upd_total)), device=device
                ),
                "pseudo_update_score_ok_ratio": torch.as_tensor(float(upd_score_ok) / float(max(1.0, upd_total)), device=device),
                "pseudo_update_inside_ok_ratio": torch.as_tensor(
                    float(upd_inside_ok) / float(max(1.0, upd_total)), device=device
                ),
                "pseudo_update_wh_ok_ratio": torch.as_tensor(float(upd_wh_ok) / float(max(1.0, upd_total)), device=device),
                "pseudo_update_clip_up_ratio": torch.as_tensor(float(upd_clip_up) / float(max(1.0, upd_total)), device=device),
                "pseudo_update_clip_down_ratio": torch.as_tensor(
                    float(upd_clip_down) / float(max(1.0, upd_total)), device=device
                ),
                "pseudo_update_mean_max_scale_ok": torch.as_tensor(
                    float(upd_sum_mean_max_scale_ok) / float(max(1.0, upd_ok)), device=device
                ),
                "pseudo_update_mean_area_ratio_ok": torch.as_tensor(
                    float(upd_sum_mean_area_ratio_ok) / float(max(1.0, upd_ok)), device=device
                ),
                "pseudo_update_score_topk_used_ratio": torch.as_tensor(
                    float(upd_score_topk_used) / float(max(1.0, upd_images)), device=device
                ),
                "pseudo_update_score_thresh_eff": torch.as_tensor(
                    float(upd_sum_score_thresh_eff) / float(max(1.0, upd_total)), device=device
                ),
                "pseudo_update_frozen_ratio": torch.as_tensor(float(upd_frozen_images) / float(max(1.0, upd_images)), device=device),
                "pseudo_update_tgt_total": torch.as_tensor(float(upd_tgt_total), device=device),
                "pseudo_update_tgt_updated_ratio": torch.as_tensor(
                    float(upd_tgt_updated) / float(max(1.0, upd_tgt_total)), device=device
                ),
            }
        )
        return losses
