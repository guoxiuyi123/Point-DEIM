import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.solver.det_engine import _ScaleMemoryBank, _build_point_fixed_targets, _ensure_point_teacher_bag, _random_block_mask


def test_point_teacher_fixed_targets_shapes_and_ranges():
    targets = [
        {
            "boxes": torch.tensor(
                [
                    [0.30, 0.40, 0.01, 0.01],
                    [0.70, 0.20, 0.01, 0.01],
                ],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([1, 3], dtype=torch.int64),
        }
    ]
    model_inputs = torch.zeros((1, 3, 640, 640), dtype=torch.float32)
    point_teacher = {
        "enabled": True,
        "fixed_box_wh_px": [20, 20],
        "Bag": {
            "enabled": True,
            "bag_size": 8,
            "aspect_ratios": [0.5, 1.0, 2.0],
            "wh_px": [6, 8, 10, 12, 16, 20],
            "jitter": 0.1,
        },
    }

    out = _build_point_fixed_targets(targets, point_teacher, model_inputs=model_inputs)
    assert isinstance(out, list) and len(out) == 1
    t = out[0]
    assert t["boxes"].shape == (2, 4)
    assert torch.isfinite(t["boxes"]).all()
    assert (t["boxes"][:, :2] >= 0).all() and (t["boxes"][:, :2] <= 1).all()
    assert (t["boxes"][:, 2:] > 0).all()
    assert "mil_boxes" in t and "mil_scores" in t
    assert t["mil_boxes"].shape == (2, 8, 4)
    assert t["mil_scores"].shape == (2, 8)
    assert torch.isfinite(t["mil_boxes"]).all()
    assert torch.isfinite(t["mil_scores"]).all()
    assert torch.allclose(t["mil_boxes"][:, :, :2], t["boxes"][:, None, :2], atol=1e-6)


def test_point_teacher_bag_is_added_if_missing():
    targets = [
        {
            "boxes": torch.tensor(
                [
                    [0.30, 0.40, 0.05, 0.03],
                    [0.70, 0.20, 0.04, 0.04],
                ],
                dtype=torch.float32,
            ),
            "labels": torch.tensor([1, 3], dtype=torch.int64),
        }
    ]
    model_inputs = torch.zeros((1, 3, 800, 600), dtype=torch.float32)
    point_teacher = {
        "enabled": True,
        "Bag": {
            "enabled": True,
            "bag_size": 6,
            "aspect_ratios": [1.0],
            "wh_px": [12, 16, 24],
            "jitter": 0.0,
        },
    }
    out = _ensure_point_teacher_bag(targets, point_teacher, model_inputs=model_inputs)
    t = out[0]
    assert "mil_boxes" in t and "mil_scores" in t
    assert t["mil_boxes"].shape == (2, 6, 4)
    assert t["mil_scores"].shape == (2, 6)
    assert torch.allclose(t["mil_boxes"][:, :, :2], t["boxes"][:, None, :2], atol=1e-6)


def test_random_block_mask_modifies_tensor_when_prob_one():
    x = torch.ones((2, 3, 64, 64), dtype=torch.float32)
    cfg = {"enabled": True, "prob": 1.0, "num_blocks": 4, "min_ratio": 0.2, "max_ratio": 0.3, "fill": 0.0}
    y = _random_block_mask(x, cfg)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert (y != x).any()


def test_scale_memory_bank_updates_and_drives_bag():
    bank = _ScaleMemoryBank(num_classes=5, init_mean_wh_px=(20.0, 20.0), init_std_wh_px=(10.0, 10.0), min_std_px=(0.35, 0.35))
    labels = torch.tensor([1, 1, 1], dtype=torch.int64)
    wh_px = torch.tensor([[8.0, 10.0], [9.0, 11.0], [7.0, 9.0]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.95], dtype=torch.float32)
    bank.update(labels, wh_px, scores=scores, beta=0.5, score_thresh=0.0, min_wh_px=2.0, max_wh_px=128.0)
    m = bank.mean_wh_px[1]
    assert float(m[0].item()) < 20.0 and float(m[1].item()) < 20.0

    targets = [
        {
            "boxes": torch.tensor([[0.30, 0.40, 0.05, 0.03]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
    ]
    model_inputs = torch.zeros((1, 3, 640, 640), dtype=torch.float32)
    point_teacher = {
        "enabled": True,
        "Bag": {"enabled": True, "bag_size": 8, "aspect_ratios": [1.0], "wh_px": [6, 8, 10], "jitter": 0.0},
        "DSPE": {"enabled": True, "explore_ratio": 0.0, "explore_wh_px": [6, 8, 10], "pseudo_min_wh_px": 2, "pseudo_max_wh_px": 128},
    }
    out = _ensure_point_teacher_bag(targets, point_teacher, model_inputs=model_inputs, point_teacher_state={"scale_bank": bank, "num_classes": 5})
    t = out[0]
    assert t["mil_boxes"].shape == (1, 8, 4)
    wh_norm = t["mil_boxes"][0, :, 2:]
    wh_px_out = wh_norm * torch.tensor([640.0, 640.0], dtype=torch.float32)
    assert (wh_px_out >= 2.0).all() and (wh_px_out <= 128.0).all()
