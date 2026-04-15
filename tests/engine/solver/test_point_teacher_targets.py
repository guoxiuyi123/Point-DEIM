import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.solver.det_engine import _build_point_fixed_targets, _ensure_point_teacher_bag, _random_block_mask


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

