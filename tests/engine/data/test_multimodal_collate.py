import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.data.dataloader import BatchMultimodalCollateFunction


def _make_item(value: float, *, h=8, w=8):
    sample = {
        "rgb": torch.full((3, h, w), value, dtype=torch.float32),
        "npy": torch.full((1, h, w), value, dtype=torch.float32),
    }
    target = {
        "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "area": torch.tensor([200.0], dtype=torch.float32),
        "iscrowd": torch.tensor([0], dtype=torch.int64),
        "image_id": torch.tensor([0], dtype=torch.int64),
        "orig_size": torch.tensor([w, h], dtype=torch.int64),
    }
    return sample, target


def _make_pattern_item(v_shift: float = 0.0, *, h=16, w=16):
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    base = (yy + xx).float() + v_shift
    sample = {
        "rgb": torch.stack([base, base, base], dim=0),
        "npy": base.unsqueeze(0),
    }
    target = {
        "boxes": torch.tensor([[0.5, 0.5, 0.375, 0.375]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "area": torch.tensor([400.0], dtype=torch.float32),
        "iscrowd": torch.tensor([0], dtype=torch.int64),
        "image_id": torch.tensor([0], dtype=torch.int64),
        "orig_size": torch.tensor([w, h], dtype=torch.int64),
    }
    return sample, target


def test_batch_multimodal_collate_stacks_modalities():
    collate = BatchMultimodalCollateFunction()
    collate.set_epoch(0)
    samples, targets = collate([_make_item(0.2), _make_item(0.8)])

    assert samples["rgb"].shape == (2, 3, 8, 8)
    assert samples["npy"].shape == (2, 1, 8, 8)
    assert len(targets) == 2


def test_batch_multimodal_collate_mixup_updates_both_modalities(monkeypatch):
    collate = BatchMultimodalCollateFunction(mixup_prob=1.0, mixup_epochs=[0, 10], copyblend_prob=0.0)
    collate.set_epoch(0)

    monkeypatch.setattr("engine.data.dataloader.random.random", lambda: 0.0)
    monkeypatch.setattr("engine.data.dataloader.random.uniform", lambda a, b: 0.5)

    samples, targets = collate([_make_item(0.0), _make_item(1.0)])

    assert torch.allclose(samples["rgb"][0], torch.full((3, 8, 8), 0.5))
    assert torch.allclose(samples["npy"][0], torch.full((1, 8, 8), 0.5))
    assert "mixup" in targets[0]


def test_batch_multimodal_collate_copyblend_branch_runs(monkeypatch):
    collate = BatchMultimodalCollateFunction(
        mixup_prob=0.0,
        copyblend_prob=1.0,
        copyblend_epochs=[0, 10],
        copyblend_type="copy",
        area_threshold=0,
        num_objects=1,
    )
    collate.set_epoch(0)

    monkeypatch.setattr("engine.data.dataloader.random.random", lambda: 0.0)
    monkeypatch.setattr("engine.data.dataloader.random.randint", lambda a, b: a)
    monkeypatch.setattr("engine.data.dataloader.random.sample", lambda seq, k: [seq[0]])

    samples, targets = collate([_make_item(0.2), _make_item(0.8)])

    assert samples["rgb"].shape == (2, 3, 8, 8)
    assert samples["npy"].shape == (2, 1, 8, 8)
    assert len(targets[0]["boxes"]) >= 1


def test_batch_multimodal_collate_multiscale_resizes_both_modalities(monkeypatch):
    collate = BatchMultimodalCollateFunction(base_size=64, base_size_repeat=1, stop_epoch=5)
    collate.set_epoch(0)

    monkeypatch.setattr("engine.data.dataloader.random.choice", lambda seq: seq[0])
    samples, _ = collate([_make_item(0.2), _make_item(0.8)])
    assert samples["rgb"].shape[-2:] == (32, 32)
    assert samples["npy"].shape[-2:] == (32, 32)

    collate.set_epoch(6)
    samples_after_stop, _ = collate([_make_item(0.2), _make_item(0.8)])
    assert samples_after_stop["rgb"].shape[-2:] == (8, 8)
    assert samples_after_stop["npy"].shape[-2:] == (8, 8)


def test_batch_multimodal_collate_copyblend_with_expand_changes_patch(monkeypatch):
    base_collate = BatchMultimodalCollateFunction(
        mixup_prob=0.0,
        copyblend_prob=1.0,
        copyblend_epochs=[0, 10],
        copyblend_type="copy",
        area_threshold=0,
        num_objects=1,
        with_expand=False,
    )
    expand_collate = BatchMultimodalCollateFunction(
        mixup_prob=0.0,
        copyblend_prob=1.0,
        copyblend_epochs=[0, 10],
        copyblend_type="copy",
        area_threshold=0,
        num_objects=1,
        with_expand=True,
        expand_ratios=[0.25, 0.25],
    )
    base_collate.set_epoch(0)
    expand_collate.set_epoch(0)

    monkeypatch.setattr("engine.data.dataloader.random.random", lambda: 0.0)
    monkeypatch.setattr("engine.data.dataloader.random.randint", lambda a, b: a)
    monkeypatch.setattr("engine.data.dataloader.random.sample", lambda seq, k: [seq[0]])
    monkeypatch.setattr("engine.data.dataloader.random.uniform", lambda a, b: (a + b) / 2.0)

    items = [_make_pattern_item(10.0), _make_pattern_item(0.0)]
    samples_base, _ = base_collate(items)
    samples_expand, _ = expand_collate(items)

    rgb_diff = (samples_expand["rgb"] - samples_base["rgb"]).abs().sum().item()
    npy_diff = (samples_expand["npy"] - samples_base["npy"]).abs().sum().item()
    assert rgb_diff > 0
    assert npy_diff > 0
