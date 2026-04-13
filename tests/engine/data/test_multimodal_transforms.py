import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.data._misc import convert_to_tv_tensor
from engine.data.transforms.multimodal_container import MultimodalCompose
from engine.data.transforms.multimodal_mosaic import MultimodalMosaic


class _DatasetStub:
    def __init__(self, epoch=0):
        self.epoch = epoch

    def __len__(self):
        return 4

    def load_item(self, idx):
        sample = {
            "rgb": torch.full((3, 8, 8), idx + 1, dtype=torch.float32),
            "npy": torch.full((1, 8, 8), idx + 1, dtype=torch.float32),
        }
        target = {
            "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "area": torch.tensor([200.0], dtype=torch.float32),
            "iscrowd": torch.tensor([0], dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "orig_size": torch.tensor([8, 8], dtype=torch.int64),
        }
        return sample, target


class RandomPhotometricDistort(nn.Module):
    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]

        if isinstance(inputs, tuple) and len(inputs) == 3:
            sample, target, dataset = inputs
            sample["rgb"] = sample["rgb"] + 1.0
            sample["npy"] = sample["npy"] + 10.0
            return sample, target, dataset

        rgb, target = inputs
        return rgb + 1.0, target


class MultimodalMosaicMarker(nn.Module):
    def forward(self, inputs):
        sample, target, dataset = inputs
        target = dict(target)
        target["mosaic"] = torch.tensor([1])
        return sample, target, dataset


class RandomZoomOut(nn.Module):
    def forward(self, inputs):
        sample, target, dataset = inputs
        target = dict(target)
        target["zoom"] = torch.tensor([1])
        return sample, target, dataset


class DummyStop(nn.Module):
    def forward(self, inputs):
        sample, target, dataset = inputs
        target = dict(target)
        target["dummy_stop"] = target.get("dummy_stop", torch.tensor([0])) + 1
        return sample, target, dataset


def test_multimodal_compose_applies_color_to_rgb_only():
    compose = MultimodalCompose(ops=[RandomPhotometricDistort()], policy={"name": "default"})
    dataset = _DatasetStub(epoch=0)
    sample, target = dataset.load_item(0)

    out_sample, _, _ = compose(sample, target, dataset)
    assert torch.equal(out_sample["rgb"], sample["rgb"] + 1.0)
    assert torch.equal(out_sample["npy"], sample["npy"])


def test_multimodal_compose_mosaic_and_zoomout_are_mutually_exclusive(monkeypatch):
    compose = MultimodalCompose(
        ops=[MultimodalMosaicMarker(), RandomZoomOut()],
        policy={"name": "stop_epoch", "ops": [], "epoch": [0, 5, 10]},
        mosaic_prob=1.0,
    )
    dataset = _DatasetStub(epoch=2)
    sample, target = dataset.load_item(0)

    monkeypatch.setattr("engine.data.transforms.multimodal_container.random.random", lambda: 0.0)
    _, out_target, _ = compose(sample, target, dataset)

    assert "mosaic" in out_target
    assert "zoom" not in out_target


def test_multimodal_compose_stop_sample_policy():
    compose = MultimodalCompose(
        ops=[DummyStop()],
        policy={"name": "stop_sample", "ops": ["DummyStop"], "sample": 1},
    )
    dataset = _DatasetStub(epoch=0)

    sample_1, target_1 = dataset.load_item(0)
    sample_2, target_2 = dataset.load_item(1)

    _, out_target_1, _ = compose(sample_1, target_1, dataset)
    _, out_target_2, _ = compose(sample_2, target_2, dataset)

    assert "dummy_stop" in out_target_1
    assert "dummy_stop" not in out_target_2


def test_multimodal_mosaic_probability_skip():
    dataset = _DatasetStub(epoch=0)
    sample, target = dataset.load_item(0)
    transform = MultimodalMosaic(output_size=8, probability=0.0, use_cache=False)

    out_sample, out_target, _ = transform(sample, target, dataset)
    assert torch.equal(out_sample["rgb"], sample["rgb"])
    assert torch.equal(out_sample["npy"], sample["npy"])
    assert torch.equal(out_target["boxes"], target["boxes"])


def test_multimodal_mosaic_use_cache_and_use_dataset_paths():
    dataset = _DatasetStub(epoch=0)
    sample, target = dataset.load_item(0)

    transform_cache = MultimodalMosaic(output_size=8, probability=1.0, use_cache=True, max_cached_images=10)
    cached_sample, cached_target, _ = transform_cache(sample, target, dataset)

    assert cached_sample["rgb"].shape[-2:] == (16, 16)
    assert cached_sample["npy"].shape[-2:] == (16, 16)
    assert cached_target["boxes"].shape[0] == 4

    transform_dataset = MultimodalMosaic(output_size=8, probability=1.0, use_cache=False)
    sampled_sample, sampled_target, _ = transform_dataset(sample, target, dataset)

    assert sampled_sample["rgb"].shape[-2:] == (16, 16)
    assert sampled_sample["npy"].shape[-2:] == (16, 16)
    assert sampled_target["boxes"].shape[0] == 4


def test_multimodal_mosaic_rotation_syncs_rgb_and_npy():
    dataset = _DatasetStub(epoch=0)

    torch.manual_seed(1)
    sample, target = dataset.load_item(0)
    sample["rgb"] = sample["rgb"].clone()
    sample["rgb"][0, :6, 1] = 1.0
    sample["rgb"][0, 5, 1:6] = 1.0
    sample["rgb"][1] = sample["rgb"][0]
    sample["rgb"][2] = sample["rgb"][0]
    sample["npy"] = sample["rgb"][0:1].clone()

    transform_no_rot = MultimodalMosaic(
        output_size=8,
        probability=1.0,
        use_cache=False,
        rotation_range=0,
        translation_range=(0.0, 0.0),
        scaling_range=(1.0, 1.0),
        fill_value=0,
    )
    torch.manual_seed(1)
    out_no_rot, _, _ = transform_no_rot(sample, target, dataset)

    transform_rot = MultimodalMosaic(
        output_size=8,
        probability=1.0,
        use_cache=False,
        rotation_range=45,
        translation_range=(0.0, 0.0),
        scaling_range=(1.0, 1.0),
        fill_value=0,
    )
    torch.manual_seed(1)
    out_rot, _, _ = transform_rot(sample, target, dataset)

    rgb_delta = (out_rot["rgb"] - out_no_rot["rgb"]).abs().sum().item()
    npy_delta = (out_rot["npy"] - out_no_rot["npy"]).abs().sum().item()

    assert rgb_delta > 0
    assert npy_delta > 0
    assert torch.allclose(out_rot["rgb"][0], out_rot["npy"][0])


def test_multimodal_compose_geometry_sync_for_pil_rgb_and_tensor_npy():
    compose = MultimodalCompose(
        ops=[{"type": "RandomHorizontalFlip", "p": 1.0}],
        policy={"name": "default"},
    )
    dataset = _DatasetStub(epoch=0)

    base = np.arange(16, dtype=np.uint8).reshape(4, 4)
    rgb = Image.fromarray(np.stack([base, base, base], axis=-1))
    npy = torch.tensor(base, dtype=torch.float32).unsqueeze(0)

    sample = {"rgb": rgb, "npy": npy}
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "area": torch.tensor([16.0], dtype=torch.float32),
        "iscrowd": torch.tensor([0], dtype=torch.int64),
        "image_id": torch.tensor([1], dtype=torch.int64),
        "orig_size": torch.tensor([4, 4], dtype=torch.int64),
    }

    out_sample, _, _ = compose(sample, target, dataset)
    rgb_out = torch.tensor(np.array(out_sample["rgb"]))[:, :, 0].float()
    npy_out = out_sample["npy"][0].float()
    expected = torch.flip(torch.tensor(base, dtype=torch.float32), dims=[1])

    assert torch.allclose(rgb_out, expected)
    assert torch.allclose(npy_out, expected)


def test_multimodal_compose_random_iou_crop_is_compatible_with_dataset_tuple():
    compose = MultimodalCompose(
        ops=[{"type": "RandomIoUCrop", "p": 1.0}],
        policy={"name": "default"},
    )
    dataset = _DatasetStub(epoch=0)

    base = np.arange(64, dtype=np.uint8).reshape(8, 8)
    rgb = Image.fromarray(np.stack([base, base, base], axis=-1))
    npy = torch.tensor(base, dtype=torch.float32).unsqueeze(0)
    sample = {"rgb": rgb, "npy": npy}
    target = {
        "boxes": convert_to_tv_tensor(
            torch.tensor([[1.0, 1.0, 6.0, 6.0]], dtype=torch.float32),
            key="boxes",
            spatial_size=(8, 8),
        ),
        "labels": torch.tensor([1], dtype=torch.int64),
        "area": torch.tensor([25.0], dtype=torch.float32),
        "iscrowd": torch.tensor([0], dtype=torch.int64),
        "image_id": torch.tensor([1], dtype=torch.int64),
        "orig_size": torch.tensor([8, 8], dtype=torch.int64),
    }

    out_sample, _, _ = compose(sample, target, dataset)
    out_rgb = out_sample["rgb"]
    if isinstance(out_rgb, Image.Image):
        rgb_w, rgb_h = out_rgb.size
    else:
        rgb_h, rgb_w = out_rgb.shape[-2:]

    npy_h, npy_w = out_sample["npy"].shape[-2:]
    assert (rgb_h, rgb_w) == (npy_h, npy_w)


def test_multimodal_compose_normalize_npy_minmax_per_sample():
    compose = MultimodalCompose(
        ops=[{"type": "NormalizeNPYMinMax"}],
        policy={"name": "default"},
    )
    dataset = _DatasetStub(epoch=0)

    sample = {
        "rgb": torch.zeros(3, 2, 2, dtype=torch.float32),
        "npy": torch.tensor([[[2.0, 4.0], [6.0, 10.0]]], dtype=torch.float32),
    }
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }
    out_sample, _, _ = compose(sample, target, dataset)
    out_npy = out_sample["npy"]

    assert torch.isclose(out_npy.min(), torch.tensor(0.0))
    assert torch.isclose(out_npy.max(), torch.tensor(1.0))
    assert torch.isclose(out_npy[0, 0, 0], torch.tensor(0.0))
    assert torch.isclose(out_npy[0, 1, 1], torch.tensor(1.0))


def test_multimodal_compose_normalize_npy_constant_sample_to_zero():
    compose = MultimodalCompose(
        ops=[{"type": "NormalizeNPYMinMax"}],
        policy={"name": "default"},
    )
    dataset = _DatasetStub(epoch=0)

    sample = {
        "rgb": torch.zeros(3, 2, 2, dtype=torch.float32),
        "npy": torch.full((1, 2, 2), 7.0, dtype=torch.float32),
    }
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }
    out_sample, _, _ = compose(sample, target, dataset)
    out_npy = out_sample["npy"]

    assert torch.allclose(out_npy, torch.zeros_like(out_npy))
