import sys
import types
import importlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_featuremap_module(monkeypatch):
    fake_cv2 = types.SimpleNamespace(COLORMAP_AUTUMN=0)
    fake_cv2.applyColorMap = lambda array, _: np.stack([array, array, array], axis=-1)

    fake_transforms = types.ModuleType("torchvision.transforms")
    fake_torchvision = types.ModuleType("torchvision")
    fake_torchvision.transforms = fake_transforms

    fake_engine_core = types.ModuleType("engine.core")
    fake_engine_core.YAMLConfig = object

    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", fake_transforms)
    monkeypatch.setitem(sys.modules, "engine.core", fake_engine_core)
    sys.modules.pop("featuremap", None)

    return importlib.import_module("featuremap")


def test_save_layer_feature_maps_writes_channel_and_sum_images(tmp_path, monkeypatch):
    featuremap = _load_featuremap_module(monkeypatch)
    featmap = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]
        ]
    )

    featuremap.save_layer_feature_maps(featmap, tmp_path)

    channel_zero = tmp_path / "0.png"
    channel_one = tmp_path / "1.png"
    channel_sum = tmp_path / "sum.png"

    assert channel_zero.exists()
    assert channel_one.exists()
    assert channel_sum.exists()

    saved = Image.open(channel_sum)
    assert saved.size == (2, 2)
