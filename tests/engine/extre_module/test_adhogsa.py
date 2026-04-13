import importlib
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_adhogsa_class():
    try:
        module = importlib.import_module("engine.extre_module.custom_nn.transformer.ADHOGSA")
    except ModuleNotFoundError:
        return None
    return getattr(module, "ADHOGSA", None)


def test_adhogsa_preserves_shape_for_odd_spatial_dims():
    adhogsa_cls = _load_adhogsa_class()
    assert adhogsa_cls is not None, "ADHOGSA should be implemented under custom_nn/transformer"

    module = adhogsa_cls(dim=32, num_heads=8, patch_sizes=(3, 7))
    x = torch.randn(2, 32, 21, 19)

    out = module(x)

    assert out.shape == x.shape


def test_adhogsa_rejects_invalid_head_configuration():
    adhogsa_cls = _load_adhogsa_class()
    assert adhogsa_cls is not None, "ADHOGSA should be implemented under custom_nn/transformer"

    with pytest.raises(ValueError, match="divisible by num_heads"):
        adhogsa_cls(dim=30, num_heads=8)


def test_adhogsa_supports_backward():
    adhogsa_cls = _load_adhogsa_class()
    assert adhogsa_cls is not None, "ADHOGSA should be implemented under custom_nn/transformer"

    module = adhogsa_cls(dim=16, num_heads=4, patch_sizes=(2, 4))
    x = torch.randn(1, 16, 12, 10, requires_grad=True)

    module(x).mean().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
