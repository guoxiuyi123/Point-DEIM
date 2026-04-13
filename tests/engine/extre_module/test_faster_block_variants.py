import importlib
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_variant(name, cls_name):
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError:
        return None
    return getattr(module, cls_name, None)


def test_faster_msgla_block_preserves_shape_for_odd_spatial_dims():
    cls = _load_variant(
        "engine.extre_module.custom_nn.module.faster_ms_gla_block",
        "Faster_MSGLA_Block",
    )
    assert cls is not None, "Faster_MSGLA_Block should be implemented under custom_nn/module"

    module = cls(16, 32, kernel_sizes=(3, 5, 7), drop_path=0.0)
    x = torch.randn(2, 16, 21, 19)

    out = module(x)

    assert out.shape == (2, 32, 21, 19)


def test_faster_cga_block_preserves_shape_with_layer_scale():
    cls = _load_variant(
        "engine.extre_module.custom_nn.module.faster_cga_block",
        "Faster_CGA_Block",
    )
    assert cls is not None, "Faster_CGA_Block should be implemented under custom_nn/module"

    module = cls(24, 24, mlp_ratio=2.0, layer_scale_init_value=0.1, drop_path=0.0)
    x = torch.randn(2, 24, 17, 15)

    out = module(x)

    assert out.shape == x.shape


@pytest.mark.parametrize(
    ("name", "cls_name"),
    [
        ("engine.extre_module.custom_nn.module.faster_ms_gla_block", "Faster_MSGLA_Block"),
        ("engine.extre_module.custom_nn.module.faster_cga_block", "Faster_CGA_Block"),
    ],
)
def test_faster_block_variants_support_backward(name, cls_name):
    cls = _load_variant(name, cls_name)
    assert cls is not None, f"{cls_name} should be implemented under custom_nn/module"

    module = cls(16, 24, drop_path=0.0)
    x = torch.randn(1, 16, 13, 11, requires_grad=True)

    module(x).mean().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_faster_msgla_block_rejects_even_kernel_sizes():
    cls = _load_variant(
        "engine.extre_module.custom_nn.module.faster_ms_gla_block",
        "Faster_MSGLA_Block",
    )
    assert cls is not None, "Faster_MSGLA_Block should be implemented under custom_nn/module"

    with pytest.raises(ValueError, match="odd"):
        cls(16, 16, kernel_sizes=(3, 4, 7))
