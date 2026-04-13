import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_cidaf_class():
    try:
        module = importlib.import_module("engine.extre_module.custom_nn.featurefusion.CIDAF")
    except ModuleNotFoundError:
        return None
    return getattr(module, "CIDAF", None)


def test_cidaf_preserves_same_scale_output_shape():
    cidaf_cls = _load_cidaf_class()
    assert cidaf_cls is not None, "CIDAF should be implemented under custom_nn/featurefusion"

    module = cidaf_cls([32, 16], 24)
    x_top = torch.randn(2, 32, 20, 20)
    x_bottom = torch.randn(2, 16, 20, 20)

    out = module([x_top, x_bottom])

    assert out.shape == (2, 24, 20, 20)


def test_cidaf_rejects_mismatched_spatial_shapes():
    cidaf_cls = _load_cidaf_class()
    assert cidaf_cls is not None, "CIDAF should be implemented under custom_nn/featurefusion"

    module = cidaf_cls([32, 16], 24)
    x_top = torch.randn(1, 32, 20, 20)
    x_bottom = torch.randn(1, 16, 10, 10)

    try:
        module([x_top, x_bottom])
    except ValueError as exc:
        assert "same spatial shape" in str(exc)
    else:
        raise AssertionError("CIDAF should reject mismatched same-scale inputs")
