import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_wdaf_class():
    try:
        module = importlib.import_module("engine.extre_module.custom_nn.featurefusion.WDAF")
    except ModuleNotFoundError:
        return None
    return getattr(module, "WDAF", None)


def test_wdaf_preserves_output_shape_for_odd_spatial_dims():
    wdaf_cls = _load_wdaf_class()
    assert wdaf_cls is not None, "WDAF should be implemented under custom_nn/featurefusion"

    module = wdaf_cls([32, 16], 24)
    x1 = torch.randn(2, 32, 21, 19)
    x2 = torch.randn(2, 16, 21, 19)

    out = module([x1, x2])

    assert out.shape == (2, 24, 21, 19)


def test_wdaf_rejects_mismatched_spatial_shapes():
    wdaf_cls = _load_wdaf_class()
    assert wdaf_cls is not None, "WDAF should be implemented under custom_nn/featurefusion"

    module = wdaf_cls([32, 16], 24)
    x1 = torch.randn(1, 32, 20, 20)
    x2 = torch.randn(1, 16, 10, 10)

    try:
        module([x1, x2])
    except ValueError as exc:
        assert "same spatial shape" in str(exc)
    else:
        raise AssertionError("WDAF should reject mismatched same-scale inputs")
