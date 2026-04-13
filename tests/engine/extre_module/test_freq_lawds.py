import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_freq_lawds_class():
    try:
        module = importlib.import_module("engine.extre_module.custom_nn.downsample.FreqLAWDS")
    except ModuleNotFoundError:
        return None
    return getattr(module, "FreqLAWDS", None)


def test_freq_lawds_downsamples_odd_spatial_dims():
    freq_lawds_cls = _load_freq_lawds_class()
    assert freq_lawds_cls is not None, "FreqLAWDS should be implemented under custom_nn/downsample"

    module = freq_lawds_cls(16, 24)
    x = torch.randn(2, 16, 21, 19)

    out = module(x)

    assert out.shape == (2, 24, 11, 10)


def test_freq_lawds_supports_backward():
    freq_lawds_cls = _load_freq_lawds_class()
    assert freq_lawds_cls is not None, "FreqLAWDS should be implemented under custom_nn/downsample"

    module = freq_lawds_cls(16, 24)
    x = torch.randn(1, 16, 20, 18, requires_grad=True)

    module(x).mean().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
