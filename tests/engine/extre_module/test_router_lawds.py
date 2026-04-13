import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_router_lawds_class():
    try:
        module = importlib.import_module("engine.extre_module.custom_nn.downsample.RouterLAWDS")
    except ModuleNotFoundError:
        return None
    return getattr(module, "RouterLAWDS", None)


def test_router_lawds_downsamples_odd_spatial_dims():
    router_lawds_cls = _load_router_lawds_class()
    assert router_lawds_cls is not None, "RouterLAWDS should be implemented under custom_nn/downsample"

    module = router_lawds_cls(16, 24)
    x = torch.randn(2, 16, 23, 17)

    out = module(x)

    assert out.shape == (2, 24, 12, 9)


def test_router_lawds_supports_backward():
    router_lawds_cls = _load_router_lawds_class()
    assert router_lawds_cls is not None, "RouterLAWDS should be implemented under custom_nn/downsample"

    module = router_lawds_cls(16, 24)
    x = torch.randn(1, 16, 20, 18, requires_grad=True)

    module(x).mean().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
