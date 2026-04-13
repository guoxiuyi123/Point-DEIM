import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_edge_lawds_class():
    try:
        module = importlib.import_module("engine.extre_module.custom_nn.downsample.EdgeLAWDS")
    except ModuleNotFoundError:
        return None
    return getattr(module, "EdgeLAWDS", None)


def test_edge_lawds_downsamples_odd_spatial_dims():
    edge_lawds_cls = _load_edge_lawds_class()
    assert edge_lawds_cls is not None, "EdgeLAWDS should be implemented under custom_nn/downsample"

    module = edge_lawds_cls(16, 24)
    x = torch.randn(2, 16, 25, 21)

    out = module(x)

    assert out.shape == (2, 24, 13, 11)


def test_edge_lawds_supports_backward():
    edge_lawds_cls = _load_edge_lawds_class()
    assert edge_lawds_cls is not None, "EdgeLAWDS should be implemented under custom_nn/downsample"

    module = edge_lawds_cls(16, 24)
    x = torch.randn(1, 16, 20, 18, requires_grad=True)

    module(x).mean().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_edge_lawds_handles_non_multiple_channel_groupings():
    edge_lawds_cls = _load_edge_lawds_class()
    assert edge_lawds_cls is not None, "EdgeLAWDS should be implemented under custom_nn/downsample"

    module = edge_lawds_cls(50, 24)
    x = torch.randn(1, 50, 19, 15)

    out = module(x)

    assert out.shape == (1, 24, 10, 8)
