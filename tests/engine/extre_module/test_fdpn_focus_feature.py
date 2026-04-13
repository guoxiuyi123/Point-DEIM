import importlib
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_focus_feature_class():
    try:
        module = importlib.import_module("engine.extre_module.custom_nn.neck.FDPN")
    except ModuleNotFoundError:
        return None
    return getattr(module, "FocusFeature", None)


def _load_dynamic_frequency_focus_feature_class():
    try:
        module = importlib.import_module("engine.extre_module.custom_nn.neck.FDPN")
    except ModuleNotFoundError:
        return None
    return getattr(module, "DynamicFrequencyFocusFeature", None)


def _load_alignment_guided_focus_feature_class():
    try:
        module = importlib.import_module("engine.extre_module.custom_nn.neck.FDPN")
    except ModuleNotFoundError:
        return None
    return getattr(module, "AlignmentGuidedFocusFeature", None)


def _build_inputs():
    return [
        torch.randn(2, 32, 10, 10),
        torch.randn(2, 32, 20, 20),
        torch.randn(2, 32, 40, 40),
    ]


def test_focus_feature_preserves_middle_scale_shape():
    focus_feature_cls = _load_focus_feature_class()
    assert focus_feature_cls is not None, "FocusFeature should be implemented under custom_nn/neck/FDPN.py"

    module = focus_feature_cls([32, 32, 32], [3, 5, 7, 9])
    out = module(_build_inputs())

    assert out.shape == (2, 32, 20, 20)


def test_focus_feature_keeps_original_constructor_without_variant_arg():
    focus_feature_cls = _load_focus_feature_class()
    assert focus_feature_cls is not None, "FocusFeature should be implemented under custom_nn/neck/FDPN.py"

    with pytest.raises(TypeError):
        focus_feature_cls([32, 32, 32], [3, 5, 7, 9], variant="dynamic_freq")


def test_dynamic_frequency_focus_feature_preserves_middle_scale_shape():
    dynamic_focus_cls = _load_dynamic_frequency_focus_feature_class()
    assert dynamic_focus_cls is not None, "DynamicFrequencyFocusFeature should be implemented under custom_nn/neck/FDPN.py"

    module = dynamic_focus_cls([32, 32, 32], [3, 5, 7, 9])
    out = module(_build_inputs())

    assert out.shape == (2, 32, 20, 20)


def test_alignment_guided_focus_feature_preserves_middle_scale_shape():
    alignment_focus_cls = _load_alignment_guided_focus_feature_class()
    assert alignment_focus_cls is not None, "AlignmentGuidedFocusFeature should be implemented under custom_nn/neck/FDPN.py"

    module = alignment_focus_cls([32, 32, 32], [3, 5, 7, 9])
    out = module(_build_inputs())

    assert out.shape == (2, 32, 20, 20)
