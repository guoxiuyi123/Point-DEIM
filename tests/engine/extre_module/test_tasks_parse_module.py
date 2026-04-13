import sys
from pathlib import Path
import importlib

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import engine.extre_module.tasks as tasks


class _FakeFeatureInfo:
    def channels(self):
        return [48, 96]


class _FakeTimmBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_info = _FakeFeatureInfo()


def test_parse_module_timm_string_defaults_pretrained_false_when_args_empty(monkeypatch):
    called = {}

    def _fake_create_model(name, pretrained, features_only):
        called["name"] = name
        called["pretrained"] = pretrained
        called["features_only"] = features_only
        return _FakeTimmBackbone()

    monkeypatch.setattr(tasks.timm, "create_model", _fake_create_model)

    module, channels, _, _ = tasks.parse_module(
        d={},
        i=0,
        f=-1,
        m="mobilenetv4_conv_small_050",
        args=[],
        ch=[3],
    )

    assert isinstance(module, _FakeTimmBackbone)
    assert channels == [48, 96]
    assert called["name"] == "mobilenetv4_conv_small_050"
    assert called["pretrained"] is False
    assert called["features_only"] is True


def test_parse_module_builds_cidaf_with_featurefusion_channel_rules(monkeypatch):
    try:
        cidaf_module = importlib.import_module("engine.extre_module.custom_nn.featurefusion.CIDAF")
    except ModuleNotFoundError:
        cidaf_module = None

    cidaf_cls = getattr(cidaf_module, "CIDAF", None) if cidaf_module is not None else None
    assert cidaf_cls is not None, "CIDAF should be implemented before parse_module can build it"

    called = {}

    def _fake_create_model(name, pretrained, features_only):
        called["name"] = name
        called["pretrained"] = pretrained
        called["features_only"] = features_only
        return _FakeTimmBackbone()

    monkeypatch.setattr(tasks.timm, "create_model", _fake_create_model)

    module, channels, _, parsed_args = tasks.parse_module(
        d={},
        i=0,
        f=[0, 1],
        m="CIDAF",
        args=[24],
        ch=[32, 16],
    )

    assert isinstance(module, cidaf_cls)
    assert channels == 24
    assert parsed_args[0] == [32, 16]
    assert called == {}


def test_parse_module_builds_wdaf_with_featurefusion_channel_rules(monkeypatch):
    try:
        wdaf_module = importlib.import_module("engine.extre_module.custom_nn.featurefusion.WDAF")
    except ModuleNotFoundError:
        wdaf_module = None

    wdaf_cls = getattr(wdaf_module, "WDAF", None) if wdaf_module is not None else None
    assert wdaf_cls is not None, "WDAF should be implemented before parse_module can build it"

    called = {}

    def _fake_create_model(name, pretrained, features_only):
        called["name"] = name
        called["pretrained"] = pretrained
        called["features_only"] = features_only
        return _FakeTimmBackbone()

    monkeypatch.setattr(tasks.timm, "create_model", _fake_create_model)

    module, channels, _, parsed_args = tasks.parse_module(
        d={},
        i=0,
        f=[0, 1],
        m="WDAF",
        args=[24],
        ch=[32, 16],
    )

    assert isinstance(module, wdaf_cls)
    assert channels == 24
    assert parsed_args[0] == [32, 16]
    assert called == {}


def test_parse_module_builds_freq_lawds_with_single_input_channel_rules():
    try:
        freq_module = importlib.import_module("engine.extre_module.custom_nn.downsample.FreqLAWDS")
    except ModuleNotFoundError:
        freq_module = None

    freq_cls = getattr(freq_module, "FreqLAWDS", None) if freq_module is not None else None
    assert freq_cls is not None, "FreqLAWDS should be implemented before parse_module can build it"

    module, channels, _, parsed_args = tasks.parse_module(
        d={},
        i=0,
        f=0,
        m="FreqLAWDS",
        args=[24],
        ch=[16],
    )

    assert isinstance(module, freq_cls)
    assert channels == 24
    assert parsed_args[:2] == [16, 24]


def test_parse_module_builds_router_lawds_with_single_input_channel_rules():
    try:
        router_module = importlib.import_module("engine.extre_module.custom_nn.downsample.RouterLAWDS")
    except ModuleNotFoundError:
        router_module = None

    router_cls = getattr(router_module, "RouterLAWDS", None) if router_module is not None else None
    assert router_cls is not None, "RouterLAWDS should be implemented before parse_module can build it"

    module, channels, _, parsed_args = tasks.parse_module(
        d={},
        i=0,
        f=0,
        m="RouterLAWDS",
        args=[24],
        ch=[16],
    )

    assert isinstance(module, router_cls)
    assert channels == 24
    assert parsed_args[:2] == [16, 24]


def test_parse_module_builds_edge_lawds_with_single_input_channel_rules():
    try:
        edge_module = importlib.import_module("engine.extre_module.custom_nn.downsample.EdgeLAWDS")
    except ModuleNotFoundError:
        edge_module = None

    edge_cls = getattr(edge_module, "EdgeLAWDS", None) if edge_module is not None else None
    assert edge_cls is not None, "EdgeLAWDS should be implemented before parse_module can build it"

    module, channels, _, parsed_args = tasks.parse_module(
        d={},
        i=0,
        f=0,
        m="EdgeLAWDS",
        args=[24],
        ch=[16],
    )

    assert isinstance(module, edge_cls)
    assert channels == 24
    assert parsed_args[:2] == [16, 24]


def test_parse_module_builds_dynamic_frequency_focus_feature():
    try:
        fdpn_module = importlib.import_module("engine.extre_module.custom_nn.neck.FDPN")
    except ModuleNotFoundError:
        fdpn_module = None

    dynamic_focus_cls = getattr(fdpn_module, "DynamicFrequencyFocusFeature", None) if fdpn_module is not None else None
    assert dynamic_focus_cls is not None, "DynamicFrequencyFocusFeature should be implemented before parse_module can build it"

    module, channels, _, parsed_args = tasks.parse_module(
        d={},
        i=0,
        f=[0, 1, 2],
        m="DynamicFrequencyFocusFeature",
        args=[[3, 5, 7, 9]],
        ch=[32, 32, 32],
    )

    assert isinstance(module, dynamic_focus_cls)
    assert channels == 32
    assert parsed_args[0] == [32, 32, 32]
    assert parsed_args[1] == [3, 5, 7, 9]


def test_parse_module_builds_alignment_guided_focus_feature():
    try:
        fdpn_module = importlib.import_module("engine.extre_module.custom_nn.neck.FDPN")
    except ModuleNotFoundError:
        fdpn_module = None

    alignment_focus_cls = getattr(fdpn_module, "AlignmentGuidedFocusFeature", None) if fdpn_module is not None else None
    assert alignment_focus_cls is not None, "AlignmentGuidedFocusFeature should be implemented before parse_module can build it"

    module, channels, _, parsed_args = tasks.parse_module(
        d={},
        i=0,
        f=[0, 1, 2],
        m="AlignmentGuidedFocusFeature",
        args=[[3, 5, 7, 9]],
        ch=[32, 32, 32],
    )

    assert isinstance(module, alignment_focus_cls)
    assert channels == 32
    assert parsed_args[0] == [32, 32, 32]
    assert parsed_args[1] == [3, 5, 7, 9]
