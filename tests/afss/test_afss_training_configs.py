import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig


def test_dfine_afss_training_config_builds_afss_wrapped_train_dataset():
    cfg = YAMLConfig("configs/afss_training/dfine_hgnetv2_n_custom_afss.yml")
    train_loader = cfg.train_dataloader

    assert train_loader.dataset.__class__.__name__ == "AFSSDataset"
    assert "AFSS" in cfg.yaml_cfg
    assert "afss_update_dataloader" in cfg.yaml_cfg


def test_deim_afss_training_config_builds_afss_wrapped_train_dataset():
    cfg = YAMLConfig("configs/afss_training/deim_hgnetv2_n_custom_afss.yml")
    train_loader = cfg.train_dataloader

    assert train_loader.dataset.__class__.__name__ == "AFSSDataset"
    assert "AFSS" in cfg.yaml_cfg
    assert "afss_update_dataloader" in cfg.yaml_cfg
