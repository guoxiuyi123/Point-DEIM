import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig


def test_segmentation_afss_config_builds_afss_wrapped_train_dataset():
    cfg = YAMLConfig("configs/test/dfine_hgnetv2_s_seg_afss.yml")

    assert cfg.train_dataloader.dataset.__class__.__name__ == "AFSSDataset"
    assert cfg.yaml_cfg["postprocessor"] == "SegPostProcessor"
    assert "AFSS" in cfg.yaml_cfg
    assert "afss_update_dataloader" in cfg.yaml_cfg
