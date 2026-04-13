import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core.yaml_utils import load_config


def test_dfine_mm_train_collate_uses_multimodal_collate():
    cfg_path = ROOT / "configs" / "test" / "dfine_hgnetv2_n_mm.yml"
    cfg = load_config(str(cfg_path))

    assert cfg["train_dataloader"]["dataset"]["type"] == "MultimodalCocoDetection"
    assert cfg["train_dataloader"]["collate_fn"]["type"] == "BatchMultimodalCollateFunction"


def test_multimodal_dataset_config_includes_npy_minmax_normalization():
    cfg_path = ROOT / "configs" / "test" / "dfine_hgnetv2_n_mm.yml"
    cfg = load_config(str(cfg_path))

    train_ops = [op["type"] for op in cfg["train_dataloader"]["dataset"]["transforms"]["ops"]]
    val_ops = [op["type"] for op in cfg["val_dataloader"]["dataset"]["transforms"]["ops"]]

    assert "NormalizeNPYMinMax" in train_ops
    assert "NormalizeNPYMinMax" in val_ops
