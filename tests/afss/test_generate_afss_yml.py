import sys
from pathlib import Path
import os

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.utils.generate_afss_yml as generate_afss_yml
from engine.core.yaml_utils import load_config


def test_generate_detection_afss_yml_from_base_config(tmp_path):
    out_path = generate_afss_yml.generate_afss_yaml(
        config_path=ROOT / "configs" / "dfine" / "dfine_hgnetv2_n_custom.yml",
        output_dir=tmp_path,
    )

    assert out_path.name == "dfine_hgnetv2_n_custom_afss.yml"
    raw = yaml.safe_load(out_path.read_text())
    expected_include = os.path.relpath(ROOT / "configs" / "dfine" / "dfine_hgnetv2_n_custom.yml", tmp_path)
    assert raw["__include__"] == [expected_include]
    assert raw["train_dataloader"]["dataset"]["type"] == "AFSSDataset"
    assert raw["afss_update_dataloader"]["dataset"]["return_masks"] is False
    assert raw["AFSS"]["enabled"] is True

    merged = load_config(str(out_path))
    assert merged["train_dataloader"]["dataset"]["type"] == "AFSSDataset"
    assert merged["afss_update_dataloader"]["dataset"]["return_masks"] is False


def test_generate_segmentation_afss_yml_from_base_config(tmp_path):
    out_path = generate_afss_yml.generate_afss_yaml(
        config_path=ROOT / "configs" / "test" / "dfine_hgnetv2_s_seg.yml",
        output_dir=tmp_path,
    )

    raw = yaml.safe_load(out_path.read_text())
    expected_include = os.path.relpath(ROOT / "configs" / "test" / "dfine_hgnetv2_s_seg.yml", tmp_path)
    assert raw["__include__"] == [expected_include]
    assert raw["train_dataloader"]["dataset"]["type"] == "AFSSDataset"
    assert raw["train_dataloader"]["dataset"]["dataset"]["return_masks"] is True
    assert raw["afss_update_dataloader"]["dataset"]["return_masks"] is True

    merged = load_config(str(out_path))
    assert merged["train_dataloader"]["dataset"]["type"] == "AFSSDataset"
    assert merged["afss_update_dataloader"]["dataset"]["return_masks"] is True


def test_generate_multimodal_afss_yml_from_base_config(tmp_path):
    out_path = generate_afss_yml.generate_afss_yaml(
        config_path=ROOT / "configs" / "test" / "dfine_hgnetv2_n_mm.yml",
        output_dir=tmp_path,
    )

    raw = yaml.safe_load(out_path.read_text())
    expected_include = os.path.relpath(ROOT / "configs" / "test" / "dfine_hgnetv2_n_mm.yml", tmp_path)

    assert raw["__include__"] == [expected_include]
    assert raw["train_dataloader"]["dataset"]["type"] == "AFSSDataset"
    assert raw["train_dataloader"]["dataset"]["dataset"]["type"] == "MultimodalCocoDetection"
    assert raw["afss_update_dataloader"]["dataset"]["type"] == "MultimodalCocoDetection"
    assert raw["afss_update_dataloader"]["dataset"]["transforms"]["type"] == "MultimodalCompose"
    assert raw["afss_update_dataloader"]["collate_fn"]["type"] == "BatchMultimodalCollateFunction"

    update_ops = [op["type"] for op in raw["afss_update_dataloader"]["dataset"]["transforms"]["ops"]]
    assert update_ops == ["Resize", "ConvertPILImage", "NormalizeNPYMinMax"]

    merged = load_config(str(out_path))
    assert merged["train_dataloader"]["dataset"]["type"] == "AFSSDataset"
    assert merged["train_dataloader"]["dataset"]["dataset"]["type"] == "MultimodalCocoDetection"
    assert merged["afss_update_dataloader"]["dataset"]["type"] == "MultimodalCocoDetection"
    assert merged["afss_update_dataloader"]["collate_fn"]["type"] == "BatchMultimodalCollateFunction"


def test_generate_afss_yml_refuses_to_overwrite_existing_file(tmp_path):
    out_path = tmp_path / "dfine_hgnetv2_n_custom_afss.yml"
    out_path.write_text("existing")

    with pytest.raises(FileExistsError):
        generate_afss_yml.generate_afss_yaml(
            config_path=ROOT / "configs" / "dfine" / "dfine_hgnetv2_n_custom.yml",
            output_dir=tmp_path,
        )
