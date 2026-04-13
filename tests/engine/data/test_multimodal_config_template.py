import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core import create
from engine.core.yaml_utils import load_config, merge_config
from engine.data.dataloader import BatchMultimodalCollateFunction
from engine.data.dataset.multimodal_coco_dataset import MultimodalCocoDetection
from engine.data.transforms.multimodal_container import MultimodalCompose


def _build_tiny_coco(tmp_path: Path):
    rgb_dir = tmp_path / "rgb"
    npy_dir = tmp_path / "npy"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    categories = [{"id": 1, "name": "person"}]

    for image_id in (1, 2):
        width, height = 8, 8
        file_name = f"img_{image_id}.jpg"
        Image.fromarray(np.full((height, width, 3), image_id * 20, dtype=np.uint8)).save(rgb_dir / file_name)
        np.save(npy_dir / f"img_{image_id}.npy", np.full((height, width), float(image_id), dtype=np.float32))

        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [1.0, 1.0, 3.0, 3.0],
                "area": 9.0,
                "iscrowd": 0,
                "segmentation": [[1.0, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 4.0]],
            }
        )

    ann_path = tmp_path / "instances.json"
    ann_path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))
    return rgb_dir, npy_dir, ann_path


def test_multimodal_template_config_can_build_components(tmp_path: Path):
    cfg_path = ROOT / "configs" / "dataset" / "multimodal_rgb_npy_detection.yml"
    assert cfg_path.exists(), f"Missing config template: {cfg_path}"

    cfg = load_config(str(cfg_path))
    cfg = merge_config(cfg, inplace=False, overwrite=False)

    rgb_dir, npy_dir, ann_path = _build_tiny_coco(tmp_path)

    # tiny test json only has category id=1, so expected num_classes is 2 (max_id + 1)
    cfg["num_classes"] = 2
    cfg["train_dataloader"]["dataset"]["img_folder"] = str(rgb_dir)
    cfg["train_dataloader"]["dataset"]["npy_folder"] = str(npy_dir)
    cfg["train_dataloader"]["dataset"]["ann_file"] = str(ann_path)

    dataset_cfg = cfg["train_dataloader"]["dataset"]
    collate_cfg = cfg["train_dataloader"]["collate_fn"]
    cfg["_tmp_multimodal_dataset"] = deepcopy(dataset_cfg)
    cfg["_tmp_multimodal_collate"] = deepcopy(collate_cfg)

    dataset = create("_tmp_multimodal_dataset", cfg)
    collate = create("_tmp_multimodal_collate", cfg)

    assert isinstance(dataset, MultimodalCocoDetection)
    assert isinstance(dataset._transforms, MultimodalCompose)
    assert isinstance(collate, BatchMultimodalCollateFunction)

    sample, target = dataset[0]
    assert set(sample.keys()) == {"rgb", "npy"}

    collate.set_epoch(0)
    batched_sample, batched_targets = collate([(sample, target), dataset[1]])
    assert batched_sample["rgb"].shape[0] == 2
    assert batched_sample["npy"].shape[0] == 2
    assert len(batched_targets) == 2
    assert isinstance(batched_sample["npy"], torch.Tensor)
