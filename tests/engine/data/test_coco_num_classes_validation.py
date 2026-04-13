import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.data.dataset.coco_dataset import CocoDetection
from engine.data.dataset.multimodal_coco_dataset import MultimodalCocoDetection


def _build_tiny_coco(tmp_path: Path, category_ids):
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    width, height = 8, 6
    file_name = "img_1.jpg"
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(img_dir / file_name)

    categories = [{"id": cid, "name": f"cls_{cid}"} for cid in category_ids]
    annotations = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": category_ids[0],
            "bbox": [1.0, 1.0, 2.0, 2.0],
            "area": 4.0,
            "iscrowd": 0,
        }
    ]
    images = [{"id": 1, "file_name": file_name, "width": width, "height": height}]

    ann_path = tmp_path / "instances.json"
    ann_path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))
    return img_dir, ann_path


def _write_dummy_npy_dir(tmp_path: Path):
    npy_dir = tmp_path / "npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_dir / "img_1.npy", np.zeros((6, 8), dtype=np.float32))
    return npy_dir


def test_coco_num_classes_validation_accepts_zero_based_ids(tmp_path: Path):
    img_dir, ann_path = _build_tiny_coco(tmp_path, category_ids=[0, 1, 2])

    dataset = CocoDetection(
        img_folder=str(img_dir),
        ann_file=str(ann_path),
        transforms=None,
        num_classes=3,
    )

    assert len(dataset.ids) == 1


def test_coco_num_classes_validation_raises_when_num_classes_is_too_small(tmp_path: Path):
    img_dir, ann_path = _build_tiny_coco(tmp_path, category_ids=[1, 2, 3])

    with pytest.raises(ValueError) as exc_info:
        _ = CocoDetection(
            img_folder=str(img_dir),
            ann_file=str(ann_path),
            transforms=None,
            num_classes=3,
        )

    msg = str(exc_info.value)
    assert "num_classes mismatch" in msg
    assert "configured num_classes=3" in msg
    assert "expected_num_classes=4" in msg
    assert "category_ids=[1, 2, 3]" in msg


def test_multimodal_coco_num_classes_validation_raises_too(tmp_path: Path):
    img_dir, ann_path = _build_tiny_coco(tmp_path, category_ids=[1, 2])
    npy_dir = _write_dummy_npy_dir(tmp_path)

    with pytest.raises(ValueError, match="expected_num_classes=3"):
        _ = MultimodalCocoDetection(
            img_folder=str(img_dir),
            npy_folder=str(npy_dir),
            ann_file=str(ann_path),
            transforms=None,
            num_classes=2,
        )


def test_coco_num_classes_validation_is_skipped_when_remap_enabled(tmp_path: Path):
    img_dir, ann_path = _build_tiny_coco(tmp_path, category_ids=[1, 2, 3])

    dataset = CocoDetection(
        img_folder=str(img_dir),
        ann_file=str(ann_path),
        transforms=None,
        remap_mscoco_category=True,
        num_classes=3,
    )

    assert len(dataset.ids) == 1
