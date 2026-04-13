import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.data.dataset.multimodal_coco_dataset import MultimodalCocoDetection


def _build_tiny_coco(
    tmp_path: Path,
    *,
    invalid_npy_shape=False,
    mismatch_shape=False,
    modality_format="npy",
    also_write_npy=False,
):
    rgb_dir = tmp_path / "rgb"
    npy_dir = tmp_path / "npy"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    categories = [{"id": 1, "name": "person"}]

    for image_id in (1, 2):
        width, height = 6, 4
        file_name = f"img_{image_id}.jpg"
        Image.fromarray(np.full((height, width, 3), image_id * 30, dtype=np.uint8)).save(rgb_dir / file_name)

        if invalid_npy_shape and image_id == 1:
            modality = np.zeros((1, height, width), dtype=np.float32)
        else:
            nh, nw = (height, width)
            if mismatch_shape and image_id == 2:
                nh, nw = (height + 1, width + 2)
            modality = np.full((nh, nw), float(image_id), dtype=np.float32)

        if modality_format == "npy":
            np.save(npy_dir / f"img_{image_id}.npy", modality)
        elif modality_format == "npz_arr0":
            np.savez(npy_dir / f"img_{image_id}.npz", arr_0=modality)
            if also_write_npy:
                np.save(npy_dir / f"img_{image_id}.npy", modality + 100)
        elif modality_format == "npz_missing_arr0":
            np.savez(npy_dir / f"img_{image_id}.npz", depth=modality)
        else:
            raise ValueError(f"Unsupported modality_format: {modality_format}")

        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [1.0, 1.0, 2.0, 2.0],
                "area": 4.0,
                "iscrowd": 0,
                "segmentation": [[1.0, 1.0, 3.0, 1.0, 3.0, 3.0, 1.0, 3.0]],
            }
        )

    ann_path = tmp_path / "instances.json"
    ann_path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))
    return rgb_dir, npy_dir, ann_path


def test_multimodal_dataset_returns_expected_protocol(tmp_path: Path):
    rgb_dir, npy_dir, ann_path = _build_tiny_coco(tmp_path)
    dataset = MultimodalCocoDetection(
        img_folder=str(rgb_dir),
        npy_folder=str(npy_dir),
        ann_file=str(ann_path),
        transforms=None,
        return_masks=False,
    )

    sample, target = dataset.load_item(0)
    assert set(sample.keys()) == {"rgb", "npy"}
    assert sample["rgb"].size == (6, 4)
    assert isinstance(sample["npy"], torch.Tensor)
    assert sample["npy"].shape == (1, 4, 6)
    assert sample["npy"].dtype == torch.float32

    for key in ("boxes", "labels", "area", "iscrowd", "orig_size", "image_id", "idx"):
        assert key in target


def test_multimodal_dataset_return_masks_and_remap(tmp_path: Path):
    rgb_dir, npy_dir, ann_path = _build_tiny_coco(tmp_path)
    dataset = MultimodalCocoDetection(
        img_folder=str(rgb_dir),
        npy_folder=str(npy_dir),
        ann_file=str(ann_path),
        transforms=None,
        return_masks=True,
        remap_mscoco_category=True,
    )

    sample, target = dataset.load_item(0)
    assert "masks" in target
    assert target["masks"].shape[0] == target["boxes"].shape[0]
    assert target["labels"].tolist() == [0]
    assert sample["npy"].shape[-2:] == (4, 6)


def test_multimodal_dataset_validates_npy_shape(tmp_path: Path):
    rgb_dir, npy_dir, ann_path = _build_tiny_coco(tmp_path, invalid_npy_shape=True)
    dataset = MultimodalCocoDetection(
        img_folder=str(rgb_dir),
        npy_folder=str(npy_dir),
        ann_file=str(ann_path),
        transforms=None,
    )

    with pytest.raises(ValueError, match="2D"):
        dataset.load_item(0)


def test_multimodal_dataset_size_mismatch_error_and_resize(tmp_path: Path):
    rgb_dir, npy_dir, ann_path = _build_tiny_coco(tmp_path, mismatch_shape=True)

    dataset_error = MultimodalCocoDetection(
        img_folder=str(rgb_dir),
        npy_folder=str(npy_dir),
        ann_file=str(ann_path),
        transforms=None,
        npy_size_mismatch="error",
    )
    with pytest.raises(ValueError, match="size mismatch"):
        dataset_error.load_item(1)

    dataset_resize = MultimodalCocoDetection(
        img_folder=str(rgb_dir),
        npy_folder=str(npy_dir),
        ann_file=str(ann_path),
        transforms=None,
        npy_size_mismatch="resize",
    )
    sample, _ = dataset_resize.load_item(1)
    assert sample["npy"].shape == (1, 4, 6)


def test_multimodal_dataset_npy_ram_cache(tmp_path: Path, monkeypatch):
    rgb_dir, npy_dir, ann_path = _build_tiny_coco(tmp_path)
    dataset = MultimodalCocoDetection(
        img_folder=str(rgb_dir),
        npy_folder=str(npy_dir),
        ann_file=str(ann_path),
        transforms=None,
        ram_cache=False,
    )
    dataset.ram_cache = True

    calls = {"count": 0}
    real_load = np.load

    def _counted_load(*args, **kwargs):
        calls["count"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", _counted_load)

    sample1, _ = dataset.load_item(0)
    sample2, _ = dataset.load_item(0)
    assert calls["count"] == 1
    assert torch.equal(sample1["npy"], sample2["npy"])


def test_multimodal_dataset_reads_npz_arr0(tmp_path: Path):
    rgb_dir, npy_dir, ann_path = _build_tiny_coco(tmp_path, modality_format="npz_arr0")
    dataset = MultimodalCocoDetection(
        img_folder=str(rgb_dir),
        npy_folder=str(npy_dir),
        ann_file=str(ann_path),
        transforms=None,
    )

    sample, _ = dataset.load_item(0)
    assert sample["npy"].shape == (1, 4, 6)
    assert torch.all(sample["npy"] == 1.0)


def test_multimodal_dataset_npz_missing_arr0_raises(tmp_path: Path):
    rgb_dir, npy_dir, ann_path = _build_tiny_coco(tmp_path, modality_format="npz_missing_arr0")
    dataset = MultimodalCocoDetection(
        img_folder=str(rgb_dir),
        npy_folder=str(npy_dir),
        ann_file=str(ann_path),
        transforms=None,
    )

    with pytest.raises(ValueError, match="arr_0"):
        dataset.load_item(0)


def test_multimodal_dataset_prefers_npy_over_npz(tmp_path: Path):
    rgb_dir, npy_dir, ann_path = _build_tiny_coco(
        tmp_path,
        modality_format="npz_arr0",
        also_write_npy=True,
    )
    dataset = MultimodalCocoDetection(
        img_folder=str(rgb_dir),
        npy_folder=str(npy_dir),
        ann_file=str(ann_path),
        transforms=None,
    )

    sample, _ = dataset.load_item(0)
    # npy file stores value 101.0 while npz arr_0 stores value 1.0
    assert torch.all(sample["npy"] == 101.0)
