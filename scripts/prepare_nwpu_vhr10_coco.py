from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from PIL import Image


def _sorted_image_files(img_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files


def _guess_image_id_from_name(name: str) -> int | None:
    stem = Path(name).stem
    if stem.isdigit():
        return int(stem) - 1
    return None


def _load_src(src_root: Path) -> Tuple[List[Path], Dict[str, Any]]:
    img_dir = src_root / "images"
    ann_path = src_root / "annotations" / "train.json"
    if not img_dir.is_dir():
        raise FileNotFoundError(str(img_dir))
    if not ann_path.is_file():
        raise FileNotFoundError(str(ann_path))

    img_files = _sorted_image_files(img_dir)
    with ann_path.open("r", encoding="utf-8") as f:
        ann = json.load(f)
    return img_files, ann


def _ensure_coco_images(img_files: Sequence[Path], ann: Dict[str, Any]) -> List[Dict[str, Any]]:
    images = ann.get("images")
    if isinstance(images, list) and len(images) > 0:
        return images

    out: List[Dict[str, Any]] = []
    for p in img_files:
        image_id = _guess_image_id_from_name(p.name)
        if image_id is None:
            raise ValueError(f"Cannot infer image_id from filename: {p.name}")
        with Image.open(p) as im:
            w, h = im.size
        out.append({"id": image_id, "file_name": p.name, "width": int(w), "height": int(h)})

    out.sort(key=lambda x: int(x["id"]))
    return out


def _reindex_annotations(annotations: Sequence[Dict[str, Any]], start_id: int = 1) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cur = int(start_id)
    for a in annotations:
        na = dict(a)
        na["id"] = cur
        cur += 1
        out.append(na)
    return out


def _split_ids(ids: Sequence[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    ids = list(ids)
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    n_val = int(round(len(ids) * float(val_ratio)))
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    return train_ids, val_ids


def _filter_by_image_ids(
    images: Sequence[Dict[str, Any]],
    annotations: Sequence[Dict[str, Any]],
    keep_ids: Iterable[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    keep = set(int(x) for x in keep_ids)
    out_images = [im for im in images if int(im["id"]) in keep]
    out_anns = [a for a in annotations if int(a["image_id"]) in keep]
    out_images.sort(key=lambda x: int(x["id"]))
    return out_images, out_anns


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _safe_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(str(src), str(dst))


def _link_split_images(img_dir: Path, out_dir: Path, images: Sequence[Dict[str, Any]]) -> None:
    for im in images:
        name = str(im["file_name"])
        src = img_dir / name
        if not src.is_file():
            raise FileNotFoundError(str(src))
        _safe_link(src, out_dir / name)


def _limit_images(
    images: Sequence[Dict[str, Any]],
    annotations: Sequence[Dict[str, Any]],
    max_images: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if max_images <= 0 or max_images >= len(images):
        return list(images), list(annotations)
    ids = [int(im["id"]) for im in images]
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    keep = set(ids[: int(max_images)])
    return _filter_by_image_ids(images, annotations, keep)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        type=str,
        default="/home/pc/gxy/dataset/dataset_nwpu/NWPU-VHR-10-DET-train",
        help="原始 NWPU-VHR-10-DET-train 目录（包含 images/ 与 annotations/train.json）",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="dataset/nwpu_vhr10",
        help="输出到 Point 目录下的 dataset/nwpu_vhr10",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-train", type=int, default=50, help="额外输出 train 的 smoke 子集 json（<=0 表示不输出）")
    parser.add_argument("--smoke-val", type=int, default=50, help="额外输出 val 的 smoke 子集 json（<=0 表示不输出）")
    args = parser.parse_args()

    point_root = Path(__file__).resolve().parents[1]
    src_root = Path(args.src_root).expanduser().resolve()
    out_root = (point_root / args.out_root).resolve()

    img_files, ann = _load_src(src_root)
    categories = ann.get("categories")
    if not isinstance(categories, list) or len(categories) == 0:
        raise ValueError("annotations/train.json missing categories")

    annotations = ann.get("annotations")
    if not isinstance(annotations, list) or len(annotations) == 0:
        raise ValueError("annotations/train.json missing annotations")

    images = _ensure_coco_images(img_files, ann)
    all_ids = [int(im["id"]) for im in images]
    train_ids, val_ids = _split_ids(all_ids, val_ratio=float(args.val_ratio), seed=int(args.seed))

    src_img_dir = src_root / "images"
    train_img_dir = out_root / "images" / "train"
    val_img_dir = out_root / "images" / "val"
    ann_dir = out_root / "annotations"

    train_images, train_anns = _filter_by_image_ids(images, annotations, train_ids)
    val_images, val_anns = _filter_by_image_ids(images, annotations, val_ids)

    _link_split_images(src_img_dir, train_img_dir, train_images)
    _link_split_images(src_img_dir, val_img_dir, val_images)

    train_obj = {
        "images": train_images,
        "annotations": _reindex_annotations(train_anns, start_id=1),
        "categories": categories,
    }
    val_obj = {
        "images": val_images,
        "annotations": _reindex_annotations(val_anns, start_id=1),
        "categories": categories,
    }

    _write_json(ann_dir / "instances_train.json", train_obj)
    _write_json(ann_dir / "instances_val.json", val_obj)

    if int(args.smoke_train) > 0:
        s_images, s_anns = _limit_images(train_images, train_anns, int(args.smoke_train), seed=int(args.seed))
        _write_json(
            ann_dir / "instances_train_smoke.json",
            {"images": s_images, "annotations": _reindex_annotations(s_anns, start_id=1), "categories": categories},
        )
    if int(args.smoke_val) > 0:
        s_images, s_anns = _limit_images(val_images, val_anns, int(args.smoke_val), seed=int(args.seed) + 1)
        _write_json(
            ann_dir / "instances_val_smoke.json",
            {"images": s_images, "annotations": _reindex_annotations(s_anns, start_id=1), "categories": categories},
        )

    meta = {
        "src_root": str(src_root),
        "out_root": str(out_root),
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
        "num_images": len(images),
        "num_train_images": len(train_images),
        "num_val_images": len(val_images),
        "num_annotations": len(annotations),
        "num_train_annotations": len(train_anns),
        "num_val_annotations": len(val_anns),
        "categories": [{"id": c.get("id"), "name": c.get("name")} for c in categories],
    }
    _write_json(out_root / "prep_meta.json", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

