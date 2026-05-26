from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from PIL import Image


_VISDRONE_CATEGORY_NAMES: List[str] = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]
_VISDRONE_CATEGORIES: List[Dict[str, Any]] = [{"id": int(i), "name": n} for i, n in enumerate(_VISDRONE_CATEGORY_NAMES)]
_VISDRONE_RAWID2ID: Dict[int, int] = {int(i + 1): int(i) for i in range(len(_VISDRONE_CATEGORY_NAMES))}


def _sorted_image_files(img_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files


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


@dataclass(frozen=True)
class _ParsedObj:
    category_id: int
    x: float
    y: float
    w: float
    h: float


@dataclass
class _Stats:
    total_images: int = 0
    total_anns: int = 0
    skipped_empty_files: int = 0
    skipped_parse_error_lines: int = 0
    skipped_invalid_cat: int = 0
    skipped_ignored: int = 0
    skipped_nonpositive_wh: int = 0
    clipped_boxes: int = 0


def _parse_visdrone_annotation_file(path: Path, valid_cats: Iterable[int]) -> Tuple[List[_ParsedObj], _Stats]:
    valid = set(int(x) for x in valid_cats)
    stats = _Stats()
    out: List[_ParsedObj] = []
    if not path.is_file():
        raise FileNotFoundError(str(path))
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        stats.skipped_empty_files += 1
        return out, stats
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 6:
            stats.skipped_parse_error_lines += 1
            continue
        try:
            x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            score = float(parts[4])
            cat = int(float(parts[5]))
        except Exception:
            stats.skipped_parse_error_lines += 1
            continue
        if cat == 0 or score <= 0.0:
            stats.skipped_ignored += 1
            continue
        if cat not in valid:
            stats.skipped_invalid_cat += 1
            continue
        if w <= 0.0 or h <= 0.0:
            stats.skipped_nonpositive_wh += 1
            continue
        out.append(_ParsedObj(category_id=cat, x=x, y=y, w=w, h=h))
    return out, stats


def _clip_xywh(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float, bool]:
    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))
    x2 = min(float(img_w), float(x) + float(w))
    y2 = min(float(img_h), float(y) + float(h))
    cw = float(x2 - x1)
    ch = float(y2 - y1)
    clipped = (abs(x1 - x) > 0.0) or (abs(y1 - y) > 0.0) or (abs(cw - w) > 0.0) or (abs(ch - h) > 0.0)
    return x1, y1, cw, ch, clipped


def _build_split(
    split_root: Path,
    split_name: str,
    eps: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], _Stats]:
    images_dir = split_root / "images"
    ann_dir = split_root / "annotations"
    if not images_dir.is_dir():
        raise FileNotFoundError(str(images_dir))
    if not ann_dir.is_dir():
        raise FileNotFoundError(str(ann_dir))

    img_files = _sorted_image_files(images_dir)
    if len(img_files) == 0:
        raise ValueError(f"Empty split: {split_root}")

    stats = _Stats(total_images=len(img_files))
    valid_cats = list(_VISDRONE_RAWID2ID.keys())
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    ann_id = 1
    for image_id, img_path in enumerate(img_files):
        with Image.open(img_path) as im:
            w, h = im.size
        w_i, h_i = int(w), int(h)
        images.append(
            {
                "id": int(image_id),
                "idx": int(image_id),
                "file_name": img_path.name,
                "width": w_i,
                "height": h_i,
                "split": str(split_name),
            }
        )

        txt = ann_dir / f"{img_path.stem}.txt"
        objs, s = _parse_visdrone_annotation_file(txt, valid_cats)
        stats.skipped_empty_files += s.skipped_empty_files
        stats.skipped_parse_error_lines += s.skipped_parse_error_lines
        stats.skipped_invalid_cat += s.skipped_invalid_cat
        stats.skipped_ignored += s.skipped_ignored
        stats.skipped_nonpositive_wh += s.skipped_nonpositive_wh

        for o in objs:
            x, y, bw, bh, clipped = _clip_xywh(o.x, o.y, o.w, o.h, img_w=w_i, img_h=h_i)
            if bw <= 0.0 or bh <= 0.0:
                stats.skipped_nonpositive_wh += 1
                continue
            if clipped:
                stats.clipped_boxes += 1
            cx = x + bw * 0.5
            cy = y + bh * 0.5
            if cx < -eps or cy < -eps or cx > float(w_i) + eps or cy > float(h_i) + eps:
                raise ValueError(f"Point out of range: {img_path.name} ({cx}, {cy}) vs ({w_i}, {h_i})")

            px = float(cx) / float(w_i)
            py = float(cy) / float(h_i)
            if px < -eps or py < -eps or px > 1.0 + eps or py > 1.0 + eps:
                raise ValueError(f"Point(norm) out of range: {img_path.name} ({px}, {py})")
            px = min(max(0.0, px), 1.0)
            py = min(max(0.0, py), 1.0)

            annotations.append(
                {
                    "id": int(ann_id),
                    "idx": int(image_id),
                    "image_id": int(image_id),
                    "category_id": int(_VISDRONE_RAWID2ID[int(o.category_id)]),
                    "bbox": [float(x), float(y), float(bw), float(bh)],
                    "segmentation": [
                        [
                            float(x),
                            float(y),
                            float(x + bw),
                            float(y),
                            float(x + bw),
                            float(y + bh),
                            float(x),
                            float(y + bh),
                        ]
                    ],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                    "points": [float(px), float(py)],
                }
            )
            ann_id += 1
            stats.total_anns += 1
    return images, annotations, stats


def _build_smoke_train_subset(
    train_images: Sequence[Dict[str, Any]],
    train_anns: Sequence[Dict[str, Any]],
    smoke_count: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    n = max(0, int(smoke_count))
    if n <= 0 or len(train_images) == 0:
        return [], []
    n = min(n, len(train_images))
    smoke_images: List[Dict[str, Any]] = []
    old2new: Dict[int, int] = {}
    for new_id, im in enumerate(train_images[:n]):
        old_id = int(im["id"])
        old2new[old_id] = int(new_id)
        smoke_images.append(
            {
                "id": int(new_id),
                "idx": int(new_id),
                "file_name": str(im["file_name"]),
                "width": int(im["width"]),
                "height": int(im["height"]),
                "split": str(im.get("split", "train")),
            }
        )

    smoke_anns: List[Dict[str, Any]] = []
    ann_id = 1
    for ann in train_anns:
        old_img_id = int(ann["image_id"])
        if old_img_id not in old2new:
            continue
        new_img_id = int(old2new[old_img_id])
        smoke_anns.append(
            {
                "id": int(ann_id),
                "idx": int(new_img_id),
                "image_id": int(new_img_id),
                "category_id": int(ann["category_id"]),
                "bbox": list(ann["bbox"]),
                "segmentation": list(ann["segmentation"]),
                "area": float(ann["area"]),
                "iscrowd": int(ann["iscrowd"]),
                "points": list(ann["points"]),
            }
        )
        ann_id += 1
    return smoke_images, smoke_anns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        type=str,
        default="/home/pc/gxy/dataset/VisDrone2019/VisDrone2019",
        help="原始 VisDrone2019 根目录（包含 VisDrone2019-DET-train/ 与 VisDrone2019-DET-val/）",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="dataset/visdrone2019",
        help="输出到 Point-DEIM 目录下的 dataset/visdrone2019",
    )
    parser.add_argument("--eps", type=float, default=1e-6, help="一致性校验容差")
    parser.add_argument("--smoke-count", type=int, default=64, help="生成 smoke 子集的图片数量（从 train 的排序结果中取前 N 张）")
    args = parser.parse_args()

    point_root = Path(__file__).resolve().parents[1]
    src_root = Path(args.src_root).expanduser().resolve()
    out_root = (point_root / args.out_root).resolve()

    train_root = src_root / "VisDrone2019-DET-train"
    val_root = src_root / "VisDrone2019-DET-val"
    if not train_root.is_dir():
        raise FileNotFoundError(str(train_root))
    if not val_root.is_dir():
        raise FileNotFoundError(str(val_root))

    train_images, train_anns, train_stats = _build_split(train_root, split_name="train", eps=float(args.eps))
    val_images, val_anns, val_stats = _build_split(val_root, split_name="val", eps=float(args.eps))

    out_train_img_dir = out_root / "images" / "train"
    out_val_img_dir = out_root / "images" / "val"
    out_smoke_img_dir = out_root / "images_smoke"
    out_ann_dir = out_root / "annotations"

    _link_split_images(train_root / "images", out_train_img_dir, train_images)
    _link_split_images(val_root / "images", out_val_img_dir, val_images)

    train_obj = {"images": train_images, "annotations": train_anns, "categories": list(_VISDRONE_CATEGORIES)}
    val_obj = {"images": val_images, "annotations": val_anns, "categories": list(_VISDRONE_CATEGORIES)}
    smoke_images, smoke_anns = _build_smoke_train_subset(train_images, train_anns, smoke_count=int(args.smoke_count))
    smoke_obj = {"images": smoke_images, "annotations": smoke_anns, "categories": list(_VISDRONE_CATEGORIES)}
    if len(smoke_images) > 0:
        _link_split_images(train_root / "images", out_smoke_img_dir, smoke_images)

    _write_json(out_ann_dir / "instances_train.json", train_obj)
    _write_json(out_ann_dir / "instances_val.json", val_obj)
    if len(smoke_images) > 0:
        _write_json(out_ann_dir / "instances_train_smoke.json", smoke_obj)

    meta = {
        "src_root": str(src_root),
        "out_root": str(out_root),
        "eps": float(args.eps),
        "smoke_count": int(args.smoke_count),
        "train": train_stats.__dict__,
        "val": val_stats.__dict__,
        "categories": [{"id": c.get("id"), "name": c.get("name")} for c in _VISDRONE_CATEGORIES],
    }
    _write_json(out_root / "prep_meta.json", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
