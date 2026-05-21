from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


COCO_BBOX_NAMES = ["ap", "ap50", "ap75", "aps", "apm", "apl", "ar", "ar50", "ar75", "ars", "arm", "arl"]
DIAG_KEYS = ["point_match_ratio", "point_matched", "point_num_points", "pseudo_score_thresh"]


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _extract_coco_stats(record: Dict[str, Any]) -> Optional[List[float]]:
    stats = record.get("test_coco_eval_bbox")
    if isinstance(stats, list) and stats:
        return [_safe_float(v) for v in stats]
    return None


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _best_by_ap(rows: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[float], Optional[int]]:
    best_row = None
    best_ap = None
    best_idx = None
    for i, r in enumerate(rows):
        stats = _extract_coco_stats(r)
        ap = stats[0] if stats and len(stats) > 0 else None
        if ap is None:
            continue
        if best_ap is None or ap > best_ap:
            best_ap = ap
            best_row = r
            best_idx = i

    return best_row, best_ap, best_idx


def _last_scalar_before(rows: List[Dict[str, Any]], last_i: int, key: str) -> Optional[float]:
    last_i = min(last_i, len(rows) - 1)
    for i in range(last_i, -1, -1):
        v = _safe_float(rows[i].get(key))
        if v is not None:
            return v
    return None


def _last_scalar_before_any(rows: List[Dict[str, Any]], last_i: int, keys: List[str]) -> Optional[float]:
    for k in keys:
        v = _last_scalar_before(rows, last_i, k)
        if v is not None:
            return v
    return None


def collect(outputs_dir: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for log_path in outputs_dir.rglob("log.txt"):
        exp_dir = log_path.parent
        rows = _read_jsonl(log_path)
        best_row, best_ap, best_row_i = _best_by_ap(rows)
        last_row = rows[-1] if rows else None

        item: Dict[str, Any] = {
            "exp_dir": str(exp_dir),
            "log_path": str(log_path),
            "best_ap": best_ap,
            "best_epoch": best_row.get("epoch") if best_row else None,
            "last_epoch": last_row.get("epoch") if last_row else None,
        }

        best_stats = _extract_coco_stats(best_row) if best_row else None
        if best_stats:
            item["best_coco_eval_bbox"] = best_stats
            item["best_coco_eval_bbox_named"] = {
                COCO_BBOX_NAMES[i]: best_stats[i] for i in range(min(len(best_stats), len(COCO_BBOX_NAMES)))
            }

        if rows:
            best_i = best_row_i if best_row_i is not None else (len(rows) - 1)
            item["best_diag"] = {
                k: _last_scalar_before_any(rows, best_i, [k, f"train_{k}"]) for k in DIAG_KEYS
            }
            item["last_diag"] = {
                k: _last_scalar_before_any(rows, len(rows) - 1, [k, f"train_{k}"]) for k in DIAG_KEYS
            }

        args_json = exp_dir / "args.json"
        if args_json.exists():
            try:
                item["args_json"] = json.loads(args_json.read_text(encoding="utf-8"))
            except Exception:
                item["args_json"] = None

        results.append(item)

    results.sort(key=lambda x: x["exp_dir"])
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    results = collect(outputs_dir)

    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
