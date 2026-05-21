from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from collect_results import COCO_BBOX_NAMES, collect


def _fmt(x: Any, ndigits: int = 4) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return ""


def _basename(exp_dir: str) -> str:
    return Path(exp_dir).name


def _row(metrics: Dict[str, Any], keys: List[str], diag_keys: List[str], diag_from: str) -> List[str]:
    named = metrics.get("best_coco_eval_bbox_named") or {}
    diag = metrics.get(f"{diag_from}_diag") or {}
    return [_basename(metrics["exp_dir"])] + [_fmt(named.get(k)) for k in keys] + [_fmt(diag.get(k)) for k in diag_keys]


def to_markdown(rows: List[List[str]], headers: List[str]) -> str:
    out: List[str] = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--out-md", type=str, default="")
    parser.add_argument(
        "--cols",
        type=str,
        default="ap,ap50,ap75,aps,apm,apl",
        help="逗号分隔，来自 COCOeval.stats",
    )
    parser.add_argument(
        "--diag-cols",
        type=str,
        default="",
        help="逗号分隔，可选输出的诊断字段（例如 point_match_ratio,pseudo_score_thresh）",
    )
    parser.add_argument(
        "--diag-from",
        type=str,
        default="best",
        choices=["best", "last"],
        help="诊断字段取值来源：best（以 best AP 对应的时间点）或 last（日志最后一次出现）",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    results = collect(outputs_dir)

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    for c in cols:
        if c not in COCO_BBOX_NAMES:
            raise ValueError(f"Unknown col: {c}")

    diag_cols = [c.strip() for c in args.diag_cols.split(",") if c.strip()]
    headers = ["exp"] + cols + diag_cols
    rows = [_row(r, cols, diag_cols, args.diag_from) for r in results]
    md = to_markdown(rows, headers)

    if args.out_md:
        out_path = Path(args.out_md).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
    else:
        print(md, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
