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


def _row(metrics: Dict[str, Any], keys: List[str]) -> List[str]:
    named = metrics.get("best_coco_eval_bbox_named") or {}
    return [_basename(metrics["exp_dir"])] + [_fmt(named.get(k)) for k in keys]


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
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir).expanduser().resolve()
    results = collect(outputs_dir)

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    for c in cols:
        if c not in COCO_BBOX_NAMES:
            raise ValueError(f"Unknown col: {c}")

    headers = ["exp"] + cols
    rows = [_row(r, cols) for r in results]
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

