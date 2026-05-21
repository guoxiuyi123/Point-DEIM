from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _default_deim_entry(point_root: Path) -> Path:
    return (point_root / "deim_entry.py").resolve()


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str)
    parser.add_argument("-t", "--tuning", type=str)
    parser.add_argument("-d", "--device", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--summary-dir", type=str)
    parser.add_argument("--test-only", action="store_true", default=False)

    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("-m", "--mode", type=str, default="det", choices=["det", "mask"])

    parser.add_argument("-u", "--update", nargs="+")

    parser.add_argument("--print-method", type=str, default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int)

    parser.add_argument("--deim-entry", type=str)
    return parser


def main() -> int:
    point_root = Path(__file__).resolve().parent
    args = _build_argparser().parse_args()

    deim_entry = Path(args.deim_entry).expanduser().resolve() if args.deim_entry else _default_deim_entry(point_root)
    if not deim_entry.exists():
        raise FileNotFoundError(str(deim_entry))

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (point_root / config_path).resolve()
    else:
        config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(str(config_path))

    cmd = [sys.executable, str(deim_entry)]
    forwarded = list(sys.argv[1:])
    for i in range(len(forwarded)):
        if forwarded[i] in ("-c", "--config") and i + 1 < len(forwarded):
            forwarded[i + 1] = str(config_path)
            break
    cmd += forwarded

    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    proc = subprocess.run(cmd, cwd=str(point_root), env=env)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
