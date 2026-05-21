from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    point_root = Path(__file__).resolve().parent
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfigdir")
    deim_root = (point_root / "DEIM").resolve()
    deim_train = (deim_root / "train.py").resolve()

    sys.path.insert(0, str(point_root))
    sys.path.insert(0, str(deim_root))

    import point_ext

    sys.argv[0] = str(deim_train)
    runpy.run_path(str(deim_train), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
