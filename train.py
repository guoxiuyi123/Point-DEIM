from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _default_deim_train(point_root: Path) -> Path:
    return (point_root / "deim_entry.py").resolve()


def _preset_config(point_root: Path, preset: str) -> Path:
    mapping = {
        "nwpu_vhr10_point_only": point_root / "configs" / "nwpu_vhr10" / "deim_hgnetv2_n_point_only.yml",
        "nwpu_vhr10_full_box": point_root / "configs" / "nwpu_vhr10" / "deim_hgnetv2_n_full_box.yml",
        "nwpu_vhr10_point_only_smoke": point_root / "configs" / "nwpu_vhr10" / "deim_hgnetv2_n_point_only_smoke.yml",
        "nwpu_vhr10_full_box_smoke": point_root / "configs" / "nwpu_vhr10" / "deim_hgnetv2_n_full_box_smoke.yml",
        "nwpu_vhr10_point_only_sa_init": point_root
        / "configs"
        / "nwpu_vhr10"
        / "deim_hgnetv2_n_point_only_sa_init.yml",
        "nwpu_vhr10_point_only_score_sched": point_root
        / "configs"
        / "nwpu_vhr10"
        / "deim_hgnetv2_n_point_only_score_sched.yml",
        "nwpu_vhr10_point_only_pg_crop": point_root
        / "configs"
        / "nwpu_vhr10"
        / "deim_hgnetv2_n_point_only_pg_crop.yml",
        "nwpu_vhr10_point_only_smallobj_all": point_root
        / "configs"
        / "nwpu_vhr10"
        / "deim_hgnetv2_n_point_only_smallobj_all.yml",
        "nwpu_vhr10_point_only_smallobj_all_smoke": point_root
        / "configs"
        / "nwpu_vhr10"
        / "deim_hgnetv2_n_point_only_smallobj_all_smoke.yml",
    }
    if preset not in mapping:
        raise ValueError(f"Unknown preset: {preset}")
    return mapping[preset].resolve()


def main() -> int:
    point_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        type=str,
        choices=[
            "nwpu_vhr10_point_only",
            "nwpu_vhr10_full_box",
            "nwpu_vhr10_point_only_smoke",
            "nwpu_vhr10_full_box_smoke",
            "nwpu_vhr10_point_only_sa_init",
            "nwpu_vhr10_point_only_score_sched",
            "nwpu_vhr10_point_only_pg_crop",
            "nwpu_vhr10_point_only_smallobj_all",
            "nwpu_vhr10_point_only_smallobj_all_smoke",
        ],
        help="使用内置实验配置",
    )
    parser.add_argument("--config", type=str, help="直接指定 YAML 配置路径（优先级高于 --preset）")
    parser.add_argument("--deim-train", type=str, default=str(_default_deim_train(point_root)))
    parser.add_argument("--seed", type=int)

    args, passthrough = parser.parse_known_args()
    if passthrough[:1] == ["--"]:
        passthrough = passthrough[1:]

    if args.config:
        config_path = Path(args.config).expanduser().resolve()
    elif args.preset:
        config_path = _preset_config(point_root, args.preset)
    else:
        parser.error("必须指定 --preset 或 --config")

    deim_train = Path(args.deim_train).expanduser().resolve()
    if not deim_train.exists():
        raise FileNotFoundError(str(deim_train))

    cmd = [sys.executable, str(deim_train), "-c", str(config_path)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    cmd += passthrough

    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    proc = subprocess.run(cmd, cwd=str(point_root), env=env)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
