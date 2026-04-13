import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.solver.det_solver import DetSolver


def _build_solver(tmp_path):
    cfg = SimpleNamespace()
    solver = DetSolver(cfg)
    solver.output_dir = tmp_path
    solver.train_dataloader = SimpleNamespace(
        collate_fn=SimpleNamespace(stop_epoch=5, ema_restart_decay=0.9999),
        dataset=SimpleNamespace(
            category2name={0: "obj"},
            remap_mscoco_category=False,
        ),
    )
    return solver


def test_checkpoint_paths_include_last_after_stop_epoch(tmp_path):
    solver = _build_solver(tmp_path)

    checkpoint_paths = solver._build_checkpoint_paths(epoch=6, checkpoint_freq=4)

    assert tmp_path / "last.pth" in checkpoint_paths


def test_load_best_stg1_missing_is_safe(tmp_path):
    solver = _build_solver(tmp_path)

    loaded = solver._load_best_stg1_if_available()

    assert loaded is False


def test_best_stat_roundtrip_in_state_dict(monkeypatch, tmp_path):
    solver = _build_solver(tmp_path)
    solver.best_stat = {
        "epoch": 8,
        "avg_metric": 0.66,
        "coco_eval_bbox": 0.66,
    }

    monkeypatch.setattr("engine.solver.det_solver.BaseSolver.state_dict", lambda self: {"dummy": 1})
    monkeypatch.setattr("engine.solver.det_solver.BaseSolver.load_state_dict", lambda self, state: None)

    state = solver.state_dict()

    assert "best_stat" in state
    assert state["best_stat"]["avg_metric"] == 0.66

    solver2 = _build_solver(tmp_path)
    solver2.load_state_dict(state)

    assert solver2.best_stat["epoch"] == 8
    assert solver2.best_stat["avg_metric"] == 0.66


def test_bootstrap_best_stat_from_eval_only_when_missing(tmp_path):
    solver = _build_solver(tmp_path)
    solver.best_stat = {"epoch": 3, "avg_metric": 0.9, "coco_eval_bbox": 0.9}

    solver._bootstrap_best_stat_from_eval(
        test_stats={"coco_eval_bbox": [0.1]},
        epoch=10,
    )

    assert solver.best_stat["epoch"] == 3
    assert solver.best_stat["avg_metric"] == 0.9
