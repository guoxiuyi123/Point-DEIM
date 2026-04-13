import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.solver.det_solver import DetSolver


class DummyAFSS:
    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"states": {0: {"precision": 1.0, "recall": 1.0, "last_used_epoch": 3}}}

    def load_state_dict(self, state):
        self.loaded = state


def test_det_solver_state_dict_includes_afss_state_when_present(monkeypatch):
    solver = DetSolver.__new__(DetSolver)
    solver.best_stat = {"epoch": 1}
    solver.afss = DummyAFSS()
    solver.train_dataloader = type(
        "Loader",
        (),
        {"dataset": type("Dataset", (), {"category2name": {0: "cls"}, "remap_mscoco_category": False})()},
    )()
    solver.last_epoch = 2

    monkeypatch.setattr("engine.solver.det_solver.BaseSolver.state_dict", lambda self: {"dummy": 1})

    state = DetSolver.state_dict(solver)

    assert "afss_state" in state


def test_det_solver_loads_afss_state_when_present(monkeypatch):
    solver = DetSolver.__new__(DetSolver)
    solver.best_stat = {"epoch": -1}
    solver.afss = DummyAFSS()

    monkeypatch.setattr("engine.solver.det_solver.BaseSolver.load_state_dict", lambda self, state: None)

    DetSolver.load_state_dict(
        solver,
        {"last_epoch": 0, "afss_state": {"states": {1: {"precision": 0.5, "recall": 0.5, "last_used_epoch": 4}}}},
    )

    assert solver.afss.loaded["states"][1]["last_used_epoch"] == 4


def test_build_afss_restores_resume_state(monkeypatch, tmp_path):
    state_path = tmp_path / "resume.pth"
    solver = DetSolver.__new__(DetSolver)
    solver.cfg = SimpleNamespace(
        resume=str(state_path),
        yaml_cfg={"AFSS": {"enabled": True, "random_seed": 0, "dump_refresh_json": False}},
        build_dataloader=lambda name: None,
    )
    solver.train_dataloader = SimpleNamespace(dataset=SimpleNamespace(base_dataset=[0, 1, 2]))
    solver.afss = None
    solver.afss_update_dataloader = None

    monkeypatch.setattr(
        "engine.solver.det_solver.torch.load",
        lambda path, map_location="cpu", weights_only=False: {
            "afss_state": {"states": {1: {"precision": 0.7, "recall": 0.8, "last_used_epoch": 5}}}
        },
    )

    DetSolver._build_afss(solver)

    assert solver.afss.states[1]["precision"] == 0.7
    assert solver.afss.states[1]["last_used_epoch"] == 5
