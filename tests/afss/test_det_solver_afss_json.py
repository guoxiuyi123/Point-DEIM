import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.solver.afss import AFSSController
from engine.solver.det_solver import DetSolver


class _DummySampler:
    def set_epoch(self, epoch):
        self.epoch = epoch


class _DummyLoader:
    def __init__(self, batch, dataset):
        self._batch = batch
        self.dataset = dataset
        self.sampler = _DummySampler()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        yield self._batch


class _DummySegModel(torch.nn.Module):
    def forward(self, model_inputs):
        return {"unused": model_inputs}


class _DummySegPostprocessor:
    def __call__(self, outputs, current_sizes, for_eval=True):
        return [
            {
                "labels": torch.tensor([0], dtype=torch.int64),
                "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float32),
                "scores": torch.tensor([0.9], dtype=torch.float32),
                "masks": torch.tensor(
                    [[[1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1],
                      [1, 1, 1, 1]]],
                    dtype=torch.float32,
                ),
            }
        ]


class _DummyCoco:
    def loadImgs(self, image_id):
        return [{"file_name": f"image_{image_id}.jpg"}]


class _DummyBaseDataset:
    def __init__(self):
        self.ids = [101]
        self.coco = _DummyCoco()
        self.img_folder = "/tmp/images"

    def __len__(self):
        return len(self.ids)


class _DummyTrainDataset:
    def __init__(self):
        self.active_indices = [0]
        self.base_dataset = _DummyBaseDataset()

    def __len__(self):
        return 1


def _build_solver(tmp_path, dump_refresh_json):
    solver = DetSolver.__new__(DetSolver)
    solver.afss = AFSSController(random_seed=0)
    solver.afss.initialize_states(total_samples=1)
    solver.ema = None
    solver.model = _DummySegModel()
    solver.model.train()
    solver.device = torch.device("cpu")
    solver.postprocessor = _DummySegPostprocessor()
    solver.output_dir = tmp_path
    solver.cfg = type("Cfg", (), {"yaml_cfg": {"AFSS": {"dump_refresh_json": dump_refresh_json}}})()
    train_dataset = _DummyTrainDataset()
    solver.train_dataloader = type("TrainLoader", (), {"dataset": train_dataset})()
    solver.afss_update_dataloader = _DummyLoader(
        (
            torch.zeros((1, 3, 4, 4), dtype=torch.float32),
            [
                {
                    "idx": torch.tensor([0], dtype=torch.int64),
                    "labels": torch.tensor([0], dtype=torch.int64),
                    "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float32),
                    "masks": torch.tensor(
                        [[[1, 1, 1, 1],
                          [1, 1, 1, 1],
                          [1, 1, 1, 1],
                          [1, 1, 1, 1]]],
                        dtype=torch.float32,
                    ),
                }
            ],
        ),
        dataset=train_dataset,
    )
    return solver


def test_refresh_afss_scores_writes_reference_style_json_when_enabled(tmp_path):
    solver = _build_solver(tmp_path, dump_refresh_json=True)

    solver._refresh_afss_scores(epoch=5)

    json_path = tmp_path / "afss" / "refresh_epoch0005.json"
    assert json_path.exists()

    payload = json.loads(json_path.read_text())
    assert sorted(payload.keys()) == ["active_count", "active_images", "counts", "epoch", "images", "thresholds"]
    assert payload["epoch"] == 5
    assert payload["active_count"] == 1
    assert payload["active_images"] == ["/tmp/images/image_101.jpg"]
    assert payload["counts"]["easy"] == 1
    assert payload["thresholds"] == {"easy": 0.85, "moderate": 0.55}

    image_entry = payload["images"][0]
    assert image_entry["im_file"] == "/tmp/images/image_101.jpg"
    assert image_entry["last_eval_epoch"] == 5
    assert image_entry["last_used_epoch"] == -1
    assert image_entry["level"] == "easy"
    assert image_entry["task_score"] == 1.0
    assert image_entry["metrics"]["box"] == {"precision": 1.0, "recall": 1.0}
    assert image_entry["metrics"]["mask"] == {"precision": 1.0, "recall": 1.0}


def test_refresh_afss_scores_skips_json_when_disabled(tmp_path):
    solver = _build_solver(tmp_path, dump_refresh_json=False)

    solver._refresh_afss_scores(epoch=5)

    assert not (tmp_path / "afss" / "refresh_epoch0005.json").exists()
