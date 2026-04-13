import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.solver.afss import AFSSController
from engine.solver.det_solver import DetSolver


def test_compute_image_mask_precision_recall_matches_perfect_prediction():
    target = {
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
    result = {
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

    precision, recall = AFSSController.compute_image_mask_precision_recall(
        result,
        target,
        iou_threshold=0.5,
        conf_threshold=0.25,
    )

    assert precision == 1.0
    assert recall == 1.0


def test_compute_image_mask_precision_recall_resizes_target_masks_to_prediction_shape():
    target = {
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
    result = {
        "labels": torch.tensor([0], dtype=torch.int64),
        "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float32),
        "scores": torch.tensor([0.9], dtype=torch.float32),
        "masks": torch.tensor(
            [[[1, 1],
              [1, 1]]],
            dtype=torch.float32,
        ),
    }

    precision, recall = AFSSController.compute_image_mask_precision_recall(
        result,
        target,
        iou_threshold=0.5,
        conf_threshold=0.25,
    )

    assert precision == 1.0
    assert recall == 1.0


class _DummySegLoader:
    def __init__(self, batch):
        self._batch = batch
        self.sampler = type("Sampler", (), {"set_epoch": lambda self, epoch: None})()

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


def test_refresh_afss_scores_updates_joint_segmentation_metrics():
    solver = DetSolver.__new__(DetSolver)
    solver.afss = AFSSController(random_seed=0)
    solver.afss.initialize_states(total_samples=1)
    solver.afss_update_dataloader = _DummySegLoader(
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
        )
    )
    solver.train_dataloader = type("TrainLoader", (), {"dataset": type("Dataset", (), {"active_indices": [0], "__len__": lambda self: 1})()})()
    solver.ema = None
    solver.model = _DummySegModel()
    solver.model.train()
    solver.device = torch.device("cpu")
    solver.postprocessor = _DummySegPostprocessor()

    solver._refresh_afss_scores(epoch=5)

    assert solver.afss.states[0]["box_precision"] == 1.0
    assert solver.afss.states[0]["mask_precision"] == 1.0
    assert solver.afss.states[0]["score"] == 1.0
