import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.solver.afss import AFSSController


def test_afss_controller_uses_minimum_of_box_and_mask_metrics_for_level():
    controller = AFSSController(easy_threshold=0.85, moderate_threshold=0.55, random_seed=0)
    controller.initialize_states(total_samples=2)

    controller.update_image_metrics(
        0,
        box_precision=0.95,
        box_recall=0.92,
        mask_precision=0.90,
        mask_recall=0.88,
    )
    controller.update_image_metrics(
        1,
        box_precision=0.95,
        box_recall=0.92,
        mask_precision=0.90,
        mask_recall=0.40,
    )

    assert controller.states[0]["score"] == 0.88
    assert controller.level_for_index(0) == "easy"
    assert controller.states[1]["score"] == 0.40
    assert controller.level_for_index(1) == "hard"


def test_afss_controller_roundtrips_mask_metrics_in_state_dict():
    controller = AFSSController(random_seed=0)
    controller.initialize_states(total_samples=1)
    controller.update_image_metrics(
        0,
        box_precision=0.7,
        box_recall=0.8,
        mask_precision=0.9,
        mask_recall=0.6,
    )

    restored = AFSSController(random_seed=0)
    restored.load_state_dict(controller.state_dict())

    assert restored.states[0]["mask_recall"] == 0.6
    assert restored.states[0]["score"] == 0.6
