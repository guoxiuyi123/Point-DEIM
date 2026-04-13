import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.solver.afss import AFSSController


def test_afss_controller_keeps_hard_samples_and_applies_easy_moderate_budgets():
    controller = AFSSController(
        easy_threshold=0.85,
        moderate_threshold=0.55,
        easy_ratio=0.02,
        moderate_ratio=0.4,
        easy_review_gap=10,
        moderate_gap=3,
        update_interval=5,
        warmup_epochs=5,
        random_seed=0,
    )

    controller.initialize_states(total_samples=10)
    controller.states[0]["precision"] = 0.95
    controller.states[0]["recall"] = 0.95
    controller.states[0]["last_used_epoch"] = 0
    controller.states[1]["precision"] = 0.70
    controller.states[1]["recall"] = 0.80
    controller.states[1]["last_used_epoch"] = 0
    controller.states[2]["precision"] = 0.30
    controller.states[2]["recall"] = 0.40
    controller.states[2]["last_used_epoch"] = 0

    selected = controller.select_indices_for_epoch(epoch=12)

    assert 2 in selected
    assert len(selected) >= 1
    assert controller.level_for_index(0) == "easy"
    assert controller.level_for_index(1) == "moderate"
    assert controller.level_for_index(2) == "hard"


def test_afss_controller_roundtrips_state_dict():
    controller = AFSSController(random_seed=0)
    controller.initialize_states(total_samples=3)
    controller.states[1]["precision"] = 0.9
    state = controller.state_dict()

    restored = AFSSController(random_seed=0)
    restored.load_state_dict(state)

    assert restored.states[1]["precision"] == 0.9
