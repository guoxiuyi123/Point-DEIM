import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig
from engine.solver.afss import AFSSController


def test_afss_smoke_refresh_decision_matches_warmup_and_interval():
    controller = AFSSController(update_interval=5, warmup_epochs=5)
    controller.initialize_states(total_samples=4)

    assert controller.should_refresh_scores(0) is False
    assert controller.should_refresh_scores(4) is False
    assert controller.should_refresh_scores(5) is True
    assert controller.should_refresh_scores(6) is False
    assert controller.should_refresh_scores(10) is True


def test_baseline_config_remains_non_afss():
    cfg = YAMLConfig("configs/deim/deim_hgnetv2_n_custom.yml")

    assert "AFSS" not in cfg.yaml_cfg
