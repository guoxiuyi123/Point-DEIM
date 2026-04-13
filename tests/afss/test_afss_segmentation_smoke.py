import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig
from engine.solver.afss import AFSSController


def test_segmentation_joint_score_uses_lowest_metric():
    controller = AFSSController(random_seed=0)
    controller.initialize_states(total_samples=1)
    controller.update_image_metrics(
        0,
        box_precision=0.9,
        box_recall=0.8,
        mask_precision=0.7,
        mask_recall=0.6,
    )

    assert controller.states[0]["score"] == 0.6


def test_detection_afss_config_still_uses_box_postprocessor():
    cfg = YAMLConfig("configs/deim/deim_hgnetv2_n_custom_afss.yml")
    assert cfg.yaml_cfg["postprocessor"] == "PostProcessor"
