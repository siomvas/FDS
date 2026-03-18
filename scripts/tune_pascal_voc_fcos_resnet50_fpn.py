from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fl_training.tuning.optuna_proxy import run_preset_cli
from fl_training.tuning.presets import PRESETS


if __name__ == "__main__":
    raise SystemExit(run_preset_cli(PRESETS["pascal_voc_fcos_resnet50_fpn"]))
