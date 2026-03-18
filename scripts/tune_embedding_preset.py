from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from partitioning.tuning.optuna_proxy import run_preset_cli
from partitioning.tuning.presets import PRESETS


def _parse_preset_key(argv: list[str]) -> tuple[str, list[str]]:
    if len(argv) < 2:
        raise SystemExit("Usage: tune_embedding_preset.py <preset_key> [preset args...]")
    return argv[1], argv[2:]


if __name__ == "__main__":
    preset_key, preset_argv = _parse_preset_key(sys.argv)
    if preset_key not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise SystemExit(f"Unknown embedding preset '{preset_key}'. Valid presets: {valid}")
    raise SystemExit(run_preset_cli(PRESETS[preset_key], preset_argv))
