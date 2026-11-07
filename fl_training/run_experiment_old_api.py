import logging

import hydra
from omegaconf import DictConfig


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Backward-compatible entrypoint that delegates to the current FL runner."""
    from fl_training import fl_job

    log.warning("`run_experiment_old_api.py` is deprecated; delegating to the current FL job runner.")
    fl_job.run_job(cfg)


if __name__ == "__main__":
    main()
