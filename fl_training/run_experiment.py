import logging

import hydra
from omegaconf import DictConfig

# Configure logging
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Delay heavy FL stack imports until runtime to keep module import smoke lightweight.
    from fl_training import fl_job

    log.info(f"Starting experiment: {cfg.name}")
    log.info(f"Output directory: {cfg.paths.output_dir}")
    fl_job.run_job(cfg)

if __name__ == "__main__":
    main()
