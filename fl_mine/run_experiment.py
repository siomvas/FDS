import sys
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Configure logging
log = logging.getLogger(__name__)

def setup_python_path(cfg: DictConfig):
    original_cwd = Path(hydra.utils.get_original_cwd())
    data_root = (original_cwd / cfg.paths.data_loading_root).resolve().parent
    if str(data_root) not in sys.path:
        log.info(f"Adding {data_root} to sys.path to enable data module imports")
        sys.path.append(str(data_root))
    else:
        log.debug(f"{data_root} already in sys.path")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    log.info(f"Starting experiment: {cfg.name}")
    log.info(f"Output directory: {cfg.paths.output_dir}")
    
    setup_python_path(cfg)

    try:
        import fl_job
    except ImportError as e:
        log.error("Could not import fl_job. Make sure it is in the same directory.")
        raise e
    fl_job.run_job(cfg)

if __name__ == "__main__":
    main()