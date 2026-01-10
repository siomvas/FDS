"""NVFlare 2.7 job configuration generator.

Converts user-friendly YAML experiment configs to NVFlare JSON job structure.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any

from .experiment_config import FDSExperimentConfig


class JobConfigGenerator:
    """Generate NVFlare 2.7 JSON job configs from YAML experiment configs."""

    def __init__(self, template_dir: Path = None):
        """Initialize generator.

        Args:
            template_dir: Path to job templates directory
        """
        if template_dir is None:
            template_dir = Path(__file__).parent.parent / "jobs" / "templates"
        self.template_dir = Path(template_dir)

    def generate(
        self,
        yaml_config_path: str,
        output_dir: str,
        client_script_path: str = "train.py"
    ) -> Path:
        """Generate NVFlare job config from YAML.

        Args:
            yaml_config_path: Path to YAML experiment config
            output_dir: Directory to write job config
            client_script_path: Path to client training script

        Returns:
            Path to generated job config JSON
        """
        # Load YAML config
        config = FDSExperimentConfig.from_yaml(yaml_config_path)

        # Select template based on task type
        template_file = self.template_dir / f"{config.experiment.task_type}_job_template.json"
        if not template_file.exists():
            raise ValueError(f"Template not found for task type: {config.experiment.task_type}")

        # Load template
        with open(template_file, 'r') as f:
            template_content = f.read()

        # Build replacement dict
        replacements = self._build_replacements(config, client_script_path)

        # Replace placeholders
        job_config_content = self._replace_placeholders(template_content, replacements)

        # Write output
        output_path = Path(output_dir) / f"{config.experiment.name}_job_config.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            # Parse and re-dump to ensure valid JSON with proper types
            job_config = json.loads(job_config_content)
            json.dump(job_config, f, indent=2)

        return output_path

    def _build_replacements(
        self,
        config: FDSExperimentConfig,
        client_script_path: str
    ) -> Dict[str, Any]:
        """Build placeholder replacement dictionary.

        Args:
            config: FDS experiment configuration
            client_script_path: Path to client script

        Returns:
            Dict mapping placeholder names to values
        """
        fl = config.federated_learning
        train = config.training
        exp = config.experiment

        # Min clients defaults to num_clients
        min_clients = fl.min_clients if fl.min_clients is not None else fl.num_clients

        return {
            "EXPERIMENT_NAME": exp.name,
            "NUM_ROUNDS": fl.num_rounds,
            "NUM_CLIENTS": fl.num_clients,
            "MIN_CLIENTS": min_clients,
            "WAIT_TIME": fl.wait_time,
            "AGGREGATOR_TYPE": fl.aggregator,
            "AGGREGATOR_CONFIG": json.dumps(fl.aggregator_config),
            "DATASET_SPLITS": exp.dataset_splits,
            "MODEL_NAME": train.model,
            "MODEL_INIT_PATH": f"{train.model}_init.pt",
            "CLIENT_SCRIPT_PATH": client_script_path,
            "LEARNING_RATE": train.learning_rate,
            "LOCAL_EPOCHS": train.local_epochs,
            "BATCH_SIZE": train.batch_size,
            "AMP_ENABLED": str(train.amp_enabled).lower(),
            "GRAD_TRACKING": str(train.grad_tracking).lower(),
            "SEED": train.seed
        }

    def _replace_placeholders(
        self,
        template: str,
        replacements: Dict[str, Any]
    ) -> str:
        """Replace {{PLACEHOLDER}} with values.

        Args:
            template: Template string with {{PLACEHOLDER}} syntax
            replacements: Dict mapping placeholder names to values

        Returns:
            Template with placeholders replaced
        """
        result = template
        for key, value in replacements.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))

        # Check for unreplaced placeholders
        unreplaced = re.findall(r'\{\{(\w+)\}\}', result)
        if unreplaced:
            raise ValueError(f"Unreplaced placeholders: {unreplaced}")

        return result
