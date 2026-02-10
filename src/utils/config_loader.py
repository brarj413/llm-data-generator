import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str | Path) -> dict:
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.debug(f"Successfully loaded config from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(
            f"Error loading YAML file {config_path}: {e}")
        raise


def load_main_config(config_path: str = "config/current_vllm_experiment.yaml") -> dict:
    return load_yaml_config(config_path)


def get_dataset_config(active_dataset_key: str) -> dict:
    config_path = Path(f"config/dataset_specific_params/{active_dataset_key}.yaml")
    return load_yaml_config(config_path)


def get_paths_config() -> dict:
    return load_yaml_config("config/paths_config.yaml")


def get_text_prompts_config() -> dict:
    return load_yaml_config("config/prompts.yaml")


def get_vllm_experiment_plans_config() -> dict:
    return load_yaml_config("config/vllm_experiment_plans.yaml")


def get_vllm_models_and_tests_config() -> dict:
    return load_yaml_config("config/vllm_models_and_tests.yaml")
