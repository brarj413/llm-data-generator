import logging
from pathlib import Path
from src.utils.config_loader import get_paths_config, get_dataset_config

logger = logging.getLogger(__name__)


class PathConstructor:
    def __init__(self, experiment_plan_config: dict, current_experiment_config_key: str):
        self.experiment_plan_config = experiment_plan_config
        self.current_experiment_config_key = current_experiment_config_key
        self.paths_cfg = get_paths_config()
        self.dataset_key_from_plan = experiment_plan_config.get('dataset_key_for_images', "plant_doc")
        self.dataset_cfg = get_dataset_config(self.dataset_key_from_plan)
        self.dataset_name = self.dataset_cfg['dataset_name']
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_root_abs = self.project_root / self.paths_cfg['data_root']

    def get_dataset_base_dir(self) -> Path:
        return self.data_root_abs / self.dataset_name

    def get_image_dir(self) -> Path:
        return self.get_dataset_base_dir() / self.dataset_cfg['image_subfolder']

    def get_metadata_path(self) -> Path:
        return self.get_dataset_base_dir() / self.dataset_cfg['metadata_filename']

    def get_base_output_dir_for_current_test(self) -> Path:
        base_output_path = self.get_dataset_base_dir() / "text" / self.current_experiment_config_key
        base_output_path.mkdir(parents=True, exist_ok=True)
        return base_output_path

    def get_image_description_output_dir_for_split(self, split_name: str) -> Path:
        output_dir = self.get_base_output_dir_for_current_test() / split_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def get_text_summaries_output_dir_for_split(self, split_name: str) -> Path:
        return self.get_image_description_output_dir_for_split(split_name)

    def get_performance_metrics_file_path(self) -> Path:
        output_dir = self.get_base_output_dir_for_current_test()
        return output_dir / f"performance_metrics_{self.current_experiment_config_key}.csv"

    def get_resource_utilization_file_path(self) -> Path:
        output_dir = self.get_base_output_dir_for_current_test()
        return output_dir / f"resource_utilization_{self.current_experiment_config_key}.csv"
