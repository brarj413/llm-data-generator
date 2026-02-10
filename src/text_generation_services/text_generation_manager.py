import logging
import pandas as pd
import time
import os
import psutil
import subprocess
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Tuple
from src.utils.config_loader import get_text_prompts_config, get_dataset_config
from src.utils.path_constructor import PathConstructor
from .vllm_service_client import VLLMServiceClient
from src.data_management.dataset_loader import load_image_paths_from_metadata, load_text_files_for_summarization

logger = logging.getLogger(__name__)


def get_gpu_utilization():
    if not torch.cuda.is_available():
        return []
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, text=True, check=True, encoding='utf-8'
        )
        gpu_stats = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            util, mem_used, mem_total = line.split(', ')
            gpu_stats.append({
                'gpu_util_percent': float(util),
                'gpu_mem_used_mb': float(mem_used),
                'gpu_mem_total_mb': float(mem_total)
            })
        return gpu_stats
    except Exception as e:
        logger.warning(f"nvidia-smi command stats fail: {e}")
        return []


class VLLMExperimentRunner:
    def __init__(self, experiment_plan_config: dict, current_experiment_key: str, experiment_config: dict):
        self.experiment_plan_config = experiment_plan_config
        self.current_experiment_key = current_experiment_key
        self.experiment_config = experiment_config
        self.prompts_cfg = get_text_prompts_config()
        self.paths_constructor = PathConstructor(self.experiment_plan_config, self.current_experiment_key)
        self.model_id = self.experiment_config['model_id']
        self.task_type = self.experiment_config['task_type']
        self.prompt_key = self.experiment_config['prompt_key']
        self.vllm_engine_args = self.experiment_config.get('vllm_engine_args', {})
        self.sampling_params = self.experiment_config.get('sampling_params', {})
        self.llm_client: Optional[VLLMServiceClient] = None

        self.project_root_dir = self.paths_constructor.project_root
        if not self.project_root_dir.exists() or not self.project_root_dir.is_dir():
            self.project_root_dir = Path(__file__).resolve().parents[2]
            logger.warning(f"PathConstructor.project_root not found, using fallback: {self.project_root_dir}")

        self.dataset_cfg = get_dataset_config(experiment_plan_config.get('dataset_key_for_images', "plant_doc"))
        self.metadata_path = self.paths_constructor.get_metadata_path()
        self.num_images_to_process = experiment_plan_config.get('num_images_to_process', -1)
        self.num_texts_to_summarize = experiment_plan_config.get('num_texts_to_summarize', -1)
        self.all_performance_metrics: List[Dict[str, Any]] = []
        self.resource_log: List[Dict[str, Any]] = []

        logger.info(f"ExperimentRunner initialized for experiment: {self.current_experiment_key}")
        logger.info(f"Model ID: {self.model_id}, Task: {self.task_type}")

    def _init_llm_client(self):
        if self.llm_client and self.llm_client.get_model_id() == self.model_id:
            self.llm_client.last_request_metrics = {}
            return

        if self.llm_client:
            self.llm_client.unload_model()

        logger.info(f"Initializing VLLMServiceClient for experiment {self.current_experiment_key} with model {self.model_id}")
        self.llm_client = VLLMServiceClient(
            model_id=self.model_id,
            vllm_engine_args=self.vllm_engine_args.copy(),
            sampling_params_dict=self.sampling_params.copy(),
            experiment_config=self.experiment_config.copy()
        )

    def _log_resource_snapshot(self, stage: str, item_id: Optional[str] = None):
        process = psutil.Process(os.getpid())
        cpu_percent = process.cpu_percent(interval=None)
        snapshot = {
            "timestamp_ns": time.time_ns(),
            "stage": stage,
            "item_id": str(item_id) if item_id is not None else None,
            "cpu_percent_process": cpu_percent,
            "ram_rss_mb": process.memory_info().rss / (1024 * 1024),
            "ram_vms_mb": process.memory_info().vms / (1024 * 1024),
        }
        try:
            snapshot["cpu_percent_system"] = psutil.cpu_percent(interval=None)
        except Exception as e:
            snapshot["cpu_percent_system"] = None

        gpu_stats = get_gpu_utilization()
        for i, stats in enumerate(gpu_stats):
            snapshot[f"gpu_{i}_util_percent"] = stats['gpu_util_percent']
            snapshot[f"gpu_{i}_mem_used_mb"] = stats['gpu_mem_used_mb']
            snapshot[f"gpu_{i}_mem_total_mb"] = stats['gpu_mem_total_mb']
        self.resource_log.append(snapshot)

    def run_image_to_text_generation(self) -> Optional[Path]:
        if self.task_type != "image-to-text":
            logger.info(f"Skipping image-to-text for {self.current_experiment_key} because task type is {self.task_type}")
            return None

        self._init_llm_client()
        if not self.llm_client or not self.llm_client.llm_engine:
            logger.error(f"LLM client failed to initialize for experiment {self.current_experiment_key}.")
            return None

        image_data_loaded = load_image_paths_from_metadata(self.metadata_path, self.project_root_dir, self.num_images_to_process)
        if not image_data_loaded:
            logger.warning(f"Error loading images for experiment {self.current_experiment_key}.")
            return self.paths_constructor.get_base_output_dir_for_current_test()

        prompt_data = self.prompts_cfg['prompts'][self.prompt_key]
        prompt_content_llm: str
        if self.experiment_config.get('use_chat_template', False):
            prompt_content_llm = prompt_data.get('user_instruction_text', '')
            if not prompt_content_llm:
                logger.error(f"Prompt key '{self.prompt_key}' is configured to use 'chat template' mode. 'user_instruction_text' is missing.")
                return None
        else:
            prompt_content_llm = prompt_data.get('user_prompt_template', '')
            if not prompt_content_llm:
                logger.error(f"Prompt key '{self.prompt_key}' is configured to use 'user prompt template' mode. 'user_prompt_template' is missing.")
                return None

        logger.info(
            f"Starting image-to-text generation for {len(image_data_loaded)} images using {self.model_id} for experiment {self.current_experiment_key}.")
        self._log_resource_snapshot(stage="start_image_to_text_generation_for_experiment_key_" + self.current_experiment_key)

        successful_generations = 0
        for image_id, image_absolute_path in tqdm(image_data_loaded, desc=f"Img2Txt: {self.current_experiment_key}"):
            self._log_resource_snapshot(stage="before_generate_image", item_id=image_id)
            item_perf_metrics = {"item_id": image_id, "model_id": self.model_id, "task_type": "image-to-text"}
            start_item_time = time.time()
            try:
                image_pil_object = Image.open(image_absolute_path)
                try:
                    path_relative_to_project = image_absolute_path.relative_to(self.project_root_dir)
                    if len(path_relative_to_project.parts) > 3 and path_relative_to_project.parts[3] in ["train", "test", "val"]:
                        split_name = path_relative_to_project.parts[3]
                    else:
                        parent_dir_name = image_absolute_path.parent.name.lower()
                        if parent_dir_name in ["train", "test", "val"]:
                            split_name = parent_dir_name
                        else:
                            logger.warning(f"Could not determine split for {image_absolute_path} from path.")
                            split_name = "unknown"
                except ValueError:
                    logger.error(f"Error when making {image_absolute_path} image path relative to {self.project_root_dir} project path.")
                    split_name = "unknown"

                output_dir_for_split = self.paths_constructor.get_image_description_output_dir_for_split(split_name)
                generated_text = self.llm_client.generate_text_for_image(image_pil_object, prompt_content_llm)
                logger.info(f"Image ID: {image_id} (Split: {split_name}) - Raw VLM text: '{generated_text[:100]}...'")

                item_request_metrics = self.llm_client.get_last_request_metrics()
                item_perf_metrics.update(item_request_metrics)

                if generated_text and "Error:" not in generated_text:
                    try:
                        img_number_part = image_id
                        if '_' in image_id:
                            img_number_part = image_id.split('_')[-1]

                        if img_number_part.isdigit():
                            output_file_name = f"text_{img_number_part}.txt"
                        else:
                            output_file_name = f"{image_id}_desc.txt"
                    except Exception as e:
                        logger.warning(f"Could not parse image_id '{image_id}' from image filename.")
                        output_file_name = f"{image_id}_desc.txt"

                    with open(output_dir_for_split / output_file_name, 'w', encoding='utf-8') as f:
                        f.write(generated_text)
                    logger.info(f"Successfully wrote description for {image_id} to {output_dir_for_split / output_file_name}")
                    successful_generations += 1
                else:
                    logger.warning(f"Skipping file write for {image_id} due to error response from model: '{generated_text}'")
                    item_perf_metrics["error_message"] = generated_text if generated_text else "Empty response from model"
            except Exception as e:
                logger.error(f"Failed to process image {image_absolute_path} for experiment {self.current_experiment_key}: {e}", exc_info=True)
                item_perf_metrics["error_message"] = str(e)

            end_item_time = time.time()
            item_perf_metrics["total_item_wall_time_ms"] = (end_item_time - start_item_time) * 1000
            self.all_performance_metrics.append(item_perf_metrics)
            self._log_resource_snapshot(stage="after_generate_image", item_id=image_id)

        self._log_resource_snapshot(stage="end_image_to_text_generation_for_experiment_key_" + self.current_experiment_key)
        logger.info(
            f"Image-to-text generation finished for {self.current_experiment_key} experiment. Generated {successful_generations}/{len(image_data_loaded)} descriptions.")
        return self.paths_constructor.get_base_output_dir_for_current_test()

    def run_text_to_text_summarization(self, input_text_base_dir_for_vlm_test: Path) -> Optional[Path]:
        if self.task_type != "text-to-text":
            logger.info(f"Skipping text-to-text for {self.current_experiment_key} because task type is {self.task_type}")
            return None

        if not input_text_base_dir_for_vlm_test or not input_text_base_dir_for_vlm_test.exists():
            logger.warning(
                f"Input VLM directory for descriptions not found: {input_text_base_dir_for_vlm_test} in summarization experiment {self.current_experiment_key}.")
            return None

        self._init_llm_client()
        if not self.llm_client or not self.llm_client.llm_engine:
            logger.error(f"LLM client failed to initialize for summarization experiment {self.current_experiment_key}.")
            return None

        all_texts_to_summarize: List[Tuple[str, str, str]] = []
        for split_name_enum in ["train", "test", "val", "unknown"]:
            input_desc_dir_for_split = input_text_base_dir_for_vlm_test / split_name_enum
            if input_desc_dir_for_split.exists():
                loaded_texts = load_text_files_for_summarization(input_desc_dir_for_split, -1)
                for text_file_stem, content in loaded_texts:
                    all_texts_to_summarize.append((text_file_stem, content, split_name_enum))

        if self.num_texts_to_summarize != -1 and len(all_texts_to_summarize) > self.num_texts_to_summarize:
            all_texts_to_summarize = all_texts_to_summarize[:self.num_texts_to_summarize]

        if not all_texts_to_summarize:
            logger.warning(
                f"No text files loaded from subdirectories of {input_text_base_dir_for_vlm_test} directory for experiment {self.current_experiment_key}.")
            return None

        prompt_template_content = self.prompts_cfg['prompts'][self.prompt_key]['user_prompt_template']
        logger.info(
            f"Starting text-to-text summarization for {len(all_texts_to_summarize)} texts using {self.model_id} for experiment {self.current_experiment_key}.")
        self._log_resource_snapshot(stage="start_text_to_text_summarization_for_experiment_key_" + self.current_experiment_key)

        successful_generations = 0
        for text_file_stem, text_content, original_split_name in tqdm(all_texts_to_summarize, desc=f"Txt2Txt: {self.current_experiment_key}"):
            self._log_resource_snapshot(stage="before_generate_summary", item_id=text_file_stem)
            item_perf_metrics = {"item_id": text_file_stem, "model_id": self.model_id, "task_type": "text-to-text"}
            start_item_time = time.time()
            try:
                generated_summary = self.llm_client.generate_text_from_text(text_content, prompt_template_content)
                item_request_metrics = self.llm_client.get_last_request_metrics()
                item_perf_metrics.update(item_request_metrics)

                if generated_summary and "Error:" not in generated_summary:
                    output_dir_for_summary_split = self.paths_constructor.get_image_description_output_dir_for_split(original_split_name)
                    output_file_name_summary = f"{text_file_stem}_summary.txt"
                    with open(output_dir_for_summary_split / output_file_name_summary, 'w', encoding='utf-8') as f:
                        f.write(generated_summary)
                    logger.info(f"Successfully wrote summary for {text_file_stem} to {output_dir_for_summary_split / output_file_name_summary}")
                    successful_generations += 1
                else:
                    logger.warning(f"Skipping file write for summary of {text_file_stem} due to error in generated text: '{generated_summary}'")
                    item_perf_metrics["error_message"] = generated_summary if generated_summary else "Empty response from model"
            except Exception as e:
                logger.error(f"Failed to summarize text_id {text_file_stem} for experiment {self.current_experiment_key}: {e}", exc_info=True)
                item_perf_metrics["error_message"] = str(e)

            end_item_time = time.time()
            item_perf_metrics["total_item_wall_time_ms"] = (end_item_time - start_item_time) * 1000
            self.all_performance_metrics.append(item_perf_metrics)
            self._log_resource_snapshot(stage="after_generate_summary", item_id=text_file_stem)

        self._log_resource_snapshot(stage="end_text_to_text_summarization_for_experiment_key_" + self.current_experiment_key)
        logger.info(
            f"Text-to-text summarization finished for {self.current_experiment_key}. Generated {successful_generations}/{len(all_texts_to_summarize)} summaries.")
        return self.paths_constructor.get_base_output_dir_for_current_test()

    def save_results(self):
        perf_metrics_path = self.paths_constructor.get_performance_metrics_file_path()
        if self.all_performance_metrics:
            try:
                df_performance = pd.DataFrame(self.all_performance_metrics)
                df_performance.to_csv(perf_metrics_path, index=False)
                logger.info(f"Performance metrics for {self.current_experiment_key} experiment saved to: {perf_metrics_path}")
            except Exception as e:
                logger.error(f"Failed to save performance metrics for {self.current_experiment_key} experiment: {e}")
        else:
            logger.warning(f"No performance metrics collected for {self.current_experiment_key} experiment.")

        resource_log_path = self.paths_constructor.get_resource_utilization_file_path()
        if self.resource_log:
            try:
                df_resource = pd.DataFrame(self.resource_log)
                df_resource.to_csv(resource_log_path, index=False)
                logger.info(f"Resource utilization log for {self.current_experiment_key} experiment saved to: {resource_log_path}")
            except Exception as e:
                logger.error(f"Failed to save resource utilization log for {self.current_experiment_key} experiment: {e}")
        else:
            logger.warning(f"No resource utilization data collected for {self.current_experiment_key} experiment.")

    def cleanup_client(self):
        if self.llm_client:
            self.llm_client.unload_model()
            self.llm_client = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
