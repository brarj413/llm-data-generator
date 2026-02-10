import logging
from pathlib import Path
from typing import Dict, Optional
from src.utils.config_loader import get_vllm_experiment_plans_config, get_vllm_models_and_tests_config
from src.text_generation_services.text_generation_manager import VLLMExperimentRunner

logger = logging.getLogger(__name__)


def pipeline_run_vllm_experiment(main_setup_config: dict):
    logger.info("Starting vLLM experiment pipeline...")

    active_plan_key = main_setup_config['active_experiment_plan_key']
    all_plans_config = get_vllm_experiment_plans_config()

    if active_plan_key not in all_plans_config:
        logger.error(f"Active experiment plan key '{active_plan_key}' not found in vllm_experiment_plans.yaml.")
        return

    current_plan_config = all_plans_config[active_plan_key]
    current_plan_config["active_experiment_plan_key"] = active_plan_key
    logger.info(f"Running experiment plan: '{active_plan_key}' - {current_plan_config.get('description', '')}")

    all_models_and_tests_config = get_vllm_models_and_tests_config()
    description_output_base_dirs: Dict[str, Path] = {}

    logger.info("Running image-to-text tasks...")
    for experiment_key in current_plan_config.get('experiments_to_run', []):
        if experiment_key not in all_models_and_tests_config:
            logger.warning(f"Experiment configuration key '{experiment_key}' not found in vllm_models_and_tests.yaml file.")
            continue

        experiment_config = all_models_and_tests_config[experiment_key]
        if experiment_config.get('task_type') == "image-to-text":
            logger.info(f"Preparing image-to-text experiment for: {experiment_key}")
            runner = None
            try:
                runner = VLLMExperimentRunner(
                    experiment_plan_config=current_plan_config.copy(),
                    current_experiment_key=experiment_key,
                    experiment_config=experiment_config.copy()
                )

                output_base_dir_for_vlm_experiment = runner.run_image_to_text_generation()
                if output_base_dir_for_vlm_experiment and output_base_dir_for_vlm_experiment.exists():
                    description_output_base_dirs[experiment_key] = output_base_dir_for_vlm_experiment
                else:
                    logger.warning(f"VLM experiment {experiment_key} did not produce output directory.")
                runner.save_results()
            except Exception as e:
                logger.error(f"Error running image-to-text experiment {experiment_key}: {e}", exc_info=True)
            finally:
                if runner:
                    runner.cleanup_client()
            logger.info(f"Finished image-to-text experiment {experiment_key}")

    logger.info("Running text-to-text summarization tasks")
    for experiment_key in current_plan_config.get('experiments_to_run', []):
        if experiment_key not in all_models_and_tests_config:
            continue

        experiment_config = all_models_and_tests_config[experiment_key]
        if experiment_config.get('task_type') == "text-to-text":
            logger.info(f"Preparing text-to-text experiment for: {experiment_key}")

            input_desc_source_key = experiment_config.get("input_description_source_key")
            input_base_dir_for_summaries: Optional[Path] = None

            if input_desc_source_key and input_desc_source_key in description_output_base_dirs:
                input_base_dir_for_summaries = description_output_base_dirs[input_desc_source_key]
                logger.info(
                    f"Using descriptions from VLM experiment '{input_desc_source_key}', path: {input_base_dir_for_summaries}) for summarization task '{experiment_key}'.")
            elif not input_desc_source_key and description_output_base_dirs:
                fallback_source_key = next(iter(description_output_base_dirs.keys()))
                input_base_dir_for_summaries = description_output_base_dirs[fallback_source_key]
                logger.warning(
                    f"No key 'input_description_source_key' for summarization experiment {experiment_key}. Using descriptions from the first VLM experiment: '{fallback_source_key}'.")

            if not input_base_dir_for_summaries:
                logger.error(f"No input description base directory found for summarization task {experiment_key}.")
                continue

            runner = None
            try:
                runner = VLLMExperimentRunner(
                    experiment_plan_config=current_plan_config.copy(),
                    current_experiment_key=experiment_key,
                    experiment_config=experiment_config.copy()
                )
                runner.run_text_to_text_summarization(input_text_base_dir_for_vlm_test=input_base_dir_for_summaries)
                runner.save_results()
            except Exception as e:
                logger.error(
                    f"Error running text-to-text experiment {experiment_key}: {e}", exc_info=True)
            finally:
                if runner:
                    runner.cleanup_client()
            logger.info(f"Finished text-to-text experiment for: {experiment_key}")

    logger.info("vLLM experiment pipeline finished.")
