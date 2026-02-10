import logging
import os
from src.utils.config_loader import load_main_config
from src.pipelines.run_vllm_experiment import pipeline_run_vllm_experiment
from loggers import setup_loggers
from dotenv import load_dotenv

setup_loggers()
logger = logging.getLogger("main")


def main():
    logger.info("Starting the LLM inference with vLLM.")
    load_dotenv()

    try:
        current_experiment_path = "config/current_vllm_experiment.yaml"
        if not os.path.exists(current_experiment_path):
            logger.error(f"Current experiment configuration file not found at {current_experiment_path} path.")
            return

        current_experiment_config = load_main_config(current_experiment_path)
        logger.info(f"Loaded vLLM experiment setup configuration: {current_experiment_config}")

        pipeline_run_vllm_experiment(current_experiment_config)

        logger.info("LLM inference finished.")

    except Exception as e:
        logger.error(f"Error during the LLM inference execution: {e}", exc_info=True)


if __name__ == "__main__":
    main()
