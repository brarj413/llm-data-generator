import logging
import time
import torch
import textwrap
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple
from transformers import AutoProcessor
from vllm import LLM, SamplingParams, RequestOutput
from .base_llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class VLLMServiceClient(BaseLLMClient):
    def __init__(self, model_id: str, vllm_engine_args: Dict[str, Any], sampling_params_dict: Dict[str, Any], experiment_config: Dict[str, Any]):
        super().__init__(model_id, vllm_engine_args, sampling_params_dict)
        self.llm_engine: Optional[LLM] = None
        self.last_request_metrics: Dict[str, Any] = {}
        self.use_chat_template = experiment_config.get('use_chat_template', False)
        self.hf_processor: Optional[AutoProcessor] = None
        self._initialize_engine()

    def _initialize_engine(self):
        if self.llm_engine is None:
            logger.info(f"Initializing vLLM engine for model: {self.model_id} with args: {self.vllm_engine_args}")
            try:
                if "dtype" in self.vllm_engine_args and isinstance(self.vllm_engine_args["dtype"], str):
                    if self.vllm_engine_args["dtype"] == "auto":
                        pass
                    elif hasattr(torch, self.vllm_engine_args["dtype"]):
                        self.vllm_engine_args["dtype"] = getattr(torch, self.vllm_engine_args["dtype"])
                    else:
                        logger.warning(f"Unknown torch dtype string: {self.vllm_engine_args['dtype']}.")

                self.llm_engine = LLM(model=self.model_id, **self.vllm_engine_args)

                if self.use_chat_template:
                    logger.info(f"Loading HuggingFace AutoProcessor for model: {self.model_id} with use_chat_template mode.")
                    trust_remote_code_for_processor = self.vllm_engine_args.get('trust_remote_code', False)
                    self.hf_processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=trust_remote_code_for_processor)
                    logger.info(f"HuggingFace AutoProcessor for {self.model_id} loaded.")

                logger.info(f"vLLM engine for {self.model_id} initialized successfully.")
            except Exception as e:
                logger.error(f"Error initializing vLLM engine for {self.model_id} model: {e}", exc_info=True)
                raise
        else:
            logger.info(f"vLLM engine for {self.model_id} already initialized.")

    def _prepare_sampling_params(self) -> SamplingParams:
        return SamplingParams(**self.sampling_params_dict)

    def _process_outputs(self, outputs: List[RequestOutput], start_time: float) -> Tuple[str, Dict[str, Any]]:
        end_time = time.time()
        generated_text = ""
        total_output_tokens = 0

        if outputs and len(outputs) > 0:
            output_item = outputs[0]
            if output_item.outputs:
                generated_text = output_item.outputs[0].text
                total_output_tokens = len(output_item.outputs[0].token_ids)
            else:
                logger.warning(f"Received empty outputs list from vLLM for model {self.model_id}")
                generated_text = "Empty output"

        e2e_latency = end_time - start_time
        tokens_per_second = total_output_tokens / e2e_latency if e2e_latency > 0 else 0

        metrics = {
            "e2e_latency_ms": e2e_latency * 1000,
            "total_output_tokens": total_output_tokens,
            "tokens_per_second": tokens_per_second,
            "num_input_tokens": len(outputs[0].prompt_token_ids) if outputs and output_item.outputs else 0,
        }
        self.last_request_metrics = metrics
        return generated_text.strip(), metrics

    def generate_text_for_image(self, image_pil_object: Image.Image, prompt_content: str) -> str:
        if not self.llm_engine:
            logger.error("vLLM engine not initialized.")
            return "Error: vLLM engine not initialized."

        if image_pil_object.mode != 'RGB':
            image_pil_object = image_pil_object.convert('RGB')

        sampling_params_obj = self._prepare_sampling_params()
        final_prompt_str: str

        if self.use_chat_template:
            if not self.hf_processor:
                error_msg_hf_processor = "HF Processor not initialized for model"
                logger.error(f"{error_msg_hf_processor} {self.model_id}.")
                return f"Error: {error_msg_hf_processor} {self.model_id}"

            user_instruction = prompt_content
            messages = [
                # {"role": "system", "content": [{"type": "text", "text": system_prompt_text}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_pil_object},
                        {"type": "text", "text": user_instruction}
                    ]
                }
            ]

            try:
                final_prompt_str = self.hf_processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
            except Exception as e:
                error_msg_chat_template = "Applying chat template fail for model"
                logger.error(f"{error_msg_chat_template} {self.model_id}: {e}", exc_info=True)
                return f"Error: {error_msg_chat_template} {self.model_id}"
        else:
            final_prompt_str = textwrap.dedent(prompt_content).strip()

        request_payload = {
            "prompt": final_prompt_str,
            "multi_modal_data": {"image": image_pil_object}
        }

        start_time = time.time()
        try:
            outputs = self.llm_engine.generate([request_payload], sampling_params=sampling_params_obj)
            generated_text, metrics = self._process_outputs(outputs, start_time)
            return generated_text
        except Exception as e:
            logger.error(f"Error during vLLM image-to-text generation: {e}", exc_info=True)
            self.last_request_metrics = {"error": str(e)}
            return f"Error: vLLM generation - {type(e).__name__}"

    def generate_text_from_text(self, text_input: str, prompt_template: str) -> str:
        if not self.llm_engine:
            error_msg_vllm_engine = "vLLM engine not initialized"
            logger.error(error_msg_vllm_engine)
            return f"Error: {error_msg_vllm_engine}"

        full_prompt = prompt_template.format(plant_description=text_input)
        sampling_params_obj = self._prepare_sampling_params()

        start_time = time.time()
        try:
            outputs = self.llm_engine.generate(prompts=[full_prompt], sampling_params=sampling_params_obj)
            generated_text, metrics = self._process_outputs(outputs, start_time)
            return generated_text
        except Exception as e:
            logger.error(f"Error during vLLM text-to-text generation: {e}", exc_info=True)
            self.last_request_metrics = {"error": str(e)}
            return f"Error: vLLM generation - {type(e).__name__}"

    def get_last_request_metrics(self) -> Dict[str, Any]:
        return self.last_request_metrics

    def unload_model(self):
        if self.llm_engine is not None:
            logger.info(f"Unloading vLLM engine for model: {self.model_id}")
            del self.llm_engine
            self.llm_engine = None
            if self.hf_processor:
                del self.hf_processor
                self.hf_processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"vLLM engine references for {self.model_id} removed.")
            logger.info("CUDA cache cleared.")
