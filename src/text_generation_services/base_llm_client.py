from abc import ABC, abstractmethod
from PIL import Image
from typing import Any, Dict


class BaseLLMClient(ABC):
    def __init__(self, model_id: str, vllm_engine_args: Dict[str, Any], sampling_params_dict: Dict[str, Any]):
        self.model_id = model_id
        self.vllm_engine_args = vllm_engine_args
        self.sampling_params_dict = sampling_params_dict
        self.llm_engine = None

    @abstractmethod
    def generate_text_for_image(self, image: Image.Image, prompt_template: str) -> str:
        pass

    @abstractmethod
    def generate_text_from_text(self, text_input: str, prompt_template: str) -> str:
        pass

    @abstractmethod
    def get_last_request_metrics(self) -> Dict[str, Any]:
        pass

    def get_model_id(self) -> str:
        return self.model_id
