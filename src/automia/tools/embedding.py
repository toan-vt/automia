from ..common import get_logger
from openai import OpenAI
from ..common import LLMClientProvider
import time
import os

class EmbeddingTool:
    # only support vllm for now
    def __init__(self, model_name: str, vllm_api_base: str, provider: LLMClientProvider):
        self._logger = get_logger("system")
        self._model_name = model_name
        self._vllm_api_base = vllm_api_base
        if provider == LLMClientProvider.VLLM:
            api_key = "dummy_key"
        elif provider == LLMClientProvider.OPENAI:
            api_key = os.environ["OPENAI_API_KEY"]
        elif provider == LLMClientProvider.GOOGLE:
            api_key = os.environ["GOOGLE_API_KEY"]
        else:
            raise ValueError(f"Invalid provider: {provider}")
        self._client = OpenAI(base_url=self._vllm_api_base, api_key=api_key)
        self._logger.info(f"Initializing EmbeddingTool with model: {model_name} and vllm API base: {vllm_api_base}")

        if not self._test_request():
            raise RuntimeError(f"Embedding test request failed after 3 attempts")
        
    def _get_max_length(self) -> int:
        return self._client.embeddings.max_length
    
    def _test_request(self) -> bool:
        last_error = None
        for i in range(3):
            try:
                self._client.embeddings.create(
                    input="Hello, world!",
                    model=self._model_name,
                    encoding_format="float"
                )
                self._logger.info(f"Embedding test request successful on attempt {i+1}")
                return True
            except Exception as e:
                last_error = e
                self._logger.error(f"Error testing embedding request: {e}")
                time.sleep(1)
        self._logger.error(f"Embedding test request failed after 3 attempts: {last_error}")

        return False
    
    def embed(self, text: str) -> list[float]:
        last_error = None
        for _ in range(3):
            try:
                resp = self._client.embeddings.create(
                    input=text,
                    model=self._model_name,
                    encoding_format="float"
                )
                return resp.data[0].embedding
            except Exception as e:
                last_error = e
                self._logger.error(f"Error embedding text: {e}")
                time.sleep(1)
        
        self._logger.error(f"Embedding text failed after 3 attempts: {last_error}")
        return None