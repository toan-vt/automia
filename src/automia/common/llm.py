from typing import Any, Optional, Type
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os
import time
from .logger import get_logger
from enum import Enum

class LLMClientProvider(Enum):
    GOOGLE = "google"
    OPENAI = "openai"
    VLLM = "vllm"

class LLMClient:
    def __init__(self, model_name: str, provider: LLMClientProvider, **kwargs):
        self._llm_logger = get_logger("llm")
        self._system_logger = get_logger("system")
        self._model_name = model_name
        self._temperature = kwargs.get("temperature", 0.7)
        self._top_p = kwargs.get("top_p", 0.8)
        self._top_k = kwargs.get("top_k", 20)
        self._min_p = kwargs.get("min_p", 0.0)
        self._presence_penalty = kwargs.get("presence_penalty", 0.75)
        self._max_tokens = kwargs.get("max_tokens", 8196*4)
        self._base_url = kwargs.get("base_url", None)
        
        if provider == LLMClientProvider.VLLM:
            if self._base_url is not None:
                self._llm = ChatOpenAI(model=self._model_name, base_url=self._base_url, api_key="dummy_key", temperature=self._temperature, top_p=self._top_p, presence_penalty=self._presence_penalty, max_tokens=self._max_tokens, extra_body={"top_k": self._top_k, "min_p": self._min_p})
            else:
                raise ValueError("vLLM API URL is not provided in the constructor")
        elif provider == LLMClientProvider.GOOGLE:
            api_key = None
            if "GOOGLE_API_KEY" in os.environ:
                api_key = os.environ["GOOGLE_API_KEY"]
            else:
                raise ValueError("Google API key is not provided in environment variables or in the constructor")
            self._llm = ChatGoogleGenerativeAI(model=self._model_name)
        elif provider == LLMClientProvider.OPENAI:
            if "OPENAI_API_KEY" in os.environ:
                api_key = os.environ["OPENAI_API_KEY"]
            else:
                raise ValueError("OpenAI API key is not provided in environment variables or in the constructor")
            if self._base_url is not None:
                self._llm = ChatOpenAI(model=self._model_name, api_key=api_key, base_url=self._base_url)
            else:
                self._llm = ChatOpenAI(model=self._model_name, api_key=api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        if not self._test():
            raise RuntimeError("LLM test failed")

    def _test(self) -> bool:
        try:
            resp = self._llm.invoke("Say Hello World.")
            self._system_logger.info("LLM test successful: %s", getattr(resp, "content", str(resp)))
            return True
        except Exception as e:
            self._system_logger.error("LLM test failed: %s", e)
            return False

    def invoke_structured(self, prompt_chain, inputs: dict, schema: Type[BaseModel], op_name: str) -> Optional[BaseModel]:
        chain = prompt_chain | self._llm.with_structured_output(schema)
        return self._safe_invoke(chain, inputs, op_name)

    def _safe_invoke(self, chain, inputs: dict, op_name: str) -> Optional[Any]:
        self._llm_logger.info(f"\n-------------------------------- {op_name} START --------------------------------")        
        self._llm_logger.info("Invoking %s with inputs: %s", op_name, inputs)
        for attempt in range(3): # 3 attempts
            try:
                out = chain.invoke(inputs)
                if out is not None:
                    self._llm_logger.info("%s returned: %s", op_name, out)
                    self._llm_logger.info(f"\n-------------------------------- {op_name} END --------------------------------\n")
                    return out
                self._system_logger.warning("%s returned None on attempt %d", op_name, attempt + 1)
            except Exception as e:
                self._system_logger.warning("%s failed on attempt %d: %s", op_name, attempt + 1, e)
            time.sleep(1)
        self._system_logger.error("%s failed after 3 attempts", op_name)
        self._llm_logger.info(f"\n-------------------------------- {op_name} END --------------------------------\n")
        return None
