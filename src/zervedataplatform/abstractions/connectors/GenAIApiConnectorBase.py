from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

from zervedataplatform.abstractions.types.models.LLMData import LLMData


class GenAIApiConnectorBase(ABC):
    def __init__(self, gen_ai_api_config: dict):
        self.__config = gen_ai_api_config

    @abstractmethod
    def configure_llm(self):
        """ This will configure our config """
        pass

    @abstractmethod
    def submit_data_prompt(self, prompt: str, llm_instructions: str) -> Union[Dict[str, Optional[LLMData]], dict]:
        """ This will submit prompt to LLM and get a response"""
        pass

    @abstractmethod
    def get_base_prompt(self, prompt: str, llm_instructions: str):
        pass

    def get_config(self):
        if self.__config:
            return self.__config
        else:
            raise Exception("No config found!")

    @abstractmethod
    def submit_general_prompt(self, prompt, llm_instructions, is_json: bool):
        pass
