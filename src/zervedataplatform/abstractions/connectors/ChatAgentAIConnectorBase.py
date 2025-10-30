from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

from zervedataplatform.abstractions.connectors.AIConnectorBase import AIConnectorBase
from zervedataplatform.abstractions.types.models.LLMData import LLMData


class ChatAgentAiConnectorBase(AIConnectorBase, ABC):
    def __init__(self, ai_api_config: dict):
        super().__init__(ai_api_config)
        self.__config = ai_api_config

    @abstractmethod
    def submit_data_prompt(self, prompt: str, llm_instructions: str) -> Union[Dict[str, Optional[LLMData]], dict]:
        """ This will submit prompt to LLM and get a response"""
        pass

    @abstractmethod
    def get_base_prompt(self, prompt: str, llm_instructions: str):
        pass

    @abstractmethod
    def submit_general_prompt(self, prompt, llm_instructions, is_json: bool):
        pass
