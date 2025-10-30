from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

from zervedataplatform.abstractions.connectors.AIConnectorBase import AIConnectorBase
from zervedataplatform.abstractions.types.models.LLMData import LLMData


class EmbeddingsAiConnectorBase(AIConnectorBase, ABC):
    def __init__(self, ai_api_config: dict):
        super().__init__(ai_api_config)
        self.__config = ai_api_config

    @abstractmethod
    def get_embeddings(self):
        pass