from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, List
import pandas as pd

from zervedataplatform.abstractions.connectors.AIConnectorBase import AIConnectorBase
from zervedataplatform.abstractions.types.models.LLMData import LLMData


class EmbeddingsAiConnectorBase(AIConnectorBase, ABC):
    def __init__(self, ai_api_config: dict):
        super().__init__(ai_api_config)
        self.__config = ai_api_config

    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def generate_embeddings_batch(self, texts: Union[pd.Series, List[str]]) -> pd.Series:
        pass

    @abstractmethod
    def embed_query(self, text: str):
        pass

    @abstractmethod
    def get_dimensions(self) -> Union[int | None]:
        pass