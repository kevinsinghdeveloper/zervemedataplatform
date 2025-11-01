from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

from zervedataplatform.abstractions.types.models.LLMData import LLMData


class AIConnectorBase(ABC):
    def __init__(self, ai_api_config: dict):
        self.__config = ai_api_config

    @abstractmethod
    def configure_model(self):
        """ This will configure our config """
        pass

    def get_config(self):
        if self.__config:
            return self.__config
        else:
            raise Exception("No config found!")
