from abc import ABC, abstractmethod


class AiApiConnectorBase(ABC):
    def __init__(self,  ai_api_config_path: str):
        self._config_path = ai_api_config_path

    @abstractmethod
    def initialize_api(self) -> None:
        pass
