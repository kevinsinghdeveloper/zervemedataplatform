from typing import Any

from abstractions.connectors.GenAIApiConnectorBase import GenAIApiConnectorBase
from connectors.ai.GeminiGenAiConnector import GeminiGenAiConnector
from connectors.ai.OpenAiConnector import OpenAiConnector
from connectors.ai.LangChainLLMConnector import LangChainLLMConnector

class GenAIManager:
    def __init__(self, api_config):
        self.__api_config = api_config
        # move to a static so we can use n number of apis
        self.__gen_ai_connector = GenAIManager.get_api_connector(api_config)

    def submit_data_prompt(self, prompt: str, llm_instructions: str) -> tuple[Any, Any]:
        data, usage_data = self.__gen_ai_connector.submit_data_prompt(
            llm_instructions=llm_instructions,
            prompt=prompt
        )

        return data, usage_data

    @staticmethod
    def get_api_connector(api_config) -> GenAIApiConnectorBase:
        """
        Factory method to get the appropriate API connector.
        Auto-detects the connector type based on config structure.

        Supports:
        - LangChain connector (recommended) - detects by 'provider' key
          Supports multiple providers: OpenAI, Ollama, HuggingFace, etc.
        - Legacy OpenAI connector - detects by 'openai' nested key
        - Gemini connector - detects by 'gemini' nested key

        Config examples:
        1. LangChain (flat structure with 'provider'):
           {"provider": "ollama", "model_name": "llama3.2", ...}

        2. Legacy nested structure:
           {"openai": {"model_name": "gpt-4", ...}}
           {"gemini": {"model_name": "gemini-pro", ...}}
        """
        if not isinstance(api_config, dict):
            raise ValueError("api_config must be a dictionary")

        # Auto-detect based on config structure
        # Priority 1: Check for 'provider' key (LangChain flat config)
        if 'provider' in api_config:
            return LangChainLLMConnector(api_config)

        # Priority 2: Check for nested legacy configs
        if 'openai' in api_config:
            return OpenAiConnector(api_config['openai'])
        elif 'gemini' in api_config:
            return GeminiGenAiConnector(api_config['gemini'])

        # Priority 3: Assume it's a LangChain config without explicit provider
        # (for backward compatibility or configs that omit provider key)
        if 'model_name' in api_config:
            return LangChainLLMConnector(api_config)

        # If we can't determine the type, raise an error
        raise ValueError(
            "Unable to determine API connector type from config. "
            "Config should have either:\n"
            "  - 'provider' key for LangChain connector\n"
            "  - 'openai' or 'gemini' nested key for legacy connectors\n"
            f"Got config keys: {list(api_config.keys())}"
        )