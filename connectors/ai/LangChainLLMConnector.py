from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from data_platform.abstractions.connectors.GenAIApiConnectorBase import GenAIApiConnectorBase
from data_platform.utils.Utility import Utility


class LangChainLLMConnector(GenAIApiConnectorBase):
    """
    Connector for LLMs using LangChain. Supports multiple backends:
    - OpenAI (gpt-4, gpt-3.5-turbo, etc.)
    - Ollama (llama3.2, mistral, etc.)
    - OpenAI-compatible APIs (LM Studio, vLLM, LocalAI)
    - HuggingFace models
    - And more through LangChain

    Expected config format:
    {
        "provider": "ollama",  # "openai", "ollama", "openai_compatible", "huggingface"
        "model_name": "llama3.2",  # or "gpt-4", "llama3.2:3b", "mistral", etc.
        "base_url": "http://localhost:11434",  # Optional, provider-specific
        "api_key": "sk-...",  # Required for OpenAI, optional for others
        "gen_config": {
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 0.9,
            "top_k": 40
        },
        "format": "json"  # Optional, for structured output
    }
    """

    def __init__(self, gen_ai_api_config: dict):
        super().__init__(gen_ai_api_config)
        self.__model = None
        self.__model_name = None
        self.__provider = None
        self.__gen_config = None
        self.__format = None

        self.configure_llm()

    def configure_llm(self):
        """Configure the LangChain LLM based on provider"""
        config = self.get_config()
        self.__model_name = config["model_name"]
        self.__provider = config.get("provider", "ollama")
        self.__gen_config = config.get('gen_config', {})
        self.__format = config.get("format", None)

        try:
            if self.__provider == "ollama":
                # Use ChatOllama for chat models
                self.__model = ChatOllama(
                    model=self.__model_name,
                    base_url=config.get("base_url", "http://localhost:11434"),
                    temperature=self.__gen_config.get("temperature", 0.7),
                    num_predict=self.__gen_config.get("max_tokens", 500),
                    top_p=self.__gen_config.get("top_p", 0.9),
                    top_k=self.__gen_config.get("top_k", 40),
                    repeat_penalty=self.__gen_config.get("repeat_penalty", 1.1),  # Add this
                    format=self.__format,
                    timeout=1000,  # Add timeout for long responses
                )
                Utility.log(f"Successfully configured Ollama model: {self.__model_name}")
            elif self.__provider == "openai_compatible":
                # Use ChatOpenAI for OpenAI-compatible APIs (LM Studio, vLLM, etc.)
                self.__model = ChatOpenAI(
                    model=self.__model_name,
                    base_url=config.get("base_url"),
                    api_key=config.get("api_key", "not-needed"),
                    temperature=self.__gen_config.get("temperature", 0.7),
                    max_tokens=self.__gen_config.get("max_tokens", 500),
                    top_p=self.__gen_config.get("top_p", 0.9)
                )
                Utility.log(f"Successfully configured OpenAI-compatible model: {self.__model_name}")

            elif self.__provider == "huggingface":
                # Use HuggingFace models
                from langchain_community.llms import HuggingFacePipeline
                self.__model = HuggingFacePipeline.from_model_id(
                    model_id=self.__model_name,
                    task="text-generation",
                    model_kwargs={
                        "temperature": self.__gen_config.get("temperature", 0.7),
                        "max_length": self.__gen_config.get("max_tokens", 500),
                    }
                )
                Utility.log(f"Successfully configured HuggingFace model: {self.__model_name}")

            else:
                raise ValueError(f"Unsupported provider: {self.__provider}")

        except Exception as e:
            Utility.error_log(f"Error configuring LLM: {e}")
            raise

    def get_base_prompt(self, prompt: str, llm_instructions: str):
        """Format prompts for LangChain"""
        system_prompt = {"role": "system", "content": llm_instructions}
        user_prompt = {"role": "user", "content": prompt}
        return system_prompt, user_prompt

    def submit_general_prompt(self, prompt: str, llm_instructions: str, is_json: bool = False):
        """Submit a prompt using LangChain"""
        try:
            # Create messages for chat models
            messages = [
                SystemMessage(content=llm_instructions),
                HumanMessage(content=prompt)
            ]

            # Invoke the model
            if is_json and self.__format != "json":
                # Append JSON instruction to the system message if not already configured
                messages[0] = SystemMessage(
                    content=f"{llm_instructions}\n\nIMPORTANT: Return your response as valid JSON."
                )

            response = self.__model.invoke(messages)

            # Return in a format consistent with the original implementation
            return {
                "message": {"content": response.content},
                "model": self.__model_name,
                "provider": self.__provider
            }

        except Exception as e:
            Utility.error_log(f"Error submitting prompt to {self.__provider}: {e}")
            raise

    def __process_and_extract_response(self, response: dict):
        """Extract response message and usage data"""
        response_message = response.get("message", {}).get("content", "")

        # Usage data format (LangChain doesn't always provide this)
        usage_data = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        return response_message, usage_data

    def submit_data_prompt(self, prompt: str, llm_instructions: str):
        """Submit a data prompt and return the processed response"""
        response = self.submit_general_prompt(prompt, llm_instructions, is_json=True)

        response_message, usage_data = self.__process_and_extract_response(response)

        return response_message, usage_data
