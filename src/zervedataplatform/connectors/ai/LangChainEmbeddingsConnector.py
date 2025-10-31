import pandas as pd
from typing import Union, List
from zervedataplatform.abstractions.connectors.EmbeddingsAIConnectorBase import EmbeddingsAiConnectorBase
from zervedataplatform.utils.Utility import Utility


class LangChainEmbeddingsConnector(EmbeddingsAiConnectorBase):
    """
    Connector for embeddings using various backends.

    Expected config format:
    {
        "provider": "sentence_transformers",  # "sentence_transformers", "openai", "huggingface", "remote_server"
        "model_name": "all-MiniLM-L6-v2",  # or "text-embedding-ada-002", "BAAI/bge-small-en-v1.5"
        "api_key": "sk-...",  # Required for OpenAI, optional for others
        "batch_size": 32,  # Optional, for batch processing
        "normalize": True,  # Optional, normalize embeddings to unit length
        "show_progress": False,  # Optional, show progress bar during encoding

        # For remote_server provider:
        "host": "localhost",  # Remote server host
        "port": 8080,  # Remote server port
        "base_url": "http://localhost:8080/embed",  # Or use full base_url instead of host+port
        "timeout": 30  # Request timeout in seconds
    }
    """

    def __init__(self, ai_api_config: dict):
        super().__init__(ai_api_config)
        self.__model = None
        self.__model_name = None
        self.__provider = None
        self.__batch_size = 32
        self.__normalize = True
        self.__show_progress = False
        self.__base_url = None
        self.__timeout = 30

        self.configure_model()

    def configure_model(self):
        """Configure the embeddings model based on provider"""
        config = self.get_config()
        self.__model_name = config.get("model_name", None)
        self.__provider = config.get("provider", None)
        self.__batch_size = config.get("batch_size", 32)
        self.__normalize = config.get("normalize", True)
        self.__show_progress = config.get("show_progress", False)
        self.__timeout = config.get("timeout", 30)

        if not self.__provider:
            Utility.error_log("Please specify an embedding provider")
            raise Exception("Please specify an embedding provider")

        try:
            if self.__provider == "sentence_transformers":
                if not self.__model_name:
                    raise ValueError("sentence_transformers provider requires 'model_name'")
                from sentence_transformers import SentenceTransformer
                self.__model = SentenceTransformer(self.__model_name)
                Utility.log(f"Successfully configured SentenceTransformer model: {self.__model_name}")

            elif self.__provider == "openai":
                if not self.__model_name:
                    raise ValueError("openai provider requires 'model_name'")
                from langchain_openai import OpenAIEmbeddings
                api_key = config.get("api_key")
                if not api_key:
                    raise ValueError("OpenAI provider requires 'api_key' in config")
                self.__model = OpenAIEmbeddings(
                    model=self.__model_name,
                    openai_api_key=api_key
                )
                Utility.log(f"Successfully configured OpenAI embeddings: {self.__model_name}")

            elif self.__provider == "huggingface":
                if not self.__model_name:
                    raise ValueError("huggingface provider requires 'model_name'")
                from langchain_huggingface import HuggingFaceEmbeddings
                self.__model = HuggingFaceEmbeddings(
                    model_name=self.__model_name
                )
                Utility.log(f"Successfully configured HuggingFace embeddings: {self.__model_name}")

            elif self.__provider == "remote_server":
                # Build base_url from host/port or use provided base_url
                if config.get("base_url"):
                    self.__base_url = config.get("base_url")
                elif config.get("host") and config.get("port"):
                    host = config.get("host")
                    port = config.get("port")
                    self.__base_url = f"http://{host}:{port}/embed"
                else:
                    raise ValueError("remote_server provider requires either 'base_url' or 'host' and 'port'")

                Utility.log(f"Successfully configured remote embedding server: {self.__base_url}")

            else:
                raise ValueError(f"Unsupported provider: {self.__provider}")

        except Exception as e:
            Utility.error_log(f"Error configuring embeddings model: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Core method: Get embeddings for a single text string.
        Provider-specific implementation.

        Args:
            text: Single text string to embed

        Returns:
            List of floats representing the embedding
        """
        try:
            if self.__provider == "sentence_transformers":
                embedding = self.__model.encode(
                    text,
                    show_progress_bar=False,
                    normalize_embeddings=self.__normalize
                )
                return embedding.tolist()

            elif self.__provider in ["openai", "huggingface"]:
                # LangChain embeddings interface
                return self.__model.embed_query(text)

            elif self.__provider == "remote_server":
                # Call remote embedding server
                import requests
                response = requests.post(
                    self.__base_url,
                    json={"text": text},
                    timeout=self.__timeout
                )
                response.raise_for_status()
                result = response.json()
                # Handle various response formats
                if isinstance(result, list):
                    return result
                elif "embedding" in result:
                    return result["embedding"]
                elif "embeddings" in result:
                    return result["embeddings"][0]
                else:
                    raise ValueError(f"Unexpected response format from server: {result}")

        except Exception as e:
            Utility.error_log(f"Error generating embedding: {e}")
            raise

    def get_embeddings(self, text: str) -> List[float]:
        """
        Get embeddings for a single text string.
        Delegates to embed_query (no duplication).

        Args:
            text: Single text string to embed

        Returns:
            List of floats representing the embedding
        """
        return self.embed_query(text)

    def generate_embeddings_batch(self, texts: Union[pd.Series, List[str]]) -> pd.Series:
        """
        Generate embeddings for a batch of texts (entire column).
        Uses get_embeddings for each text in the batch.

        Args:
            texts: Pandas Series or list of text strings

        Returns:
            Pandas Series where each element is a list of floats (embedding vector)
        """
        try:
            # Convert to list if it's a Series
            text_list = texts.tolist() if isinstance(texts, pd.Series) else texts

            # Generate embeddings using get_embeddings for each text
            embeddings = [self.get_embeddings(text) for text in text_list]

            return pd.Series(embeddings)

        except Exception as e:
            Utility.error_log(f"Error generating batch embeddings: {e}")
            raise
