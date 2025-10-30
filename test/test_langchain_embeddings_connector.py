import unittest
from unittest.mock import MagicMock, patch, Mock
import pandas as pd
import numpy as np
from zervedataplatform.connectors.ai.LangChainEmbeddingsConnector import LangChainEmbeddingsConnector

# Patch paths for lazy imports
SENTENCE_TRANSFORMER_PATH = 'sentence_transformers.SentenceTransformer'
OPENAI_EMBEDDINGS_PATH = 'langchain_openai.OpenAIEmbeddings'
HF_EMBEDDINGS_PATH = 'langchain_huggingface.HuggingFaceEmbeddings'
REQUESTS_PATH = 'requests.post'  # Mock requests.post specifically
PANDAS_UDF_PATH = 'pyspark.sql.functions.pandas_udf'


class TestLangChainEmbeddingsConnector(unittest.TestCase):
    """Test cases for LangChainEmbeddingsConnector"""

    def setUp(self):
        """Set up test fixtures"""
        self.sentence_transformers_config = {
            "provider": "sentence_transformers",
            "model_name": "all-MiniLM-L6-v2",
            "batch_size": 32,
            "normalize": True,
            "show_progress": False
        }

        self.openai_config = {
            "provider": "openai",
            "model_name": "text-embedding-ada-002",
            "api_key": "sk-test-key"
        }

        self.huggingface_config = {
            "provider": "huggingface",
            "model_name": "BAAI/bge-small-en-v1.5"
        }

        self.remote_server_config_host_port = {
            "provider": "remote_server",
            "host": "localhost",
            "port": 8080,
            "batch_size": 32,
            "timeout": 30
        }

        self.remote_server_config_base_url = {
            "provider": "remote_server",
            "base_url": "http://embedding-server:8080/api/v1/embed",
            "timeout": 30
        }

    # Test initialization - SentenceTransformers
    @patch(SENTENCE_TRANSFORMER_PATH)
    def test_init_sentence_transformers_provider(self, mock_sentence_transformer):
        """Test initialization with SentenceTransformers provider"""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        connector = LangChainEmbeddingsConnector(self.sentence_transformers_config)

        self.assertIsNotNone(connector)
        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")

    # Test initialization - OpenAI
    @patch(OPENAI_EMBEDDINGS_PATH)
    def test_init_openai_provider(self, mock_openai_embeddings):
        """Test initialization with OpenAI provider"""
        mock_model = MagicMock()
        mock_openai_embeddings.return_value = mock_model

        connector = LangChainEmbeddingsConnector(self.openai_config)

        self.assertIsNotNone(connector)
        mock_openai_embeddings.assert_called_once_with(
            model="text-embedding-ada-002",
            openai_api_key="sk-test-key"
        )

    # Test initialization - HuggingFace
    @patch(HF_EMBEDDINGS_PATH)
    def test_init_huggingface_provider(self, mock_hf_embeddings):
        """Test initialization with HuggingFace provider"""
        mock_model = MagicMock()
        mock_hf_embeddings.return_value = mock_model

        connector = LangChainEmbeddingsConnector(self.huggingface_config)

        self.assertIsNotNone(connector)
        mock_hf_embeddings.assert_called_once_with(
            model_name="BAAI/bge-small-en-v1.5"
        )

    # Test initialization - Remote Server
    def test_init_remote_server_provider_with_host_port(self):
        """Test initialization with remote_server provider using host and port"""
        connector = LangChainEmbeddingsConnector(self.remote_server_config_host_port)

        self.assertIsNotNone(connector)
        # Verify base_url was constructed
        self.assertEqual(connector._LangChainEmbeddingsConnector__base_url, "http://localhost:8080/embed")

    def test_init_remote_server_provider_with_base_url(self):
        """Test initialization with remote_server provider using base_url"""
        connector = LangChainEmbeddingsConnector(self.remote_server_config_base_url)

        self.assertIsNotNone(connector)
        self.assertEqual(
            connector._LangChainEmbeddingsConnector__base_url,
            "http://embedding-server:8080/api/v1/embed"
        )

    # Test configuration validation
    def test_init_without_provider_raises_error(self):
        """Test initialization without provider raises exception"""
        invalid_config = {
            "model_name": "test-model"
        }

        with self.assertRaises(Exception) as context:
            LangChainEmbeddingsConnector(invalid_config)

        self.assertIn("provider", str(context.exception).lower())

    @patch(SENTENCE_TRANSFORMER_PATH)
    def test_init_sentence_transformers_without_model_name_raises_error(self, mock_st):
        """Test SentenceTransformers without model_name raises error"""
        invalid_config = {
            "provider": "sentence_transformers"
        }

        with self.assertRaises(ValueError) as context:
            LangChainEmbeddingsConnector(invalid_config)

        self.assertIn("model_name", str(context.exception))

    def test_init_openai_without_api_key_raises_error(self):
        """Test OpenAI without api_key raises error"""
        invalid_config = {
            "provider": "openai",
            "model_name": "text-embedding-ada-002"
        }

        with self.assertRaises(ValueError) as context:
            LangChainEmbeddingsConnector(invalid_config)

        self.assertIn("api_key", str(context.exception))

    def test_init_remote_server_without_host_or_base_url_raises_error(self):
        """Test remote_server without host/port or base_url raises error"""
        invalid_config = {
            "provider": "remote_server"
        }

        with self.assertRaises(ValueError) as context:
            LangChainEmbeddingsConnector(invalid_config)

        self.assertIn("base_url", str(context.exception).lower())

    def test_init_invalid_provider_raises_error(self):
        """Test initialization with unsupported provider raises ValueError"""
        invalid_config = {
            "provider": "invalid_provider",
            "model_name": "test-model"
        }

        with self.assertRaises(ValueError) as context:
            LangChainEmbeddingsConnector(invalid_config)

        self.assertIn("Unsupported provider", str(context.exception))

    # Test embed_query - SentenceTransformers
    @patch(SENTENCE_TRANSFORMER_PATH)
    def test_embed_query_sentence_transformers(self, mock_sentence_transformer):
        """Test embed_query with SentenceTransformers provider"""
        mock_model = MagicMock()
        mock_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        mock_model.encode.return_value = mock_embedding
        mock_sentence_transformer.return_value = mock_model

        connector = LangChainEmbeddingsConnector(self.sentence_transformers_config)
        result = connector.embed_query("Hello world")

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        self.assertEqual(result, [0.1, 0.2, 0.3, 0.4])
        mock_model.encode.assert_called_once_with(
            "Hello world",
            show_progress_bar=False,
            normalize_embeddings=True
        )

    # Test embed_query - OpenAI
    @patch(OPENAI_EMBEDDINGS_PATH)
    def test_embed_query_openai(self, mock_openai_embeddings):
        """Test embed_query with OpenAI provider"""
        mock_model = MagicMock()
        mock_model.embed_query.return_value = [0.5, 0.6, 0.7, 0.8]
        mock_openai_embeddings.return_value = mock_model

        connector = LangChainEmbeddingsConnector(self.openai_config)
        result = connector.embed_query("Hello world")

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        self.assertEqual(result, [0.5, 0.6, 0.7, 0.8])
        mock_model.embed_query.assert_called_once_with("Hello world")

    # Test embed_query - Remote Server
    @patch(REQUESTS_PATH)
    def test_embed_query_remote_server_direct_array(self, mock_requests):
        """Test embed_query with remote_server provider returning direct array"""
        mock_response = MagicMock()
        mock_response.json.return_value = [0.9, 0.8, 0.7, 0.6]
        mock_response.raise_for_status = MagicMock()
        mock_requests.return_value = mock_response

        connector = LangChainEmbeddingsConnector(self.remote_server_config_host_port)
        result = connector.embed_query("Hello world")

        self.assertEqual(result, [0.9, 0.8, 0.7, 0.6])
        mock_requests.assert_called_once_with(
            "http://localhost:8080/embed",
            json={"text": "Hello world"},
            timeout=30
        )

    @patch(REQUESTS_PATH)
    def test_embed_query_remote_server_with_embedding_key(self, mock_requests):
        """Test embed_query with remote_server provider returning {"embedding": [...]}"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status = MagicMock()
        mock_requests.return_value = mock_response

        connector = LangChainEmbeddingsConnector(self.remote_server_config_host_port)
        result = connector.embed_query("Test text")

        self.assertEqual(result, [0.1, 0.2, 0.3])

    @patch(REQUESTS_PATH)
    def test_embed_query_remote_server_with_embeddings_key(self, mock_requests):
        """Test embed_query with remote_server provider returning {"embeddings": [[...]]}"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.4, 0.5, 0.6]]}
        mock_response.raise_for_status = MagicMock()
        mock_requests.return_value = mock_response

        connector = LangChainEmbeddingsConnector(self.remote_server_config_host_port)
        result = connector.embed_query("Test text")

        self.assertEqual(result, [0.4, 0.5, 0.6])

    # Test get_embeddings delegates to embed_query
    @patch(SENTENCE_TRANSFORMER_PATH)
    def test_get_embeddings_delegates_to_embed_query(self, mock_sentence_transformer):
        """Test get_embeddings correctly delegates to embed_query"""
        mock_model = MagicMock()
        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_model.encode.return_value = mock_embedding
        mock_sentence_transformer.return_value = mock_model

        connector = LangChainEmbeddingsConnector(self.sentence_transformers_config)
        result = connector.get_embeddings("Test text")

        self.assertEqual(result, [0.1, 0.2, 0.3])
        # Verify embed_query was called (via encode)
        mock_model.encode.assert_called_once()

    # Test generate_embeddings_batch - SentenceTransformers
    @patch(SENTENCE_TRANSFORMER_PATH)
    def test_generate_embeddings_batch_sentence_transformers(self, mock_sentence_transformer):
        """Test generate_embeddings_batch with SentenceTransformers provider"""
        mock_model = MagicMock()
        mock_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model

        connector = LangChainEmbeddingsConnector(self.sentence_transformers_config)

        texts = pd.Series(["Text 1", "Text 2", "Text 3"])
        result = connector.generate_embeddings_batch(texts)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], [0.1, 0.2, 0.3])
        self.assertEqual(result[1], [0.4, 0.5, 0.6])
        self.assertEqual(result[2], [0.7, 0.8, 0.9])

        mock_model.encode.assert_called_once_with(
            ["Text 1", "Text 2", "Text 3"],
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )

    # Test generate_embeddings_batch - OpenAI
    @patch(OPENAI_EMBEDDINGS_PATH)
    def test_generate_embeddings_batch_openai(self, mock_openai_embeddings):
        """Test generate_embeddings_batch with OpenAI provider"""
        mock_model = MagicMock()
        mock_model.embed_documents.return_value = [
            [0.1, 0.2],
            [0.3, 0.4]
        ]
        mock_openai_embeddings.return_value = mock_model

        connector = LangChainEmbeddingsConnector(self.openai_config)

        texts = ["Text 1", "Text 2"]
        result = connector.generate_embeddings_batch(texts)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 2)
        mock_model.embed_documents.assert_called_once_with(["Text 1", "Text 2"])

    # Test generate_embeddings_batch - Remote Server
    @patch(REQUESTS_PATH)
    def test_generate_embeddings_batch_remote_server_direct_array(self, mock_requests):
        """Test generate_embeddings_batch with remote_server returning direct array"""
        mock_response = MagicMock()
        mock_response.json.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_response.raise_for_status = MagicMock()
        mock_requests.return_value = mock_response

        connector = LangChainEmbeddingsConnector(self.remote_server_config_host_port)

        texts = pd.Series(["Text 1", "Text 2", "Text 3"])
        result = connector.generate_embeddings_batch(texts)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], [0.1, 0.2])

        mock_requests.assert_called_once_with(
            "http://localhost:8080/embed",
            json={"texts": ["Text 1", "Text 2", "Text 3"]},
            timeout=30
        )

    @patch(REQUESTS_PATH)
    def test_generate_embeddings_batch_remote_server_with_embeddings_key(self, mock_requests):
        """Test generate_embeddings_batch with remote_server returning {"embeddings": [...]}"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]]
        }
        mock_response.raise_for_status = MagicMock()
        mock_requests.return_value = mock_response

        connector = LangChainEmbeddingsConnector(self.remote_server_config_host_port)

        texts = ["Text 1", "Text 2"]
        result = connector.generate_embeddings_batch(texts)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[1], [0.3, 0.4])

    # Test get_pandas_udf for Spark integration
    @patch(SENTENCE_TRANSFORMER_PATH)
    @patch(PANDAS_UDF_PATH)
    def test_get_pandas_udf(self, mock_pandas_udf, mock_sentence_transformer):
        """Test get_pandas_udf returns a Pandas UDF function"""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        connector = LangChainEmbeddingsConnector(self.sentence_transformers_config)

        # Mock the pandas_udf decorator to return the function
        def mock_udf_decorator(return_type):
            def decorator(func):
                return func
            return decorator

        mock_pandas_udf.side_effect = mock_udf_decorator

        udf = connector.get_pandas_udf()

        self.assertIsNotNone(udf)
        # Verify pandas_udf was called
        mock_pandas_udf.assert_called_once()

    # Test error handling
    @patch(SENTENCE_TRANSFORMER_PATH)
    def test_embed_query_handles_exception(self, mock_sentence_transformer):
        """Test embed_query handles exceptions gracefully"""
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Model error")
        mock_sentence_transformer.return_value = mock_model

        connector = LangChainEmbeddingsConnector(self.sentence_transformers_config)

        with self.assertRaises(Exception) as context:
            connector.embed_query("Test text")

        self.assertIn("Model error", str(context.exception))

    @patch(REQUESTS_PATH)
    def test_embed_query_remote_server_handles_http_error(self, mock_requests):
        """Test embed_query with remote_server handles HTTP errors"""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500 Error")
        mock_requests.return_value = mock_response

        connector = LangChainEmbeddingsConnector(self.remote_server_config_host_port)

        with self.assertRaises(Exception) as context:
            connector.embed_query("Test text")

        self.assertIn("500", str(context.exception))

    @patch(REQUESTS_PATH)
    def test_embed_query_remote_server_handles_unexpected_response(self, mock_requests):
        """Test embed_query with remote_server handles unexpected response format"""
        mock_response = MagicMock()
        mock_response.json.return_value = {"unexpected_key": "unexpected_value"}
        mock_response.raise_for_status = MagicMock()
        mock_requests.return_value = mock_response

        connector = LangChainEmbeddingsConnector(self.remote_server_config_host_port)

        with self.assertRaises(ValueError) as context:
            connector.embed_query("Test text")

        self.assertIn("Unexpected response format", str(context.exception))

    # Test integration scenarios
    @patch(SENTENCE_TRANSFORMER_PATH)
    def test_full_workflow_sentence_transformers(self, mock_sentence_transformer):
        """Test complete workflow from initialization to batch embeddings"""
        mock_model = MagicMock()

        # Single embedding
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_sentence_transformer.return_value = mock_model

        connector = LangChainEmbeddingsConnector(self.sentence_transformers_config)

        # Test single embedding
        single_result = connector.get_embeddings("Single text")
        self.assertEqual(len(single_result), 3)

        # Test batch embeddings
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        batch_texts = pd.Series(["Text 1", "Text 2", "Text 3"])
        batch_result = connector.generate_embeddings_batch(batch_texts)

        self.assertEqual(len(batch_result), 3)
        self.assertIsInstance(batch_result, pd.Series)

    @patch(REQUESTS_PATH)
    def test_full_workflow_remote_server(self, mock_requests):
        """Test complete workflow with remote server"""
        # Single embedding response
        mock_response_single = MagicMock()
        mock_response_single.json.return_value = [0.1, 0.2, 0.3]
        mock_response_single.raise_for_status = MagicMock()

        # Batch embedding response
        mock_response_batch = MagicMock()
        mock_response_batch.json.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]]
        }
        mock_response_batch.raise_for_status = MagicMock()

        mock_requests.side_effect = [mock_response_single, mock_response_batch]

        connector = LangChainEmbeddingsConnector(self.remote_server_config_base_url)

        # Single embedding
        single_result = connector.embed_query("Test text")
        self.assertEqual(single_result, [0.1, 0.2, 0.3])

        # Batch embeddings
        batch_result = connector.generate_embeddings_batch(["Text 1", "Text 2"])
        self.assertEqual(len(batch_result), 2)

    # Test different model configurations
    @patch(SENTENCE_TRANSFORMER_PATH)
    def test_different_sentence_transformer_models(self, mock_sentence_transformer):
        """Test different SentenceTransformer model configurations"""
        mock_sentence_transformer.return_value = MagicMock()

        model_names = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2"
        ]

        for model_name in model_names:
            config = self.sentence_transformers_config.copy()
            config["model_name"] = model_name

            connector = LangChainEmbeddingsConnector(config)
            self.assertIsNotNone(connector)

    def test_remote_server_different_urls(self):
        """Test remote_server with different URL configurations"""
        test_configs = [
            {"provider": "remote_server", "host": "localhost", "port": 8080},
            {"provider": "remote_server", "host": "192.168.1.100", "port": 9000},
            {"provider": "remote_server", "base_url": "https://api.company.com/embed"}
        ]

        for config in test_configs:
            connector = LangChainEmbeddingsConnector(config)
            self.assertIsNotNone(connector)


if __name__ == '__main__':
    unittest.main()
