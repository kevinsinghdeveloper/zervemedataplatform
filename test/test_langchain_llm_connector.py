import unittest
from unittest.mock import MagicMock, patch
from zervedataplatform.connectors.ai.LangChainLLMConnector import LangChainLLMConnector


class TestLangChainLLMConnector(unittest.TestCase):
    """Test cases for LangChainLLMConnector"""

    def setUp(self):
        """Set up test fixtures"""
        self.ollama_config = {
            "provider": "ollama",
            "model_name": "llama3.2",
            "base_url": "http://localhost:11434",
            "gen_config": {
                "temperature": 0.7,
                "max_tokens": 500,
                "top_p": 0.9,
                "top_k": 40
            },
            "format": "json"
        }

        self.openai_config = {
            "provider": "openai_compatible",
            "model_name": "gpt-4",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test-key",
            "gen_config": {
                "temperature": 0.7,
                "max_tokens": 500,
                "top_p": 0.9
            }
        }

        self.huggingface_config = {
            "provider": "huggingface",
            "model_name": "meta-llama/Llama-2-7b-chat-hf",
            "gen_config": {
                "temperature": 0.7,
                "max_tokens": 500
            }
        }

    # Test initialization
    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_init_ollama_provider(self, mock_chat_ollama):
        """Test initialization with Ollama provider"""
        mock_model = MagicMock()
        mock_chat_ollama.return_value = mock_model

        connector = LangChainLLMConnector(self.ollama_config)

        self.assertIsNotNone(connector)
        mock_chat_ollama.assert_called_once()

    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOpenAI')
    def test_init_openai_compatible_provider(self, mock_chat_openai):
        """Test initialization with OpenAI-compatible provider"""
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model

        connector = LangChainLLMConnector(self.openai_config)

        self.assertIsNotNone(connector)
        mock_chat_openai.assert_called_once()

    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_init_with_missing_config_fields(self, mock_chat_ollama):
        """Test initialization with minimal config"""
        mock_model = MagicMock()
        mock_chat_ollama.return_value = mock_model

        minimal_config = {
            "model_name": "llama3.2"
        }

        connector = LangChainLLMConnector(minimal_config)

        self.assertIsNotNone(connector)

    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_init_with_invalid_provider_raises_error(self, mock_chat_ollama):
        """Test initialization with unsupported provider raises ValueError"""
        invalid_config = {
            "provider": "invalid_provider",
            "model_name": "test-model"
        }

        with self.assertRaises(ValueError) as context:
            LangChainLLMConnector(invalid_config)

        self.assertIn("Unsupported provider", str(context.exception))

    # Test configuration
    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_get_config(self, mock_chat_ollama):
        """Test get_config returns the configuration"""
        mock_chat_ollama.return_value = MagicMock()
        connector = LangChainLLMConnector(self.ollama_config)

        config = connector.get_config()

        self.assertEqual(config, self.ollama_config)
        self.assertEqual(config["model_name"], "llama3.2")

    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_configure_llm_ollama(self, mock_chat_ollama):
        """Test configure_llm sets up Ollama correctly"""
        mock_model = MagicMock()
        mock_chat_ollama.return_value = mock_model

        connector = LangChainLLMConnector(self.ollama_config)

        mock_chat_ollama.assert_called_once_with(
            model="llama3.2",
            base_url="http://localhost:11434",
            temperature=0.7,
            num_predict=500,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            format="json",
            timeout=1000
        )

    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOpenAI')
    def test_configure_llm_openai_compatible(self, mock_chat_openai):
        """Test configure_llm sets up OpenAI-compatible API correctly"""
        mock_model = MagicMock()
        mock_chat_openai.return_value = mock_model

        connector = LangChainLLMConnector(self.openai_config)

        mock_chat_openai.assert_called_once_with(
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="sk-test-key",
            temperature=0.7,
            max_tokens=500,
            top_p=0.9
        )

    # Test prompt formatting
    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_get_base_prompt(self, mock_chat_ollama):
        """Test get_base_prompt formats prompts correctly"""
        mock_chat_ollama.return_value = MagicMock()
        connector = LangChainLLMConnector(self.ollama_config)

        system_prompt, user_prompt = connector.get_base_prompt(
            "What is AI?",
            "You are a helpful assistant."
        )

        self.assertEqual(system_prompt["role"], "system")
        self.assertEqual(system_prompt["content"], "You are a helpful assistant.")
        self.assertEqual(user_prompt["role"], "user")
        self.assertEqual(user_prompt["content"], "What is AI?")

    # Test prompt submission
    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_submit_general_prompt(self, mock_chat_ollama):
        """Test submit_general_prompt sends prompt to model"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is a test response"
        mock_model.invoke.return_value = mock_response
        mock_chat_ollama.return_value = mock_model

        connector = LangChainLLMConnector(self.ollama_config)
        response = connector.submit_general_prompt(
            "Test prompt",
            "Test instructions"
        )

        self.assertIsNotNone(response)
        self.assertEqual(response["message"]["content"], "This is a test response")
        self.assertEqual(response["model"], "llama3.2")
        self.assertEqual(response["provider"], "ollama")
        mock_model.invoke.assert_called_once()

    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_submit_general_prompt_with_json_flag(self, mock_chat_ollama):
        """Test submit_general_prompt with is_json=True adds JSON instruction"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"result": "test"}'
        mock_model.invoke.return_value = mock_response
        mock_chat_ollama.return_value = mock_model

        # Config without format="json"
        config = self.ollama_config.copy()
        config.pop("format")

        connector = LangChainLLMConnector(config)
        response = connector.submit_general_prompt(
            "Test prompt",
            "Test instructions",
            is_json=True
        )

        # Verify JSON instruction was added
        call_args = mock_model.invoke.call_args[0][0]
        system_message = call_args[0]
        self.assertIn("JSON", system_message.content)

    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_submit_general_prompt_handles_exception(self, mock_chat_ollama):
        """Test submit_general_prompt handles exceptions gracefully"""
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("API Error")
        mock_chat_ollama.return_value = mock_model

        connector = LangChainLLMConnector(self.ollama_config)

        with self.assertRaises(Exception) as context:
            connector.submit_general_prompt("Test prompt", "Test instructions")

        self.assertIn("API Error", str(context.exception))

    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_submit_general_prompt_retries_on_failure(self, mock_chat_ollama):
        """Test submit_general_prompt retries on transient failures"""
        mock_model = MagicMock()

        # First two calls fail, third succeeds
        mock_response = MagicMock()
        mock_response.content = "Success after retry"
        mock_model.invoke.side_effect = [
            ConnectionError("Network error"),
            TimeoutError("Timeout error"),
            mock_response
        ]
        mock_chat_ollama.return_value = mock_model

        connector = LangChainLLMConnector(self.ollama_config)
        response = connector.submit_general_prompt("Test prompt", "Test instructions")

        # Should succeed on third attempt
        self.assertEqual(response["message"]["content"], "Success after retry")
        # Verify it was called 3 times (2 failures + 1 success)
        self.assertEqual(mock_model.invoke.call_count, 3)

    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_submit_general_prompt_fails_after_max_retries(self, mock_chat_ollama):
        """Test submit_general_prompt fails after exhausting retries"""
        mock_model = MagicMock()

        # All calls fail
        mock_model.invoke.side_effect = ConnectionError("Network error")
        mock_chat_ollama.return_value = mock_model

        connector = LangChainLLMConnector(self.ollama_config)

        with self.assertRaises(ConnectionError) as context:
            connector.submit_general_prompt("Test prompt", "Test instructions")

        self.assertIn("Network error", str(context.exception))
        # Should have tried 3 times (max retries)
        self.assertEqual(mock_model.invoke.call_count, 3)

    # Test data prompt submission
    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_submit_data_prompt(self, mock_chat_ollama):
        """Test submit_data_prompt returns processed response"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"color": ["red", "blue"]}'
        mock_model.invoke.return_value = mock_response
        mock_chat_ollama.return_value = mock_model

        connector = LangChainLLMConnector(self.ollama_config)
        response_message, usage_data = connector.submit_data_prompt(
            "Extract colors",
            "You extract colors from text"
        )

        self.assertEqual(response_message, '{"color": ["red", "blue"]}')
        self.assertIsInstance(usage_data, dict)
        self.assertIn("prompt_tokens", usage_data)
        self.assertIn("completion_tokens", usage_data)
        self.assertIn("total_tokens", usage_data)

    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_submit_data_prompt_with_empty_response(self, mock_chat_ollama):
        """Test submit_data_prompt handles empty response"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = ""
        mock_model.invoke.return_value = mock_response
        mock_chat_ollama.return_value = mock_model

        connector = LangChainLLMConnector(self.ollama_config)
        response_message, usage_data = connector.submit_data_prompt(
            "Test prompt",
            "Test instructions"
        )

        self.assertEqual(response_message, "")
        self.assertIsInstance(usage_data, dict)

    # Test response processing
    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_process_and_extract_response(self, mock_chat_ollama):
        """Test __process_and_extract_response extracts data correctly"""
        mock_chat_ollama.return_value = MagicMock()
        connector = LangChainLLMConnector(self.ollama_config)

        response = {
            "message": {"content": "Test response"},
            "model": "llama3.2",
            "provider": "ollama"
        }

        # Use the private method through submit_data_prompt
        mock_model = connector._LangChainLLMConnector__model
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_model.invoke.return_value = mock_response

        response_message, usage_data = connector.submit_data_prompt(
            "Test",
            "Test"
        )

        self.assertEqual(response_message, "Test response")
        self.assertEqual(usage_data["prompt_tokens"], 0)
        self.assertEqual(usage_data["completion_tokens"], 0)
        self.assertEqual(usage_data["total_tokens"], 0)

    # Test different providers
    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_ollama_provider_with_different_models(self, mock_chat_ollama):
        """Test Ollama provider works with different model names"""
        mock_chat_ollama.return_value = MagicMock()

        for model_name in ["llama3.2", "llama3.2:3b", "mistral", "codellama"]:
            config = self.ollama_config.copy()
            config["model_name"] = model_name

            connector = LangChainLLMConnector(config)
            self.assertIsNotNone(connector)

    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOpenAI')
    def test_openai_compatible_provider_with_different_urls(self, mock_chat_openai):
        """Test OpenAI-compatible provider works with different base URLs"""
        mock_chat_openai.return_value = MagicMock()

        base_urls = [
            "https://api.openai.com/v1",
            "http://localhost:1234/v1",
            "http://192.168.1.100:8000/v1"
        ]

        for base_url in base_urls:
            config = self.openai_config.copy()
            config["base_url"] = base_url

            connector = LangChainLLMConnector(config)
            self.assertIsNotNone(connector)

    # Test error handling
    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_configure_llm_handles_connection_error(self, mock_chat_ollama):
        """Test configure_llm handles connection errors during initialization"""
        mock_chat_ollama.side_effect = Exception("Connection refused")

        with self.assertRaises(Exception):
            LangChainLLMConnector(self.ollama_config)

    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_missing_model_name_raises_error(self, mock_chat_ollama):
        """Test missing model_name in config raises KeyError"""
        invalid_config = {
            "provider": "ollama",
            "base_url": "http://localhost:11434"
        }

        with self.assertRaises(KeyError):
            LangChainLLMConnector(invalid_config)

    # Test integration scenarios
    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOllama')
    def test_full_workflow_ollama(self, mock_chat_ollama):
        """Test complete workflow from initialization to response"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"characteristics": ["durable", "lightweight"]}'
        mock_model.invoke.return_value = mock_response
        mock_chat_ollama.return_value = mock_model

        # Initialize
        connector = LangChainLLMConnector(self.ollama_config)

        # Get config
        config = connector.get_config()
        self.assertEqual(config["model_name"], "llama3.2")

        # Submit prompt
        response_message, usage_data = connector.submit_data_prompt(
            "Analyze this product",
            "You are a product analyst"
        )

        # Verify response
        self.assertIn("characteristics", response_message)
        self.assertIsInstance(usage_data, dict)

    @patch('zervedataplatform.connectors.ai.LangChainLLMConnector.ChatOpenAI')
    def test_full_workflow_openai_compatible(self, mock_chat_openai):
        """Test complete workflow with OpenAI-compatible provider"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"result": "success"}'
        mock_model.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_model

        # Initialize
        connector = LangChainLLMConnector(self.openai_config)

        # Submit prompt
        response_message, usage_data = connector.submit_data_prompt(
            "Test prompt",
            "Test instructions"
        )

        # Verify
        self.assertEqual(response_message, '{"result": "success"}')
        mock_model.invoke.assert_called_once()


if __name__ == '__main__':
    unittest.main()
