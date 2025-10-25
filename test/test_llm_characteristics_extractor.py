import unittest
import json
import tempfile
import os
import shutil
from unittest.mock import Mock, patch

from data_platform.connectors.ai.llm_characteristics_extractor import LLMCharacteristicsExtractor
from data_platform.abstractions.types.models.LLMProductRequestData import (
    LLMProductRequestData,
    LLMCharacteristicOption
)


class TestLLMCharacteristicsExtractor(unittest.TestCase):
    """Test cases for LLMCharacteristicsExtractor"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for config files
        self.temp_dir = tempfile.mkdtemp()

        # Create LLM extractor config
        self.llm_ex_config = {
            "system_prompt": "You are a product analyst. Extract characteristics from products.",
            "examples": "Example: Product: Nike Shoes\nOutput: {\"color\": [\"red\", \"blue\"]}"
        }

        # Create category definition config
        self.category_def = {
            "footwear": {
                "color": {
                    "description": "The color of the product",
                    "is_multi": True,
                    "options": ["red", "blue", "green", "black", "white"]
                },
                "material": {
                    "description": "The material used",
                    "is_multi": True,
                    "options": ["leather", "synthetic", "canvas"]
                },
                "size": {
                    "description": "Available sizes",
                    "is_multi": True,
                    "options": ["7", "8", "9", "10", "11", "12"]
                }
            },
            "clothing": {
                "color": {
                    "description": "The color of the clothing",
                    "is_multi": True,
                    "options": ["red", "blue", "green", "black", "white"]
                }
            }
        }

        # Write category definition to file
        self.category_config_path = os.path.join(self.temp_dir, "category_config.json")
        with open(self.category_config_path, 'w') as f:
            json.dump(self.category_def, f)

        # Gen AI API config
        self.gen_ai_config = {
            "provider": "ollama",
            "model_name": "llama3.2",
            "base_url": "http://localhost:11434",
            "gen_config": {
                "temperature": 0.7,
                "max_tokens": 2000
            },
            "format": "json"
        }

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)

    @patch('data_platform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('data_platform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
    def test_initialization(self, mock_read_json, mock_llm_connector):
        """Test LLMCharacteristicsExtractor initialization"""
        mock_read_json.return_value = self.category_def

        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=[self.category_config_path],
            gen_ai_api_config=self.gen_ai_config
        )

        # Verify LangChain connector was initialized
        mock_llm_connector.assert_called_once_with(gen_ai_api_config=self.gen_ai_config)

        # Verify category config was loaded
        mock_read_json.assert_called_once_with(self.category_config_path)

    @patch('data_platform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('data_platform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
    def test_get_all_characteristics_with_valid_product(self, mock_read_json, mock_llm_connector):
        """Test get_all_characteristics extracts characteristics correctly"""
        mock_read_json.return_value = self.category_def

        # Mock LLM response
        mock_llm = Mock()
        mock_llm.submit_data_prompt.return_value = (
            '{"color": ["red", "blue"]}',
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        mock_llm_connector.return_value = mock_llm

        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=[self.category_config_path],
            gen_ai_api_config=self.gen_ai_config
        )

        # Create product data
        prod_data = LLMProductRequestData(
            super_category="footwear",
            product_title="Nike Air Max",
            product_information="Red and blue running shoes made of synthetic material"
        )

        result = extractor.get_all_characteristics(prod_data)

        # Verify characteristics were extracted
        self.assertIsInstance(result, dict)
        self.assertIn("color", result)

        # Verify LLM was called for each characteristic
        self.assertEqual(mock_llm.submit_data_prompt.call_count, 3)  # color, material, size

    @patch('data_platform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('data_platform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
    def test_get_all_characteristics_with_no_product_information(self, mock_read_json, mock_llm_connector):
        """Test get_all_characteristics with no product information"""
        mock_read_json.return_value = self.category_def

        mock_llm = Mock()
        mock_llm_connector.return_value = mock_llm

        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=[self.category_config_path],
            gen_ai_api_config=self.gen_ai_config
        )

        # Product without information
        prod_data = LLMProductRequestData(
            super_category="footwear",
            product_title="Nike Air Max",
            product_information=None
        )

        result = extractor.get_all_characteristics(prod_data)

        # Should return empty dict for all characteristics
        self.assertEqual(result, {'color': [], 'material': [], 'size': []})

    @patch('data_platform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('data_platform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
    def test_get_all_characteristics_with_invalid_category(self, mock_read_json, mock_llm_connector):
        """Test get_all_characteristics with category not in config"""
        mock_read_json.return_value = self.category_def

        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=[self.category_config_path],
            gen_ai_api_config=self.gen_ai_config
        )

        # Category not in config
        prod_data = LLMProductRequestData(
            super_category="electronics",  # Not in config
            product_title="Laptop",
            product_information="A high-end laptop"
        )

        result = extractor.get_all_characteristics(prod_data)

        # Should return empty list
        self.assertEqual(result, [])

    @patch('data_platform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('data_platform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
    def test_llm_response_with_markdown_code_blocks(self, mock_read_json, mock_llm_connector):
        """Test parsing LLM response with markdown code blocks"""
        mock_read_json.return_value = self.category_def

        # Mock LLM response with markdown
        mock_llm = Mock()
        mock_llm.submit_data_prompt.return_value = (
            '```json\n{"color": ["red", "blue"]}\n```',
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        mock_llm_connector.return_value = mock_llm

        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=[self.category_config_path],
            gen_ai_api_config=self.gen_ai_config
        )

        prod_data = LLMProductRequestData(
            super_category="footwear",
            product_title="Nike Air Max",
            product_information="Red and blue shoes"
        )

        result = extractor.get_all_characteristics(prod_data)

        # Should still parse correctly
        self.assertIn("color", result)

    @patch('data_platform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('data_platform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
    def test_llm_response_with_incomplete_json(self, mock_read_json, mock_llm_connector):
        """Test parsing incomplete JSON response from LLM"""
        mock_read_json.return_value = self.category_def

        # Mock LLM with incomplete JSON
        mock_llm = Mock()
        mock_llm.submit_data_prompt.return_value = (
            '{"color": ["red", "blue"',  # Missing closing brackets
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        mock_llm_connector.return_value = mock_llm

        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=[self.category_config_path],
            gen_ai_api_config=self.gen_ai_config
        )

        prod_data = LLMProductRequestData(
            super_category="footwear",
            product_title="Nike Air Max",
            product_information="Red and blue shoes"
        )

        result = extractor.get_all_characteristics(prod_data)

        # Should attempt to fix and parse, or return empty
        self.assertIsInstance(result, dict)

    @patch('data_platform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('data_platform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
    def test_llm_response_with_empty_string(self, mock_read_json, mock_llm_connector):
        """Test handling empty response from LLM"""
        mock_read_json.return_value = self.category_def

        # Mock LLM with empty response
        mock_llm = Mock()
        mock_llm.submit_data_prompt.return_value = (
            '',
            {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}
        )
        mock_llm_connector.return_value = mock_llm

        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=[self.category_config_path],
            gen_ai_api_config=self.gen_ai_config
        )

        prod_data = LLMProductRequestData(
            super_category="footwear",
            product_title="Nike Air Max",
            product_information="Red and blue shoes"
        )

        result = extractor.get_all_characteristics(prod_data)

        # Should handle gracefully and return empty lists
        for char in result.values():
            self.assertEqual(char, [])

    @patch('data_platform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('data_platform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
    def test_llm_response_as_dict_with_lists(self, mock_read_json, mock_llm_connector):
        """Test parsing dict response with list values"""
        mock_read_json.return_value = self.category_def

        # Mock LLM response as dict
        mock_llm = Mock()
        mock_llm.submit_data_prompt.return_value = (
            '{"colors": ["red", "blue"], "sizes": ["10", "11"]}',
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        mock_llm_connector.return_value = mock_llm

        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=[self.category_config_path],
            gen_ai_api_config=self.gen_ai_config
        )

        prod_data = LLMProductRequestData(
            super_category="footwear",
            product_title="Nike Air Max",
            product_information="Red and blue shoes"
        )

        result = extractor.get_all_characteristics(prod_data)

        # Should flatten dict values into lists
        self.assertIsInstance(result, dict)

    @patch('data_platform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('data_platform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
    def test_llm_response_with_single_quotes(self, mock_read_json, mock_llm_connector):
        """Test parsing response with Python single quotes (not valid JSON)"""
        mock_read_json.return_value = self.category_def

        # Mock LLM response with single quotes
        mock_llm = Mock()
        mock_llm.submit_data_prompt.return_value = (
            "{'color': ['red', 'blue']}",  # Python dict notation
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )
        mock_llm_connector.return_value = mock_llm

        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=[self.category_config_path],
            gen_ai_api_config=self.gen_ai_config
        )

        prod_data = LLMProductRequestData(
            super_category="footwear",
            product_title="Nike Air Max",
            product_information="Red and blue shoes"
        )

        result = extractor.get_all_characteristics(prod_data)

        # Should use ast.literal_eval as fallback
        self.assertIsInstance(result, dict)

    @patch('data_platform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('data_platform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
    def test_get_config_returns_serializable_config(self, mock_read_json, mock_llm_connector):
        """Test get_config returns configuration for serialization"""
        mock_read_json.return_value = self.category_def

        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=[self.category_config_path],
            gen_ai_api_config=self.gen_ai_config
        )

        config = extractor.get_config()

        # Verify config structure
        self.assertIn('config', config)
        self.assertIn('category_maps', config)
        self.assertIn('ai_config', config)

        self.assertEqual(config['config'], self.llm_ex_config)
        self.assertEqual(config['category_maps'], [self.category_config_path])
        self.assertEqual(config['ai_config'], self.gen_ai_config)

    @patch('data_platform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('data_platform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
    def test_transform_data_for_prompt_static_method(self, mock_read_json, mock_llm_connector):
        """Test transform_data_for_prompt formats data correctly"""
        prod_data = LLMProductRequestData(
            super_category="footwear",
            product_title="Nike Air Max",
            product_information="Red and blue running shoes"
        )

        char_data = LLMCharacteristicOption(
            characteristic="color",
            description="The color of the product",
            is_multi=True,
            options=["red", "blue", "green"]
        )

        prompt = LLMCharacteristicsExtractor.transform_data_for_prompt(prod_data, char_data)

        # Verify prompt contains product data
        self.assertIn("footwear", prompt)
        self.assertIn("Nike Air Max", prompt)
        self.assertIn("color", prompt)

    @patch('data_platform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('data_platform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
    def test_multiple_category_configs(self, mock_read_json, mock_llm_connector):
        """Test loading multiple category configuration files"""
        # Create second config file
        second_config = {
            "accessories": {
                "type": {
                    "description": "Type of accessory",
                    "is_multi": False,
                    "options": ["watch", "belt", "hat"]
                }
            }
        }

        second_config_path = os.path.join(self.temp_dir, "accessories_config.json")
        with open(second_config_path, 'w') as f:
            json.dump(second_config, f)

        # Mock reading both files
        mock_read_json.side_effect = [self.category_def, second_config]

        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=[self.category_config_path, second_config_path],
            gen_ai_api_config=self.gen_ai_config
        )

        # Verify both configs were loaded
        self.assertEqual(mock_read_json.call_count, 2)

    @patch('data_platform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('data_platform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
    def test_llm_exception_handling(self, mock_read_json, mock_llm_connector):
        """Test that LLM exceptions are raised (not caught at this level)"""
        mock_read_json.return_value = self.category_def

        # Mock LLM to raise exception
        mock_llm = Mock()
        mock_llm.submit_data_prompt.side_effect = Exception("LLM API Error")
        mock_llm_connector.return_value = mock_llm

        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=[self.category_config_path],
            gen_ai_api_config=self.gen_ai_config
        )

        prod_data = LLMProductRequestData(
            super_category="footwear",
            product_title="Nike Air Max",
            product_information="Red shoes"
        )

        # LLM exceptions should propagate up
        # The code doesn't catch exceptions during the LLM call itself
        with self.assertRaises(Exception) as context:
            extractor.get_all_characteristics(prod_data)

        self.assertIn("LLM API Error", str(context.exception))


if __name__ == '__main__':
    unittest.main()