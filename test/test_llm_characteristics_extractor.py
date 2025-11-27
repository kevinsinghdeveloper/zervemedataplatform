import unittest
import json
import tempfile
import os
import shutil
import time
from unittest.mock import Mock, patch
from collections import namedtuple

from zervedataplatform.connectors.ai.llm_characteristics_extractor import LLMCharacteristicsExtractor
from zervedataplatform.abstractions.types.models.LLMProductRequestData import (
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

    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
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

    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
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

    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
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

    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
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

    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
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

    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
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

    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
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

    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
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

    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
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

    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
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

    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
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

    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
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

    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.LangChainLLMConnector')
    @patch('zervedataplatform.connectors.ai.llm_characteristics_extractor.Utility.read_in_json_file')
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


class TestLLMCharacteristicsExtractorFunctional(unittest.TestCase):
    """
    Functional tests for LLMCharacteristicsExtractor.
    These tests hit a live llama model running locally.

    Requirements:
    - Ollama running at http://localhost:11434
    - llama3.2 model available

    To run only these tests:
        python -m pytest test_llm_characteristics_extractor.py::TestLLMCharacteristicsExtractorFunctional -v

    To skip these tests, use:
        python -m pytest test_llm_characteristics_extractor.py -k "not Functional"
    """

    def setUp(self):
        """Set up test fixtures for functional tests"""
        # LLM extractor config
        self.llm_ex_config = {
            "system_prompt": "You are a Product Characteristics Extractor Tool.\n\nPurpose:\nExtract product characteristics from product information and return them in strict JSON format.\n\nSTRICT RULES:\n1. Return ONLY valid JSON - a dictionary mapping characteristic names to value arrays\n2. Format: {\"characteristic_name\": [\"value1\", \"value2\"], \"other_char\": [\"value\"]}\n3. Use double quotes only - NO single quotes\n4. NO markdown, explanations, or extra text outside the JSON\n5. For multi-value characteristics (is_multi=true): include ALL matching values\n6. For single-value characteristics (is_multi=false): include ONLY the most relevant value\n7. If no valid value found for a characteristic: use empty array []\n8. If product information is missing/null/\"N/A\": return empty arrays for all characteristics\n\nVALUE SELECTION LOGIC (CRITICAL):\n9. IF options list is provided and non-empty: you MUST choose ONLY from those options - NO EXCEPTIONS\n10. IF options list is empty or contains only \"<open_ended>\": freely determine/extract relevant values from product information\n11. NEVER extract values outside the provided options when options are specified\n12. Match values case-insensitively from options\n13. When product contains descriptive terms (e.g., \"organic cotton\", \"navy blue\"), map them to the closest exact option (e.g., \"cotton\", \"blue\")\n14. For boolean options (true/false), return string representation: [\"true\"] or [\"false\"]\n15. When information is ambiguous, prioritize specs > description > other fields\n16. REMEMBER: Your output values must be EXACT MATCHES to the provided options list - do not add qualifiers, prefixes, or suffixes",
            "examples": "Examples:\n\nExample 1 - Strict Options (must choose from list):\nInput:\nProduct Data:\n   {\"product_title\": \"Running Shoes\", \"product_information\": \"Navy blue mesh athletic shoes\"}\nCharacteristic Information:\n   {\"characteristic\": \"color\", \"is_multi\":True, \"options\": [\"red\", \"blue\", \"black\", \"white\"]}\n\nExpected Output:\n{\"color\": [\"blue\"]}\n\nExample 2 - Open-ended (free extraction):\nInput:\nProduct Data:\n   {\"product_title\": \"Running Shoes\", \"product_information\": \"Navy blue mesh athletic shoes\"}\nCharacteristic Information:\n   {\"characteristic\": \"color\", \"is_multi\":True, \"options\": [\"<open_ended>\"]}\n\nExpected Output:\n{\"color\": [\"navy blue\"]}\n\nExample 3 - Fuzzy Matching (descriptive terms must map to exact options):\nInput:\nProduct Data:\n   {\"product_title\": \"T-Shirt\", \"product_information\": \"Made from 100% organic cotton\"}\nCharacteristic Information:\n   {\"characteristic\": \"material\", \"is_multi\":True, \"options\": [\"cotton\", \"polyester\", \"wool\"]}\n\nExpected Output:\n{\"material\": [\"cotton\"]}\nNOTE: Product says \"organic cotton\" but you MUST return \"cotton\" since that's the exact option available.\n\nExample 4 - WRONG Output (shows what NOT to do):\nInput:\nProduct Data:\n   {\"product_title\": \"Organic Shirt\", \"product_information\": \"100% organic cotton fabric\"}\nCharacteristic Information:\n   {\"characteristic\": \"material\", \"is_multi\":True, \"options\": [\"cotton\", \"polyester\", \"silk\"]}\n\nWRONG Output (DO NOT DO THIS):\n{\"material\": [\"organic cotton\"]}  ❌ WRONG - \"organic cotton\" is NOT in the options list!\n\nCORRECT Output:\n{\"material\": [\"cotton\"]}  ✓ CORRECT - \"cotton\" is in the options list\n\nExample 5 - Multi-Characteristic with Mixed Constraints:\nInput:\nProduct Data:\n   {\"super_category\": \"footwear\", \"product_title\": \"Nike Air Max\", \"product_information\": \"Black mesh running shoes with gel cushioning\"}\nCharacteristic Information:\n   - color: {\"is_multi\":True, \"options\": [\"<open_ended>\"]}\n   - material: {\"is_multi\":True, \"options\": [\"leather\", \"mesh\", \"synthetic\"]}\n   - uses: {\"is_multi\":True, \"options\": [\"running\", \"walking\", \"casual\"]}\n\nExpected Output:\n{\"color\": [\"black\"], \"material\": [\"mesh\"], \"uses\": [\"running\"]}\n\nExample 6 - Boolean Characteristic:\nInput:\nProduct Data:\n   {\"product_title\": \"Adidas Ultraboost\"}\nCharacteristic Information:\n   {\"characteristic\": \"popular_brand\", \"is_multi\":False, \"options\": [true,False]}\n\nExpected Output:\n{\"popular_brand\": [\"true\"]}\n\n===========================================\nFINAL REMINDER BEFORE YOU START:\nWhen the options list is provided (not \"<open_ended>\"):\n- SCAN the options list\n- ONLY use values from that exact list\n- If product says \"organic cotton\" and options are [\"cotton\", \"polyester\"], return \"cotton\" (not \"organic cotton\")\n- If product says \"navy blue\" and options are [\"red\", \"blue\", \"black\"], return \"blue\" (not \"navy blue\")\n- STRIP all adjectives/qualifiers and match to the base option\n==========================================="
        }

        # Category definition config with multiple categories
        # In real-world usage, this would be loaded from JSON files using Utility.read_in_json_file()
        # and passed as a list of dictionaries to the constructor
        self.category_def = [
    {
      "footwear": {
        "color": {
          "description": "The primary color or color combinations of the footwear.",
          "is_multi": True,
          "options": [
            "<open_ended>"
          ]
        },
        "uses": {
          "description": "The intended activities or purposes for which the footwear is designed.",
          "is_multi": True,
          "options": [
            "hiking",
            "running",
            "basketball",
            "casual",
            "formal",
            "training",
            "walking",
            "work"
          ]
        },
        "comfort_feel": {
          "description": "The level of comfort and sensation experienced while wearing the footwear.",
          "is_multi": True,
          "options": [
            "soft",
            "firm",
            "cushioned",
            "supportive",
            "lightweight"
          ]
        },
        "material": {
          "description": "The primary material used in the construction of the footwear.",
          "is_multi": True,
          "options": [
            "leather",
            "suede",
            "mesh",
            "synthetic",
            "canvas",
            "rubber",
            "knit"
          ]
        },
        "popular_brand": {
          "description": "Indicates whether the footwear belongs to a well-known or popular brand.",
          "is_multi": False,
          "options": [
           True,
           False
          ]
        },
        "shape": {
          "description": "The general structure or design of the footwear.",
          "is_multi": True,
          "options": [
            "low top",
            "high top",
            "mid top",
            "slip-on",
            "lace-up",
            "boot"
          ]
        },
        "style": {
          "description": "The aesthetic and functional style of the footwear.",
          "is_multi": True,
          "options": [
            "performance",
            "luxury",
            "retro",
            "minimalist",
            "streetwear",
            "athleisure",
            "formal"
          ]
        },
        "heel_height": {
          "description": "The height of the heel, which affects posture and style.",
          "is_multi": True,
          "options": [
            "high",
            "medium",
            "low",
            "flat",
            "platform"
          ]
        }
      }
    },
    {
      "smartphones": {
        "brand": {
          "description": "The manufacturer or brand of the smartphone.",
          "is_multi": False,
          "options": [
            "Apple",
            "Samsung",
            "Google",
            "OnePlus",
            "Xiaomi",
            "Motorola",
            "Nokia",
            "Sony",
            "LG",
            "Other"
          ]
        },
        "operating_system": {
          "description": "The operating system running on the smartphone.",
          "is_multi": False,
          "options": [
            "iOS",
            "Android"
          ]
        },
        "screen_size": {
          "description": "The display size category of the smartphone.",
          "is_multi": False,
          "options": [
            "small (under 6 inches)",
            "medium (6-6.5 inches)",
            "large (6.5-7 inches)",
            "extra large (over 7 inches)"
          ]
        },
        "storage_capacity": {
          "description": "Internal storage capacity of the smartphone.",
          "is_multi": True,
          "options": [
            "32GB",
            "64GB",
            "128GB",
            "256GB",
            "512GB",
            "1TB"
          ]
        },
        "camera_quality": {
          "description": "The camera system quality level.",
          "is_multi": False,
          "options": [
            "basic",
            "good",
            "excellent",
            "professional"
          ]
        },
        "5g_capable": {
          "description": "Whether the smartphone supports 5G connectivity.",
          "is_multi": False,
          "options": [
           True,
           False
          ]
        },
        "battery_life": {
          "description": "Expected battery performance category.",
          "is_multi": False,
          "options": [
            "short (under 3000mAh)",
            "average (3000-4500mAh)",
            "long (4500-5500mAh)",
            "extra long (over 5500mAh)"
          ]
        },
        "price_tier": {
          "description": "The market positioning and price range of the device.",
          "is_multi": False,
          "options": [
            "budget",
            "mid-range",
            "flagship",
            "ultra-premium"
          ]
        }
      }
    },
    {
      "televisions": {
        "screen_size": {
          "description": "The diagonal screen size of the television.",
          "is_multi": False,
          "options": [
            "32 inch",
            "40 inch",
            "43 inch",
            "50 inch",
            "55 inch",
            "65 inch",
            "75 inch",
            "77 inch",
            "85 inch",
            "over 85 inch"
          ]
        },
        "resolution": {
          "description": "The display resolution of the television.",
          "is_multi": False,
          "options": [
            "HD (720p)",
            "Full HD (1080p)",
            "4K (2160p)",
            "8K (4320p)"
          ]
        },
        "display_technology": {
          "description": "The type of display panel technology used.",
          "is_multi": False,
          "options": [
            "LED",
            "QLED",
            "OLED",
            "Mini-LED",
            "Micro-LED",
            "LCD"
          ]
        },
        "smart_tv": {
          "description": "Whether the TV has smart/streaming capabilities built-in.",
          "is_multi": False,
          "options": [
           True,
           False
          ]
        },
        "hdr_support": {
          "description": "High Dynamic Range formats supported by the TV.",
          "is_multi": True,
          "options": [
            "HDR10",
            "HDR10+",
            "Dolby Vision",
            "HLG",
            "none"
          ]
        },
        "refresh_rate": {
          "description": "The screen refresh rate in Hz.",
          "is_multi": False,
          "options": [
            "60Hz",
            "120Hz",
            "144Hz",
            "240Hz"
          ]
        },
        "brand": {
          "description": "The manufacturer or brand of the television.",
          "is_multi": False,
          "options": [
            "Samsung",
            "LG",
            "Sony",
            "TCL",
            "Hisense",
            "Vizio",
            "Panasonic",
            "Philips",
            "Other"
          ]
        }
      }
    },
    {
      "computers": {
        "form_factor": {
          "description": "The physical form and portability of the computer.",
          "is_multi": False,
          "options": [
            "laptop",
            "desktop",
            "all-in-one",
            "mini pc",
            "workstation"
          ]
        },
        "processor_brand": {
          "description": "The CPU manufacturer.",
          "is_multi": False,
          "options": [
            "Intel",
            "AMD",
            "Apple Silicon",
            "Other"
          ]
        },
        "processor_tier": {
          "description": "The performance tier of the processor.",
          "is_multi": False,
          "options": [
            "entry level (i3, Ryzen 3)",
            "mid range (i5, Ryzen 5)",
            "high end (i7, Ryzen 7)",
            "extreme (i9, Ryzen 9, Threadripper)"
          ]
        },
        "ram_capacity": {
          "description": "The amount of system memory (RAM).",
          "is_multi": False,
          "options": [
            "4GB",
            "8GB",
            "16GB",
            "32GB",
            "64GB",
            "128GB or more"
          ]
        },
        "storage_type": {
          "description": "The type of primary storage drive.",
          "is_multi": True,
          "options": [
            "HDD",
            "SSD",
            "NVMe SSD",
            "Hybrid"
          ]
        },
        "graphics_card": {
          "description": "The GPU configuration of the computer.",
          "is_multi": False,
          "options": [
            "integrated",
            "dedicated entry (GTX 1650, RX 6500)",
            "dedicated mid (RTX 3060/4060, RX 6600/7600)",
            "dedicated high (RTX 3080/4070, RX 6800/7800)",
            "dedicated extreme (RTX 4080/4090, RX 7900)"
          ]
        },
        "use_case": {
          "description": "The primary intended use or target audience.",
          "is_multi": True,
          "options": [
            "gaming",
            "business",
            "content creation",
            "programming",
            "general use",
            "student",
            "professional workstation"
          ]
        },
        "operating_system": {
          "description": "The pre-installed operating system.",
          "is_multi": False,
          "options": [
            "Windows",
            "macOS",
            "Linux",
            "Chrome OS",
            "None"
          ]
        },
        "screen_size": {
          "description": "Display size for laptops or all-in-ones.",
          "is_multi": False,
          "options": [
            "11-13 inch",
            "14-15 inch",
            "16-17 inch",
            "18+ inch",
            "N/A (desktop)"
          ]
        }
      }
    },
    {
      "headphones_audio": {
        "type": {
          "description": "The form factor and style of the audio device.",
          "is_multi": False,
          "options": [
            "over-ear headphones",
            "on-ear headphones",
            "in-ear earbuds",
            "wireless earbuds",
            "gaming headset",
            "studio monitor headphones",
            "bluetooth speaker",
            "smart speaker",
            "soundbar"
          ]
        },
        "connectivity": {
          "description": "How the device connects to audio sources.",
          "is_multi": True,
          "options": [
            "wired (3.5mm)",
            "wired (USB)",
            "bluetooth",
            "wifi",
            "wireless (proprietary)"
          ]
        },
        "noise_cancellation": {
          "description": "Active noise cancellation capability.",
          "is_multi": False,
          "options": [
            "active (ANC)",
            "passive",
            "none"
          ]
        },
        "battery_life": {
          "description": "Expected battery duration for wireless devices.",
          "is_multi": False,
          "options": [
            "short (under 10 hours)",
            "medium (10-20 hours)",
            "long (20-40 hours)",
            "extra long (over 40 hours)",
            "N/A (wired)"
          ]
        },
        "brand": {
          "description": "The manufacturer or brand.",
          "is_multi": False,
          "options": [
            "Sony",
            "Bose",
            "Apple",
            "Samsung",
            "Sennheiser",
            "Audio-Technica",
            "JBL",
            "Beats",
            "Anker",
            "Other"
          ]
        },
        "use_case": {
          "description": "The primary intended use.",
          "is_multi": True,
          "options": [
            "music listening",
            "gaming",
            "calls/meetings",
            "sports/fitness",
            "travel",
            "studio/professional",
            "home theater"
          ]
        },
        "water_resistance": {
          "description": "Water and sweat resistance rating.",
          "is_multi": False,
          "options": [
            "none",
            "splash resistant",
            "IPX4 (sweat resistant)",
            "IPX7 (waterproof)",
            "IP67 or higher"
          ]
        }
      }
    },
    {
      "tablets": {
        "brand": {
          "description": "The manufacturer or brand of the tablet.",
          "is_multi": False,
          "options": [
            "Apple",
            "Samsung",
            "Microsoft",
            "Amazon",
            "Lenovo",
            "Google",
            "Other"
          ]
        },
        "operating_system": {
          "description": "The operating system running on the tablet.",
          "is_multi": False,
          "options": [
            "iPadOS",
            "Android",
            "Windows",
            "Fire OS"
          ]
        },
        "screen_size": {
          "description": "The display size category of the tablet.",
          "is_multi": False,
          "options": [
            "small (7-8 inches)",
            "medium (9-10 inches)",
            "large (11-12 inches)",
            "extra large (13+ inches)"
          ]
        },
        "storage_capacity": {
          "description": "Internal storage capacity of the tablet.",
          "is_multi": True,
          "options": [
            "32GB",
            "64GB",
            "128GB",
            "256GB",
            "512GB",
            "1TB"
          ]
        },
        "cellular_capable": {
          "description": "Whether the tablet supports cellular connectivity.",
          "is_multi": False,
          "options": [
           True,
           False
          ]
        },
        "stylus_support": {
          "description": "Support for digital pen/stylus input.",
          "is_multi": False,
          "options": [
            "yes (included)",
            "yes (sold separately)",
            "no"
          ]
        },
        "use_case": {
          "description": "The primary intended use.",
          "is_multi": True,
          "options": [
            "entertainment",
            "productivity",
            "creative work",
            "education",
            "reading",
            "kids"
          ]
        }
      }
    },
    {
      "smartwatches_wearables": {
        "type": {
          "description": "The category of wearable device.",
          "is_multi": False,
          "options": [
            "smartwatch",
            "fitness tracker",
            "hybrid watch",
            "fitness band"
          ]
        },
        "brand": {
          "description": "The manufacturer or brand.",
          "is_multi": False,
          "options": [
            "Apple",
            "Samsung",
            "Garmin",
            "Fitbit",
            "Google",
            "Amazfit",
            "Polar",
            "Huawei",
            "Other"
          ]
        },
        "compatibility": {
          "description": "Compatible smartphone platforms.",
          "is_multi": True,
          "options": [
            "iOS",
            "Android",
            "both"
          ]
        },
        "fitness_features": {
          "description": "Health and fitness tracking capabilities.",
          "is_multi": True,
          "options": [
            "heart rate",
            "GPS",
            "sleep tracking",
            "step counter",
            "SpO2",
            "ECG",
            "temperature",
            "stress monitoring",
            "workout modes"
          ]
        },
        "battery_life": {
          "description": "Expected battery duration.",
          "is_multi": False,
          "options": [
            "1 day",
            "2-3 days",
            "4-7 days",
            "1-2 weeks",
            "over 2 weeks"
          ]
        },
        "display_type": {
          "description": "The type of display screen.",
          "is_multi": False,
          "options": [
            "AMOLED",
            "LCD",
            "E-ink",
            "OLED",
            "none (band only)"
          ]
        },
        "water_resistance": {
          "description": "Water resistance rating.",
          "is_multi": False,
          "options": [
            "none",
            "splash resistant",
            "swim proof (5ATM)",
            "dive proof (10ATM or higher)"
          ]
        }
      }
    },
    {
      "cameras": {
        "type": {
          "description": "The category and form factor of the camera.",
          "is_multi": False,
          "options": [
            "DSLR",
            "mirrorless",
            "action camera",
            "point-and-shoot",
            "webcam",
            "instant camera",
            "film camera"
          ]
        },
        "sensor_size": {
          "description": "The image sensor size.",
          "is_multi": False,
          "options": [
            "full frame",
            "APS-C",
            "Micro Four Thirds",
            "1 inch",
            "smaller than 1 inch"
          ]
        },
        "megapixels": {
          "description": "Resolution in megapixels.",
          "is_multi": False,
          "options": [
            "under 12MP",
            "12-24MP",
            "24-36MP",
            "36-50MP",
            "over 50MP"
          ]
        },
        "video_capability": {
          "description": "Maximum video resolution.",
          "is_multi": True,
          "options": [
            "HD (1080p)",
            "4K",
            "6K",
            "8K",
            "slow motion",
            "none"
          ]
        },
        "lens_mount": {
          "description": "The camera lens mount system (for interchangeable lens cameras).",
          "is_multi": False,
          "options": [
            "Canon EF/RF",
            "Nikon F/Z",
            "Sony E",
            "Fujifilm X",
            "Micro Four Thirds",
            "Pentax K",
            "fixed lens",
            "N/A"
          ]
        },
        "use_case": {
          "description": "The primary intended use.",
          "is_multi": True,
          "options": [
            "professional photography",
            "vlogging",
            "action/sports",
            "wildlife",
            "portrait",
            "landscape",
            "video calls",
            "streaming",
            "casual/beginner"
          ]
        },
        "stabilization": {
          "description": "Image stabilization capabilities.",
          "is_multi": True,
          "options": [
            "in-body (IBIS)",
            "optical (lens)",
            "electronic",
            "none"
          ]
        }
      }
    },
    {
      "gaming_consoles": {
        "brand": {
          "description": "The console manufacturer.",
          "is_multi": False,
          "options": [
            "Sony",
            "Microsoft",
            "Nintendo",
            "Valve",
            "Other"
          ]
        },
        "console_generation": {
          "description": "The generation or model of the gaming console.",
          "is_multi": False,
          "options": [
            "PlayStation 5",
            "PlayStation 4",
            "Xbox Series X/S",
            "Xbox One",
            "Nintendo Switch",
            "Steam Deck",
            "Other"
          ]
        },
        "storage_capacity": {
          "description": "Internal storage size.",
          "is_multi": True,
          "options": [
            "500GB",
            "1TB",
            "2TB",
            "expandable"
          ]
        },
        "resolution_support": {
          "description": "Maximum display resolution supported.",
          "is_multi": True,
          "options": [
            "1080p",
            "1440p",
            "4K",
            "8K"
          ]
        },
        "form_factor": {
          "description": "Whether the console is portable or stationary.",
          "is_multi": False,
          "options": [
            "home console",
            "handheld",
            "hybrid"
          ]
        },
        "vr_capable": {
          "description": "Virtual reality headset compatibility.",
          "is_multi": False,
          "options": [
           True,
           False
          ]
        },
        "disc_drive": {
          "description": "Physical media support.",
          "is_multi": False,
          "options": [
            "yes",
            "no (digital only)"
          ]
        }
      }
    },
    {
      "monitors": {
        "screen_size": {
          "description": "The diagonal screen size of the monitor.",
          "is_multi": False,
          "options": [
            "21-24 inch",
            "25-27 inch",
            "28-32 inch",
            "34-38 inch (ultrawide)",
            "over 38 inch"
          ]
        },
        "resolution": {
          "description": "The display resolution.",
          "is_multi": False,
          "options": [
            "Full HD (1920x1080)",
            "QHD (2560x1440)",
            "4K (3840x2160)",
            "5K (5120x2880)",
            "ultrawide (2560x1080)",
            "ultrawide QHD (3440x1440)",
            "ultrawide 4K (3840x1600)"
          ]
        },
        "panel_type": {
          "description": "The display panel technology.",
          "is_multi": False,
          "options": [
            "IPS",
            "VA",
            "TN",
            "OLED",
            "Mini-LED"
          ]
        },
        "refresh_rate": {
          "description": "The screen refresh rate in Hz.",
          "is_multi": False,
          "options": [
            "60Hz",
            "75Hz",
            "120Hz",
            "144Hz",
            "165Hz",
            "240Hz",
            "360Hz or higher"
          ]
        },
        "response_time": {
          "description": "Pixel response time in milliseconds.",
          "is_multi": False,
          "options": [
            "1ms",
            "2-4ms",
            "5ms or higher"
          ]
        },
        "use_case": {
          "description": "The primary intended use.",
          "is_multi": True,
          "options": [
            "gaming",
            "professional/color work",
            "general use",
            "productivity",
            "content creation"
          ]
        },
        "hdr_support": {
          "description": "High Dynamic Range support.",
          "is_multi": False,
          "options": [
           True,
           False
          ]
        },
        "adaptive_sync": {
          "description": "Variable refresh rate technology.",
          "is_multi": True,
          "options": [
            "G-Sync",
            "FreeSync",
            "G-Sync Compatible",
            "none"
          ]
        },
        "curved": {
          "description": "Whether the monitor has a curved screen.",
          "is_multi": False,
          "options": [
           True,
           False
          ]
        }
      }
    },
    {
      "keyboards_mice": {
        "device_type": {
          "description": "The category of input device.",
          "is_multi": False,
          "options": [
            "keyboard",
            "mouse",
            "keyboard and mouse combo"
          ]
        },
        "connectivity": {
          "description": "How the device connects to the computer.",
          "is_multi": False,
          "options": [
            "wired (USB)",
            "wired (USB-C)",
            "wireless (Bluetooth)",
            "wireless (2.4GHz dongle)",
            "dual mode (wired/wireless)"
          ]
        },
        "keyboard_type": {
          "description": "The type of keyboard mechanism (for keyboards).",
          "is_multi": False,
          "options": [
            "mechanical",
            "membrane",
            "scissor switch",
            "optical",
            "N/A"
          ]
        },
        "switch_type": {
          "description": "Mechanical switch type (for mechanical keyboards).",
          "is_multi": False,
          "options": [
            "linear (red)",
            "tactile (brown)",
            "clicky (blue)",
            "silent",
            "low profile",
            "N/A"
          ]
        },
        "form_factor": {
          "description": "The size and layout (for keyboards).",
          "is_multi": False,
          "options": [
            "full size (100%)",
            "tenkeyless (80%)",
            "compact (75%)",
            "60%",
            "ergonomic",
            "N/A"
          ]
        },
        "mouse_sensor": {
          "description": "The sensor type (for mice).",
          "is_multi": False,
          "options": [
            "optical",
            "laser",
            "N/A"
          ]
        },
        "dpi_range": {
          "description": "Maximum DPI/sensitivity (for mice).",
          "is_multi": False,
          "options": [
            "under 3000",
            "3000-6000",
            "6000-12000",
            "12000-20000",
            "over 20000",
            "N/A"
          ]
        },
        "use_case": {
          "description": "The primary intended use.",
          "is_multi": True,
          "options": [
            "gaming",
            "office/productivity",
            "programming",
            "creative work",
            "ergonomic/health",
            "travel"
          ]
        },
        "rgb_lighting": {
          "description": "RGB backlighting capability.",
          "is_multi": False,
          "options": [
            "full RGB",
            "single color backlight",
            "none"
          ]
        },
        "ergonomic_design": {
          "description": "Special ergonomic features.",
          "is_multi": False,
          "options": [
           True,
           False
          ]
        }
      }
    },
    {
      "smart_home": {
        "device_type": {
          "description": "The category of smart home device.",
          "is_multi": False,
          "options": [
            "smart bulb",
            "smart light strip",
            "smart plug",
            "smart switch",
            "security camera",
            "video doorbell",
            "smart thermostat",
            "smart lock",
            "smart speaker",
            "hub/controller"
          ]
        },
        "voice_assistant": {
          "description": "Compatible voice assistants.",
          "is_multi": True,
          "options": [
            "Alexa",
            "Google Assistant",
            "Siri/HomeKit",
            "none"
          ]
        },
        "connectivity": {
          "description": "Wireless connectivity protocol.",
          "is_multi": True,
          "options": [
            "WiFi",
            "Bluetooth",
            "Zigbee",
            "Z-Wave",
            "Thread",
            "Matter"
          ]
        },
        "power_source": {
          "description": "How the device is powered.",
          "is_multi": False,
          "options": [
            "AC plug",
            "battery",
            "wired (low voltage)",
            "solar",
            "hybrid"
          ]
        },
        "hub_required": {
          "description": "Whether a separate hub is required for operation.",
          "is_multi": False,
          "options": [
           True,
           False
          ]
        },
        "indoor_outdoor": {
          "description": "Intended installation location.",
          "is_multi": False,
          "options": [
            "indoor only",
            "outdoor rated",
            "both"
          ]
        },
        "video_resolution": {
          "description": "Video recording quality (for cameras/doorbells).",
          "is_multi": False,
          "options": [
            "720p",
            "1080p",
            "2K",
            "4K",
            "N/A"
          ]
        }
      }
    },
    {
      "power_charging": {
        "device_type": {
          "description": "The category of power/charging device.",
          "is_multi": False,
          "options": [
            "power bank",
            "wall charger",
            "car charger",
            "wireless charger",
            "charging cable",
            "charging station/dock"
          ]
        },
        "capacity": {
          "description": "Battery capacity in mAh (for power banks).",
          "is_multi": False,
          "options": [
            "under 5000mAh",
            "5000-10000mAh",
            "10000-20000mAh",
            "20000-30000mAh",
            "over 30000mAh",
            "N/A"
          ]
        },
        "output_power": {
          "description": "Maximum charging power output in watts.",
          "is_multi": False,
          "options": [
            "under 18W",
            "18-30W",
            "30-65W",
            "65-100W",
            "over 100W"
          ]
        },
        "port_types": {
          "description": "Available charging ports.",
          "is_multi": True,
          "options": [
            "USB-A",
            "USB-C",
            "Lightning",
            "Micro USB",
            "wireless"
          ]
        },
        "fast_charging": {
          "description": "Fast charging protocol support.",
          "is_multi": True,
          "options": [
            "USB Power Delivery (PD)",
            "Quick Charge",
            "MagSafe",
            "Qi wireless",
            "proprietary",
            "none"
          ]
        },
        "number_of_ports": {
          "description": "Number of simultaneous charging ports.",
          "is_multi": False,
          "options": [
            "1",
            "2",
            "3",
            "4 or more"
          ]
        },
        "portability": {
          "description": "Size and portability factor.",
          "is_multi": False,
          "options": [
            "pocket size",
            "compact",
            "standard",
            "large/stationary"
          ]
        }
      }
    },
    {
      "storage": {
        "device_type": {
          "description": "The category of storage device.",
          "is_multi": False,
          "options": [
            "external HDD",
            "external SSD",
            "internal SSD",
            "internal HDD",
            "USB flash drive",
            "SD card",
            "microSD card",
            "NAS drive"
          ]
        },
        "capacity": {
          "description": "Storage capacity.",
          "is_multi": False,
          "options": [
            "under 128GB",
            "128-256GB",
            "256GB-512GB",
            "512GB-1TB",
            "1-2TB",
            "2-4TB",
            "4-8TB",
            "over 8TB"
          ]
        },
        "interface": {
          "description": "Connection interface type.",
          "is_multi": False,
          "options": [
            "USB 2.0",
            "USB 3.0/3.1",
            "USB 3.2",
            "USB-C",
            "Thunderbolt 3/4",
            "SATA",
            "NVMe (M.2)",
            "SD/microSD slot"
          ]
        },
        "form_factor": {
          "description": "Physical size and form.",
          "is_multi": False,
          "options": [
            "2.5 inch",
            "3.5 inch",
            "M.2",
            "portable/pocket",
            "card",
            "stick"
          ]
        },
        "read_speed": {
          "description": "Approximate read speed category.",
          "is_multi": False,
          "options": [
            "under 100MB/s",
            "100-500MB/s",
            "500-1000MB/s",
            "1000-3000MB/s",
            "over 3000MB/s"
          ]
        },
        "durability": {
          "description": "Physical durability features.",
          "is_multi": True,
          "options": [
            "shock resistant",
            "water resistant",
            "dust proof",
            "rugged enclosure",
            "standard"
          ]
        },
        "use_case": {
          "description": "Primary intended use.",
          "is_multi": True,
          "options": [
            "backup",
            "portable storage",
            "gaming",
            "photography/video",
            "system drive",
            "cache/scratch disk",
            "archive"
          ]
        }
      }
    }
  ]

        # Gen AI API config - points to local Ollama instance
        self.gen_ai_config = {
          "provider": "ollama",
          "model_name": "llama3.2",
          "base_url": "http://192.168.68.74:11434",
          "gen_config": {
            "temperature": 0.1,  # Low temperature for strict rule following
            "max_tokens": 2000,
            "top_p": 0.9,
            "top_k": 40
          },
          "format": "json"
        }

    def test_live_extraction_footwear_simple(self):
        """Test extraction with live LLM for simple footwear product"""
        # Pass category config as list of dicts (real-world usage)
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=self.category_def,
            gen_ai_api_config=self.gen_ai_config
        )

        prod_data = LLMProductRequestData(
            super_category="footwear",
            product_title="Nike Air Max 90",
            product_information="Classic red and white running shoes made of leather and mesh materials. Available in sizes 9, 10, and 11."
        )

        result = extractor.get_all_characteristics(prod_data)

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn("color", result)
        self.assertIn("material", result)
        self.assertIn("size", result)

        # Verify extracted characteristics contain expected values
        # Note: LLM might extract variations, so we check if reasonable values are present
        print(f"\nExtracted characteristics: {result}")

        # Color should contain red or white
        colors = [c.lower() for c in result.get("color", [])]
        self.assertTrue(any(c in colors for c in ["red", "white"]),
                       f"Expected red or white in colors, got: {colors}")

        # Material should contain leather or mesh
        materials = [m.lower() for m in result.get("material", [])]
        self.assertTrue(any(m in materials for m in ["leather", "mesh"]),
                       f"Expected leather or mesh in materials, got: {materials}")

    def test_live_extraction_footwear_complex(self):
        """Test extraction with complex footwear description"""
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=self.category_def,
            gen_ai_api_config=self.gen_ai_config
        )

        prod_data = LLMProductRequestData(
            super_category="footwear",
            product_title="Timberland Classic Boot",
            product_information="""Premium waterproof boots crafted from full-grain leather in wheat brown color.
            Features a rugged rubber lug outsole for maximum traction. These iconic boots combine style and durability,
            perfect for outdoor adventures. Available in men's sizes 8 through 13."""
        )

        result = extractor.get_all_characteristics(prod_data)

        print(f"\nExtracted characteristics for Timberland: {result}")

        # Should extract brown color (may be "brown" or "wheat brown" or similar)
        colors = [c.lower() for c in result.get("color", [])]
        colors_str = ' '.join(colors)
        self.assertTrue("brown" in colors_str, f"Expected brown in colors, got: {colors}")

        # Should extract leather material
        materials = [m.lower() for m in result.get("material", [])]
        self.assertTrue("leather" in materials, f"Expected leather in materials, got: {materials}")

    def test_live_extraction_clothing(self):
        """Test extraction with clothing product"""
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=self.category_def,
            gen_ai_api_config=self.gen_ai_config
        )

        prod_data = LLMProductRequestData(
            super_category="clothing",
            product_title="Classic Cotton T-Shirt",
            product_information="Comfortable navy blue t-shirt made from 100% organic cotton. Available in sizes S, M, L, and XL."
        )

        result = extractor.get_all_characteristics(prod_data)

        print(f"\nExtracted characteristics for clothing: {result}")

        # Verify structure
        self.assertIsInstance(result, dict)
        self.assertIn("color", result)
        self.assertIn("material", result)

        # Should extract blue (may be "blue" or "navy blue" or similar)
        colors = [c.lower() for c in result.get("color", [])]
        colors_str = ' '.join(colors)
        self.assertTrue("blue" in colors_str, f"Expected blue in colors, got: {colors}")

        # Should extract cotton
        materials = [m.lower() for m in result.get("material", [])]
        self.assertTrue("cotton" in materials, f"Expected cotton in materials, got: {materials}")

    def test_live_extraction_skewing_bug_sde_84_failure(self):
        """
        FAILURE TEST: Test that unknown category returns empty list.

        This tests the SDE-84 bug where using an undefined category ('smartphones')
        should gracefully return an empty list instead of attempting extraction.

        The category_def only has: footwear, clothing, electronics
        This test uses 'smartphones' which is NOT in the config.
        """
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=self.category_def,
            gen_ai_api_config=self.gen_ai_config
        )

        # Real product data from production - using string repr to avoid Unicode issues
        product_data = """specs={'description': 'The moto g Play from Tracfone dials up the excitement without losing function. Feel the power of the new Snapdragon 680 processor and indulge your ears with Dolby Atmos and High-Res Audio. Take spectacular selfies with an 8MP front camera and capture night photos like a true professional with Quad Pixel technology. Since you'll be taking a ton of photos - hold everything you need with 64GB of internal storage, expandable to 1TB with a microSD card. And with a long-lasting 5000mAh battery - do it all on the Network America relies on, with speed, hotspot capability, international calling and exclusive discounts. Plus every Tracfone plan comes with Unlimited Carryover, so you keep unused monthly data, texts or minutes for when you need them most*. Stay in control with a moto g Play from Tracfone. Get a single line unlimited talk & text plus data plan starting at only $20/month with no contract. To activate this device, a Tracfone plan is required. *Service must be active and in use within any six month period.', 'battery_life': 'Up to 3 days', 'battery_charging_technology': 'Rapid charging', 'battery_capacity': '5000 mAh', 'battery_standby_time': ['Up to 498 hrs', '22 days', 'Up to 444 hrs'], 'battery_talk_time': ['Up to 3000 min', 'Up to 2160 min'], 'display_size': '6.5 in', 'touch_screen': 'Yes', 'display_resolution': 'HD+', 'display_pixel_density': '269 ppi', 'display_technology': 'LCD', 'display_type': 'IPS LCD', 'display_aspect_ratio': '20:9', 'display_refresh_rate': '90Hz', 'display_highlights': 'Touch screen', 'camera_resolution': '13.0 MP', 'camera_zoom': 'High-res zoom', 'camera_video_resolution': ['HD (30fps) (Rear Macro)', 'FHD (30fps) (Front)', 'FHD (30fps) (Rear Main)'], 'camera_aperture': ['f/2.4 (Rear Macro)', 'f/1.22 (Rear Main)', 'f/2.4 (Front)', 'f/2.4 (Rear Depth)'], 'camera_highlights': 'Dual Rear Cameras', 'digital_zoom': 'High-res zoom', 'camera_lens_count': 'Dual Rear Cameras', 'focus_adjustment': 'Auto Focus (Rear Main)', 'camera_light_source': 'Single LED flash', 'ports_and_interfaces': 'USB Type-C', 'sim_support': 'Single SIM', 'wi_fi': 'Wi-Fi', 'wireless_interface': ['Wi-Fi', 'Bluetooth'], 'broadband_generation': '4G', 'data_transmission': 'LTE', 'gps_type': 'GPS', 'bluetooth': 'Bluetooth', 'cell_band': ['UMTS band 1/2/4/5/8 (3G)', 'CDMA: BC0 BC1 BC10', 'GSM band 2/3/5/8 (2G)', 'LTE band 1/2/3/4/5/7/8/12/13/14/17/18/19/20/25/26/29/30/38/39/40/41/66/71 (4G)'], 'nfc': 'No', 'connectivity_highlights': 'Hotspot', 'lte_band': 'Band 1/2/3/4/5/7/8/12/13/14/17/18/19/20/25/26/29/30/38/39/40/41/66/71', 'sim_card_type': 'Nano SIM', 'product_type': 'Smartphone', 'buttons': ['Volume buttons', 'Power button', 'Vibrate/ringer switch'], 'hardware_sensors': ['Accelerometer', 'Proximity Sensor'], 'keyboard_support': 'QWERTY Keyboard', 'supported_os': 'Android', 'os_version': 'Android 10', 'security_highlights': 'Fingerprint Sensor', 'expandable_storage': 'Yes', 'supported_flash_memory_cards': 'microSDXC', 'dimensions': '167.24 x 76.54 x 9.36 mm', 'form_factor': 'Bar', 'durability': 'Water-repellent design', 'weight': '203 g', 'body_material': 'Plastic', 'design_highlights': 'Water-repellent design', 'keyboard': 'QWERTY Keyboard', 'headphone_jack': ['Yes', '3.5mm headset jack'], 'hearing_aid_compatibility': 'Yes (HAC)', 'speaker': 'Speakerphone', 'microphone': '2 Microphones', 'supported_media': 'Music Player', 'included_accessories': '10w Charger', 'gpu': 'Qualcomm ADRENO 610', 'number_of_cpu_cores': 'Octa Core', 'processor': ['Qualcomm Snapdragon 460', 'Octa Core'], 'cpu_clock_speed': '1.8 GHz', 'epeat_rating': 'A', 'top_use_cases': ['Video chats', 'Games', 'Movies'], 'typical_users': 'Students'}
        description={'description': 'The moto g Play from Tracfone dials up the excitement without losing function. Feel the power of the new Snapdragon 680 processor and indulge your ears with Dolby Atmos and High-Res Audio. Take spectacular selfies with an 8MP front camera and capture night photos like a true professional with Quad Pixel technology. Since you'll be taking a ton of photos - hold everything you need with 64GB of internal storage, expandable to 1TB with a microSD card. And with a long-lasting 5000mAh battery - do it all on the Network America relies on, with speed, hotspot capability, international calling and exclusive discounts. Plus every Tracfone plan comes with Unlimited Carryover, so you keep unused monthly data, texts or minutes for when you need them most*. Stay in control with a moto g Play from Tracfone. Get a single line unlimited talk & text plus data plan starting at only $20/month with no contract. To activate this device, a Tracfone plan is required. *Service must be active and in use within any six month period.', 'battery_life': 'Up to 3 days', 'battery_charging_technology': 'Rapid charging', 'battery_capacity': '5000 mAh', 'battery_standby_time': ['Up to 498 hrs', '22 days', 'Up to 444 hrs'], 'battery_talk_time': ['Up to 3000 min', 'Up to 2160 min'], 'display_size': '6.5 in', 'touch_screen': 'Yes', 'display_resolution': 'HD+', 'display_pixel_density': '269 ppi', 'display_technology': 'LCD', 'display_type': 'IPS LCD', 'display_aspect_ratio': '20:9', 'display_refresh_rate': '90Hz', 'display_highlights': 'Touch screen', 'camera_resolution': '13.0 MP', 'camera_zoom': 'High-res zoom', 'camera_video_resolution': ['HD (30fps) (Rear Macro)', 'FHD (30fps) (Front)', 'FHD (30fps) (Rear Main)'], 'camera_aperture': ['f/2.4 (Rear Macro)', 'f/1.22 (Rear Main)', 'f/2.4 (Front)', 'f/2.4 (Rear Depth)'], 'camera_highlights': 'Dual Rear Cameras', 'digital_zoom': 'High-res zoom', 'camera_lens_count': 'Dual Rear Cameras', 'focus_adjustment': 'Auto Focus (Rear Main)', 'camera_light_source': 'Single LED flash', 'ports_and_interfaces': 'USB Type-C', 'sim_support': 'Single SIM', 'wi_fi': 'Wi-Fi', 'wireless_interface': ['Wi-Fi', 'Bluetooth'], 'broadband_generation': '4G', 'data_transmission': 'LTE', 'gps_type': 'GPS', 'bluetooth': 'Bluetooth', 'cell_band': ['UMTS band 1/2/4/5/8 (3G)', 'CDMA: BC0 BC1 BC10', 'GSM band 2/3/5/8 (2G)', 'LTE band 1/2/3/4/5/7/8/12/13/14/17/18/19/20/25/26/29/30/38/39/40/41/66/71 (4G)'], 'nfc': 'No', 'connectivity_highlights': 'Hotspot', 'lte_band': 'Band 1/2/3/4/5/7/8/12/13/14/17/18/19/20/25/26/29/30/38/39/40/41/66/71', 'sim_card_type': 'Nano SIM', 'product_type': 'Smartphone', 'buttons': ['Volume buttons', 'Power button', 'Vibrate/ringer switch'], 'hardware_sensors': ['Accelerometer', 'Proximity Sensor'], 'keyboard_support': 'QWERTY Keyboard', 'supported_os': 'Android', 'os_version': 'Android 10', 'security_highlights': 'Fingerprint Sensor', 'expandable_storage': 'Yes', 'supported_flash_memory_cards': 'microSDXC', 'dimensions': '167.24 x 76.54 x 9.36 mm', 'form_factor': 'Bar', 'durability': 'Water-repellent design', 'weight': '203 g', 'body_material': 'Plastic', 'design_highlights': 'Water-repellent design', 'keyboard': 'QWERTY Keyboard', 'headphone_jack': ['Yes', '3.5mm headset jack'], 'hearing_aid_compatibility': 'Yes (HAC)', 'speaker': 'Speakerphone', 'microphone': '2 Microphones', 'supported_media': 'Music Player', 'included_accessories': '10w Charger', 'gpu': 'Qualcomm ADRENO 610', 'number_of_cpu_cores': 'Octa Core', 'processor': ['Qualcomm Snapdragon 460', 'Octa Core'], 'cpu_clock_speed': '1.8 GHz', 'epeat_rating': 'A', 'top_use_cases': ['Video chats', 'Games', 'Movies'], 'typical_users': 'Students'}
        sellers=[{'position': 1, 'merchant': 'Best Buy', 'link': 'https://www.bestbuy.com/product/tracfone-motorola-moto-g-play-2024-64gb-prepaid-blue/JXJCJWF5XC/sku/6589852?utm_source=feed', 'base_price_raw': '$29.99', 'base_price': 29.99, 'base_price_parsed': {'value': 29.99, 'currency': 'USD', 'raw': '29.99'}, 'tax_price_raw': '+tax', 'tax_price': 0, 'tax_price_parsed': {'value': 0, 'currency': 'USD', 'raw': '0'}, 'shipping_price_raw': '+ $5.49', 'shipping_price': 5.49, 'shipping_price_parsed': {'value': 5.49, 'currency': 'USD', 'raw': '5.49'}, 'total_price_raw': '$35.48', 'total_price': 35.48, 'total_price_parsed': {'value': 35.48, 'currency': 'USD', 'raw': '35.48'}}, {'position': 2, 'merchant': 'Walmart - Seller', 'link': 'https://www.walmart.com/ip/Verizon-Motorola-G-Play-32GB/752843265?wmlspartner=wlpa&selectedSellerId=101222135', 'base_price_raw': '$78.00', 'base_price': 78, 'base_price_parsed': {'value': 78, 'currency': 'USD', 'raw': '78.00'}, 'tax_price_raw': '+tax', 'tax_price': 0, 'tax_price_parsed': {'value': 0, 'currency': 'USD', 'raw': '0'}, 'shipping_price_raw': 'Free', 'shipping_price': 0, 'shipping_price_parsed': {'value': 0, 'currency': 'USD', 'raw': '0'}, 'total_price_raw': '$78.00', 'total_price': 78, 'total_price_parsed': {'value': 78, 'currency': 'USD', 'raw': '78.00'}}, {'position': 3, 'merchant': 'Target', 'link': 'https://www.target.com/p/tracfone-prepaid-motorola-g-play-64gb-cdma-lte-blue/-/A-92382739?TCID=OGS&AFID=google&CPNG=Electronics&adgroup=80-2&srsltid=AfmBOooruxwLzNaUYVqVVerJAJ155MWv2nY8Ws5yk4KQwUNN4oGJHt2vfJ8', 'base_price_raw': '$39.99', 'base_price': 39.99, 'base_price_parsed': {'value': 39.99, 'currency': 'USD', 'raw': '39.99'}, 'tax_price_raw': '+tax', 'tax_price': 0, 'tax_price_parsed': {'value': 0, 'currency': 'USD', 'raw': '0'}, 'shipping_price_raw': 'Free', 'shipping_price': 0, 'shipping_price_parsed': {'value': 0, 'currency': 'USD', 'raw': '0'}, 'total_price_raw': '$39.99', 'total_price': 39.99, 'total_price_parsed': {'value': 39.99, 'currency': 'USD', 'raw': '39.99'}}]
        rank=7
        merchant=Best Buy
        price=29.99
        reviews=1400
        rating=3.9"""

        # Simulate real-world product_information construction
        product_information = product_data

        prod_data = LLMProductRequestData(
            super_category="smartphones",  # NOTE: This category is NOT in the config!
            product_title="Motorola Moto G Play",
            product_information=product_information
        )

        result = extractor.get_all_characteristics(prod_data)

        print(f"\nExtracted characteristics for unknown category 'smartphones' (SDE-84 failure test): {result}")

        # FAILURE TEST: This should either:
        # 1. Return an empty list (because "smartphones" is not in the config), OR
        # 2. Return incorrect/unexpected results if it somehow tries to process

        # The config only has: footwear, clothing, electronics
        # "smartphones" is NOT a defined category

        # Primary check: Should return empty list for unknown category
        if result == []:
            # Expected behavior - unknown category returns empty list
            self.assertEqual(result, [])
        else:
            # If it returns something, verify it's NOT the correct expected values
            # This is a failure - it shouldn't extract for unknown categories

            # If it somehow returned data, verify it's incorrect
            if isinstance(result, dict):
                # Should NOT contain Motorola brand (which is clearly in the product title)
                brands = [b.lower() for b in result.get("brand", [])]
                self.assertFalse(
                    any("motorola" in b for b in brands),
                    f"FAILURE: Should not extract brand for unknown category, but got: {brands}"
                )

            # Fail the test if we got here - unknown category should return empty list
            self.fail(f"Expected empty list for unknown category 'smartphones', got: {result}")

    def test_live_extraction_skewing_bug_sde_84_working(self):
        """Test extraction with electronics product (Motorola smartphone) - SDE-84 bug fix"""
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=self.category_def,
            gen_ai_api_config=self.gen_ai_config
        )

        # Real product data from production - using string repr to avoid Unicode issues
        product_data = """specs={'description': 'The moto g Play from Tracfone dials up the excitement without losing function. Feel the power of the new Snapdragon 680 processor and indulge your ears with Dolby Atmos and High-Res Audio. Take spectacular selfies with an 8MP front camera and capture night photos like a true professional with Quad Pixel technology. Since you'll be taking a ton of photos - hold everything you need with 64GB of internal storage, expandable to 1TB with a microSD card. And with a long-lasting 5000mAh battery - do it all on the Network America relies on, with speed, hotspot capability, international calling and exclusive discounts. Plus every Tracfone plan comes with Unlimited Carryover, so you keep unused monthly data, texts or minutes for when you need them most*. Stay in control with a moto g Play from Tracfone. Get a single line unlimited talk & text plus data plan starting at only $20/month with no contract. To activate this device, a Tracfone plan is required. *Service must be active and in use within any six month period.', 'battery_life': 'Up to 3 days', 'battery_charging_technology': 'Rapid charging', 'battery_capacity': '5000 mAh', 'battery_standby_time': ['Up to 498 hrs', '22 days', 'Up to 444 hrs'], 'battery_talk_time': ['Up to 3000 min', 'Up to 2160 min'], 'display_size': '6.5 in', 'touch_screen': 'Yes', 'display_resolution': 'HD+', 'display_pixel_density': '269 ppi', 'display_technology': 'LCD', 'display_type': 'IPS LCD', 'display_aspect_ratio': '20:9', 'display_refresh_rate': '90Hz', 'display_highlights': 'Touch screen', 'camera_resolution': '13.0 MP', 'camera_zoom': 'High-res zoom', 'camera_video_resolution': ['HD (30fps) (Rear Macro)', 'FHD (30fps) (Front)', 'FHD (30fps) (Rear Main)'], 'camera_aperture': ['f/2.4 (Rear Macro)', 'f/1.22 (Rear Main)', 'f/2.4 (Front)', 'f/2.4 (Rear Depth)'], 'camera_highlights': 'Dual Rear Cameras', 'digital_zoom': 'High-res zoom', 'camera_lens_count': 'Dual Rear Cameras', 'focus_adjustment': 'Auto Focus (Rear Main)', 'camera_light_source': 'Single LED flash', 'ports_and_interfaces': 'USB Type-C', 'sim_support': 'Single SIM', 'wi_fi': 'Wi-Fi', 'wireless_interface': ['Wi-Fi', 'Bluetooth'], 'broadband_generation': '4G', 'data_transmission': 'LTE', 'gps_type': 'GPS', 'bluetooth': 'Bluetooth', 'cell_band': ['UMTS band 1/2/4/5/8 (3G)', 'CDMA: BC0 BC1 BC10', 'GSM band 2/3/5/8 (2G)', 'LTE band 1/2/3/4/5/7/8/12/13/14/17/18/19/20/25/26/29/30/38/39/40/41/66/71 (4G)'], 'nfc': 'No', 'connectivity_highlights': 'Hotspot', 'lte_band': 'Band 1/2/3/4/5/7/8/12/13/14/17/18/19/20/25/26/29/30/38/39/40/41/66/71', 'sim_card_type': 'Nano SIM', 'product_type': 'Smartphone', 'buttons': ['Volume buttons', 'Power button', 'Vibrate/ringer switch'], 'hardware_sensors': ['Accelerometer', 'Proximity Sensor'], 'keyboard_support': 'QWERTY Keyboard', 'supported_os': 'Android', 'os_version': 'Android 10', 'security_highlights': 'Fingerprint Sensor', 'expandable_storage': 'Yes', 'supported_flash_memory_cards': 'microSDXC', 'dimensions': '167.24 x 76.54 x 9.36 mm', 'form_factor': 'Bar', 'durability': 'Water-repellent design', 'weight': '203 g', 'body_material': 'Plastic', 'design_highlights': 'Water-repellent design', 'keyboard': 'QWERTY Keyboard', 'headphone_jack': ['Yes', '3.5mm headset jack'], 'hearing_aid_compatibility': 'Yes (HAC)', 'speaker': 'Speakerphone', 'microphone': '2 Microphones', 'supported_media': 'Music Player', 'included_accessories': '10w Charger', 'gpu': 'Qualcomm ADRENO 610', 'number_of_cpu_cores': 'Octa Core', 'processor': ['Qualcomm Snapdragon 460', 'Octa Core'], 'cpu_clock_speed': '1.8 GHz', 'epeat_rating': 'A', 'top_use_cases': ['Video chats', 'Games', 'Movies'], 'typical_users': 'Students'}
        description={'description': 'The moto g Play from Tracfone dials up the excitement without losing function. Feel the power of the new Snapdragon 680 processor and indulge your ears with Dolby Atmos and High-Res Audio. Take spectacular selfies with an 8MP front camera and capture night photos like a true professional with Quad Pixel technology. Since you'll be taking a ton of photos - hold everything you need with 64GB of internal storage, expandable to 1TB with a microSD card. And with a long-lasting 5000mAh battery - do it all on the Network America relies on, with speed, hotspot capability, international calling and exclusive discounts. Plus every Tracfone plan comes with Unlimited Carryover, so you keep unused monthly data, texts or minutes for when you need them most*. Stay in control with a moto g Play from Tracfone. Get a single line unlimited talk & text plus data plan starting at only $20/month with no contract. To activate this device, a Tracfone plan is required. *Service must be active and in use within any six month period.', 'battery_life': 'Up to 3 days', 'battery_charging_technology': 'Rapid charging', 'battery_capacity': '5000 mAh', 'battery_standby_time': ['Up to 498 hrs', '22 days', 'Up to 444 hrs'], 'battery_talk_time': ['Up to 3000 min', 'Up to 2160 min'], 'display_size': '6.5 in', 'touch_screen': 'Yes', 'display_resolution': 'HD+', 'display_pixel_density': '269 ppi', 'display_technology': 'LCD', 'display_type': 'IPS LCD', 'display_aspect_ratio': '20:9', 'display_refresh_rate': '90Hz', 'display_highlights': 'Touch screen', 'camera_resolution': '13.0 MP', 'camera_zoom': 'High-res zoom', 'camera_video_resolution': ['HD (30fps) (Rear Macro)', 'FHD (30fps) (Front)', 'FHD (30fps) (Rear Main)'], 'camera_aperture': ['f/2.4 (Rear Macro)', 'f/1.22 (Rear Main)', 'f/2.4 (Front)', 'f/2.4 (Rear Depth)'], 'camera_highlights': 'Dual Rear Cameras', 'digital_zoom': 'High-res zoom', 'camera_lens_count': 'Dual Rear Cameras', 'focus_adjustment': 'Auto Focus (Rear Main)', 'camera_light_source': 'Single LED flash', 'ports_and_interfaces': 'USB Type-C', 'sim_support': 'Single SIM', 'wi_fi': 'Wi-Fi', 'wireless_interface': ['Wi-Fi', 'Bluetooth'], 'broadband_generation': '4G', 'data_transmission': 'LTE', 'gps_type': 'GPS', 'bluetooth': 'Bluetooth', 'cell_band': ['UMTS band 1/2/4/5/8 (3G)', 'CDMA: BC0 BC1 BC10', 'GSM band 2/3/5/8 (2G)', 'LTE band 1/2/3/4/5/7/8/12/13/14/17/18/19/20/25/26/29/30/38/39/40/41/66/71 (4G)'], 'nfc': 'No', 'connectivity_highlights': 'Hotspot', 'lte_band': 'Band 1/2/3/4/5/7/8/12/13/14/17/18/19/20/25/26/29/30/38/39/40/41/66/71', 'sim_card_type': 'Nano SIM', 'product_type': 'Smartphone', 'buttons': ['Volume buttons', 'Power button', 'Vibrate/ringer switch'], 'hardware_sensors': ['Accelerometer', 'Proximity Sensor'], 'keyboard_support': 'QWERTY Keyboard', 'supported_os': 'Android', 'os_version': 'Android 10', 'security_highlights': 'Fingerprint Sensor', 'expandable_storage': 'Yes', 'supported_flash_memory_cards': 'microSDXC', 'dimensions': '167.24 x 76.54 x 9.36 mm', 'form_factor': 'Bar', 'durability': 'Water-repellent design', 'weight': '203 g', 'body_material': 'Plastic', 'design_highlights': 'Water-repellent design', 'keyboard': 'QWERTY Keyboard', 'headphone_jack': ['Yes', '3.5mm headset jack'], 'hearing_aid_compatibility': 'Yes (HAC)', 'speaker': 'Speakerphone', 'microphone': '2 Microphones', 'supported_media': 'Music Player', 'included_accessories': '10w Charger', 'gpu': 'Qualcomm ADRENO 610', 'number_of_cpu_cores': 'Octa Core', 'processor': ['Qualcomm Snapdragon 460', 'Octa Core'], 'cpu_clock_speed': '1.8 GHz', 'epeat_rating': 'A', 'top_use_cases': ['Video chats', 'Games', 'Movies'], 'typical_users': 'Students'}
        sellers=[{'position': 1, 'merchant': 'Best Buy', 'link': 'https://www.bestbuy.com/product/tracfone-motorola-moto-g-play-2024-64gb-prepaid-blue/JXJCJWF5XC/sku/6589852?utm_source=feed', 'base_price_raw': '$29.99', 'base_price': 29.99, 'base_price_parsed': {'value': 29.99, 'currency': 'USD', 'raw': '29.99'}, 'tax_price_raw': '+tax', 'tax_price': 0, 'tax_price_parsed': {'value': 0, 'currency': 'USD', 'raw': '0'}, 'shipping_price_raw': '+ $5.49', 'shipping_price': 5.49, 'shipping_price_parsed': {'value': 5.49, 'currency': 'USD', 'raw': '5.49'}, 'total_price_raw': '$35.48', 'total_price': 35.48, 'total_price_parsed': {'value': 35.48, 'currency': 'USD', 'raw': '35.48'}}, {'position': 2, 'merchant': 'Walmart - Seller', 'link': 'https://www.walmart.com/ip/Verizon-Motorola-G-Play-32GB/752843265?wmlspartner=wlpa&selectedSellerId=101222135', 'base_price_raw': '$78.00', 'base_price': 78, 'base_price_parsed': {'value': 78, 'currency': 'USD', 'raw': '78.00'}, 'tax_price_raw': '+tax', 'tax_price': 0, 'tax_price_parsed': {'value': 0, 'currency': 'USD', 'raw': '0'}, 'shipping_price_raw': 'Free', 'shipping_price': 0, 'shipping_price_parsed': {'value': 0, 'currency': 'USD', 'raw': '0'}, 'total_price_raw': '$78.00', 'total_price': 78, 'total_price_parsed': {'value': 78, 'currency': 'USD', 'raw': '78.00'}}, {'position': 3, 'merchant': 'Target', 'link': 'https://www.target.com/p/tracfone-prepaid-motorola-g-play-64gb-cdma-lte-blue/-/A-92382739?TCID=OGS&AFID=google&CPNG=Electronics&adgroup=80-2&srsltid=AfmBOooruxwLzNaUYVqVVerJAJ155MWv2nY8Ws5yk4KQwUNN4oGJHt2vfJ8', 'base_price_raw': '$39.99', 'base_price': 39.99, 'base_price_parsed': {'value': 39.99, 'currency': 'USD', 'raw': '39.99'}, 'tax_price_raw': '+tax', 'tax_price': 0, 'tax_price_parsed': {'value': 0, 'currency': 'USD', 'raw': '0'}, 'shipping_price_raw': 'Free', 'shipping_price': 0, 'shipping_price_parsed': {'value': 0, 'currency': 'USD', 'raw': '0'}, 'total_price_raw': '$39.99', 'total_price': 39.99, 'total_price_parsed': {'value': 39.99, 'currency': 'USD', 'raw': '39.99'}}]
        rank=7
        merchant=Best Buy
        price=29.99
        reviews=1400
        rating=3.9"""

        # Simulate real-world product_information construction
        product_information = product_data

        prod_data = LLMProductRequestData(
            super_category="smartphones",
            product_title="Motorola Moto G Play",
            product_information=product_information
        )

        result = extractor.get_all_characteristics(prod_data)

        print(f"\nExtracted characteristics for electronics (SDE-84): {result}")

        # Verify structure
        self.assertIsInstance(result, dict)
        self.assertIn("brand", result)
        self.assertIn("color", result)

        # Should extract Motorola as brand
        brands = [b.lower() for b in result.get("brand", [])]
        self.assertTrue(any("motorola" in b for b in brands), f"Expected Motorola in brands, got: {brands}")

        # Product is described as blue in the sellers link
        # May or may not extract color depending on LLM interpretation
        # Just verify the result is a list
        colors = result.get("color", [])
        self.assertIsInstance(colors, list)

    def test_live_extraction_multiple_colors(self):
        """Test extraction with multiple colors in product"""
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=self.category_def,
            gen_ai_api_config=self.gen_ai_config
        )

        prod_data = LLMProductRequestData(
            super_category="footwear",
            product_title="Adidas Ultraboost",
            product_information="Modern running shoes featuring a black and white colorway with blue accent stripes. Constructed with textile and synthetic materials."
        )

        result = extractor.get_all_characteristics(prod_data)

        print(f"\nExtracted characteristics for multi-color product: {result}")

        # Should extract multiple colors
        colors = [c.lower() for c in result.get("color", [])]
        self.assertTrue(len(colors) > 1, f"Expected multiple colors, got: {colors}")
        self.assertTrue(any(c in colors for c in ["black", "white", "blue"]),
                       f"Expected black, white, or blue, got: {colors}")

    def test_live_extraction_electronics(self):
        """Test extraction with electronics category"""
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=self.category_def,
            gen_ai_api_config=self.gen_ai_config
        )

        prod_data = LLMProductRequestData(
            super_category="electronics",
            product_title="MacBook Pro 16-inch",
            product_information="Apple's powerful laptop in sleek space gray finish. Features a stunning Retina display and M2 Pro chip."
        )

        result = extractor.get_all_characteristics(prod_data)

        print(f"\nExtracted characteristics for electronics: {result}")

        # Verify structure
        self.assertIsInstance(result, dict)
        self.assertIn("brand", result)
        self.assertIn("color", result)

        # Should extract Apple as brand
        brand = result.get("brand", [])
        if isinstance(brand, list):
            brand_str = brand[0].lower() if brand else ""
        else:
            brand_str = str(brand).lower()

        self.assertTrue("apple" in brand_str, f"Expected Apple in brand, got: {brand}")

    def test_live_extraction_minimal_information(self):
        """Test extraction with minimal product information"""
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=self.category_def,
            gen_ai_api_config=self.gen_ai_config
        )

        prod_data = LLMProductRequestData(
            super_category="footwear",
            product_title="Black Sneakers",
            product_information="Simple black sneakers."
        )

        result = extractor.get_all_characteristics(prod_data)

        print(f"\nExtracted characteristics for minimal info: {result}")

        # Should at least extract black color
        colors = [c.lower() for c in result.get("color", [])]
        self.assertTrue("black" in colors, f"Expected black in colors, got: {colors}")

    def test_live_extraction_ambiguous_information(self):
        """Test extraction with ambiguous product information"""
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=self.category_def,
            gen_ai_api_config=self.gen_ai_config
        )

        prod_data = LLMProductRequestData(
            super_category="clothing",
            product_title="Vintage Style Shirt",
            product_information="A retro-inspired shirt with classic styling and comfortable fit."
        )

        result = extractor.get_all_characteristics(prod_data)

        print(f"\nExtracted characteristics for ambiguous info: {result}")

        # Should return empty lists for characteristics that cannot be determined
        self.assertIsInstance(result, dict)
        # All values should be lists
        for key, value in result.items():
            self.assertIsInstance(value, list, f"Expected list for {key}, got {type(value)}")

    def test_real_world_spark_row_simulation(self):
        """Test real-world usage pattern simulating Spark Row structure"""
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=self.category_def,
            gen_ai_api_config=self.gen_ai_config
        )

        # Create a mock Spark Row using namedtuple
        ProductRow = namedtuple('ProductRow', [
            'super_category', 'product_title', 'description', 'specs',
            'sellers', 'rank', 'merchant', 'price', 'reviews', 'rating'
        ])

        row = ProductRow(
            super_category='footwear',
            product_title='Nike Air Zoom Pegasus',
            description='Responsive running shoes with Zoom Air cushioning',
            specs='Weight: 10oz, Drop: 10mm, Cushioning: High',
            sellers='Nike Store, Amazon',
            rank='#1 in Running Shoes',
            merchant='Nike',
            price='$120',
            reviews='4,523',
            rating='4.7'
        )

        # Simulate real-world usage pattern
        product_information = f"""
            specs: {getattr(row, 'specs', 'N/A')}
            description: {getattr(row, 'description', 'N/A')}
            sellers: {getattr(row, 'sellers', 'N/A')}
            rank: {getattr(row, 'rank', 'N/A')}
            merchant: {getattr(row, 'merchant', 'N/A')}
            price: {getattr(row, 'price', 'N/A')}
            reviews: {getattr(row, 'reviews', 'N/A')}
            rating: {getattr(row, 'rating', 'N/A')}
        """

        characteristics_map = extractor.get_all_characteristics(
            prod_data=LLMProductRequestData(
                super_category=getattr(row, 'super_category', None),
                product_title=getattr(row, 'product_title', None),
                product_information=product_information
            )
        )

        print(f"\nReal-world row extraction: {characteristics_map}")

        # Verify results
        self.assertIsInstance(characteristics_map, dict)
        # Should have all characteristics from footwear category
        self.assertIn('color', characteristics_map)
        self.assertIn('material', characteristics_map)
        self.assertIn('size', characteristics_map)
        self.assertIn('type', characteristics_map)

    def test_data_skewing_minimal_vs_rich_content(self):
        """
        Test data skewing scenario: minimal content vs rich content products.
        Rich content products may take longer to process, causing skew.
        """
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=self.category_def,
            gen_ai_api_config=self.gen_ai_config
        )

        # Minimal content product (fast processing)
        minimal_start = time.time()
        minimal_prod = LLMProductRequestData(
            super_category="footwear",
            product_title="Black Shoes",
            product_information="Black shoes."
        )
        minimal_result = extractor.get_all_characteristics(minimal_prod)
        minimal_time = time.time() - minimal_start

        # Rich content product (potentially slower processing)
        rich_start = time.time()
        rich_prod = LLMProductRequestData(
            super_category="footwear",
            product_title="Premium Leather Oxford Dress Shoes",
            product_information="""
            Premium handcrafted oxford dress shoes made from genuine full-grain Italian leather in rich mahogany brown.
            Features Goodyear welt construction for durability and resoling capability.
            Leather-lined interior with cushioned insoles for all-day comfort.
            Stacked leather heel with rubber top lift for traction.
            Available in sizes 7-13 including half sizes.
            Perfect for formal occasions, business meetings, and special events.
            Includes dust bags and cedar shoe trees.
            Specifications:
            - Material: Full-grain leather upper, leather sole
            - Construction: Goodyear welt
            - Heel height: 1 inch
            - Toe shape: Round toe
            - Color: Mahogany brown
            - Origin: Made in Italy
            """
        )
        rich_result = extractor.get_all_characteristics(rich_prod)
        rich_time = time.time() - rich_start

        print(f"\nData skewing test:")
        print(f"  Minimal content time: {minimal_time:.2f}s - Results: {minimal_result}")
        print(f"  Rich content time: {rich_time:.2f}s - Results: {rich_result}")
        print(f"  Time difference: {abs(rich_time - minimal_time):.2f}s")

        # Both should return valid results
        self.assertIsInstance(minimal_result, dict)
        self.assertIsInstance(rich_result, dict)

        # Rich content should extract more detailed characteristics
        minimal_colors = minimal_result.get('color', [])
        rich_colors = rich_result.get('color', [])
        rich_materials = rich_result.get('material', [])

        self.assertTrue(len(rich_materials) > 0, "Rich content should extract materials")
        # Rich content likely extracted "brown" from detailed description
        colors_str = ' '.join([c.lower() for c in rich_colors])
        self.assertTrue('brown' in colors_str or 'mahogany' in colors_str,
                       f"Expected brown/mahogany in rich content, got: {rich_colors}")

    def test_data_skewing_across_categories(self):
        """
        Test data skewing across different super_categories.
        Some categories may have more characteristics, causing processing time variations.
        """
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=self.category_def,
            gen_ai_api_config=self.gen_ai_config
        )

        processing_times = {}

        # Test footwear (4 characteristics: color, material, size, type)
        start = time.time()
        footwear_result = extractor.get_all_characteristics(
            LLMProductRequestData(
                super_category="footwear",
                product_title="Red Running Shoes",
                product_information="Red mesh running shoes in size 10"
            )
        )
        processing_times['footwear'] = time.time() - start

        # Test clothing (3 characteristics: color, size, material)
        start = time.time()
        clothing_result = extractor.get_all_characteristics(
            LLMProductRequestData(
                super_category="clothing",
                product_title="Blue Cotton Shirt",
                product_information="Blue cotton shirt in size M"
            )
        )
        processing_times['clothing'] = time.time() - start

        # Test electronics (2 characteristics: brand, color)
        start = time.time()
        electronics_result = extractor.get_all_characteristics(
            LLMProductRequestData(
                super_category="electronics",
                product_title="Samsung Galaxy Phone",
                product_information="Samsung smartphone in black color"
            )
        )
        processing_times['electronics'] = time.time() - start

        print(f"\nCategory processing times:")
        for category, proc_time in processing_times.items():
            print(f"  {category}: {proc_time:.2f}s")

        # Verify all categories return proper structure
        self.assertEqual(len(footwear_result), 4, "Footwear should have 4 characteristics")
        self.assertEqual(len(clothing_result), 3, "Clothing should have 3 characteristics")
        self.assertEqual(len(electronics_result), 2, "Electronics should have 2 characteristics")

    def test_result_ordering_for_spark_structtype(self):
        """
        Test that results maintain consistent key ordering for Spark StructType.
        This is CRITICAL as PySpark StructType is position-based, not key-based.
        """
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=self.category_def,
            gen_ai_api_config=self.gen_ai_config
        )

        # Run extraction multiple times
        results = []
        for i in range(3):
            result = extractor.get_all_characteristics(
                LLMProductRequestData(
                    super_category="footwear",
                    product_title=f"Test Shoes {i}",
                    product_information="Red leather shoes in size 10"
                )
            )
            results.append(result)

        # All results should have same keys in same order
        keys_list = [list(r.keys()) for r in results]

        print(f"\nKey ordering test:")
        for i, keys in enumerate(keys_list):
            print(f"  Run {i+1}: {keys}")

        # All should have identical key ordering
        for i in range(1, len(keys_list)):
            self.assertEqual(keys_list[0], keys_list[i],
                           f"Key order mismatch: {keys_list[0]} != {keys_list[i]}")

        # Simulate Spark tuple conversion (CRITICAL for avoiding column misalignment)
        characteristics_col_names = list(results[0].keys())
        for result in results:
            result_values = []
            for col_name in characteristics_col_names:
                value = result.get(col_name, [])
                if isinstance(value, list):
                    result_values.append(", ".join(str(v) for v in value) if value else None)
                elif value is not None:
                    result_values.append(str(value))
                else:
                    result_values.append(None)

            result_tuple = tuple(result_values)
            print(f"  Tuple for Spark: {result_tuple}")
            self.assertEqual(len(result_tuple), len(characteristics_col_names))

    def test_null_and_missing_attributes_row(self):
        """
        Test handling of Spark Rows with missing or null attributes.
        This is common in real-world data.
        """
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=self.category_def,
            gen_ai_api_config=self.gen_ai_config
        )

        # Row with missing description and specs
        PartialRow = namedtuple('PartialRow', ['super_category', 'product_title'])
        row = PartialRow(super_category='footwear', product_title='Mystery Shoes')

        # Simulate real-world getattr with defaults
        description = getattr(row, 'description', None)
        specs = getattr(row, 'specs', None)

        if description and specs:
            product_information = f"specs: {specs}, description: {description}"
        else:
            product_information = None

        result = extractor.get_all_characteristics(
            LLMProductRequestData(
                super_category=getattr(row, 'super_category', None),
                product_title=getattr(row, 'product_title', None),
                product_information=product_information
            )
        )

        print(f"\nMissing attributes test: {result}")

        # Should return empty lists for all characteristics
        self.assertIsInstance(result, dict)
        for key, value in result.items():
            self.assertEqual(value, [], f"Expected empty list for {key} when no product_information")

    def test_concurrent_processing_simulation(self):
        """
        Test simulating concurrent processing of multiple products.
        This helps identify potential race conditions or state issues.
        """
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=self.category_def,
            gen_ai_api_config=self.gen_ai_config
        )

        # Simulate processing multiple products sequentially (as Spark would in a partition)
        products = [
            LLMProductRequestData(
                super_category="footwear",
                product_title="Red Sneakers",
                product_information="Red canvas sneakers"
            ),
            LLMProductRequestData(
                super_category="clothing",
                product_title="Blue Jeans",
                product_information="Blue denim jeans"
            ),
            LLMProductRequestData(
                super_category="footwear",
                product_title="Black Boots",
                product_information="Black leather boots"
            ),
            LLMProductRequestData(
                super_category="electronics",
                product_title="Apple iPhone",
                product_information="Apple smartphone in white"
            ),
        ]

        results = []
        for i, prod in enumerate(products):
            start = time.time()
            result = extractor.get_all_characteristics(prod)
            elapsed = time.time() - start
            results.append({
                'product': prod.product_title,
                'category': prod.super_category,
                'result': result,
                'time': elapsed
            })
            print(f"  Processed {prod.product_title}: {elapsed:.2f}s")

        # Verify all processed correctly
        self.assertEqual(len(results), len(products))

        # Check for consistent results by category
        footwear_results = [r for r in results if r['category'] == 'footwear']
        for fr in footwear_results:
            self.assertEqual(set(fr['result'].keys()), {'color', 'material', 'size', 'type'})

    def test_edge_case_very_long_product_information(self):
        """
        Test edge case with very long product information.
        This might cause data skewing if LLM takes longer to process.
        """
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=self.category_def,
            gen_ai_api_config=self.gen_ai_config
        )

        # Create very long product description
        long_description = """
        Premium athletic running shoes designed for marathon runners and serious athletes.
        """ + " ".join([f"Feature {i}: Advanced cushioning technology." for i in range(50)])

        start = time.time()
        result = extractor.get_all_characteristics(
            LLMProductRequestData(
                super_category="footwear",
                product_title="Ultra Marathon Running Shoes",
                product_information=long_description
            )
        )
        elapsed = time.time() - start

        print(f"\nLong description test:")
        print(f"  Description length: {len(long_description)} chars")
        print(f"  Processing time: {elapsed:.2f}s")
        print(f"  Results: {result}")

        # Should still return valid results
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 4)  # All footwear characteristics

    def test_data_skewing_analysis_comprehensive(self):
        """
        Comprehensive data skewing analysis with statistical reporting.
        This test helps identify skewing patterns for Spark optimization.
        """
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=self.category_def,
            gen_ai_api_config=self.gen_ai_config
        )

        # Create diverse product samples representing real-world distribution
        test_products = [
            # Footwear - varying complexity
            ("footwear", "Basic Black Sneakers", "Black sneakers", "minimal"),
            ("footwear", "Nike Air Max", "Red and white Nike running shoes with mesh upper", "medium"),
            ("footwear", "Premium Leather Boots",
             "Handcrafted brown leather boots with Goodyear welt construction, sizes 8-13, premium quality", "rich"),

            # Clothing - varying complexity
            ("clothing", "T-Shirt", "Cotton t-shirt", "minimal"),
            ("clothing", "Denim Jeans", "Blue denim jeans in size M", "medium"),

            # Electronics - varying complexity
            ("electronics", "Phone", "Smartphone", "minimal"),
            ("electronics", "Samsung TV", "Samsung 55-inch smart TV in black", "medium"),
        ]

        results = []
        for category, title, description, complexity in test_products:
            start = time.time()
            result = extractor.get_all_characteristics(
                LLMProductRequestData(
                    super_category=category,
                    product_title=title,
                    product_information=description
                )
            )
            elapsed = time.time() - start

            results.append({
                'category': category,
                'title': title,
                'complexity': complexity,
                'num_chars': len(description),
                'num_characteristics': len(result),
                'processing_time': elapsed,
                'result': result
            })

        # Calculate statistics
        print("\n" + "="*80)
        print("DATA SKEWING ANALYSIS REPORT")
        print("="*80)

        # Group by category
        by_category = {}
        for r in results:
            if r['category'] not in by_category:
                by_category[r['category']] = []
            by_category[r['category']].append(r)

        print("\n1. PROCESSING TIME BY CATEGORY:")
        print("-" * 80)
        for category, items in by_category.items():
            times = [i['processing_time'] for i in items]
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            num_chars = items[0]['num_characteristics']

            print(f"\n{category.upper()}:")
            print(f"  Characteristics: {num_chars}")
            print(f"  Avg time: {avg_time:.2f}s")
            print(f"  Min time: {min_time:.2f}s")
            print(f"  Max time: {max_time:.2f}s")
            print(f"  Range: {max_time - min_time:.2f}s")
            print(f"  Skew factor: {max_time / min_time if min_time > 0 else 0:.2f}x")

        # Group by complexity
        by_complexity = {}
        for r in results:
            if r['complexity'] not in by_complexity:
                by_complexity[r['complexity']] = []
            by_complexity[r['complexity']].append(r)

        print("\n2. PROCESSING TIME BY CONTENT COMPLEXITY:")
        print("-" * 80)
        for complexity in ['minimal', 'medium', 'rich']:
            if complexity in by_complexity:
                items = by_complexity[complexity]
                times = [i['processing_time'] for i in items]
                avg_time = sum(times) / len(times)
                print(f"\n{complexity.upper()}:")
                print(f"  Avg time: {avg_time:.2f}s")
                print(f"  Sample count: {len(items)}")

        # Overall statistics
        all_times = [r['processing_time'] for r in results]
        print("\n3. OVERALL STATISTICS:")
        print("-" * 80)
        print(f"Total products tested: {len(results)}")
        print(f"Average processing time: {sum(all_times) / len(all_times):.2f}s")
        print(f"Min processing time: {min(all_times):.2f}s")
        print(f"Max processing time: {max(all_times):.2f}s")
        print(f"Overall skew factor: {max(all_times) / min(all_times) if min(all_times) > 0 else 0:.2f}x")

        # Skewing recommendations
        print("\n4. SPARK OPTIMIZATION RECOMMENDATIONS:")
        print("-" * 80)
        max_skew = max(all_times) / min(all_times) if min(all_times) > 0 else 0

        if max_skew > 5.0:
            print("⚠️  CRITICAL: Severe data skewing detected (>5x difference)")
            print("   - Consider using repartition() by category before processing")
            print("   - Use coalesce() to reduce partitions for small categories")
            print("   - Consider salting the partition key to distribute load")
        elif max_skew > 3.0:
            print("⚠️  WARNING: Significant data skewing detected (>3x difference)")
            print("   - Monitor partition sizes during processing")
            print("   - Consider adaptive query execution (AQE) if using Spark 3.0+")
        else:
            print("✓  Acceptable skewing levels (<3x difference)")

        print("\n" + "="*80)

        # All products should process successfully
        self.assertEqual(len(results), len(test_products))
        for result in results:
            self.assertIsInstance(result['result'], dict)
            self.assertGreater(result['num_characteristics'], 0)


if __name__ == '__main__':
    unittest.main()