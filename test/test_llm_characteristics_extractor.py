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
            "system_prompt": """You are a product analyst. Extract characteristics from product descriptions.
            Return ONLY valid JSON in the exact format shown in examples. Do not include any markdown formatting or code blocks.
            
            Example 1:
            Product: Red leather Nike Air Max running shoes, available in sizes 9, 10, and 11
            Characteristic: color
            Output: {"color": ["red"]}
            
            Example 2:
            Product: Black and white canvas sneakers made of cotton canvas
            Characteristic: material
            Output: {"material": ["canvas"]}
            """,
            "examples": ""
        }

        # Category definition config with multiple categories
        # In real-world usage, this would be loaded from JSON files using Utility.read_in_json_file()
        # and passed as a list of dictionaries to the constructor
        self.category_def = {
            "footwear": {
                "color": {
                    "description": "The color or colors of the footwear",
                    "is_multi": True,
                    "options": ["red", "blue", "green", "black", "white", "brown", "gray", "yellow", "orange", "purple"]
                },
                "material": {
                    "description": "The material the footwear is made from",
                    "is_multi": True,
                    "options": ["leather", "synthetic", "canvas", "suede", "mesh", "rubber", "textile"]
                },
                "size": {
                    "description": "Available shoe sizes",
                    "is_multi": True,
                    "options": ["6", "7", "8", "9", "10", "11", "12", "13"]
                },
                "type": {
                    "description": "Type of footwear",
                    "is_multi": False,
                    "options": ["sneakers", "boots", "sandals", "dress shoes", "running shoes", "casual shoes"]
                }
            },
            "clothing": {
                "color": {
                    "description": "The color or colors of the clothing item",
                    "is_multi": True,
                    "options": ["red", "blue", "green", "black", "white", "brown", "gray", "yellow", "orange", "purple"]
                },
                "size": {
                    "description": "Available clothing sizes",
                    "is_multi": True,
                    "options": ["XS", "S", "M", "L", "XL", "XXL"]
                },
                "material": {
                    "description": "The fabric or material",
                    "is_multi": True,
                    "options": ["cotton", "polyester", "wool", "silk", "denim", "leather"]
                }
            },
            "electronics": {
                "brand": {
                    "description": "The brand or manufacturer",
                    "is_multi": False,
                    "options": ["Apple", "Samsung", "Sony", "LG", "Dell", "HP", "Lenovo"]
                },
                "color": {
                    "description": "The color of the device",
                    "is_multi": True,
                    "options": ["black", "white", "silver", "gray", "blue", "red"]
                }
            }
        }

        # Gen AI API config - points to local Ollama instance
        self.gen_ai_config = {
          "provider": "ollama",
          "model_name": "llama3.2",
          "base_url": "http://192.168.68.74:11434",
          "gen_config": {
            "temperature": 0.7,
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
            llm_characteristics_configs=[self.category_def],
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
            llm_characteristics_configs=[self.category_def],
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
            llm_characteristics_configs=[self.category_def],
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

    def test_live_extraction_multiple_colors(self):
        """Test extraction with multiple colors in product"""
        extractor = LLMCharacteristicsExtractor(
            llm_ex_config=self.llm_ex_config,
            llm_characteristics_configs=[self.category_def],
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
            llm_characteristics_configs=[self.category_def],
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
            llm_characteristics_configs=[self.category_def],
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
            llm_characteristics_configs=[self.category_def],
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
            llm_characteristics_configs=[self.category_def],
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
            llm_characteristics_configs=[self.category_def],
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
            llm_characteristics_configs=[self.category_def],
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
            llm_characteristics_configs=[self.category_def],
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
            llm_characteristics_configs=[self.category_def],
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
            llm_characteristics_configs=[self.category_def],
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
            llm_characteristics_configs=[self.category_def],
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
            llm_characteristics_configs=[self.category_def],
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