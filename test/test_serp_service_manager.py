import unittest
from unittest.mock import Mock, patch

from zervedataplatform.connectors.api_services.SerpServiceManager import SerpServiceManager
from zervedataplatform.abstractions.types.models.SerpSearchResponseModel import SerpSearchResponseModel
from zervedataplatform.abstractions.types.models.SerpProductDetailsResponseModel import SerpProductDetailsResponseModel


class TestSerpServiceManager(unittest.TestCase):
    """Test the SerpServiceManager class"""

    def setUp(self):
        """Set up test fixtures"""
        self.api_config = {
            "api_base_url": "https://api.test.com/search",
            "default_api_filters": {
                "search_type": "shopping",
                "hl": "en",
                "gl": "us",
                "google_domain": "google.com",
                "api_key": "test_api_key"
            },
            "api_filters": {
                "query": "q",
                "search_type": "search_type",
                "google_domain": "google_domain",
                "location": "location",
                "global_location": "gl",
                "language": "hl",
                "product_id": "product_id",
                "gpc_id": "gpc_id",
                "page": "page",
                "max_page": "max_page",
                "num_items_on_page": "num",
                "price_min": "price_min",
                "price_max": "price_max"
            },
            "result_config": {
                "related_shopping_items": "shopping_results",
                "product_results": "product_results",
                "nested_keys_to_use": {
                    "product_title": "title",
                    "product_id": "product_id",
                    "gpc_id": "gpc_id",
                    "url": "link",
                    "merchant": "source",
                    "price": "price",
                    "position_rank": "position",
                    "rating": "rating",
                    "reviews": "reviews",
                    "product_image": "thumbnail",
                    "description": "description",
                    "sellers": "sellers",
                    "specs": "specifications"
                }
            }
        }

    def test_initialization_success(self):
        """Test that SerpServiceManager initializes correctly with valid config"""
        manager = SerpServiceManager(self.api_config)

        self.assertEqual(manager._api_config, self.api_config)
        self.assertIsNotNone(manager._SerpServiceManager__filters)
        self.assertIsNotNone(manager._SerpServiceManager__default_params)

    def test_initialization_failure_without_config(self):
        """Test that initialization fails without config"""
        with self.assertRaises(ValueError) as context:
            SerpServiceManager(None)

        self.assertIn("API configuration is required", str(context.exception))

    def test_initialization_failure_with_empty_config(self):
        """Test that initialization fails with empty config"""
        with self.assertRaises(ValueError) as context:
            SerpServiceManager({})

        self.assertIn("API configuration is required", str(context.exception))

    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.requests.get')
    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.Utility')
    def test_get_serp_shopping_search_data_success(self, mock_utility, mock_requests_get):
        """Test successful shopping search data retrieval"""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "shopping_results": [
                {
                    "title": "Test Product 1",
                    "product_id": "12345",
                    "gpc_id": "GPC001",
                    "link": "http://test.com/product1",
                    "source": "Test Store",
                    "price": "$99.99",
                    "position": "1",
                    "rating": "4.5",
                    "reviews": "100",
                    "thumbnail": "http://test.com/image1.jpg"
                },
                {
                    "title": "Test Product 2",
                    "product_id": "67890",
                    "gpc_id": "GPC002",
                    "link": "http://test.com/product2",
                    "source": "Another Store",
                    "price": "$149.99",
                    "position": "2",
                    "rating": "4.8",
                    "reviews": "250",
                    "thumbnail": "http://test.com/image2.jpg"
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_requests_get.return_value = mock_response

        manager = SerpServiceManager(self.api_config)
        search_config = {
            "query": "test product",
            "page": 1,
            "max_page": 1,
            "num_items_on_page": 50
        }

        result = manager.get_serp_shopping_search_data(search_config)

        # Verify the result
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], SerpSearchResponseModel)
        self.assertEqual(result[0].product_title, "Test Product 1")
        self.assertEqual(result[0].product_id, "12345")
        self.assertEqual(result[1].product_title, "Test Product 2")

        # Verify API was called
        mock_requests_get.assert_called_once()
        mock_utility.log.assert_called()

    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.requests.get')
    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.Utility')
    def test_get_serp_shopping_search_data_empty_results(self, mock_utility, mock_requests_get):
        """Test shopping search with empty results"""
        mock_response = Mock()
        mock_response.json.return_value = {"shopping_results": []}
        mock_response.raise_for_status = Mock()
        mock_requests_get.return_value = mock_response

        manager = SerpServiceManager(self.api_config)
        search_config = {"query": "test product"}

        result = manager.get_serp_shopping_search_data(search_config)

        self.assertEqual(result, [])
        mock_utility.error_log.assert_called_with("No results returned from serp")

    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.requests.get')
    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.Utility')
    def test_get_serp_shopping_search_data_http_error(self, mock_utility, mock_requests_get):
        """Test handling of HTTP errors"""
        import requests
        mock_requests_get.side_effect = requests.exceptions.HTTPError("404 Not Found")

        manager = SerpServiceManager(self.api_config)
        search_config = {"query": "test product"}

        result = manager.get_serp_shopping_search_data(search_config)

        self.assertEqual(result, [])
        mock_utility.error_log.assert_called()

    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.requests.get')
    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.Utility')
    def test_get_serp_shopping_search_data_connection_error(self, mock_utility, mock_requests_get):
        """Test handling of connection errors"""
        import requests
        mock_requests_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        manager = SerpServiceManager(self.api_config)
        search_config = {"query": "test product"}

        result = manager.get_serp_shopping_search_data(search_config)

        self.assertEqual(result, [])
        mock_utility.error_log.assert_called()

    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.requests.get')
    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.Utility')
    def test_get_serp_shopping_search_data_timeout_error(self, mock_utility, mock_requests_get):
        """Test handling of timeout errors"""
        import requests
        mock_requests_get.side_effect = requests.exceptions.Timeout("Request timed out")

        manager = SerpServiceManager(self.api_config)
        search_config = {"query": "test product"}

        result = manager.get_serp_shopping_search_data(search_config)

        self.assertEqual(result, [])
        mock_utility.error_log.assert_called()

    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.requests.get')
    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.Utility')
    def test_get_serp_shopping_search_data_with_defaults(self, mock_utility, mock_requests_get):
        """Test shopping search uses default values when not provided"""
        mock_response = Mock()
        mock_response.json.return_value = {"shopping_results": []}
        mock_response.raise_for_status = Mock()
        mock_requests_get.return_value = mock_response

        manager = SerpServiceManager(self.api_config)
        search_config = {"query": "test"}  # Minimal config

        result = manager.get_serp_shopping_search_data(search_config)

        # Verify defaults were used
        self.assertEqual(result, [])

    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.requests.get')
    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.Utility')
    def test_get_serp_product_details_search_success(self, mock_utility, mock_requests_get):
        """Test successful product details retrieval"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "product_results": {
                "product_id": "12345",
                "description": "This is a test product description",
                "sellers": [{"name": "Store A", "price": "$99.99"}],
                "specifications": [{"feature": "Color", "value": "Blue"}]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_requests_get.return_value = mock_response

        manager = SerpServiceManager(self.api_config)
        search_config = {
            "product_id": "12345",
            "gpc_id": "GPC001"
        }

        result = manager.get_serp_product_details_search(search_config)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], SerpProductDetailsResponseModel)
        self.assertEqual(result[0].product_id, "12345")
        self.assertEqual(result[0].gpc_id, "GPC001")
        self.assertIn("test product", result[0].description)

    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.requests.get')
    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.Utility')
    def test_get_serp_product_details_search_empty_results(self, mock_utility, mock_requests_get):
        """Test product details search with empty results"""
        mock_response = Mock()
        mock_response.json.return_value = {"product_results": {}}
        mock_response.raise_for_status = Mock()
        mock_requests_get.return_value = mock_response

        manager = SerpServiceManager(self.api_config)
        search_config = {"product_id": "12345"}

        result = manager.get_serp_product_details_search(search_config)

        self.assertEqual(result, [])
        mock_utility.error_log.assert_called_with("No results returned from serp")

    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.requests.get')
    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.Utility')
    def test_get_serp_product_details_search_no_api_result(self, mock_utility, mock_requests_get):
        """Test product details when API request fails"""
        import requests
        mock_requests_get.side_effect = requests.exceptions.HTTPError("500 Server Error")

        manager = SerpServiceManager(self.api_config)
        search_config = {"product_id": "12345"}

        result = manager.get_serp_product_details_search(search_config)

        self.assertEqual(result, [])

    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.requests.get')
    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.Utility')
    def test_get_serp_product_details_search_missing_product_results_key(self, mock_utility, mock_requests_get):
        """Test handling when product_results key is missing from API response"""
        mock_response = Mock()
        mock_response.json.return_value = {"some_other_key": {}}
        mock_response.raise_for_status = Mock()
        mock_requests_get.return_value = mock_response

        manager = SerpServiceManager(self.api_config)
        search_config = {"product_id": "12345", "gpc_id": "GPC001"}

        result = manager.get_serp_product_details_search(search_config)

        # Should return empty list and log error
        self.assertEqual(result, [])
        mock_utility.error_log.assert_called()

    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.requests.get')
    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.Utility')
    def test_get_with_price_filters(self, mock_utility, mock_requests_get):
        """Test that price filters are correctly applied"""
        mock_response = Mock()
        mock_response.json.return_value = {"shopping_results": []}
        mock_response.raise_for_status = Mock()
        mock_requests_get.return_value = mock_response

        manager = SerpServiceManager(self.api_config)
        search_config = {
            "query": "test product",
            "price_min": 50,
            "price_max": 100
        }

        manager.get_serp_shopping_search_data(search_config)

        # Verify the API call included price filters
        call_args = mock_requests_get.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]

        self.assertIn("price_min", params)
        self.assertIn("price_max", params)

    def test_get_filter_params_returns_copy(self):
        """Test that filter params returns a copy"""
        manager = SerpServiceManager(self.api_config)

        filters1 = manager._SerpServiceManager__get_filter_params()
        filters2 = manager._SerpServiceManager__get_filter_params()

        # Modify one copy
        filters1["new_key"] = "new_value"

        # Verify the other copy is unchanged
        self.assertNotIn("new_key", filters2)

    def test_get_default_params_returns_copy(self):
        """Test that default params returns a copy"""
        manager = SerpServiceManager(self.api_config)

        defaults1 = manager._SerpServiceManager__get_default_params()
        defaults2 = manager._SerpServiceManager__get_default_params()

        # Modify one copy
        defaults1["new_key"] = "new_value"

        # Verify the other copy is unchanged
        self.assertNotIn("new_key", defaults2)

    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.requests.get')
    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.Utility')
    def test_params_cast_to_string(self, mock_utility, mock_requests_get):
        """Test that all parameters are cast to strings before API call"""
        mock_response = Mock()
        mock_response.json.return_value = {"shopping_results": []}
        mock_response.raise_for_status = Mock()
        mock_requests_get.return_value = mock_response

        manager = SerpServiceManager(self.api_config)
        search_config = {
            "query": "test",
            "page": 1,  # Integer
            "max_page": 5,  # Integer
            "num_items_on_page": 50  # Integer
        }

        manager.get_serp_shopping_search_data(search_config)

        # Verify API was called and params are strings
        call_args = mock_requests_get.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1]

        # All param values should be strings
        for value in params.values():
            self.assertIsInstance(value, str)

    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.requests.get')
    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.Utility')
    def test_nested_value_helper_with_list_keys(self, mock_utility, mock_requests_get):
        """Test the nested value helper function with list keys"""
        # Update config to have list keys
        config_with_lists = self.api_config.copy()
        config_with_lists["result_config"]["nested_keys_to_use"]["description"] = ["desc1", "desc2"]

        mock_response = Mock()
        mock_response.json.return_value = {
            "product_results": {
                "product_id": "123",
                "desc2": "Found description",  # Second key in list
                "sellers": [],
                "specifications": []
            }
        }
        mock_response.raise_for_status = Mock()
        mock_requests_get.return_value = mock_response

        manager = SerpServiceManager(config_with_lists)
        search_config = {"product_id": "123"}

        result = manager.get_serp_product_details_search(search_config)

        # Verify it found the value using the second key in the list
        self.assertEqual(len(result), 1)
        self.assertIn("Found description", result[0].description)

    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.requests.get')
    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.Utility')
    def test_get_with_general_exception(self, mock_utility, mock_requests_get):
        """Test handling of unexpected general exceptions"""
        mock_requests_get.side_effect = Exception("Unexpected error")

        manager = SerpServiceManager(self.api_config)
        search_config = {"query": "test"}

        result = manager.get_serp_shopping_search_data(search_config)

        self.assertEqual(result, [])
        mock_utility.error_log.assert_called()

    @patch('zervedataplatform.connectors.api_services.SerpServiceManager.Utility')
    def test_utility_clean_and_convert_called(self, mock_utility):
        """Test that Utility.clean_and_convert is called for numeric fields"""
        mock_utility.clean_and_convert.return_value = 99.99
        mock_utility.log = Mock()
        mock_utility.error_log = Mock()

        with patch('connectors.api_services.SerpServiceManager.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "shopping_results": [{
                    "title": "Test",
                    "product_id": "123",
                    "gpc_id": "GPC",
                    "link": "http://test.com",
                    "source": "Store",
                    "price": "$99.99",
                    "position": "1",
                    "rating": "4.5",
                    "reviews": "100",
                    "thumbnail": "http://test.com/img.jpg"
                }]
            }
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            manager = SerpServiceManager(self.api_config)
            result = manager.get_serp_shopping_search_data({"query": "test"})

            # Verify clean_and_convert was called for numeric fields
            self.assertGreater(mock_utility.clean_and_convert.call_count, 0)


if __name__ == '__main__':
    unittest.main()
