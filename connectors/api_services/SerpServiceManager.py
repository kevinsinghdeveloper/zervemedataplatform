from dataclasses import asdict
from typing import Union

import requests

from abstractions.types.models.SerpJobProductDetailsRequestResourceModel import \
    SerpJobProductDetailsRequestResourceModel
from abstractions.types.models.SerpJobRequestResourceModel import SerpJobRequestResourceModel
from abstractions.types.models.SerpJobSearchRequestResourceModel import \
    SerpJobSearchRequestResourceModel
from abstractions.types.models.SerpProductDetailsResponseModel import SerpProductDetailsResponseModel
from abstractions.types.models.SerpSearchResponseModel import SerpSearchResponseModel
from utils.Utility import Utility

DEFAULT_PAGE = 1
DEFAULT_MAX_PAGE = 1
DEFAULT_NUM_OF_ITEMS_IN_PAGE = 50

class SerpServiceManager:
    def __init__(self, api_config: dict):
        if not api_config:
            raise ValueError("API configuration is required")

        self._api_config = api_config
        self.__filters = self._api_config["api_filters"]
        self.__default_params = self._api_config["default_api_filters"]

    def __get_filter_params(self) -> dict:
        return self.__filters.copy()

    def __get_default_params(self) -> dict:
        return self.__default_params.copy()

    def __get(self, rec_resource_model: SerpJobRequestResourceModel) -> Union[list, None]:
        request_data = asdict(rec_resource_model)

        filters = self.__get_filter_params()

        # fill in other params
        param_filters = self.__get_default_params()

        for k, v in filters.items():
            r = request_data.get(k)

            if r:
                param_filters[v] = r

        # cast all items to str
        param_filters = {k: str(param_filters[k]) for k in param_filters}

        api_result = None
        try:
            api_result = requests.get(self._api_config["api_base_url"], param_filters)
            api_result.raise_for_status()  # Raise an HTTPError for bad responses (4xx, 5xx)
        except requests.exceptions.HTTPError as http_err:
            Utility.error_log(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            Utility.error_log(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            Utility.error_log(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            Utility.error_log(f"An error occurred: {req_err}")
        except Exception as err:
            Utility.error_log(f"An unexpected error occurred: {err}")

        if api_result is None:
            Utility.error_log("Failed to get response from serp")
            return []

        api_result = api_result.json()

        return api_result

    def get_serp_shopping_search_data(self, search_config) -> Union[list[SerpSearchResponseModel] | None]:
        page = search_config.get("page", DEFAULT_PAGE)
        max_page = search_config.get("max_page", DEFAULT_MAX_PAGE)

        serp_request = SerpJobSearchRequestResourceModel(
            query=search_config['query'],
            page=page,
            max_page=max_page,
            num_items_on_page=search_config.get("num_items_on_page", DEFAULT_NUM_OF_ITEMS_IN_PAGE),
            price_min=search_config.get("price_min", None),
            price_max=search_config.get("price_max", None)
        )
        Utility.log(f"Running search query `{serp_request.query}` for page `{serp_request.page}` of `{serp_request.max_page}`")
        api_result = self.__get(serp_request)

        related_shopping_items_key = self._api_config['result_config']["related_shopping_items"]
        nested_keys_to_use = self._api_config['result_config']["nested_keys_to_use"]

        if not api_result or related_shopping_items_key not in api_result:
            Utility.error_log(f"Failed to get shopping search results from serp for query {serp_request.query}")
            return []

        related_shopping_items = api_result[related_shopping_items_key]

        if len(related_shopping_items) == 0:
            Utility.error_log("No results returned from serp")
            return []

        serp_response_items = []
        for item in related_shopping_items:
            serp_response = SerpSearchResponseModel(
                product_title=item.get(nested_keys_to_use['product_title'], None),
                product_id=item.get(nested_keys_to_use['product_id'], None),
                gpc_id=item.get(nested_keys_to_use['gpc_id'], None),
                url=item.get(nested_keys_to_use['url'], None),
                merchant=item.get(nested_keys_to_use['merchant'], None),
                price=Utility.clean_and_convert(item.get(nested_keys_to_use['price'], 0)),
                position_rank=Utility.clean_and_convert(item.get(nested_keys_to_use['position_rank'], 0)),
                rating=Utility.clean_and_convert(item.get(nested_keys_to_use['rating'], 0)),
                reviews=Utility.clean_and_convert(item.get(nested_keys_to_use['reviews'], 0)),
                product_image=item.get(nested_keys_to_use['product_image'], None),
            )

            serp_response_items.append(serp_response)

        return serp_response_items

    def get_serp_product_details_search(self, search_config) -> Union[list[SerpProductDetailsResponseModel] | None]:
        serp_request = SerpJobProductDetailsRequestResourceModel(
            product_id=search_config['product_id'],
            gpc_id=search_config.get('gpc_id', None)
        )

        api_result = self.__get(serp_request)

        if not api_result or self._api_config['result_config']["product_results"] not in api_result:
            Utility.error_log(f"Failed to get product details from serp for product id {serp_request.product_id}, gpc id {serp_request.gpc_id}")
            return []

        product_results = self._api_config['result_config']["product_results"]
        nested_keys_to_use = self._api_config['result_config']["nested_keys_to_use"]

        product_details_data = api_result[product_results]

        if len(product_details_data) == 0:
            Utility.error_log("No results returned from serp")
            return []

        # Helper function to get nested values
        def get_nested_value(data, keys):
            if isinstance(keys, list):
                # Try each key in the list until we find a value
                for key in keys:
                    value = data.get(key)
                    if value is not None:
                        return value
                return None
            else:
                # Single key
                return data.get(keys, None)

        serp_response = SerpProductDetailsResponseModel(
            product_id=get_nested_value(product_details_data, nested_keys_to_use['product_id']),
            gpc_id=search_config.get('gpc_id', None),
            description=str(get_nested_value(product_details_data, nested_keys_to_use['description'])),
            sellers=str(get_nested_value(product_details_data, nested_keys_to_use['sellers'])),
            specs=str(get_nested_value(product_details_data, nested_keys_to_use['specs']))
        )

        return [serp_response]