import ast
import json
from dataclasses import asdict
from typing import Any

from zervedataplatform.abstractions.types.models import LLMProductRequestData
from zervedataplatform.abstractions.types.models.LLMProductRequestData import LLMCharacteristicOption
from zervedataplatform.connectors.ai.LangChainLLMConnector import LangChainLLMConnector
from zervedataplatform.utils.Utility import Utility


class LLMCharacteristicsExtractor:
    def __init__(self, llm_ex_config: dict, llm_characteristics_configs: [str], gen_ai_api_config: dict):
        self.__system_prompt = llm_ex_config.get("system_prompt")
        self.__examples = llm_ex_config.get("examples", "")
        self.__llm_characteristics_configs = llm_characteristics_configs
        self.__char_category_config = {}
        self.__llm = LangChainLLMConnector(gen_ai_api_config=gen_ai_api_config)

        # Store original config for serialization
        self.__llm_ex_config = llm_ex_config
        self.__gen_ai_api_config = gen_ai_api_config

        self.__configure_characteristic_category_def_maps()

    def __configure_characteristic_category_def_maps(self):
        char_category_config = {}
        for config in self.__llm_characteristics_configs:
            config = Utility.read_in_json_file(config)
            char_category_config.update(config)

        self.__char_category_config = char_category_config

    def get_all_characteristics(self, prod_data: LLMProductRequestData) -> list[Any] | dict[Any, Any]:
        characteristics_from_config = self.__char_category_config.get(prod_data.super_category)
        characteristics_map = {}


        if not characteristics_from_config:
            return []

        if prod_data.product_information:
            for characteristic, desc in characteristics_from_config.items():
                characteristic_data = LLMCharacteristicOption(
                    characteristic=characteristic,
                    description=desc.get("description", ""),
                    is_multi=desc.get("is_multi", False),
                    options=desc.get("options", [])
                )

                char_list = self.__get_characteristic(prod_data, characteristic_data)

                if char_list:
                    characteristics_map[characteristic_data.characteristic] = char_list

        # add missing characteristics with empty list
        for characteristic in characteristics_from_config.keys():
            if characteristic not in characteristics_map:
                characteristics_map[characteristic] = []

        return characteristics_map


    def __get_characteristic(self, prod_data: LLMProductRequestData,
                             characteristic_data: LLMCharacteristicOption) -> list:

        prompt = LLMCharacteristicsExtractor.transform_data_for_prompt(prod_data, characteristic_data)

        res, _ = self.__llm.submit_data_prompt(prompt=prompt,
                                             llm_instructions=self.__system_prompt + "\n" + self.__examples)

        # Ensure 'res' is not empty or just whitespace
        if not res or res.strip() == "":
            Utility.log(f"Empty response for characteristic: {characteristic_data.characteristic}")
            return []

        # If 'res' is a string, parse it into a Python object (dict or list)
        try:
            if isinstance(res, str):
                # Clean up the response - remove markdown code blocks if present
                res = res.strip()
                if res.startswith("```json"):
                    res = res[7:]  # Remove ```json
                if res.startswith("```"):
                    res = res[3:]  # Remove ```
                if res.endswith("```"):
                    res = res[:-3]  # Remove trailing ```
                res = res.strip()

                # Try to fix common JSON issues before parsing
                # If JSON is incomplete, try to close it
                if not res.endswith('}') and not res.endswith(']'):
                    # Count open/close braces and brackets
                    open_braces = res.count('{')
                    close_braces = res.count('}')
                    open_brackets = res.count('[')
                    close_brackets = res.count(']')

                    # Add missing closing characters
                    if open_braces > close_braces:
                        res += '}' * (open_braces - close_braces)
                    if open_brackets > close_brackets:
                        res += ']' * (open_brackets - close_brackets)

                    Utility.log(f"Attempted to fix incomplete JSON for '{characteristic_data.characteristic}'")

                # Try to parse as JSON first, then fall back to Python literal evaluation
                try:
                    res = json.loads(res)
                except json.JSONDecodeError as e:
                    # JSON failed, try Python literal eval (handles single quotes)
                    try:
                        Utility.log(f"JSON parse failed, trying ast.literal_eval for '{characteristic_data.characteristic}'")
                        res = ast.literal_eval(res)
                    except (ValueError, SyntaxError) as eval_error:
                        Utility.error_log(f"Failed to parse response for characteristic '{characteristic_data.characteristic}'")
                        Utility.error_log(f"JSON error: {e}")
                        Utility.error_log(f"ast.literal_eval error: {eval_error}")
                        Utility.error_log(f"Response was: {res[:500]}...")  # Log first 500 chars
                        return []

            # If 'res' is a dictionary, extract and flatten all list values
            if isinstance(res, dict):
                res = [item for v in res.values() if isinstance(v, list) for item in v]

            # If 'res' is a list of dictionaries, extract and flatten list values from the first dict
            elif isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
                res = [item for v in res[0].values() if isinstance(v, list) for item in v]

        except Exception as e:
            Utility.error_log(f"Error processing LLM response for '{characteristic_data.characteristic}': {e}")
            res = []

        # Ensure 'res' is always a list
        return res if isinstance(res, list) else []

    @staticmethod
    def transform_data_for_prompt(data: LLMProductRequestData,
                                  characteristic_data: LLMCharacteristicOption):
        prompt = f"""
            Product Data:
                {asdict(data)}

            Characteristic Information:
                {asdict(characteristic_data)}
        """

        return prompt

    def get_config(self) -> dict:
        """
        Returns the configuration needed to recreate this extractor.
        This is used for Spark serialization - instead of serializing the object,
        we serialize its configuration.
        """
        return {
            'config': self.__llm_ex_config,
            'category_maps': self.__llm_characteristics_configs,
            'ai_config': self.__gen_ai_api_config
        }




