import json
from typing import Dict, Optional, Union

from data_platform.abstractions.connectors.GenAIApiConnectorBase import GenAIApiConnectorBase

import google.generativeai as genai

from data_platform.abstractions.types.models.LLMData import LLMData
from data_platform.utils.DataTransformationUtility import DataTransformationUtility
from data_platform.utils.Utility import Utility


class GeminiGenAiConnector(GenAIApiConnectorBase):
    def __init__(self, gen_ai_api_config: dict):
        super().__init__(gen_ai_api_config)
        self.__model = None

    def configure_llm(self):
        config = self.get_config()
        genai.configure(api_key=config['api_key'])

        self.__model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-latest",
            generation_config=config['gen_config'],
        )

    def submit_general_prompt(self, prompt: str, llm_instructions: str, is_json: bool = False) -> Union[str, dict]:
        prompt = self.get_base_prompt(prompt, llm_instructions)

        response = self.__model.generate_content(prompt)

        usage_data = response.usage_metadata

        response = response.text

        response = response.replace('```json', '').replace('```python', '').replace('```', '')

        #response = response.replace("'", '"') # replace single quote with double
        if is_json:
            response = json.loads(response)

        return response, usage_data

    def submit_data_prompt(self, prompt: str, llm_instructions: str) -> \
            Union[Dict[str, Optional[LLMData]], dict]:

        response, usage_data = self.submit_general_prompt(prompt, llm_instructions, True)

        Utility.log(f"Response {response}")

        cleaned_data = DataTransformationUtility.LLMTransformations.transform_llm_output(response)

        return cleaned_data, usage_data

    def get_base_prompt(self, prompt: str, llm_instructions: str) -> str:
        new_prompt = f"""
        Instructions:
        {llm_instructions}
        
        Data to process (please follow Instructions section above on how to process and return data):
        {prompt}
        """

        return new_prompt
