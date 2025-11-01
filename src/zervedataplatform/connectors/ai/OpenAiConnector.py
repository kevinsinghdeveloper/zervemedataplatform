from openai import OpenAI

from zervedataplatform.abstractions.connectors.ChatAgentAIConnectorBase import ChatAgentAiConnectorBase


class OpenAiConnector(ChatAgentAiConnectorBase):
    def __init__(self, gen_ai_api_config: dict):
        super().__init__(gen_ai_api_config)
        self.__model_name = None
        self.__genai_config = None
        self.__model = None

        self.configure_model()

    def configure_model(self):
        config = self.get_config()
        api_key = config["api_key"]

        self.__model = OpenAI(api_key=api_key)
        self.__genai_config = config['gen_config']
        self.__model_name = config['model_name']


    def submit_general_prompt(self, prompt: str, llm_instructions: str, is_json: bool = False):
        system_prompt, base_prompt = self.get_base_prompt(prompt, llm_instructions)

        response = self.__model.chat.completions.create(
            model=self.__model_name,
            messages=[
                system_prompt, base_prompt
            ],
            response_format={"type": "json_object"},
            temperature=self.__genai_config.get("temperature", 1),
            max_tokens=self.__genai_config.get("max_output_tokens", 500)
        )

        return response

    def get_base_prompt(self, prompt: str, llm_instructions: str):
        system_prompt = {"role": "system", "content": llm_instructions}
        base_prompt = {"role": "user", "content": prompt}

        return system_prompt, base_prompt

    def __process_and_extract_response(self, response):
        # single choice n == 1
        response_message = response.choices[0].message.content
        usage_data = response.usage

        return response_message, usage_data

    def submit_data_prompt(self, prompt: str, llm_instructions: str):
        response = self.submit_general_prompt(prompt, llm_instructions)

        response_message, usage_data = self.__process_and_extract_response(response)

        # TODO transform response message

        return response_message, usage_data



