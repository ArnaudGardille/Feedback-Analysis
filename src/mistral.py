import json
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from pydantic import BaseModel, ValidationError, ConfigDict
from typing import Type, Optional
from dotenv import load_dotenv

load_dotenv()
MISTRAL_API_KEY = "GFBjsGogmbv0LuMWjJewXBXwyN7QeKNj"


from dotenv import load_dotenv
import nest_asyncio

nest_asyncio.apply()
from tqdm.notebook import tqdm

load_dotenv()
MISTRAL_API_KEY = "GFBjsGogmbv0LuMWjJewXBXwyN7QeKNj"


class MistralLanguageModel:
    def __init__(self, api_key=MISTRAL_API_KEY, model="mistral-tiny", temperature=0.0):
        if api_key is None:
            raise ValueError(
                "The Mistral API KEY must be provided either as "
                "an argument or as an environment variable named 'MISTRAL_API_KEY'"
            )  # noqa

        self.api_key = api_key
        self.model = model
        self.temperature = temperature

        # self.client = MistralAsyncClient(api_key=self.api_key)
        self.client = MistralClient(api_key=self.api_key)

    def async_generation(
        self,
        prompt: str,  # async
        output_format: Optional[Type[BaseModel]] = None,
        max_tokens: int = None,
    ):
        system_message = "You are a helpful assistant."
        if output_format:
            system_message += f" Respond in a JSON format that contains the following keys: {self._model_structure_repr(output_format)}. You must only return a JSON, nothing else. You are strictly forbidden to return anything else, like an explanation."  # noqa

        params = {
            "model": self.model,
            "messages": [
                ChatMessage(role="system", content=system_message),
                ChatMessage(role="user", content=prompt),
            ],
            "temperature": self.temperature,
        }

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        return self.client.chat(**params)

    def generate(
        self,
        prompts: list[str],
        output_format: Optional[Type[BaseModel]] = None,
        max_tokens: int = None,
    ):
        """loop = asyncio.get_event_loop()
        tasks = [loop.create_task(self.async_generation(prompt, output_format)) for prompt in prompts]

        responses = loop.run_until_complete(asyncio.gather(*tasks))
        responses = [self.async_generation(prompt, output_format) for prompt in prompts]
        """
        responses = []
        for prompt in tqdm(prompts):
            res = (
                self.async_generation(prompt, output_format).choices[0].message.content
            )
            print(res)
            assert self._is_valid_json_for_model(res, output_format)
            responses.append(res)
        # responses = [response.choices[0].message.content for response in responses]
        return responses

        print(responses)
        if output_format:
            for res in responses:
                assert self._is_valid_json_for_model(res, output_format)

        return responses

    def _model_structure_repr(self, model: Type[BaseModel]) -> str:
        fields = model.__annotations__
        return ", ".join(f"{key}: {value}" for key, value in fields.items())

    def _is_valid_json_for_model(self, text: str, model: Type[BaseModel]) -> bool:  # noqa
        """
        Check if a text is valid JSON and if it respects the provided BaseModel. # noqa
        """
        model.model_config = ConfigDict(strict=True)

        try:
            parsed_data = json.loads(text)
            model(**parsed_data)
            return True
        except (json.JSONDecodeError, ValidationError):
            parsed_data = json.loads(text)
            model(**parsed_data)
            return False


class Output(BaseModel):
    first_name: str
    last_name: str
    city: str


llm = MistralLanguageModel()
prompts = [
    'Extract the requested  information from the following sentence: "Alice Johnson is visiting Rome."',
    'Extract the requested  information from the following sentence: "Emmanuel Macron is visiting New York."',
    'Extract the requested  information from the following sentence: "Growing Ananas is visiting Paris."',
]
response = llm.generate(prompts, output_format=Output)

print(response)
