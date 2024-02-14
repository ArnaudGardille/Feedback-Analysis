import instructor
import os
from tqdm.notebook import tqdm

import asyncio
import nest_asyncio
nest_asyncio.apply()

MAX_RETRIES = 0
TEMPERATURE = 0.2
EMBEDDING_DIMENSION = 10
CUSTOM_ENBEDDING_MODEL = False

LLL_PROVIDER = "MISTRAL_AI"
MISTRAL_API_KEY = "GFBjsGogmbv0LuMWjJewXBXwyN7QeKNj"


if LLL_PROVIDER == "OPEN_AI":
    from openai import OpenAI, AsyncOpenAI
    client = AsyncOpenAI()

    async def get_async_analysis(client, prompt, response_model):
        response: response_model = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Tu est un assistant spélialisé dans l'analyse de commentaires, et qui ne renvoit que des fichiers JSON."},
                {"role": "user", "content": str(prompt)},
            ],
            response_format={ "type": "json_object" },
            model=GENERATION_ENGINE,
            temperature=TEMPERATURE,
            max_retries=MAX_RETRIES,
            response_model=response_model,
            )
        return response #.choices[0].message.content

    def apply_async_analysis(prompts, response_models):
        if type(response_models) is not list:
            response_models = [response_models for _ in prompts]
        loop = asyncio.get_event_loop()
        tasks = [loop.create_task(get_async_analysis(prompt, response_model)) for (prompt, response_model) in zip(prompts, response_models)]
        res =  loop.run_until_complete(asyncio.gather(*tasks))
        return res
    
elif LLL_PROVIDER == "MISTRAL_AI":
    from mistralai.async_client import MistralAsyncClient
    from mistralai.models.chat_completion import ChatMessage

    api_key = MISTRAL_API_KEY
    model = "mistral-tiny"

    client = MistralAsyncClient(api_key=api_key)
    

    messages = [
        ChatMessage(role="user", content="What is the best French cheese?")
    ]

    # With async
    async_response = client.chat(model=model, messages=messages, temperature=TEMPERATURE)

    async for chunk in async_response: 
        print(chunk.choices[0].delta.content)

    async def get_async_analysis(client, prompt, response_model):
        response: response_model = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Tu est un assistant spélialisé dans l'analyse de commentaires, et qui ne renvoit que des fichiers JSON."},
                {"role": "user", "content": str(prompt)},
            ],
            response_format={ "type": "json_object" },
            model=GENERATION_ENGINE,
            temperature=TEMPERATURE,
            max_retries=MAX_RETRIES,
            response_model=response_model,
            )
        return response #.choices[0].message.content

    def apply_async_analysis(prompts, response_models):
        if type(response_models) is not list:
            response_models = [response_models for _ in prompts]
        loop = asyncio.get_event_loop()
        tasks = [loop.create_task(get_async_analysis(prompt, response_model)) for (prompt, response_model) in zip(prompts, response_models)]
        res =  loop.run_until_complete(asyncio.gather(*tasks))
        return res
           
elif LLL_PROVIDER == "LOCAL":
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-xxx")
else:
    raise NameError("invalid model provider")

if CUSTOM_ENBEDDING_MODEL:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('OrdalieTech/Solon-embeddings-large-0.1')
GENERATION_ENGINE = "gpt-4-turbo-preview"
EMBEDDING_ENGINE = "text-embedding-3-large"

GENERATION_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"



client = instructor.patch(client)



#%% LLMs

async def get_async_analysis(client, prompt, response_model):
    response: response_model = await client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Tu est un assistant spélialisé dans l'analyse de commentaires, et qui ne renvoit que des fichiers JSON."},
            {"role": "user", "content": str(prompt)},
        ],
        response_format={ "type": "json_object" },
        model=GENERATION_ENGINE,
        temperature=TEMPERATURE,
        max_retries=MAX_RETRIES,
        response_model=response_model,
        )
    return response #.choices[0].message.content

def apply_async_analysis(prompts, response_models):
    if type(response_models) is not list:
        response_models = [response_models for _ in prompts]
    loop = asyncio.get_event_loop()
    tasks = [loop.create_task(get_async_analysis(prompt, response_model)) for (prompt, response_model) in zip(prompts, response_models)]
    res =  loop.run_until_complete(asyncio.gather(*tasks))
    return res

def get_analysis(prompt, response_model):
    response = client.chat.completions.create(
        messages=[
            #{"role": "system", "content": "Tu est analyste marketing"},
            {"role": "user", "content": str(prompt)},
        ],
        model=GENERATION_MODEL,
        #response_format={ "type": "json_object" },
        response_model=response_model,
        )
    return response


def apply_analysis(prompts, response_models, bar=False):
    if type(response_models) is not list:
        response_models = [response_models for _ in prompts]
    res = []
    iterator = zip(prompts, response_models)
    if bar:
        iterator = tqdm(iterator)
    for (prompt, response_model) in iterator:
        res.append(get_analysis(prompt, response_model))
    return res