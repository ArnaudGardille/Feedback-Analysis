import asyncio
import instructor
import os

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]


MAX_RETRIES = 0
TEMPERATURE = 0.5
EMBEDDING_DIMENSION = 10
CUSTOM_ENBEDDING_MODEL = False

#client = AsyncOpenAI()
client = instructor.patch(openai.AsyncOpenAI())

if CUSTOM_ENBEDDING_MODEL:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('OrdalieTech/Solon-embeddings-large-0.1')
GENERATION_ENGINE = "gpt-4-turbo-preview"
EMBEDDING_ENGINE = "text-embedding-3-large"

import nest_asyncio
nest_asyncio.apply()

#%% LLMs
    
async def get_analysis(prompt, response_model):
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
    tasks = [loop.create_task(get_analysis(prompt, response_model)) for (prompt, response_model) in zip(prompts, response_models)]
    res =  loop.run_until_complete(asyncio.gather(*tasks))
    return res
