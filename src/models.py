import instructor

import asyncio
import nest_asyncio

nest_asyncio.apply()

MAX_RETRIES = 0
TEMPERATURE = 0.2
EMBEDDING_DIMENSION = 10
CUSTOM_ENBEDDING_MODEL = False

LLL_PROVIDER = "OPEN_AI"
AZURE = False

if LLL_PROVIDER == "OPEN_AI":
    EMBEDDING_ENGINE = "text-embedding-3-large"
    if AZURE:
        from openai import AsyncAzureOpenAI

        client = AsyncAzureOpenAI(
            azure_endpoint="https://vigieinstance.openai.azure.com/",
            api_version="2023-07-01-preview",
            api_key="6e612f025340400d827a519b0549cff6",
        )
        GENERATION_ENGINE = "dep"

    else:
        from openai import OpenAI, AsyncOpenAI

        client = AsyncOpenAI(
            api_key="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",
            organization="org-EYbk8L8UD8kpRGOeDarxXD55",
        )
        GENERATION_ENGINE = "gpt-4-turbo-preview"

    client = instructor.patch(client)

    async def get_async_analysis(prompt, response_model):
        response: response_model = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Tu est un assistant spélialisé dans l'analyse de commentaires, et qui ne renvoit que des fichiers JSON.",
                },
                {"role": "user", "content": str(prompt)},
            ],
            response_format={"type": "json_object"},
            model=GENERATION_ENGINE,
            temperature=TEMPERATURE,
            # max_retries=MAX_RETRIES,
            response_model=response_model,
        )
        return response  # .choices[0].message.content

    def apply_async_analysis(prompts, response_models):
        if type(response_models) is not list:
            response_models = [response_models for _ in prompts]
        loop = asyncio.get_event_loop()
        tasks = [
            loop.create_task(get_async_analysis(prompt, response_model))
            for (prompt, response_model) in zip(prompts, response_models)
        ]
        res = loop.run_until_complete(asyncio.gather(*tasks))
        return res


elif LLL_PROVIDER == "MISTRAL_AI":
    MISTRAL_API_KEY = "GFBjsGogmbv0LuMWjJewXBXwyN7QeKNj"
    EMBEDDING_ENGINE = ""  # "text-embedding-3-large"
    GENERATION_ENGINE = "mistralai/Mistral-7B-Instruct-v0.2"

    from mistralai.async_client import MistralClient
    from mistralai.models.chat_completion import ChatMessage

    api_key = MISTRAL_API_KEY
    model = "mistral-tiny"

    client = MistralClient(api_key=api_key)

    messages = [ChatMessage(role="user", content="What is the best French cheese?")]

    async def get_async_analysis(prompt, response_model):
        response: response_model = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Tu est un assistant spélialisé dans l'analyse de commentaires, et qui ne renvoit que des fichiers JSON.",
                },
                {"role": "user", "content": str(prompt)},
            ],
            # response_format={ "type": "json_object" },
            model=GENERATION_ENGINE,
            temperature=TEMPERATURE,
            max_retries=MAX_RETRIES,
            response_model=response_model,
        )
        return response  # .choices[0].message.content

    def apply_async_analysis(prompts, response_models):
        if type(response_models) is not list:
            response_models = [response_models for _ in prompts]
        loop = asyncio.get_event_loop()
        tasks = [
            loop.create_task(get_async_analysis(prompt, response_model))
            for (prompt, response_model) in zip(prompts, response_models)
        ]
        res = loop.run_until_complete(asyncio.gather(*tasks))
        return res

elif LLL_PROVIDER == "LOCAL":
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-xxx")
else:
    raise NameError("invalid model provider")

if CUSTOM_ENBEDDING_MODEL:
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("OrdalieTech/Solon-embeddings-large-0.1")


def get_analysis(prompt, response_model):
    return apply_async_analysis([prompt], response_model)[0]


# %% LLMs


"""
def get_analysis(prompt, response_model):
    response = client.chat.completions.create(
        messages=[
            #{"role": "system", "content": "Tu est analyste marketing"},
            {"role": "user", "content": str(prompt)},
        ],
        model=GENERATION_ENGINE,
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
    """


"""
def tag_single_request(prompt: str, sous_categories: List[SousCategorie]) -> Aspect:
    allowed_tags = [(tag.indice, tag.nom) for tag in sous_categories]
    allowed_tags_str = ", ".join([f"`{tag}`" for tag in allowed_tags])

    return client.chat.completions.create(
        model="mixtral",
        messages=[
            {
                "role": "system",
                "content": f"Tu es {feedback_context['role']} au sein de l'entreprise {feedback_context['entreprise']}. Voici un bref rappel sur cette entreprise: \n'{feedback_context['context']}'\n\En tant que  {feedback_context['role']}, tu est spécialisé dans l'analyse de commentaire."
            },
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": f"Voici les sous-catégories: {allowed_tags_str}",
            },
        ], 
        response_model=ListAspects,  # Minimizes the hallucination of tags that are not in the allowed tags.
        validation_context={"sous_categories": sous_categories},
    )

def tag_request(request: AspectsRequest) -> AspectsResponse:
    predictions = [tag_single_request(text, request.tags) for text in request.texts]
    
    return AspectsResponse(
        texts=request.texts,
        predictions=predictions)

"""
