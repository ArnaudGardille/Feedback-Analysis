### Imports

from sentence_transformers import SentenceTransformer

from openai import AsyncOpenAI
# import asyncio


import instructor

client = instructor.patch(AsyncOpenAI())

MAX_RETRIES = 0
TEMPERATURE = 0.5
EMBEDDING_DIMENSION = 10
CUSTOM_ENBEDDING_MODEL = False
PUSH_TO_BUBBLE = False

if CUSTOM_ENBEDDING_MODEL:
    embedding_model = SentenceTransformer("OrdalieTech/Solon-embeddings-large-0.1")
GENERATION_ENGINE = "gpt-4-turbo-preview"
EMBEDDING_ENGINE = "text-embedding-3-large"

# import nest_asyncio
# nest_asyncio.apply()
