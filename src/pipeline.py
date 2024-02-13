### Imports

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from datasets import load_dataset
import umap
import altair as alt
from sentence_transformers import SentenceTransformer
from typing import List
import enum

from langchain_community.llms import Ollama
from langchain.output_parsers.regex_dict import RegexDictParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, ChatMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field, validator, create_model
from openai import AsyncOpenAI, OpenAI
#import asyncio
import os

from bubble_api import Field as BubbleField
from bubble_api import BubbleClient

import itertools
from copy import deepcopy
from tqdm.notebook import tqdm, trange
from sklearn.cluster import KMeans

import umap.umap_ as umap
import hdbscan

import openai
import instructor

openai.api_key = os.environ["OPENAI_API_KEY"]
client = instructor.patch(AsyncOpenAI())

MAX_RETRIES = 0
TEMPERATURE = 0.5
EMBEDDING_DIMENSION = 10
CUSTOM_ENBEDDING_MODEL = False
PUSH_TO_BUBBLE = False

if CUSTOM_ENBEDDING_MODEL:
    embedding_model = SentenceTransformer('OrdalieTech/Solon-embeddings-large-0.1')
GENERATION_ENGINE = "gpt-4-turbo-preview"
EMBEDDING_ENGINE = "text-embedding-3-large"

#import nest_asyncio
#nest_asyncio.apply()
