"""
This examples demonstrates the setup for Question-Answer-Retrieval.

You can input a query or a question. The script then uses semantic search
to find relevant passages in Simple English Wikipedia (as it is smaller and fits better in RAM).

As model, we use: nq-distilbert-base-v1

It was trained on the Natural Questions dataset, a dataset with real questions from Google Search
together with annotated data from Wikipedia providing the answer. For the passages, we encode the
Wikipedia article tile together with the individual text passages.

Google Colab Example: https://colab.research.google.com/drive/11GunvCqJuebfeTlgbJWkIMT0xJH6PWF1?usp=sharing
"""
import json
from sentence_transformers import SentenceTransformer, util
import time
import gzip
import os
import torch

# We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
model_name = 'nq-distilbert-base-v1'
bi_encoder = SentenceTransformer(model_name)
top_k = 5  # Number of passages we want to retrieve with the bi-encoder

# As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
# about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder

wikipedia_filepath = 'data/simplewiki-2020-11-01.jsonl.gz'

if not os.path.exists(wikipedia_filepath):
    util.http_get('http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz', wikipedia_filepath)

passages = []
with gzip.open(wikipedia_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        data = json.loads(line.strip())
        print(data['title'])
        #for paragraph in data['paragraphs']:
            # We encode the passages as [title, text]
            #print(paragraph)