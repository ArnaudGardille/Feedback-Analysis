import asyncio
from openai import AsyncOpenAI

from typing import Any, Dict, List
import time

from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain.schema import HumanMessage, LLMResult
from langchain_openai import ChatOpenAI, 

from dotenv import load_dotenv, find_dotenv
import openai
import os
from langchain.agents. agent_types import AgentType

openai.api_key = "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"

#model = ChatOpenAI() #ChatOpenAI(api_key="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")
model = ChatOpenAI() #ChatOpenAI(api_key="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")
from langchain_core.messages import HumanMessage, SystemMessage



messages = [
    SystemMessage(content="You're a helpful assistant"),
]
prompts = ["What is the square of {i} ?".format(i=i) for i in range(10)]

async def main():


    async def async_generate(prompt: str):

        response = model.invoke(prompt).content
        return response

    async def generate_concurrently(prompts: List[str]):
        tasks = [async_generate(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        return responses

    responses = await generate_concurrently(prompts)

    #for prompt, response in zip(prompts, responses):
    #    print(f"Prompt: {prompt}")
    #    print(f"Generated text: {response}")


if __name__ == "__main__":
    start = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    end = time.time()
    print(end - start)

    start = time.time()
    for prompt in prompts:
        response = model.invoke(prompt).content
    end = time.time()
    print(end - start)



