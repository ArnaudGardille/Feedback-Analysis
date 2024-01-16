import asyncio
from openai import AsyncOpenAI
import openai
openai.api_key = "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"

import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
)


async def main() -> None:
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="gpt-3.5-turbo",
    )
    print(chat_completion.choices[0].message.content)


asyncio.run(main())