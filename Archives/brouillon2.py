from openai import OpenAI
import openai
openai.api_key = "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"

client = OpenAI()

res = client.completions.create(
  model="gpt-3.5-turbo-instruct-0914",
  prompt="Say this is a test",
  max_tokens=7,
  temperature=0
)

print(res)