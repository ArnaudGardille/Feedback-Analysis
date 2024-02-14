#Note: The openai-python library support for Azure OpenAI is in preview.
      #Note: This code sample requires OpenAI Python library version 0.28.1 or lower.
import os
import openai

openai.api_type = "azure"
openai.api_base = "https://vigieinstance.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "6e612f025340400d827a519b0549cff6"

message_text = [{"role":"system","content":"Assistant is an AI chatbot that helps users turn a natural language list into JSON format. After users input a list they want in JSON format, it will provide suggested list of attribute labels if the user has not provided any, then ask the user to confirm them before creating the list."}]

completion = openai.ChatCompletion.create(
  engine="dep",
  messages = message_text,
  temperature=0.2,
  max_tokens=350,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None
)