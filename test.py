import asyncio
from ollama import AsyncClient

def safe_print(text):
    print(text.encode('ascii', 'ignore').decode(), end='', flush=True)


async def chat():
  message = {'role': 'user', 'content': 'Me fale um resumo do filme Senhor dos Aneis'}
  async for part in await AsyncClient().chat(model='gemma3:1b', messages=[message], stream=True):
    safe_print(part['message']['content'])

asyncio.run(chat())