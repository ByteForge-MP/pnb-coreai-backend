import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"

async def stream_ollama(prompt: str):

    async with httpx.AsyncClient(timeout=None) as client:

        async with client.stream(
            "POST",
            OLLAMA_URL,
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": True
            }
        ) as response:

            async for line in response.aiter_lines():
                if line:
                    yield line