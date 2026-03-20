import httpx
import logging
import time

logger = logging.getLogger("chat_logger")

OLLAMA_URL = "http://localhost:11434/api/generate"

async def stream_ollama(prompt: str, request_id: str = "unknown"):
    started_at = time.monotonic()
    logger.info("[%s] Ollama helper stream started | url=%s", request_id, OLLAMA_URL)

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
            logger.info("[%s] Ollama helper response status: %s", request_id, response.status_code)
            response.raise_for_status()
            chunk_count = 0

            async for line in response.aiter_lines():
                if line:
                    chunk_count += 1

                    if chunk_count == 1:
                        logger.info("[%s] Ollama helper first chunk received", request_id)

                    yield line

            logger.info(
                "[%s] Ollama helper stream completed in %.2fs | chunks=%s",
                request_id,
                time.monotonic() - started_at,
                chunk_count,
            )
