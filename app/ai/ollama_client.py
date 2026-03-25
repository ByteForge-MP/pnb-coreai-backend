import os
import httpx
import logging
import time

logger = logging.getLogger("chat_logger")

DEFAULT_OLLAMA_URLS = (
    "http://ollama:11434/api/generate",
    "http://host.docker.internal:11434/api/generate",
    "http://localhost:11434/api/generate",
)


def _get_ollama_urls():
    configured_url = os.getenv("OLLAMA_URL", "").strip()
    candidates = []

    if configured_url:
        candidates.append(configured_url)

    candidates.extend(DEFAULT_OLLAMA_URLS)

    normalized = []

    for url in candidates:
        final_url = url.rstrip("/")

        if not final_url.endswith("/api/generate"):
            final_url = f"{final_url}/api/generate"

        if final_url not in normalized:
            normalized.append(final_url)

    return normalized

async def stream_ollama(prompt: str, request_id: str = "unknown"):
    started_at = time.monotonic()
    timeout = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)
    errors = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        for ollama_url in _get_ollama_urls():
            try:
                logger.info("[%s] Ollama helper stream started | url=%s", request_id, ollama_url)

                async with client.stream(
                    "POST",
                    ollama_url,
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
                    return

            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.HTTPStatusError) as exc:
                logger.warning("[%s] Ollama helper failed for %s: %s", request_id, ollama_url, exc)
                errors.append(f"{ollama_url} -> {exc}")

    raise RuntimeError(
        "Could not connect to Ollama. Set OLLAMA_URL to the reachable endpoint. "
        f"Tried: {'; '.join(errors) if errors else 'no endpoints'}"
    )
