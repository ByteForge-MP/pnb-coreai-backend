import os
from app.query_expander import expand_query
import logging
import requests
import time

logger = logging.getLogger("chat_logger")

DEFAULT_SEARXNG_URLS = (
    "http://searxng:8080/search",
    "http://host.docker.internal:8080/search",
    "http://localhost:8080/search",
)


def _get_searxng_urls():
    configured_url = os.getenv("SEARXNG_URL", "").strip()
    candidates = []

    if configured_url:
        candidates.append(configured_url)

    candidates.extend(DEFAULT_SEARXNG_URLS)

    normalized = []

    for url in candidates:
        final_url = url.rstrip("/")

        if not final_url.endswith("/search"):
            final_url = f"{final_url}/search"

        if final_url not in normalized:
            normalized.append(final_url)

    return normalized


def search_searxng(query, top_k=5, request_id="unknown"):
    started_at = time.monotonic()
    logger.info("[%s] SearxNG search started | query=%r | top_k=%s", request_id, query, top_k)

    queries = expand_query(query)
    logger.info("[%s] Expanded into %s query variants", request_id, len(queries))

    docs = []
    seen_urls = set()
    image = None
    searxng_urls = _get_searxng_urls()

    for index, q in enumerate(queries, start=1):
        params = {
            "q": q,
            "format": "json",
            "categories": "general"
        }

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }

        logger.info("[%s] SearxNG request %s/%s started | q=%r", request_id, index, len(queries), q)

        response = None

        for url in searxng_urls:
            try:
                logger.info("[%s] Trying SearxNG at %s", request_id, url)
                response = requests.get(url, params=params, headers=headers, timeout=10)
                break
            except Exception as exc:
                logger.warning("[%s] SearxNG request failed via %s for q=%r: %s", request_id, url, q, exc)

        if response is None:
            logger.error("[%s] All SearxNG endpoints failed for q=%r", request_id, q)
            continue

        logger.info(
            "[%s] SearxNG request %s/%s completed | status=%s",
            request_id,
            index,
            len(queries),
            response.status_code,
        )

        time.sleep(1)

        if response.status_code != 200:
            logger.warning("[%s] SearxNG non-200 response for q=%r", request_id, q)
            continue

        results = response.json().get("results", [])
        logger.info("[%s] SearxNG returned %s results for q=%r", request_id, len(results), q)

        for r in results[:top_k]:

            title = r.get("title", "")
            snippet = r.get("content", "")
            url = r.get("url", "")
            thumbnail = r.get("thumbnail", "")

            if url in seen_urls:
                continue

            seen_urls.add(url)

            if thumbnail and image is None:
                image = thumbnail

            text = f"{title}. {snippet}"

            docs.append(text)

    # join all snippets
    combined_text = " ".join(docs)

    # split into sentences
    sentences = combined_text.split(". ")

    # keep only first 20 lines
    important_lines = sentences[:40]

    summary = "\n".join(important_lines)
    logger.info(
        "[%s] SearxNG search completed in %.2fs | docs=%s | unique_urls=%s | has_image=%s",
        request_id,
        time.monotonic() - started_at,
        len(docs),
        len(seen_urls),
        image is not None,
    )

    return {
        "summary": summary,
        "image": image
    }
