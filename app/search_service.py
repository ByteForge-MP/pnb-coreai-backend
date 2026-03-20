from app.query_expander import expand_query
import logging
import requests
import time

logger = logging.getLogger("chat_logger")


def search_searxng(query, top_k=5, request_id="unknown"):
    started_at = time.monotonic()
    logger.info("[%s] SearxNG search started | query=%r | top_k=%s", request_id, query, top_k)

    queries = expand_query(query)
    logger.info("[%s] Expanded into %s query variants", request_id, len(queries))

    docs = []
    seen_urls = set()
    image = None

    for index, q in enumerate(queries, start=1):

        # url = "http://localhost:8080/search"
        url = "http://searxng:8080/search"

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

        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
        except Exception as exc:
            logger.exception("[%s] SearxNG request failed for q=%r: %s", request_id, q, exc)
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
