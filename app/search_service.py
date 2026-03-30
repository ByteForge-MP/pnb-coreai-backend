import os
from app.query_expander import expand_query
import logging
import requests
import time

logger = logging.getLogger("chat_logger")

DEFAULT_SEARXNG_URLS = (
    "http://searxng:8080/search",
    "http://localhost:8080/search",
)


def _is_offline_mode():
    return os.getenv("OFFLINE_MODE", "false").lower() == "true"


def _truncate(value, limit=500):
    text = str(value or "").strip()

    if len(text) <= limit:
        return text

    return text[:limit] + "...[truncated]"


def _extract_image(item):
    if not isinstance(item, dict):
        return None

    for key in ("thumbnail", "img_src", "thumbnail_src", "image", "src"):
        value = str(item.get(key, "")).strip()

        if value:
            return value

    return None


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

    if _is_offline_mode():
        logger.info("[%s] SearxNG skipped because OFFLINE_MODE=true", request_id)
        return {
            "summary": "",
            "documents": [],
            "image": None,
            "sources": [],
            "error": "Web search disabled in offline mode.",
        }

    queries = expand_query(query)
    logger.info("[%s] Expanded into %s query variants", request_id, len(queries))

    docs = []
    source_docs = []
    seen_urls = set()
    sources = []
    image = None
    search_error = None
    searxng_urls = _get_searxng_urls()

    for index, q in enumerate(queries, start=1):
        params = {
            "q": q,
            "format": "json",
            "categories": "general"
        }

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "X-Forwarded-For": "127.0.0.1",
            "X-Real-IP": "127.0.0.1",
        }

        logger.info("[%s] SearxNG request %s/%s started | q=%r", request_id, index, len(queries), q)

        response = None
        response_url = None

        for url in searxng_urls:
            try:
                logger.info("[%s] Trying SearxNG at %s", request_id, url)
                response = requests.get(url, params=params, headers=headers, timeout=10)
                response_url = url

                if response.status_code == 200:
                    break

                preview = _truncate(response.text)
                logger.warning(
                    "[%s] SearxNG non-200 via %s for q=%r | status=%s | body=%r",
                    request_id,
                    url,
                    q,
                    response.status_code,
                    preview,
                )
                search_error = f"SearxNG returned HTTP {response.status_code}"
            except Exception as exc:
                logger.warning("[%s] SearxNG request failed via %s for q=%r: %s", request_id, url, q, exc)

        if response is None:
            logger.error("[%s] All SearxNG endpoints failed for q=%r", request_id, q)
            if search_error is None:
                search_error = "Could not connect to any SearxNG endpoint"
            continue

        logger.info(
            "[%s] SearxNG request %s/%s completed | status=%s | endpoint=%s",
            request_id,
            index,
            len(queries),
            response.status_code,
            response_url,
        )

        if response.status_code != 200:
            if response.status_code == 403:
                logger.warning(
                    "[%s] Stopping further query expansions after 403 for q=%r; check local SearxNG limiter/settings.",
                    request_id,
                    q,
                )
                break

            time.sleep(1)
            continue

        payload = response.json()
        results = payload.get("results", [])
        answers = payload.get("answers", [])
        infoboxes = payload.get("infoboxes", [])
        logger.info("[%s] SearxNG returned %s results for q=%r", request_id, len(results), q)
        logger.info(
            "[%s] SearxNG payload summary for q=%r | answers=%s | infoboxes=%s | suggestions=%s | corrections=%s | unresponsive_engines=%s",
            request_id,
            q,
            len(answers),
            len(infoboxes),
            len(payload.get("suggestions", []) or []),
            len(payload.get("corrections", []) or []),
            len(payload.get("unresponsive_engines", []) or []),
        )

        if answers:
            logger.info("[%s] SearxNG answers for q=%r: %s", request_id, q, answers[:3])

        if infoboxes:
            logger.info("[%s] SearxNG first infobox for q=%r: %s", request_id, q, infoboxes[0])
            if image is None:
                image = _extract_image(infoboxes[0])

        if results:
            first_result = results[0]
            logger.info(
                "[%s] SearxNG first result for q=%r | title=%r | url=%r | content=%r | thumbnail=%r | img_src=%r",
                request_id,
                q,
                first_result.get("title", ""),
                first_result.get("url", ""),
                first_result.get("content", ""),
                first_result.get("thumbnail", ""),
                first_result.get("img_src", ""),
            )
        else:
            logger.warning("[%s] SearxNG returned no results for q=%r | raw_keys=%s", request_id, q, sorted(payload.keys()))

        for answer in answers:
            answer_text = str(answer).strip()

            if answer_text:
                docs.append(answer_text)
                source_docs.append({
                    "text": answer_text,
                    "kind": "answer",
                    "query": q,
                })

        for infobox in infoboxes:
            infobox_parts = []
            title = str(infobox.get("infobox", "") or infobox.get("id", "")).strip()
            content = str(infobox.get("content", "")).strip()

            if title:
                infobox_parts.append(title)

            if content:
                infobox_parts.append(content)

            attributes = infobox.get("attributes", [])

            for attr in attributes:
                if isinstance(attr, list) and len(attr) >= 2:
                    key = str(attr[0]).strip()
                    value = str(attr[1]).strip()

                    if key and value:
                        infobox_parts.append(f"{key}: {value}")

            if infobox_parts:
                infobox_text = ". ".join(infobox_parts)
                docs.append(infobox_text)
                source_docs.append({
                    "text": infobox_text,
                    "kind": "infobox",
                    "query": q,
                })

        for r in results[:top_k]:

            title = r.get("title", "")
            snippet = r.get("content", "")
            url = r.get("url", "")
            thumbnail = _extract_image(r)

            if url in seen_urls:
                continue

            seen_urls.add(url)

            sources.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "image": thumbnail,
            })

            if thumbnail and image is None:
                image = thumbnail

            text = f"{title}. {snippet}"

            docs.append(text)
            source_docs.append({
                "text": text,
                "kind": "result",
                "query": q,
                "title": title,
                "url": url,
                "snippet": snippet,
                "image": thumbnail,
            })

        if docs:
            logger.info("[%s] SearxNG gathered usable docs after q=%r; skipping remaining expansions", request_id, q)
            break

        time.sleep(1)

    # join all snippets
    combined_text = " ".join(docs)

    # split into sentences
    sentences = combined_text.split(". ")

    # keep only first 20 lines
    important_lines = sentences[:40]

    summary = "\n".join(important_lines)
    logger.info(
        "[%s] SearxNG search completed in %.2fs | docs=%s | unique_urls=%s | sources=%s | has_image=%s | error=%s",
        request_id,
        time.monotonic() - started_at,
        len(docs),
        len(seen_urls),
        len(sources),
        image is not None,
        search_error,
    )

    return {
        "summary": summary,
        "documents": source_docs,
        "image": image,
        "sources": sources,
        "error": search_error,
    }
