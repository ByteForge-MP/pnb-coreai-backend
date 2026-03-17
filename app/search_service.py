from app.query_expander import expand_query
import requests
import time


def search_searxng(query, top_k=5):

    queries = expand_query(query)

    docs = []
    seen_urls = set()
    image = None

    for q in queries:

        url = "http://localhost:8080/search"

        params = {
            "q": q,
            "format": "json",
            "categories": "general"
        }

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)

        time.sleep(1)

        if response.status_code != 200:
            continue

        results = response.json().get("results", [])

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

    print("Summary:",summary)
    print("Image:",image)

    return {
        "summary": summary,
        "image": image
    }