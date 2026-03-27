def expand_query(query):
    candidates = [
        query,
        f"{query} details",
        f"{query} recent news",
    ]

    normalized = []

    for item in candidates:
        value = item.strip()

        if value and value not in normalized:
            normalized.append(value)

    return normalized
