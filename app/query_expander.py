import re


ABBREVIATIONS = {
    "md": "managing director",
    "ceo": "chief executive officer",
    "cfo": "chief financial officer",
    "cto": "chief technology officer",
    "coo": "chief operating officer",
    "pm": "prime minister",
    "sbi": "state bank of india",
    "pnb": "punjab national bank",
}


def _normalize_query(query):
    normalized = " ".join(query.split())
    lowered = normalized.lower()

    for short, full in ABBREVIATIONS.items():
        lowered = re.sub(rf"\b{re.escape(short)}\b", full, lowered, flags=re.IGNORECASE)

    return lowered


def expand_query(query):
    normalized_query = _normalize_query(query)
    candidates = [
        query,
        normalized_query,
        f"{normalized_query} official website",
        f"{normalized_query} details",
        f"{normalized_query} latest update",
    ]

    normalized = []

    for item in candidates:
        value = item.strip()

        if value and value not in normalized:
            normalized.append(value)

    return normalized
