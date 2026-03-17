import re

MAX_CONTEXT = 3500


def compress_context(text):

    if len(text) <= MAX_CONTEXT:
        return text

    # better sentence splitting
    sentences = re.split(r'(?<=[.!?]) +', text)

    compressed = ""

    for s in sentences:

        if len(compressed) + len(s) + 1 > MAX_CONTEXT:
            break

        compressed += s + " "

    # fallback if nothing added
    if not compressed:
        return text[:MAX_CONTEXT]

    return compressed.strip()