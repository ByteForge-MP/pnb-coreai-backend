import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5",device="cuda" if faiss.get_num_gpus() > 0 else "cpu")

CHUNK_SIZE = 120
OVERLAP = 40


def chunk_text(text):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + CHUNK_SIZE

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += CHUNK_SIZE - OVERLAP

    return chunks


def build_dynamic_embeddings(docs):

    chunks = []

    for doc in docs:

        parts = chunk_text(doc)

        for p in parts:

            chunks.append(p)

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    return index, chunks