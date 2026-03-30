import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from app.device import get_best_device

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model = None

CHUNK_SIZE = 120
OVERLAP = 40


def _is_offline_mode():
    return os.getenv("OFFLINE_MODE", "false").lower() == "true"


def _get_embedding_device():
    device = get_best_device()

    if device == "mps":
        return "cpu"

    return device


def _get_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=_get_embedding_device(),
            local_files_only=_is_offline_mode(),
        )

    return _model


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
    if not docs:
        return None, []

    chunks = []

    for doc in docs:

        parts = chunk_text(doc)

        for p in parts:

            chunks.append(p)

    model = _get_model()

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])

    index.add(embeddings)

    return index, chunks
