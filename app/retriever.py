import json
import logging
import os
import pickle
import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("chat_logger")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_INDEX_PATH = os.path.join(ROOT_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(ROOT_DIR, "chunks.pkl")
ABBREVIATIONS_PATH = os.path.join(ROOT_DIR, "abbreviations.json")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None
_index = None
_chunks = None
_abbreviations = None
_bm25 = None


def _load_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _model


def _load_abbreviations():
    global _abbreviations

    if _abbreviations is None:
        if os.path.exists(ABBREVIATIONS_PATH):
            with open(ABBREVIATIONS_PATH, "r") as f:
                _abbreviations = json.load(f)
        else:
            logger.warning("abbreviations.json not found at %s", ABBREVIATIONS_PATH)
            _abbreviations = {}

    return _abbreviations


def _load_retrieval_assets():
    global _index, _chunks, _bm25

    if _index is not None and _chunks is not None and _bm25 is not None:
        return _index, _chunks, _bm25

    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        logger.warning(
            "Retrieval assets missing. Expected %s and %s",
            FAISS_INDEX_PATH,
            CHUNKS_PATH,
        )
        _index = None
        _chunks = []
        _bm25 = None
        return _index, _chunks, _bm25

    _index = faiss.read_index(FAISS_INDEX_PATH)

    with open(CHUNKS_PATH, "rb") as f:
        _chunks = pickle.load(f)

    corpus = [chunk["text"] for chunk in _chunks]
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    _bm25 = BM25Okapi(tokenized_corpus)

    return _index, _chunks, _bm25


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def embed_query(query):
    model = _load_model()
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    return q_emb


def expand_query(query):
    abbreviations = _load_abbreviations()
    q = query.lower()
    expanded_terms = []

    for short, full in abbreviations.items():
        if short in q:
            expanded_terms.append(full)

    return q + " " + " ".join(expanded_terms)


def retrieve(query, k=3):
    smalltalk = ["hello", "hi", "hey", "good morning"]

    if query.lower().strip() in smalltalk:
        return []

    index, chunks, bm25 = _load_retrieval_assets()

    if index is None or not chunks or bm25 is None:
        return []

    model = _load_model()
    expanded_query = expand_query(query)
    vector = model.encode([expanded_query], normalize_embeddings=True)
    distances, indices = index.search(np.array(vector), 10)

    vector_scores = {}

    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        text = chunks[idx]["text"]
        vector_scores[text] = score

    tokenized_query = tokenize(expanded_query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top = np.argsort(bm25_scores)[::-1][:10]

    bm25_dict = {}

    for idx in bm25_top:
        text = chunks[idx]["text"]
        bm25_dict[text] = bm25_scores[idx]

    combined = {}

    for chunk in chunks:
        text = chunk["text"]
        v_score = vector_scores.get(text, 0)
        b_score = bm25_dict.get(text, 0)

        if v_score > 0 or b_score > 0:
            combined[text] = {
                "chunk": chunk,
                "vector": v_score,
                "bm25": b_score,
            }

    max_v = max([v["vector"] for v in combined.values()] + [1])
    max_b = max([v["bm25"] for v in combined.values()] + [1])

    for item in combined.values():
        item["vector"] /= max_v
        item["bm25"] /= max_b
        item["score"] = 0.7 * item["vector"] + 0.3 * item["bm25"]

    ranked = sorted(
        combined.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    return [r["chunk"] for r in ranked[:k]]
