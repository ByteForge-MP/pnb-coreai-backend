import logging
import os

from sentence_transformers import CrossEncoder
from app.device import get_best_device

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
_reranker = None
logger = logging.getLogger("chat_logger")


def _is_offline_mode():
    return os.getenv("OFFLINE_MODE", "false").lower() == "true"


def _get_reranker_device():
    device = get_best_device()

    if device == "mps":
        return "cpu"

    return device


def _get_reranker():
    global _reranker

    if _reranker is None:
        try:
            _reranker = CrossEncoder(
                RERANKER_MODEL_NAME,
                device=_get_reranker_device(),
                local_files_only=_is_offline_mode(),
            )
        except Exception as exc:
            logger.warning("Reranker unavailable, falling back to input order: %s", exc)
            _reranker = False

    return _reranker

def rerank(query, docs, top_k=5):
    reranker = _get_reranker()

    if not reranker:
        return docs[:top_k]

    pairs = [[query, d] for d in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [d for d, s in ranked[:top_k]]
