from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
_reranker = None


def _get_reranker():
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)

    return _reranker

def rerank(query, docs, top_k=5):
    reranker = _get_reranker()

    pairs = [[query, d] for d in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [d for d, s in ranked[:top_k]]
