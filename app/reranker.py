from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank(query, docs, top_k=5):

    pairs = [[query, d] for d in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [d for d, s in ranked[:top_k]]