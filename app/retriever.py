import faiss
import pickle
import numpy as np
import re
import json
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# -------------------------------
# LOAD FAISS INDEX
# -------------------------------
index = faiss.read_index("faiss.index")

# -------------------------------
# LOAD CHUNKS
# -------------------------------
chunks = pickle.load(open("chunks.pkl", "rb"))

# -------------------------------
# LOAD ABBREVIATIONS
# -------------------------------
with open("abbreviations.json", "r") as f:
    abbreviations = json.load(f)

# -------------------------------
# TOKENIZER
# -------------------------------
def tokenize(text):
    return re.findall(r"\w+", text.lower())

# -------------------------------
# QUERY EXPANSION
# -------------------------------
def expand_query(query):

    q = query.lower()
    expanded_terms = []

    for short, full in abbreviations.items():
        if short in q:
            expanded_terms.append(full)

    return q + " " + " ".join(expanded_terms)

# -------------------------------
# PREPARE BM25
# -------------------------------
corpus = [c["text"] for c in chunks]

tokenized_corpus = [tokenize(doc) for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

# -------------------------------
# RETRIEVAL
# -------------------------------
def retrieve(query, k=3):

    smalltalk = ["hello", "hi", "hey", "good morning"]

    if query.lower().strip() in smalltalk:
        return []

    # -------------------------------
    # EXPAND QUERY
    # -------------------------------
    expanded_query = expand_query(query)

    # -------------------------------
    # VECTOR SEARCH
    # -------------------------------
    vector = model.encode([expanded_query], normalize_embeddings=True)

    distances, indices = index.search(np.array(vector), 10)

    vector_scores = {}

    for score, idx in zip(distances[0], indices[0]):

        if idx == -1:
            continue

        text = chunks[idx]["text"]

        vector_scores[text] = score

    # -------------------------------
    # BM25 SEARCH
    # -------------------------------
    tokenized_query = tokenize(expanded_query)

    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_top = np.argsort(bm25_scores)[::-1][:10]

    bm25_dict = {}

    for idx in bm25_top:

        text = chunks[idx]["text"]

        bm25_dict[text] = bm25_scores[idx]

    # -------------------------------
    # MERGE RESULTS
    # -------------------------------
    combined = {}

    for chunk in chunks:

        text = chunk["text"]

        v_score = vector_scores.get(text, 0)
        b_score = bm25_dict.get(text, 0)

        if v_score > 0 or b_score > 0:

            combined[text] = {
                "chunk": chunk,
                "vector": v_score,
                "bm25": b_score
            }

    # -------------------------------
    # NORMALIZE SCORES
    # -------------------------------
    max_v = max([v["vector"] for v in combined.values()] + [1])
    max_b = max([v["bm25"] for v in combined.values()] + [1])

    for item in combined.values():

        item["vector"] /= max_v
        item["bm25"] /= max_b

        item["score"] = 0.7 * item["vector"] + 0.3 * item["bm25"]

    # -------------------------------
    # SORT BY FINAL SCORE
    # -------------------------------
    ranked = sorted(
        combined.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    return [r["chunk"] for r in ranked[:k]]