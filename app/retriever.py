import faiss
import pickle
import numpy as np
import re
import json
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Load FAISS
index = faiss.read_index("faiss.index")

# Load chunks
chunks = pickle.load(open("chunks.pkl", "rb"))

# -------------------------------
# LOAD ABBREVIATIONS
# -------------------------------
with open("abbreviations.json", "r") as f:
    abbreviations = json.load(f)

# -------------------------------
# TOKENIZATION FUNCTION
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

    # keep original query and append expansions
    expanded_query = q + " " + " ".join(expanded_terms)

    return expanded_query


# -------------------------------
# PREPARE BM25 CORPUS
# -------------------------------
corpus = [c["text"] for c in chunks]

tokenized_corpus = [tokenize(doc) for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)


def retrieve(query, k=3):

    smalltalk = ["hello", "hi", "hey", "good morning"]

    if query.lower().strip() in smalltalk:
        return []

    # -------------------------------
    # EXPAND QUERY
    # -------------------------------
    expanded_query = expand_query(query)

    # ---------- VECTOR SEARCH ----------
    vector = model.encode([expanded_query], normalize_embeddings=True)

    # search more candidates internally
    distances, indices = index.search(np.array(vector), 10)

    vector_results = []

    for idx in indices[0]:
        if idx != -1:
            vector_results.append(chunks[idx])

    # ---------- BM25 KEYWORD SEARCH ----------
    tokenized_query = tokenize(expanded_query)

    bm25_scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(bm25_scores)[::-1][:10]

    bm25_results = [chunks[i] for i in top_indices]

    # ---------- MERGE RESULTS ----------
    combined = vector_results + bm25_results

    # remove duplicates while preserving order
    unique = []
    seen = set()

    for item in combined:

        text = item["text"]

        if text not in seen:
            seen.add(text)
            unique.append(item)

    return unique[:k]