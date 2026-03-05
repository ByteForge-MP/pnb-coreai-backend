import faiss
import pickle
import numpy as np
import re
import json
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# -------------------------------------------------
# STEP 1: LOAD EMBEDDING MODEL
# -------------------------------------------------

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


# -------------------------------------------------
# STEP 2: LOAD FAISS INDEX
# -------------------------------------------------

index = faiss.read_index("faiss.index")


# -------------------------------------------------
# STEP 3: LOAD CHUNKS
# -------------------------------------------------

chunks = pickle.load(open("chunks.pkl", "rb"))


# -------------------------------------------------
# STEP 4: LOAD ABBREVIATIONS
# -------------------------------------------------

with open("abbreviations.json", "r") as f:
    abbreviations = json.load(f)


# -------------------------------------------------
# STEP 5: QUERY EXPANSION FUNCTION
# -------------------------------------------------

def expand_query(query):

    q = query.lower()

    expanded_terms = []

    for short, full in abbreviations.items():

        if short in q:
            expanded_terms.append(full)

    # append expansions to query
    expanded_query = q + " " + " ".join(expanded_terms)

    return expanded_query


# -------------------------------------------------
# STEP 6: TOKENIZER
# -------------------------------------------------

def tokenize(text):
    return re.findall(r"\w+", text.lower())


# -------------------------------------------------
# STEP 7: PREPARE BM25 CORPUS
# -------------------------------------------------

corpus = [c["text"] for c in chunks]

tokenized_corpus = [tokenize(doc) for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)


# -------------------------------------------------
# STEP 8: HYBRID RETRIEVAL FUNCTION
# -------------------------------------------------

def retrieve(query, k=3):

    # ------------------------------
    # QUERY EXPANSION
    # ------------------------------

    expanded_query = expand_query(query)

    # ------------------------------
    # CLEAN QUERY
    # ------------------------------

    cleaned_query = tokenize(expanded_query)

    # ------------------------------
    # VECTOR SEARCH
    # ------------------------------

    vector = model.encode([expanded_query], normalize_embeddings=True)

    distances, indices = index.search(np.array(vector), 10)

    vector_results = {}

    for score, idx in zip(distances[0], indices[0]):

        if idx == -1:
            continue

        text = chunks[idx]["text"]

        vector_results[text] = {
            "chunk": chunks[idx],
            "vector_score": float(score)
        }

    # ------------------------------
    # BM25 SEARCH
    # ------------------------------

    bm25_scores = bm25.get_scores(cleaned_query)

    bm25_top = np.argsort(bm25_scores)[::-1][:10]

    bm25_results = {}

    for idx in bm25_top:

        text = chunks[idx]["text"]

        bm25_results[text] = {
            "chunk": chunks[idx],
            "bm25_score": float(bm25_scores[idx])
        }

    # ------------------------------
    # MERGE RESULTS
    # ------------------------------

    combined = {}

    for text, item in vector_results.items():

        combined[text] = {
            "chunk": item["chunk"],
            "vector_score": item["vector_score"],
            "bm25_score": 0
        }

    for text, item in bm25_results.items():

        if text in combined:
            combined[text]["bm25_score"] = item["bm25_score"]

        else:
            combined[text] = {
                "chunk": item["chunk"],
                "vector_score": 0,
                "bm25_score": item["bm25_score"]
            }

    # ------------------------------
    # NORMALIZE SCORES
    # ------------------------------

    vector_scores = [v["vector_score"] for v in combined.values()]
    bm25_scores = [v["bm25_score"] for v in combined.values()]

    max_vector = max(vector_scores) if vector_scores else 1
    max_bm25 = max(bm25_scores) if bm25_scores else 1

    for item in combined.values():

        item["vector_score"] /= max_vector
        item["bm25_score"] /= max_bm25

        item["final_score"] = (
            0.7 * item["vector_score"] +
            0.3 * item["bm25_score"]
        )

    # ------------------------------
    # SORT RESULTS
    # ------------------------------

    ranked = sorted(
        combined.values(),
        key=lambda x: x["final_score"],
        reverse=True
    )

    return ranked[:k]


# -------------------------------------------------
# STEP 9: TEST RETRIEVAL
# -------------------------------------------------

query = "Who is the MD of SBI?"

results = retrieve(query)

print("\nQuery:", query)
print("\nRetrieved Chunks:\n")

for r in results:

    chunk = r["chunk"]

    print(chunk["text"])
    print("Source:", chunk["source"])
    print("Vector Score:", round(r["vector_score"], 3))
    print("BM25 Score:", round(r["bm25_score"], 3))
    print("Final Score:", round(r["final_score"], 3))
    print("----------------------------")