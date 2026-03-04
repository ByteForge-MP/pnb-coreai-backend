import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# load FAISS
index = faiss.read_index("faiss.index")

# load chunks
chunks = pickle.load(open("chunks.pkl", "rb"))

# prepare BM25
corpus = [c["text"] for c in chunks]
tokenized_corpus = [doc.split() for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)


def retrieve(query, k=3):

    # ---------- VECTOR SEARCH ----------
    vector = model.encode([query], normalize_embeddings=True)

    distances, indices = index.search(np.array(vector), k)

    vector_results = []

    for idx in indices[0]:
        if idx != -1:
            vector_results.append(chunks[idx])


    # ---------- BM25 SEARCH ----------
    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:k]

    bm25_results = [chunks[i] for i in top_indices]


    # ---------- MERGE ----------
    combined = vector_results + bm25_results

    unique = []
    seen = set()

    for item in combined:

        text = item["text"]

        if text not in seen:
            seen.add(text)
            unique.append(item)

    return unique[:k]


# ---------- TEST ----------
query = "Who is Ashok?"

results = retrieve(query)

print("\nQuery:", query)
print("\nRetrieved Chunks:\n")

for r in results:
    print(r["text"])
    print("Source:", r["source"])
    print("------------------")