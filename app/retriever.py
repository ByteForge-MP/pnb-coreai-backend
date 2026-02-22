import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("faiss.index")

chunks = pickle.load(open("chunks.pkl", "rb"))


def retrieve(query, k=3):

    vector = model.encode([query])

    distances, indices = index.search(np.array(vector), k)

    return [chunks[i] for i in indices[0] if i != -1]