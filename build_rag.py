import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_FOLDER = "rag_data"

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------- STEP 1: Load all txt files ----------
def load_txt_files():
    texts = []
    sources = []

    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".txt"):
            path = os.path.join(DATA_FOLDER, file)
            content = open(path, "r", encoding="utf-8").read()
            texts.append(content)
            sources.append(file)

    return texts, sources


# ---------- STEP 2: Chunk text ----------
def chunk_text(text, size=200):
    words = text.split()
    chunks = []

    for i in range(0, len(words), size):
        chunks.append(" ".join(words[i:i+size]))

    return chunks


# ---------- STEP 3: Prepare dataset ----------
def build_dataset():

    raw_texts, sources = load_txt_files()

    all_chunks = []
    metadata = []

    for text, source in zip(raw_texts, sources):
        chunks = chunk_text(text)

        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append(source)

    return all_chunks, metadata


# ---------- STEP 4: Convert to embeddings ----------
def create_embeddings(chunks):
    return model.encode(chunks)


# ---------- STEP 5: Build FAISS index ----------
def build_index(embeddings):

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)

    index.add(np.array(embeddings))

    return index


# ---------- STEP 6: Save everything ----------
def save(index, chunks, metadata):

    faiss.write_index(index, "faiss.index")

    pickle.dump(chunks, open("chunks.pkl", "wb"))
    pickle.dump(metadata, open("meta.pkl", "wb"))


# ---------- RUN ----------
if __name__ == "__main__":

    chunks, metadata = build_dataset()

    embeddings = create_embeddings(chunks)

    index = build_index(embeddings)

    save(index, chunks, metadata)

    print("RAG dataset built successfully!")