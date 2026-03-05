import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


# -------------------------------------------------
# STEP 1: LOAD EMBEDDING MODEL
# -------------------------------------------------

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


# -------------------------------------------------
# STEP 2: CONFIGURATION
# -------------------------------------------------

KB_FOLDER = "rag_data"

CHUNK_SIZE = 120        # number of words per chunk
CHUNK_OVERLAP = 40      # overlapping words


documents = []


# -------------------------------------------------
# STEP 3: LOAD DOCUMENTS
# -------------------------------------------------

for file in os.listdir(KB_FOLDER):

    if file.endswith(".txt"):

        path = os.path.join(KB_FOLDER, file)

        with open(path, "r", encoding="utf-8") as f:

            text = f.read()

        documents.append({
            "source": file,
            "content": text
        })


print("Documents loaded:", len(documents))


# -------------------------------------------------
# STEP 4: SMART CHUNKING FUNCTION
# -------------------------------------------------

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# -------------------------------------------------
# STEP 5: CREATE CHUNKS
# -------------------------------------------------

chunks = []

for doc in documents:

    text_chunks = chunk_text(doc["content"])

    for chunk in text_chunks:

        chunks.append({
            "text": chunk,
            "source": doc["source"]
        })


print("Total chunks created:", len(chunks))


# -------------------------------------------------
# STEP 6: CREATE EMBEDDINGS
# -------------------------------------------------

texts = [c["text"] for c in chunks]

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
).astype("float32")

print("Embedding shape:", embeddings.shape)


# -------------------------------------------------
# STEP 7: NORMALIZE EMBEDDINGS
# -------------------------------------------------

faiss.normalize_L2(embeddings)


# -------------------------------------------------
# STEP 8: CREATE FAISS INDEX
# -------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("FAISS index size:", index.ntotal)


# -------------------------------------------------
# STEP 9: SAVE INDEX + METADATA
# -------------------------------------------------

faiss.write_index(index, "faiss.index")

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)


print("Knowledge base built successfully.")