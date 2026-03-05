import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

KB_FOLDER = "rag_data"

documents = []

# STEP 1: LOAD DOCUMENTS
for file in os.listdir(KB_FOLDER):

    if file.endswith(".txt"):

        path = os.path.join(KB_FOLDER, file)

        with open(path, "r") as f:
            text = f.read()

        documents.append({
            "source": file,
            "content": text
        })


# STEP 2: CHUNK DOCUMENTS
def chunk_text(text):

    words = text.split(".")
    return [word.strip() for word in words if word.strip()]

chunks = []

for doc in documents:

    text_chunks = chunk_text(doc["content"])

    for chunk in text_chunks:

        chunks.append({
            "text": chunk,
            "source": doc["source"]
        })

print("Total chunks:", len(chunks))

# STEP 3: CREATE EMBEDDINGS
texts = [c["text"] for c in chunks]

embeddings = model.encode(texts).astype("float32")

print("Embeddings shape:", embeddings.shape)

faiss.normalize_L2(embeddings)

# STEP 4: CREATE FAISS INDEX
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(np.array(embeddings))


# STEP 5: SAVE DATA
faiss.write_index(index, "faiss.index")

pickle.dump(chunks, open("chunks.pkl", "wb"))

print("Knowledge base built successfully")
