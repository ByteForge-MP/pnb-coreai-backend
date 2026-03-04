import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

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
def chunk_text(text, chunk_size=40):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


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

embeddings = model.encode(texts)

print("Embeddings shape:", embeddings.shape)

# STEP 4: CREATE FAISS INDEX
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))


# STEP 5: SAVE DATA
faiss.write_index(index, "faiss.index")

pickle.dump(chunks, open("chunks.pkl", "wb"))

print("Knowledge base built successfully")
