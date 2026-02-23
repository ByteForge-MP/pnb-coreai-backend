import os
import re
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

DATA_FOLDER = "rag_data/positions"

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------- STEP 1: Load all txt files ----------
def load_txt_files():
    texts = []
    sources = []

    for path in Path(DATA_FOLDER).rglob("*.txt"):
        content = path.read_text(encoding="utf-8")
        texts.append(content)
        sources.append(str(path))   # keep full path for debugging

    return texts, sources


# ---------- STEP 2: Chunk by TOPIC + SECTION ----------
def chunk_text(text, source):

    # Extract topic
    topic_match = re.search(r"TOPIC:\s*(.*)", text)
    topic = topic_match.group(1).strip() if topic_match else "Unknown"

    # Split by SECTION
    parts = re.split(r"SECTION:\s*(.*)", text)

    chunks = []
    metadata = []

    for i in range(1, len(parts), 2):
        section = parts[i].strip()
        content = parts[i+1].strip()

        if len(content) < 10:
            continue

        chunk = f"""
TOPIC: {topic}
SECTION: {section}

{content}
""".strip()

        chunks.append(chunk)

        metadata.append({
            "source": source,
            "topic": topic,
            "section": section
        })

    return chunks, metadata


# ---------- STEP 3: Prepare dataset ----------
def build_dataset():

    raw_texts, sources = load_txt_files()

    all_chunks = []
    metadata = []

    for text, source in zip(raw_texts, sources):

        chunks, meta = chunk_text(text, source)

        all_chunks.extend(chunks)
        metadata.extend(meta)

    return all_chunks, metadata


# ---------- STEP 4: Convert to embeddings ----------
def create_embeddings(chunks):
    return model.encode(chunks, convert_to_numpy=True)


# ---------- STEP 5: Build FAISS index ----------
def build_index(embeddings):

    dim = embeddings.shape[1]

    # cosine similarity index (better than L2 for text)
    index = faiss.IndexFlatIP(dim)

    # normalize vectors for cosine search
    faiss.normalize_L2(embeddings)

    index.add(embeddings)

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