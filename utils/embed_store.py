# utils/embed_store.py

import pickle
import faiss
import os

from sentence_transformers import SentenceTransformer
import numpy as np

def load_chunks(chunk_path: str) -> list:
    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"📦 Loaded {len(chunks)} chunks.")
    return chunks

def create_embeddings(chunks: list, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings, model

def create_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"📊 FAISS index built with {index.ntotal} vectors.")
    return index

def save_index(index, index_path: str):
    faiss.write_index(index, index_path)
    print(f"💾 FAISS index saved to {index_path}")

def save_metadata(chunks: list, output_path: str):
    with open(output_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"📝 Metadata saved to {output_path}")

if __name__ == "__main__":
    CHUNKS_PATH = "data/cleaned_chunks.pkl"
    INDEX_PATH = "embeddings/faiss_index.faiss"
    METADATA_PATH = "embeddings/vector_store.pkl"

    os.makedirs("embeddings", exist_ok=True)

    print("📥 Loading document chunks...")
    chunks = load_chunks(CHUNKS_PATH)

    print("🔗 Creating embeddings...")
    embeddings, model = create_embeddings(chunks)

    print("📂 Creating FAISS index...")
    index = create_faiss_index(np.array(embeddings))

    print("💾 Saving FAISS index and metadata...")
    save_index(index, INDEX_PATH)
    save_metadata(chunks, METADATA_PATH)
