# rag/embed_utils.py

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_and_save_embeddings(doc_path, embed_path, model_name="all-MiniLM-L6-v2"):
    # Load documents
    with open(doc_path, "rb") as f:
        documents = pickle.load(f)

    # Load embedding model
    print("🚀 Loading model...")
    model = SentenceTransformer(model_name)

    # Generate embeddings
    print("🔍 Generating embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True)

    # Save embeddings and documents
    os.makedirs(os.path.dirname(embed_path), exist_ok=True)
    np.save(embed_path, embeddings)

    docs_out = embed_path.replace(".npy", "_docs.pkl")
    with open(docs_out, "wb") as f:
        pickle.dump(documents, f)

    print(f"✅ Saved {len(embeddings)} embeddings to {embed_path}")
    print(f"✅ Corresponding documents saved to {docs_out}")

if __name__ == "__main__":
    DOC_PATH = "./documents/indexed_docs.pkl"
    EMBED_PATH = "./vectorstore/embeddings.npy"
    generate_and_save_embeddings(DOC_PATH, EMBED_PATH)
