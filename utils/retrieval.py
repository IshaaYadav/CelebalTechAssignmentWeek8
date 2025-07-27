# utils/retrieval.py

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGRetriever:
    def __init__(self,
                 index_path="embeddings/faiss_index.faiss",
                 metadata_path="embeddings/vector_store.pkl",
                 model_name="all-MiniLM-L6-v2"):
        print("📦 Loading FAISS index and metadata...")
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.chunks = pickle.load(f)
        
        print(f"🔗 Loaded {len(self.chunks)} document chunks.")
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, top_k=3):
        print(f"🔍 Query: {query}")
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                "chunk": self.chunks[idx],
                "score": float(dist)
            })
        return results
