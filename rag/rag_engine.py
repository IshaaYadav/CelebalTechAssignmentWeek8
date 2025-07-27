# rag/rag_engine.py

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import openai
import os

class RAGEngine:
    def __init__(self, embedding_path, model_type="huggingface", top_k=3):
        self.embedding_path = embedding_path
        self.top_k = top_k
        self.model_type = model_type
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Load embeddings & docs
        self.embeddings = np.load(embedding_path)
        with open(embedding_path.replace(".npy", "_docs.pkl"), "rb") as f:
            self.documents = pickle.load(f)

        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        # Load language model
        if model_type == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
        elif model_type == "huggingface":
            self.llm = pipeline(
                "text-generation",
                model="tiiuae/falcon-7b-instruct",  # You can replace with smaller model if needed
                tokenizer=AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct"),
                model_kwargs={"torch_dtype": "auto", "device_map": "auto"},
                max_new_tokens=150
            )

    def retrieve_docs(self, query):
        query_vec = self.embed_model.encode([query])
        scores, indices = self.index.search(query_vec, self.top_k)
        return [self.documents[i] for i in indices[0]]

    def generate_answer(self, query):
        retrieved = self.retrieve_docs(query)
        context = "\n".join(retrieved)

        prompt = f"""Answer the question based on the below context:\n\n{context}\n\nQuestion: {query}\nAnswer:"""

        if self.model_type == "openai":
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=150
            )
            return response['choices'][0]['message']['content'].strip()
        elif self.model_type == "huggingface":
            response = self.llm(prompt, do_sample=True)
            return response[0]['generated_text'].replace(prompt, "").strip()

if __name__ == "__main__":
    rag = RAGEngine("./vectorstore/embeddings.npy", model_type="huggingface")
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = rag.generate_answer(query)
        print(f"\n🤖 Answer: {answer}")
