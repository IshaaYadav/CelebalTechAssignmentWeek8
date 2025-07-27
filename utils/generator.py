# utils/generator.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

class RAGGenerator:
    def __init__(self, model_name="google/flan-t5-base", max_input_tokens=512):
        print(f"🚀 Loading generator model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.max_input_tokens = max_input_tokens

    def build_prompt(self, query: str, chunks: list) -> str:
        context = "\n".join([f"- {item['chunk']}" for item in chunks])
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        return prompt

    def generate_answer(self, query: str, retrieved_chunks: list) -> str:
        prompt = self.build_prompt(query, retrieved_chunks)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens).to(self.device)

        # Generate
        output = self.model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7
        )

        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer
