# app/streamlit_app.py
# app/streamlit_app.py

import sys
import os

# Add root directory to sys.path so utils can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import streamlit as st
from utils.retrieval import RAGRetriever
from utils.generator import RAGGenerator

# App title
st.set_page_config(page_title="Loan QA RAG Chatbot 💬", layout="wide")
st.title("🏦 Loan Approval Q&A Chatbot (RAG powered)")
st.markdown("Ask any question related to loan approvals based on the dataset...")

# Load models
@st.cache_resource
def load_rag_components():
    retriever = RAGRetriever()
    generator = RAGGenerator()
    return retriever, generator

retriever, generator = load_rag_components()

# User input
query = st.text_input("🔍 Enter your question:", "")

if query:
    with st.spinner("Retrieving and generating answer..."):
        retrieved_chunks = retriever.retrieve(query, top_k=3)
        answer = generator.generate_answer(query, retrieved_chunks)

    # Output section
    st.subheader("🧠 Generated Answer:")
    st.success(answer)

    # Context section
    st.subheader("📄 Retrieved Context:")
    for i, item in enumerate(retrieved_chunks):
        st.markdown(f"**Chunk {i+1} (Score: {item['score']:.2f})**")
        st.code(item["chunk"], language="markdown")

import matplotlib.pyplot as plt

# Visualize similarity scores
st.subheader("📊 Similarity Scores of Retrieved Chunks")

# Prepare data
labels = [f"Chunk {i+1}" for i in range(len(retrieved_chunks))]
scores = [item["score"] for item in retrieved_chunks]

# Normalize scores (optional, FAISS uses L2 distance, lower is better)
normalized_scores = [1 - (s / max(scores)) for s in scores]

# Plot bar chart
fig, ax = plt.subplots()
ax.barh(labels, normalized_scores, color="skyblue")
ax.set_xlabel("Relevance Score (1 = most relevant)")
ax.invert_yaxis()
st.pyplot(fig)

