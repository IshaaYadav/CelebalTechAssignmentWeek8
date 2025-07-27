# 💬 Loan Approval RAG Q&A Chatbot

A **Retrieval-Augmented Generation (RAG)** powered chatbot to answer questions about **loan approval criteria** based on a real dataset. This app retrieves the most relevant data chunks using **FAISS** and answers intelligently using a **Hugging Face LLM**.
> 🔗Chatbot **[🚀 Live App on Streamlit Cloud](https://loanapprovalchatbot.streamlit.app/)**

## DEMO 🎥 
![](demo.gif)
---
## 📦 Dataset Source

- 📊 **Kaggle Dataset**: [Loan Approval Prediction](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction)
---

## 🛠️ How the Project Works
This RAG chatbot intelligently answers questions using a combination of document retrieval + generative AI.

🔁 Step-by-Step Pipeline
1. Data Cleaning & Chunking:	The raw Kaggle dataset (Training_Dataset.csv) is cleaned and each row is converted into a readable text "chunk".
2. Text Embedding:	Each chunk is embedded into a dense vector using sentence-transformers.
3. FAISS Vector Index:	These embeddings are stored in a FAISS index for fast similarity search.
4. Query Embedding:	The user query is embedded using the same transformer model.
5. Similarity Search:	FAISS retrieves the top-k most relevant chunks from the dataset based on cosine/L2 similarity.
6. Prompt Creation:	The retrieved chunks + user query are formatted into a prompt.
7. Answer Generation:	The prompt is passed to a generative model which outputs a natural language answer.
8. Display on UI:	Streamlit displays the answer, retrieved context, and a bar chart of similarity scores.
---
## 🧠 Example:
If you ask:
“What income is needed for loan approval?”
The app:
Finds top rows where income is mentioned alongside approved loans.
Passes those as context to the model.
Generates an informed, readable answer like:
“Applicants with a combined income above 6000 have a higher chance of approval, especially with good credit history.”

## 📁 Project Structure
chatbot-loan-qa/
├── data/
│
├── embeddings/
│ ├── faiss_index.faiss # FAISS vector store
│ └── vector_store.pkl # Metadata for chunks
│
├── utils/
│ ├── preprocessing.py # Clean and chunk dataset
│ ├── embed_store.py # Embedding + FAISS index creation
│ ├── retrieval.py # Top-k document retriever
│ └── generator.py # RAG-style text generator
│
├── app/
│ └── streamlit_app.py # Streamlit frontend
│
├── demo.gif
│
├── requirements.txt
└── README.md 
---

## 🔧 Technologies Used

| Component             | Tech Stack                                       |
|----------------------|--------------------------------------------------|
| Embedding            | `sentence-transformers/all-MiniLM-L6-v2`         |
| Retrieval            | `FAISS`                                          |
| Generator Model      | `google/flan-t5-base`                            |
| Frontend             | `Streamlit`                                      |
| Vector Storage       | `FAISS`                                          |
| Language             | Python                                           |

---

## 💡 Sample Questions to Ask

> Try these in the app:

- What income is needed to get a loan approved?
- Does credit history affect approval?
- Are urban applicants more likely to be approved?
- How much loan is given to graduates?

---

## 🚀 Deployment
Deployed on **Streamlit Cloud**:

📄 License This project is licensed under the MIT License.
👩‍💻 Developed by: Isha Yadav Btech CSE (AIML)
