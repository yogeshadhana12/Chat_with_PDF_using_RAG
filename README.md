# 📄 Chat with PDF using RAG (LLM App)

🚀 An AI-powered application that allows users to chat with PDF documents using Retrieval-Augmented Generation (RAG).

---

## 🔥 Live Demo
👉 Coming soon...

---

## 📌 Features
- 📄 Upload and process PDF documents
- 🔍 Semantic search using vector embeddings
- 💬 Chat interface (like ChatGPT)
- ⚡ Real-time response streaming
- 🧠 Context-aware answers using LLM

---

## 🧠 How It Works
1. PDF is loaded and split into chunks  
2. Text is converted into embeddings  
3. Stored in FAISS vector database  
4. User query retrieves relevant chunks  
5. LLM generates answer using context  

---

## 🛠 Tech Stack
- **LangChain**
- **FAISS (Vector DB)**
- **HuggingFace Embeddings**
- **Groq (LLaMA 3)**
- **Streamlit**

---

## 📂 Project Structure

chat-with-pdf-rag/
│
├── chat_with_PDF_using_RAG_with_FAISS.py    # Main ipynb file (RAG pipeline)
├── dashboard app.py    # Main Streamlit file (Dashboard)
├── requirements.txt    # List of dependencies
├── .gitignore          # Files to ignore in Git
├── README.md           # Project documentation



## ▶️ Run Locally

1. Download the project from GitHub  
2. Open terminal in the project folder  

Run:

pip install -r requirements.txt  
streamlit run app.py  
