import streamlit as st
import os
from dotenv import load_dotenv

# Load env
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Langchain imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# PDF Processing
# -----------------------------
def process_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    docs = splitter.split_documents(pages)
    return docs

# -----------------------------
# Create Vector Store
# -----------------------------
@st.cache_resource
def create_vectorstore(pdf_path):
    docs = process_pdf(pdf_path)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# -----------------------------
# Format Docs
# -----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -----------------------------
# Build RAG Chain
# -----------------------------
def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    llm = ChatGroq(model="llama-3.1-8b-instant")

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Chat with PDF", layout="wide")

st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    vectorstore = create_vectorstore("temp.pdf")
    rag_chain = create_rag_chain(vectorstore)

    # Chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.chat_input("Ask something about the PDF...")

    if user_input:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get response
        response = rag_chain.invoke(user_input)

        st.session_state.messages.append({"role": "assistant", "content": response})

    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])