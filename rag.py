import os
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()  # now GROQ_API_KEY is available

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ===============================
# GLOBALS (persist across calls)
# ===============================
VECTOR_DB = None
RAG_CHAIN = None

CHROMA_DIR = "./chroma_db"


# ===============================
# PROCESS URLS
# ===============================
def process_urls(urls: List[str]):
    global VECTOR_DB, RAG_CHAIN

    yield "Loading documents..."
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()

    yield "Splitting documents..."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    yield "Creating embeddings..."
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    yield "Creating vector database..."
    VECTOR_DB = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    retriever = VECTOR_DB.as_retriever(search_kwargs={"k": 4})

    yield "Loading LLM..."
    llm = ChatGroq(
        model="groq/compound",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a real estate research assistant.
Use ONLY the context below to answer.

Context:
{context}

Question:
{question}

Give a clear, factual answer.
"""
    )

    RAG_CHAIN = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    yield "✅ URLs processed successfully!"


# ===============================
# GENERATE ANSWER
# ===============================
def generate_answer(query: str):
    global VECTOR_DB, RAG_CHAIN

    if VECTOR_DB is None or RAG_CHAIN is None:
        raise RuntimeError("URLs not processed yet")

    response = RAG_CHAIN.invoke(query)

    docs = VECTOR_DB.similarity_search(query, k=4)
    sources = "\n".join(
        {doc.metadata.get("source", "Unknown") for doc in docs}
    )

    return response, sources
