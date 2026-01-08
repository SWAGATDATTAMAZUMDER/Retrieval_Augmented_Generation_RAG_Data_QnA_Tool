# RAG_Engine.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


def load_documents(documents_path="documents"):
    pages = []

    for file_name in os.listdir(documents_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(documents_path, file_name)
            loader = PyPDFLoader(file_path)
            pages.extend(loader.load())

    return pages


def chunk_documents(pages, chunk_size=200, overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(pages)
    return chunks


def create_or_load_vectorstore(chunks, persist_directory="vectorstore"):
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    if os.path.exists(persist_directory):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vector_store.persist()

    return vector_store


def create_qa_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0
    )

    retriever = vector_store.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="refine",
        return_source_documents=True
    )

    return qa_chain


def answer_question(query, qa_chain):
    result = qa_chain.invoke({"query": query})
    return result["result"], result["source_documents"]
