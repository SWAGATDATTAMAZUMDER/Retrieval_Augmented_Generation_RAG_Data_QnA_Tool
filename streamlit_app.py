import streamlit as st
import os

from RAG_engine import (
    load_documents,
    chunk_documents,
    create_or_load_vectorstore,
    create_qa_chain,
    answer_question,
)

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="📄 Document Q&A with RAG",
    page_icon="📄",
)

st.title("📄 Document Q&A with Gemini (RAG)")
st.write(
    "Ask questions about your PDF documents using Retrieval-Augmented Generation."
)

# =========================
# API Key Input (Gemini)
# =========================
if "GOOGLE_API_KEY" not in os.environ:
    api_key = st.text_input(
        "🔑 Enter your Google Gemini API Key",
        type="password"
    )
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

if "GOOGLE_API_KEY" not in os.environ:
    st.info("Please enter your Google Gemini API key to continue.")
    st.stop()

# =========================
# Build / Load RAG Pipeline
# =========================
if "qa_chain" not in st.session_state:
    with st.spinner("🔧 Loading documents and building vectorstore..."):
        pages = load_documents("RAG_Project")
        chunks = chunk_documents(pages)
        vectorstore = create_or_load_vectorstore(chunks)
        st.session_state.qa_chain = create_qa_chain(vectorstore)

    st.success("✅ RAG system is ready!")

# =========================
# Question Input
# =========================
question = st.text_area(
    "🔎 Ask a question about your documents:",
    placeholder="What is the National Policy for AMR Containment?",
)

if st.button("Get Answer") and question:
    with st.spinner("🧠 Thinking..."):
        answer, sources = answer_question(
            question,
            st.session_state.qa_chain
        )

    st.markdown("### 🧠 Answer")
    st.write(answer)

    if sources:
        st.markdown("### 📚 Sources")
        for i, doc in enumerate(sources, start=1):
            st.write(f"**Source {i}:** {doc.metadata.get('source', 'Unknown')}")

# =========================
# Utility: Rebuild Vectorstore
# =========================
if st.button("🔄 Rebuild Vectorstore"):
    with st.spinner("Rebuilding vectorstore..."):
        pages = load_documents("RAG_Project")
        chunks = chunk_documents(pages)
        vectorstore = create_or_load_vectorstore(chunks)
        st.session_state.qa_chain = create_qa_chain(vectorstore)

    st.success("✅ Vectorstore rebuilt successfully!")
