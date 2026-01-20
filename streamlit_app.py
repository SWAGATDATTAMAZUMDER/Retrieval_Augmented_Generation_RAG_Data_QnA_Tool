import streamlit as st
import os

# Importing our custom functiuons from the Backend File #
from RAG_engine import (
    load_vectorstore,
    create_qa_chain,
    answer_question,
)

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="📄 MediContext RAG Explorer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 MediContext RAG Explorer")
st.write(
    "Ask questions about your PDF documents using this Retrieval-Augmented Generation (RAG) System."
)

# =========================
# API Key Input (Gemini)
# =========================
with st.sidebar:
    st.header("Settings")
    # Using the "Gemini_API_KEY" to match the backend logic #
    if "Gemini_API_KEY" not in os.environ:
        api_key = st.text_input("Enter the Gemini API Key", type="password")
        if api_key:
            os.environ["Gemini_API_KEY"] = api_key

    st.info("System Status: Using pre-built database.")


# ========================= #
# Loading the vector Database (The Brain of the RAG) #
# ========================= #
if "qa_chain" not in st.session_state:
   # Ensuring API Key is present before loading #
   if "Gemini_API_KEY" in os.environ:
       with st.spinner("Connecting to your medical database....") :
           # 1) Loading the existing database, No rebuilding required #
           vectorstore = load_vectorstore() 

           if vectorstore:
               # 2) Creating the connection #
               st.session_state.qa_chain = create_qa_chain(vectorstore)
               st.success("The RAG System is ready now !")
           else:
               st.error("Database was not found. Please run 'create_db.py' first.")

# ========================= #
# Question Input #
# ========================= #
# Only to show the chat input if the system is ready #
if "qa_chain" in st.session_state: 
    question = st.text_area(
        # Ask a question about the document(s). #
        placeholder="What is the state-wide policy for AMR Containment ?",
    )

    if st.button("Get Answer") and question:
        with st.spinner (" Thinking .. .."):
            # Calling the Backend function #
            answer, sources = answer_question(
                question,
                st.session_state.qa_chain
            )
        
        st.markdown(" ## Answer ")
        st.write(answer)

        if sources:
            with st.expander(" Viewing Source Documents "):
                for i, doc in enumerate(sources, start=1):
                    st.write(f"**Source {i}:** {doc.metadata.get('sources', 'Unknown')}")
                    # To show the first 200 characters from the documents. #
                    st.write(f"_{doc.page_content[:200]}..._")
else:
    st.warning(" Please enter your API Key in the sidebar to start. ")