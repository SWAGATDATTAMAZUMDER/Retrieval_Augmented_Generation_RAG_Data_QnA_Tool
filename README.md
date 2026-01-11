# Medi-Context RAG Explorer
# Retrieval-Augmented Generation (RAG) Data QnA Tool

This project is a **Retrieval-Augmented Generation (RAG)** based Question-Answering system that allows users to query multiple **medical reference documents** and receive **context-aware, AI-generated answers**.

The core logic is implemented as a **backend RAG engine**, while a lightweight **Streamlit frontend** is used to provide a simple and interactive user interface.

---

## What This Project Does

- Loads and processes multiple medical PDF documents  
- Converts document text into vector embeddings  
- Stores embeddings in a vector database  
- Retrieves the most relevant document context for a user query  
- Uses **Google Gemini** to generate answers grounded in the source documents  

This approach helps reduce hallucinations and ensures responses remain aligned with authoritative medical references.

---

##  Features

-  **Multi-Document Support**  
  Query across multiple PDFs such as SOPs, training manuals, and treatment workflows  

-  **Context-Aware Retrieval**  
  Relevant document sections are retrieved before generating answers  

-  **Efficient Vector Search**  
  Uses Chroma / FAISS for fast semantic similarity search  

-  **Streamlit Interface**  
  Simple UI for asking questions without interacting with notebooks  

-  **Secure API Handling**  
  API keys are never hard-coded and are handled securely  

---

##  Project Structure

```
├── RAG_Engine.py         # Core backend RAG engine
├── streamlit_app.py      # Streamlit UI that interacts with the engine
├── documents/            # Medical reference PDFs
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── RESTART.md            # Guide to restarting Streamlit in Codespaces
```

---

## ⚙️ Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️. Add Documents
Place all reference PDF files inside the `documents/` folder.

### 3️. Run the Application
```bash
streamlit run streamlit_app.py
```

Once started, open the displayed URL (usually `http://localhost:8501`) to access the UI.

---

##  Tech Stack

- **Python** – Core programming language  
- **LangChain** – RAG pipeline orchestration  
- **Chroma / FAISS** – Vector database  
- **pypdf** – PDF text extraction  
- **Google Gemini API** – Large Language Model  
- **Streamlit** – Frontend interface  

---

##  Restarting the App

If you restart your Codespace or close the browser, the Streamlit app will stop running.  
Follow the steps in **[RESTART.md](./RESTART.md)** to bring it back up quickly.

---

##  Purpose & Learning Goals

This project was built as a hands-on learning exercise to explore:

- Retrieval-Augmented Generation (RAG)
- Vector databases and semantic search
- Practical LLM integration
- Backend-first AI system design with lightweight UIs

The focus is on clarity, correctness, and extensibility, rather than production deployment.

---

##  Author:
**Swagat Datta Mazumder**  
Aspiring Data Analyst & Engineer  
Exploring applied AI, data systems, and real-world problem solving
