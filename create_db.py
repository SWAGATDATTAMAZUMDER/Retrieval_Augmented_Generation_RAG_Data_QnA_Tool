import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1.) Setup Paths #
documents_folder = "./documents"
persist_directory = "./chroma_db"

def create_vector_db() :
    # To check if documents folder exists #
    if not os.path.exists(documents_folder):
        print(f"ERROR: The fodler '{documents_folder}' does not exist.")
        print("Please create a folder named 'documetns' and put your PDFs inside.")
        return
    
    # 2.) Load PDFs #
    documents = []
    pdf_files = [f for f in os.listdir(documents_folder) if f.endswith('.pdf')]

    if not pdf_files:
        print("No PDF files found in the 'documents' folder.")
        return
    
    print(f"Found {len(pdf_files)} PDFs. Loading .... ..")

    for pdf_file in pdf_files:
        file_path = os.path.join(documents_folder, pdf_file)
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
        print(f" - Loaded: {pdf_file}")

    # 3.) Splitting the PDF File into smaller chunks (Chunking) #
    # To maintain the context of the text in the PDF file, will use larger chunks for heathcare context #
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    # 4.) Creating embeddings & Saving it to Disk for future reference #
    print("Creating vector Database .... .. (This might take a minute)")
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Creating and to maintain(persist) the database #
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=persist_directory
    )

    print(f"SUCCESS! Database saved to '{persist_directory}'")

if __name__ == "__main__":
    create_vector_db()