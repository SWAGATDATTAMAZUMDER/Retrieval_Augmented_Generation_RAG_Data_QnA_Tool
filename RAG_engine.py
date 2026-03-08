# RAG_Engine.py

import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


# 1.) Loading the existing Database (The "Brain" of the engine) #
def load_vectorstore(persist_directory = "./chroma_db"):
    #To ensure matching the embedding model we used in the create_db.py#
    embedding_function = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")

    if os.path.exists(persist_directory) :
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
        )
        return vector_store
    else :
        print(f"Error : Could not find the database at the {persist_directory}")
        return None
    
# Creating the Logic Chain#
def create_qa_chain(vector_store) : 

    import asyncio

    # Ensuring Streamlit thread has an event loop #
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Ensuring the API Key is set #
    if "Gemini_API_KEY" not in os.environ:
        raise ValueError("Gemini API Key is missing !")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.environ["Gemini_API_KEY"]
    )

    # To retireve/Search for the 4 most relevant retrieved chunks #
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",#using 'stuff' as it is faster than 'refine'#
        return_source_documents=True
    )

    return qa_chain

# 3) To Ask and answer the questions #
def answer_question(query, qa_chain) :
    result = qa_chain.invoke({"query": query})
    
    print("\n The Retrieved Documents: \n")

    for i, doc in enumerate(result["source_documents"], 1):
        print(f"Source {i}:\n{doc.page_content}\n")

    return result["result"], result["source_documents"]