from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import time

SOURCE_DIR = "sources"
VECTOR_DIR = "vectorstore"
MODEL_NAME = "llama3"

def load_documents():
    documents = []
    for file in os.listdir(SOURCE_DIR):
        path = os.path.join(SOURCE_DIR, file)
        if file.endswith(".txt"):
            loader = TextLoader(path)
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            continue
        documents.extend(loader.load())
    return documents

def update_vector_store():
    docs = load_documents()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    if os.path.exists(f"{VECTOR_DIR}/index.faiss"):
        vectorstore = FAISS.load_local(VECTOR_DIR, embeddings)
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DIR)

if __name__ == "__main__":
    update_vector_store()
