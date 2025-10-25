from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
import os
from dotenv import load_dotenv

load_dotenv()

def load_document(file_path: str) -> List[Document]:
    """Carga y splittea un documento (PDF o text)."""
    if file_path.endswith('.pdf'):
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif file_path.endswith('.txt'):
        encodings = ['utf-8', 'latin-1', 'windows-1252']
        text = None
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                print(f"Archivo cargado con codificación: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise ValueError(f"No se pudo decodificar {file_path} con ninguna codificación.")
        docs = [Document(page_content=text)]
    else:
        raise ValueError(f"Formato no soportado: {file_path}")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def setup_chroma(file_path: str, collection_name: str = "faqs") -> VectorStoreRetriever:
    """Configura Chroma con embeddings del documento."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docs = load_document(file_path)
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})