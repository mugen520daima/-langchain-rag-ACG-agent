"""向量数据库"""
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

PERSIST_DIR = "data/vectorstore"
_vectorstore = None


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")


def get_vectorstore() -> Chroma | None:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    
    if Path(PERSIST_DIR).exists():
        _vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embeddings())
        return _vectorstore
    return None


def create_vectorstore(documents: list[Document]) -> Chroma:
    global _vectorstore
    _vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        persist_directory=PERSIST_DIR
    )
    return _vectorstore


def get_retriever(k: int = 3):
    vs = get_vectorstore()
    if vs is None:
        return None
    return vs.as_retriever(search_kwargs={"k": k})
