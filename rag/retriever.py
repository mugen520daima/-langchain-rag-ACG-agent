"""检索器"""
from rag.vector_store import get_retriever


def retrieve_documents(query: str, k: int = 3) -> list:
    """检索相关文档"""
    retriever = get_retriever(k=k)
    if retriever is None:
        return []
    return retriever.invoke(query)
