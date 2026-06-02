"""RAG服务入口"""
from rag.document_loader import load_documents
from rag.splitter import split_documents
from rag.vector_store import get_vectorstore, create_vectorstore, get_retriever


class RAGService:
    """RAG检索服务"""
    
    def __init__(self, knowledge_dir: str = "data/knowledge"):
        self.knowledge_dir = knowledge_dir
        self._init_vectorstore()
    
    def _init_vectorstore(self):
        if get_vectorstore() is None:
            docs = load_documents(self.knowledge_dir)
            if docs:
                chunks = split_documents(docs)
                create_vectorstore(chunks)
    
    def retrieve(self, query: str, k: int = 3) -> str:
        retriever = get_retriever(k=k)
        if retriever is None:
            return ""
        docs = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)
    
    def refresh(self):
        """重新加载知识库"""
        docs = load_documents(self.knowledge_dir)
        if docs:
            chunks = split_documents(docs)
            create_vectorstore(chunks)
