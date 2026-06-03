"""
向量数据库模块
==============

本模块封装了基于 Chroma 的向量存储功能，用于 RAG（检索增强生成）系统中的文档索引和相似度检索。

核心功能:
    - 使用 HuggingFace 的 BGE 模型将文本转换为向量（Embedding）
    - 使用 Chroma 向量数据库存储和检索文档
    - 支持持久化存储，避免重复计算 Embedding

依赖包:
    - langchain-chroma: Chroma 向量数据库的 LangChain 集成
    - langchain-huggingface: HuggingFace Embedding 模型集成
    - sentence-transformers: 本地 Embedding 模型运行时

暂时未加入新文件入知识库自动触发加载，而是在rag_service存在手动重新加载
"""
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import VECTORSTORE_PERSIST_DIR, EMBEDDING_MODEL_NAME

# 模块级单例，避免重复加载向量数据库
_vectorstore = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    获取 Embedding 模型实例
    
    使用 BAAI/bge-small-zh-v1.5 模型，这是一个针对中文优化的小型 Embedding 模型:
        - 模型大小: ~100MB
        - 向量维度: 512
        - 特点: 中文效果好，资源占用低，适合本地部署
    
    Returns:
        HuggingFaceEmbeddings: 配置好的 Embedding 模型实例
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_vectorstore() -> Chroma | None:
    """
    获取向量数据库实例（懒加载单例模式）
    
    如果向量数据库已存在于磁盘，则加载并返回；否则返回 None。
    使用模块级变量缓存实例，避免重复加载。
    
    Returns:
        Chroma | None: 向量数据库实例，如果不存在则返回 None
    """
    global _vectorstore
    
    # 如果已缓存，直接返回
    if _vectorstore is not None:
        return _vectorstore
    
    # 检查持久化目录是否存在，存在则加载
    if Path(VECTORSTORE_PERSIST_DIR).exists():
        _vectorstore = Chroma(
            persist_directory=VECTORSTORE_PERSIST_DIR,
            embedding_function=get_embeddings()
        )
        return _vectorstore
    
    return None


def create_vectorstore(documents: list[Document]) -> Chroma:
    """
    创建向量数据库并索引文档
    
    将传入的文档列表转换为向量并存储到 Chroma 数据库中。
    新创建的数据库会自动持久化到 VECTORSTORE_PERSIST_DIR 目录。
    
    Args:
        documents: 待索引的文档列表，每个文档包含 page_content（文本）和 metadata（元数据）
    
    Returns:
        Chroma: 创建好的向量数据库实例
    
    Note:
        调用此函数会覆盖已有的向量数据库缓存
    """
    global _vectorstore
    
    _vectorstore = Chroma.from_documents(
        documents=documents,                    # 待索引的文档
        embedding=get_embeddings(),             # Embedding 模型
        persist_directory=VECTORSTORE_PERSIST_DIR  # 持久化目录
    )
    return _vectorstore


def get_retriever(k: int = 3):
    """
    获取文档检索器
    
    基于向量数据库创建检索器，用于根据查询文本检索相似文档。
    
    Args:
        k: 返回的最相似文档数量，默认 3 条
    
    Returns:
        VectorStoreRetriever | None: 检索器实例，如果向量数据库不存在则返回 None
    
    Example:
        >>> retriever = get_retriever(k=5)
        >>> docs = retriever.invoke("什么是 RAG?")  # 检索 5 条最相关的文档
    """
    vs = get_vectorstore()
    if vs is None:
        return None
    
    return vs.as_retriever(search_kwargs={"k": k})
