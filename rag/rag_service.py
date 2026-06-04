"""RAG服务入口。

本模块是 RAG（Retrieval-Augmented Generation）检索服务的统一入口，
负责协调文档加载、切分、向量化、重排序和检索的完整流程。

检索流程：
    用户查询 → 向量召回(Top-K) → Rerank重排序 → 阈值过滤 → 结构化结果

核心职责：
- 初始化向量数据库（如果尚未创建）
- 提供基于语义相似度 + Rerank 的文档检索接口
- 支持知识库热更新
- 记录 Bad Case（高频检索失败的问题）
"""

import json
import logging
from pathlib import Path

from config import KNOWLEDGE_DIR, RAG_DEFAULT_K, MIN_RAG_SCORE, RERANK_MODEL_NAME, BAD_CASE_LOG_PATH
from rag.document_loader import load_documents
from rag.splitter import split_documents
from rag.vector_store import get_vectorstore, create_vectorstore
from rag.reranker import rerank

# 模块级日志器，用于记录 RAG 服务的运行状态
logger = logging.getLogger(__name__)


class RAGService:
    """RAG检索服务。

    检索流程：向量召回 → Rerank重排序 → 阈值过滤 → 结构化输出
    """

    def __init__(self, knowledge_dir: str = KNOWLEDGE_DIR):
        self.knowledge_dir = knowledge_dir
        logger.info(f"[RAGService] 初始化，知识库目录: {knowledge_dir}")
        self._init_vectorstore()

    def _init_vectorstore(self):
        """初始化向量数据库。"""
        if get_vectorstore() is None:
            logger.info("[RAGService] 向量库不存在，开始构建...")
            docs = load_documents(self.knowledge_dir)
            if docs:
                chunks = split_documents(docs)
                create_vectorstore(chunks)
                logger.info("[RAGService] 向量库构建完成")
            else:
                logger.warning("[RAGService] 未找到任何文档，向量库未创建")
        else:
            logger.debug("[RAGService] 向量库已存在，跳过初始化")

    def retrieve(self, query: str, k: int = RAG_DEFAULT_K) -> dict:
        """检索与查询最相关的文档片段，经 Rerank 重排序后返回结构化结果。

        返回 dict:
            - context: 拼接后的上下文文本（供 LLM 使用）
            - sources: 引用的来源文件列表
            - chunks: 原文片段列表（供前端高亮对比）
            - max_score: Rerank 后最高分数
            - is_confident: 是否达到可信阈值
        """
        logger.debug(f"[RAGService] retrieve 查询: {query[:50]}... k={k}")
        empty_result = {"context": "", "sources": [], "chunks": [], "max_score": 0.0, "is_confident": False}

        vs = get_vectorstore()
        if vs is None:
            logger.warning("[RAGService] 向量库为空，可能未初始化")
            return empty_result

        # Step 1: 向量召回候选集（召回更多，交给 Rerank 精排）
        recall_k = k * 3
        docs = vs.similarity_search(query, k=recall_k)
        if not docs:
            logger.info("[RAGService] 向量召回无结果")
            return empty_result

        # Step 2: Rerank 重排序
        reranked = rerank(query, docs, top_n=k)
        if not reranked:
            logger.info("[RAGService] Rerank 后无结果")
            return empty_result

        # Step 3: 提取结构化结果
        max_score = reranked[0]["score"]
        is_confident = max_score >= MIN_RAG_SCORE

        chunks = []
        for item in reranked:
            chunks.append({
                "content": item["doc"].page_content,
                "source": item["doc"].metadata.get("source", "unknown"),
                "score": item["score"],
            })

        sources = list(dict.fromkeys(c["source"] for c in chunks))
        context = "\n\n".join(c["content"] for c in chunks)

        logger.info(f"[RAGService] Rerank完成, top_score={max_score:.3f}, "
                    f"is_confident={is_confident}, sources={sources}")
        return {
            "context": context,
            "sources": sources,
            "chunks": chunks,
            "max_score": max_score,
            "is_confident": is_confident,
        }

    def log_bad_case(self, query: str, max_score: float):
        """记录检索失败的 Bad Case，用于数据飞轮闭环。"""
        bad_case_path = Path(BAD_CASE_LOG_PATH)
        bad_case_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"query": query, "max_score": max_score}
        with open(bad_case_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"[RAGService] Bad Case 已记录: {query[:50]}")

    def refresh(self):
        """重新加载知识库，手动触发重建向量库。"""
        logger.info("[RAGService] 开始刷新知识库...")
        docs = load_documents(self.knowledge_dir)
        if docs:
            chunks = split_documents(docs)
            create_vectorstore(chunks)
            logger.info("[RAGService] 知识库刷新完成")
        else:
            logger.warning("[RAGService] 刷新时未找到文档")