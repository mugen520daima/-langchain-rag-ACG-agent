"""Rerank 重排序模块。

使用 Cross-Encoder 模型对向量召回的候选文档进行精排，
提升最终送入 LLM 的上下文质量。

流程：向量召回(Top-K*3) → Rerank精排 → 取Top-N → 返回带分数的结果
"""
import logging
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from config import RERANK_MODEL_NAME

logger = logging.getLogger(__name__)

# 模块级单例，避免重复加载模型
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder | None:
    """懒加载 Rerank 模型单例。"""
    global _reranker
    if _reranker is None:
        try:
            logger.info(f"[Reranker] 加载模型: {RERANK_MODEL_NAME}")
            _reranker = CrossEncoder(RERANK_MODEL_NAME)
        except Exception as e:
            logger.error(f"[Reranker] 模型加载失败: {e}")
            return None
    return _reranker


def rerank(query: str, docs: list[Document], top_n: int = 3) -> list[dict]:
    """对候选文档进行 Rerank 重排序。

    Args:
        query: 用户查询
        docs: 向量召回的候选文档列表
        top_n: 返回排序后的前 N 条

    Returns:
        list[dict]: 每项包含 {"doc": Document, "score": float}，按分数降序
    """
    if not docs:
        return []

    reranker = _get_reranker()
    if reranker is None:
        logger.warning("[Reranker] 模型不可用，跳过重排序")
        return [{"doc": doc, "score": 0.5} for doc in docs[:top_n]]
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    # 按分数降序排列，取 top_n
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_n]

    return [{"doc": doc, "score": float(score)} for doc, score in scored_docs]
