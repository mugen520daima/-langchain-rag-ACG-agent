"""RAG服务入口。

本模块是 RAG（Retrieval-Augmented Generation）检索服务的统一入口，
负责协调文档加载、切分、向量化和检索的完整流程。

核心职责：
- 初始化向量数据库（如果尚未创建）
- 提供基于语义相似度的文档检索接口
- 支持知识库热更新
"""



"""
这段 RAGService 作为一个基础的 RAG 原型或教学示例是合格的，但如果要将其部署到真实的生产环境中，它在架构设计、检索质量、工程性能和安全性等方面还存在明显的不足。

结合工业级 RAG 系统的最佳实践，以下是该设计的核心缺陷及相应的改进建议：

一、 架构与工程设计缺陷
同步阻塞式实现
   不足：当前的 retrieve() 方法是纯同步的。在高并发场景下，向量检索会阻塞主线程，严重限制系统的 QPS（每秒查询率）。
   改进：将核心方法封装为异步版本（如 async def retrieve），或者在服务层引入线程池/协程池来并发执行检索任务。
缺乏懒加载（Lazy Loading）机制
   不足：当前在 init 中直接调用 _init_vectorstore()。如果知识库很大，每次服务启动或实例化时都会消耗大量时间（可能长达数十秒），导致服务无法快速响应。
   改进：采用懒加载策略，仅在首次真正调用 retrieve() 时才触发知识库的初始化和构建，从而加快服务器启动速度并节省不必要的资源消耗。
缺少容错与降级（Fallback）机制
   不足：代码中没有异常捕获逻辑。一旦 LLM API 超时或向量库连接断开，整个 Pipeline 会直接崩溃。
   改进：增加 Fallback 机制。例如，当高级检索失败时，自动降级为基础的关键字搜索或返回默认提示，保证服务的可用性。

二、 检索质量与数据处理缺陷
“一刀切”的分块策略（Chunking）
   不足：虽然调用了 split_documents，但通常这种基础实现只是按固定字符数切分。这会破坏文档的语义结构，导致表格被截断、代码函数不完整等问题。
   改进：将分块视为检索架构的核心部分。应根据文档类型（Markdown 标题、代码函数边界、表格完整性）进行语义分块，并合理设置重叠区（Overlap）以保持上下文连贯性。
缺乏重排序（Rerank）环节
   不足：仅依赖向量相似度（Top-K）往往不够精确。向量检索只能找到“可能相关”的片段，次优片段可能会排在最优片段前面，误导大模型。
   改进：在检索和生成之间加入重排序层（Reranker）。流程应升级为：密集检索 -> 候选集 -> 重排序器 -> 最终上下文 -> 大模型，这是提升 RAG 准确率最具性价比的手段。
忽视元数据过滤（Metadata Filtering）
   不足：仅靠文本相似度匹配，极易出现语境错误。例如用户问欧盟合规要求，却检索到了美国的政策。
   改进：结合结构化数据进行混合检索。支持按地区、产品层级、时效性或用户权限等元数据进行前置过滤，约束检索范围。

三、 输出格式与安全缺陷
返回值缺乏结构化与溯源能力
   不足：retrieve() 仅仅返回了一个拼接好的纯字符串。在实际产品中，前端通常需要展示“参考文档卡片”以供用户点击跳转原文。
   改进：返回结构化对象（如包含 answer, documents, metadata 的字典），并在 Prompt 中强制要求模型提供引用标注，以便前端渲染来源信息。
缺乏数据清洗与安全防护
   不足：直接将目录下的文件灌入向量库，容易吸入重复页眉页脚、乱码等噪声；同时未对输入输出做安全校验，存在提示注入和数据泄露风险。
   改进：在入库前建立真正的数据清洗管道（去重、格式化）；在应用层增加输入净化和护栏机制，防止恶意攻击。

总结：当前的 RAGService 解决了“从无到有”的问题，但要走向生产级，还需要在异步性能、语义分块、重排序机制以及可观测性上进行全面的架构升级。
"""


import logging

from config import KNOWLEDGE_DIR, RAG_DEFAULT_K
from rag.document_loader import load_documents
from rag.splitter import split_documents
from rag.vector_store import get_vectorstore, create_vectorstore, get_retriever

# 模块级日志器，用于记录 RAG 服务的运行状态
logger = logging.getLogger(__name__)


class RAGService:
    """RAG检索服务。

    这是对外暴露的主要接口类，封装了从知识库检索相关上下文的能力。
    上层调用者（如 Agent）只需通过 retrieve() 方法获取检索结果，
    无需关心底层的文档处理和向量存储细节。
    """

    def __init__(self, knowledge_dir: str = KNOWLEDGE_DIR):
        """初始化 RAG 服务。

        参数：
        - knowledge_dir: 知识库文档所在目录，默认从 config.KNOWLEDGE_DIR 读取。
        """
        # 知识库文档目录路径，用于加载和刷新文档
        self.knowledge_dir = knowledge_dir
        logger.info(f"[RAGService] 初始化，知识库目录: {knowledge_dir}")
        self._init_vectorstore()

    def _init_vectorstore(self):
        """初始化向量数据库。

        检查向量库是否已存在：
        - 如果已存在，直接复用，避免重复构建。
        - 如果不存在，加载文档 -> 切分 -> 创建向量库。
        """
        if get_vectorstore() is None:
            logger.info("[RAGService] 向量库不存在，开始构建...")
            # 从知识库目录加载原始文档列表
            docs = load_documents(self.knowledge_dir)
            if docs:
                logger.debug(f"[RAGService] 加载文档数: {len(docs)}")
                # 将长文档切分为适合向量化的小块
                chunks = split_documents(docs)
                logger.debug(f"[RAGService] 切分后块数: {len(chunks)}")
                create_vectorstore(chunks)
                logger.info("[RAGService] 向量库构建完成")
            else:
                logger.warning("[RAGService] 未找到任何文档，向量库未创建")
        else:
            logger.debug("[RAGService] 向量库已存在，跳过初始化")

    def retrieve(self, query: str, k: int = RAG_DEFAULT_K) -> str:
        """检索与查询最相关的文档片段。

        参数：
        - query: 用户查询文本，将被转换为向量进行相似度匹配。
        - k: 返回的最相关文档数量，默认从 config.RAG_DEFAULT_K 读取。

        返回：
        - 拼接后的上下文文本；如果检索失败或无结果，返回空字符串。
        """
        logger.debug(f"[RAGService] retrieve 查询: {query[:50]}... k={k}")
        # 获取向量检索器实例，k 指定返回结果数量
        retriever = get_retriever(k=k)
        if retriever is None:
            logger.warning("[RAGService] retriever 为空，向量库可能未初始化")
            return ""
        # 执行向量相似度检索，返回最相关的文档列表
        docs = retriever.invoke(query)
        logger.info(f"[RAGService] 检索到 {len(docs)} 条结果")
        return "\n\n".join(doc.page_content for doc in docs)

    def refresh(self):
        """重新加载知识库。

        用于知识库文档更新后，手动触发重建向量库。
        会清除旧的向量数据，重新加载和索引所有文档。
        """
        logger.info("[RAGService] 开始刷新知识库...")
        # 重新加载知识库目录下的所有文档
        docs = load_documents(self.knowledge_dir)
        if docs:
            logger.debug(f"[RAGService] 重新加载文档数: {len(docs)}")
            # 重新切分文档为小块
            chunks = split_documents(docs)
            logger.debug(f"[RAGService] 切分后块数: {len(chunks)}")
            create_vectorstore(chunks)
            logger.info("[RAGService] 知识库刷新完成")
        else:
            logger.warning("[RAGService] 刷新时未找到文档")