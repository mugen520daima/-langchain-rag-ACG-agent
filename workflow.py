"""
任务规划与路由模块
==================

本模块实现了 Agent 的意图路由功能，负责将用户输入分类到不同的处理流程。

核心功能:
    - 意图识别: 使用 LLM 判断用户查询的意图类型
    - 流程路由: 根据意图将请求分发到对应的处理管道（RAG/Tool/Chat）

路由策略:
    - rag: 需要查询知识库的问题（如番剧介绍、ACG 术语解释）
    - tool: 需要调用外部工具的请求（如搜索周边、查询观看渠道）
    - chat: 普通对话，直接由 LLM 回复（如闲聊、角色互动）

架构位置:
    用户输入 -> [Router] -> RAG Pipeline / Tool Pipeline / Chat Pipeline
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompts import ROUTER_PROMPT


def create_router(llm):
    """
    创建意图路由器（LCEL Chain）
    
    使用 LangChain Expression Language (LCEL) 构建一个简单的路由链：
        PromptTemplate -> LLM -> StrOutputParser
    
    工作流程:
        1. PromptTemplate: 将用户输入填充到路由提示词模板
        2. LLM: 调用大模型分析意图，输出 rag/tool/chat
        3. StrOutputParser: 提取纯文本结果
    
    Args:
        llm: LangChain LLM 实例（如 ChatOpenAI、Qwen 等）
    
    Returns:
        Runnable: 可调用的 LCEL Chain，输入 {"input": query}，输出意图字符串
    
    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI()
        >>> router = create_router(llm)
        >>> router.invoke({"input": "鬼灭之刃讲的是什么"})
        'rag'
    """
    # 从 prompts.py 加载路由提示词模板
    prompt = PromptTemplate.from_template(ROUTER_PROMPT)
    # 使用 LCEL 的管道语法组装 Chain: prompt | llm | parser
    return prompt | llm | StrOutputParser()


def route_query(router, query: str) -> str:
    """
    路由用户查询到合适的处理流程
    
    调用路由器分析用户意图，并将 LLM 输出标准化为预定义的路由类型。
    
    Args:
        router: 由 create_router() 创建的路由器 Chain
        query: 用户的原始查询文本
    
    Returns:
        str: 路由类型，取值为 "rag" | "tool" | "chat"
            - "rag": 触发知识库检索流程
            - "tool": 触发工具调用流程
            - "chat": 直接对话，无需额外处理
    
    Note:
        - LLM 输出可能包含额外内容，通过关键词匹配提取意图
        - 未匹配到 rag/tool 时，默认返回 chat（兜底策略）
    
    Example:
        >>> intent = route_query(router, "帮我查一下火影忍者的周边")
        >>> print(intent)
        'tool'
    """
    # 调用路由器获取 LLM 判断结果，统一转小写便于匹配
    result = router.invoke({"input": query}).strip().lower()
    
    # 通过关键词匹配确定路由类型
    if "rag" in result:
        return "rag"      # 知识库检索
    elif "tool" in result:
        return "tool"     # 工具调用
    
    return "chat"         # 默认: 直接对话
