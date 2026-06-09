"""
Agent 主逻辑模块
=================

负责创建和管理 ACG（动画/漫画/游戏）助手 Agent，包括：
- LLM 初始化（使用阿里云 DashScope 的 Qwen 系列模型）
- 工具集成（番剧查询、周边搜索、正版渠道、网络搜索、时间查询）
- 记忆管理（会话状态 SessionState + 对话历史 ConversationMemory）
- RAG 检索增强（向量召回 + Cross-Encoder Rerank + 可信度控制 + 降级策略）
- 意图路由（LLM 判断 rag/tool/chat 三路分发）
- 超时机制与服务降级（路由/RAG/Agent 三层独立超时保护）
- 流式输出（astream_events 实现逐 token 流式返回）

架构流程：
    用户输入 → 意图路由 → [RAG检索 | 工具调用 | 直接对话] → Agent执行 → 流式/整体输出
"""
import asyncio
import logging
import queue
import threading
import concurrent.futures
from functools import wraps
from typing import Any, Generator

from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import DASHSCOPE_API_KEY, CHAT_MODEL_NAME, ROUTER_TIMEOUT, RAG_TIMEOUT, AGENT_TIMEOUT
from prompts import SYSTEM_PROMPT
from memory import SessionMemory
from workflow import create_router, route_query
from tools import get_all_tools
from rag.rag_service import RAGService

logger = logging.getLogger(__name__)

# 当 Agent 执行超时或异常时返回的兜底回复
FALLBACK_REPLY = "抱歉，我刚才遇到了一点问题，没能正常回答你的问题。你可以换个方式再问我一次，或者稍后再试哦~"


def with_timeout(timeout_seconds: int, fallback: Any = None):
    """超时装饰器：将函数放入线程池执行，超时后返回 fallback 值。

    原理：使用 ThreadPoolExecutor 提交任务，通过 future.result(timeout) 控制超时。
    适用于同步阻塞调用（如 LLM invoke、RAG 检索），不适用于流式场景。

    Args:
        timeout_seconds: 超时时间（秒）
        fallback: 超时或异常时的降级返回值
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"[超时] {func.__name__} 超过 {timeout_seconds}s，触发降级")
                    return fallback
                except Exception as e:
                    logger.error(f"[异常] {func.__name__}: {e}")
                    return fallback
        return wrapper
    return decorator


class AnimeAgent:
    """ACG 智能助手 Agent。

    核心职责：
    1. 接收用户输入，通过 LLM 意图路由判断处理路径（rag/tool/chat）
    2. 若为 rag 意图，执行两阶段检索（向量召回 + Rerank 精排）获取知识库上下文
    3. 结合对话历史和检索结果，调用 Agent（含工具）生成回复
    4. 支持流式输出（chat_stream）和整体输出（chat）两种模式
    """

    def __init__(self, session_id: str = "default"):
        """初始化 Agent 各组件。

        初始化顺序：LLM → 工具 → 记忆 → RAG服务 → 意图路由器 → Agent执行器

        Args:
            session_id: 会话标识，用于隔离不同用户/会话的记忆
        """
        logger.info(f"开始初始化 AnimeAgent, session_id={session_id}...")

        # 初始化 LLM：使用阿里云 DashScope 的 OpenAI 兼容接口
        self.llm = ChatOpenAI(
            model=CHAT_MODEL_NAME,
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=60,
        )

        # 加载工具集（番剧查询、周边搜索、正版渠道、网络搜索、时间查询）
        self.tools = get_all_tools()
        logger.info(f"已加载 {len(self.tools)} 个工具: {[t.name for t in self.tools]}")

        # 初始化会话记忆（内存滑动窗口 + 结构化状态）
        self.memory = SessionMemory(session_id=session_id)

        # 初始化 RAG 检索服务（向量库 + Rerank 模型）
        try:
            self.rag_service = RAGService()
        except Exception as e:
            logger.error(f"RAG 服务初始化失败，将禁用 RAG 功能: {e}")
            self.rag_service = None

        # 创建意图路由器（LCEL Chain: PromptTemplate → LLM → StrOutputParser）
        self.router = create_router(self.llm)

        # 创建 LangChain Agent 执行器（含工具调用能力）
        self.agent_executor = self._create_agent()
        logger.info("AnimeAgent 初始化完成")
    
    def _create_agent(self) -> AgentExecutor:
        """创建 LangChain Agent 执行器。

        使用 ChatPromptTemplate 构建 Prompt 结构：
        - system: 角色设定（傲娇猫娘人设 + 工具说明 + 回复规范）
        - chat_history: 历史对话消息（MessagesPlaceholder，滑动窗口）
        - human: 当前用户输入（可能包含 RAG 上下文增强）
        - agent_scratchpad: Agent 中间推理步骤（工具调用链路）

        Returns:
            AgentExecutor: 配置好的 Agent 执行器，支持 invoke/stream/astream_events
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def _route_with_timeout(self, user_input: str) -> str:
        """带超时保护的意图路由。

        调用 LLM 分析用户输入意图（rag/tool/chat），超时或异常时降级为 chat。
        超时时间由 config.ROUTER_TIMEOUT 控制（默认 10s）。

        Args:
            user_input: 用户原始输入文本

        Returns:
            意图类型字符串："rag" | "tool" | "chat"
        """
        @with_timeout(ROUTER_TIMEOUT, fallback="chat")
        def _route(query):
            return route_query(self.router, query)
        return _route(user_input)

    def _rag_with_timeout(self, user_input: str) -> dict:
        """带超时保护的 RAG 检索。

        执行向量召回 + Rerank 精排，超时或异常时返回空结果。
        超时时间由 config.RAG_TIMEOUT 控制（默认 120s，首次加载模型需更长时间）。

        Args:
            user_input: 用户查询文本

        Returns:
            dict: 检索结果，包含 context/sources/chunks/max_score/is_confident
        """
        @with_timeout(RAG_TIMEOUT, fallback=None)
        def _retrieve(query):
            return self.rag_service.retrieve(query)
        result = _retrieve(user_input)
        return result if result else {"context": "", "sources": [], "chunks": [], "max_score": 0.0, "is_confident": False}

    def _agent_invoke_with_timeout(self, agent_input: str) -> str | None:
        """带超时保护的 Agent 同步执行。

        将增强后的输入（可能含 RAG 上下文）和对话历史发送给 Agent 执行。
        超时时间由 config.AGENT_TIMEOUT 控制（默认 120s）。

        Args:
            agent_input: 发送给 Agent 的输入文本（原始查询或 RAG 增强后的查询）

        Returns:
            模型回复文本，超时时返回 None
        """
        @with_timeout(AGENT_TIMEOUT, fallback=None)
        def _invoke(inp, history):
            result = self.agent_executor.invoke({
                "input": inp,
                "chat_history": history
            })
            return result["output"]
        return _invoke(agent_input, self.memory.history.messages)

    def chat(self, user_input: str) -> dict:
        """同步处理用户输入并返回结构化回复（非流式模式）。

        这是非流式的完整处理入口，等待 Agent 生成完整回复后一次性返回。
        含超时机制与服务降级。

        Args:
            user_input: 用户输入文本

        Returns:
            dict: 包含以下字段：
                - output (str): 模型生成的回复文本
                - rag_chunks (list): 检索原文片段列表（供前端高亮对比溯源）
                - rag_confident (bool): RAG 检索是否达到可信阈值
        """
        try:
            return self._chat_internal(user_input)
        except Exception as e:
            logger.error(f"chat 方法异常: {e}", exc_info=True)
            return {"output": FALLBACK_REPLY, "rag_chunks": [], "rag_confident": False}

    def _chat_internal(self, user_input: str) -> dict:
        """非流式模式的内部处理逻辑。

        处理流程：
            1. 意图路由 → 判断 rag/tool/chat
            2. 开始会话轮次 → 更新 memory 状态
            3. RAG 检索增强 → 三步兜底策略（可信→引用回答 / 不可信→工具降级 / 无结果→搜索兜底）
            4. Agent 执行 → 带超时的 LLM 调用（含工具调用能力）
            5. 结束会话轮次 → 保存回复到 memory
        """
        logger.info(f"收到用户输入: {user_input[:50]}...")
        
        # Step 1: 意图路由（超时降级为 chat）
        intent = self._route_with_timeout(user_input)
        logger.info(f"意图识别结果: {intent}")
        
        # Step 2: 开始本轮会话
        self.memory.start_turn(user_input, intent)
        
        # 构建发送给 Agent 的输入
        agent_input = user_input
        rag_chunks = []  # 检索到的原文片段，供前端溯源展示
        
        # Step 3: RAG 检索增强 — 三步兜底策略
        if intent == "rag" and self.rag_service is not None:
            logger.info("触发 RAG 检索...")
            rag_result = self._rag_with_timeout(user_input)

            if rag_result["context"] and rag_result["is_confident"]:
                # ✓ 知识库命中且可信 → 带引用标注回答
                logger.info(f"RAG 可信, score={rag_result['max_score']:.3f}, sources={rag_result['sources']}")
                self.memory.record_rag_contexts([rag_result["context"]])
                rag_chunks = rag_result["chunks"]
                # 构建带编号的参考片段，供模型标注 [1][2]
                numbered_chunks = "\n\n".join(
                    f"[{i+1}] {c['content']}" for i, c in enumerate(rag_chunks)
                )
                sources_str = "\n".join(
                    f"[{i+1}] {c['source']}（相关度: {c['score']:.2f}）"
                    for i, c in enumerate(rag_chunks)
                )
                agent_input = (f"参考资料（请在回答中用[1][2]等标注引用来源）：\n{numbered_chunks}\n\n"
                               f"参考来源：\n{sources_str}\n\n用户问题：{user_input}")
            elif rag_result["max_score"] > 0:
                # 兜底第1步：阈值拦截 — 有结果但置信度不足
                logger.warning(f"RAG 置信度不足 score={rag_result['max_score']:.3f}，触发意图降级")
                self.rag_service.log_bad_case(user_input, rag_result["max_score"])
                # 兜底第2步：意图路由降级 → 让 Agent 用工具搜索
                agent_input = (f"（知识库中未找到足够可靠的信息，Rerank最高分={rag_result['max_score']:.2f}，"
                               f"低于阈值。请使用 web_search 工具搜索相关信息，"
                               f"并在回答中明确告知用户：'知识库暂无相关信息，以下内容来自通用搜索'）\n\n"
                               f"用户问题：{user_input}")
            else:
                # 检索完全无结果或超时
                logger.warning("RAG 检索无结果，降级为工具搜索")
                self.rag_service.log_bad_case(user_input, 0.0)
                agent_input = (f"（知识库检索无结果。请使用 web_search 工具搜索相关信息，"
                               f"并在回答中明确告知用户：'知识库暂无相关信息，以下内容来自通用搜索'）\n\n"
                               f"用户问题：{user_input}")
        
        # Step 4: 调用 Agent 执行（超时返回兜底回复）
        logger.info("调用 Agent 执行...")
        output = self._agent_invoke_with_timeout(agent_input)
        
        if output is None:
            logger.error("Agent 执行超时或异常，返回兜底回复")
            output = FALLBACK_REPLY
        
        # Step 5: 结束本轮会话
        self.memory.finish_turn(output)
        
        logger.info(f"Agent 回复生成完成，长度: {len(output)} 字符")
        return {
            "output": output,
            "rag_chunks": rag_chunks,
            "rag_confident": bool(rag_chunks),
        }
    
    def clear_history(self):
        """清空会话历史"""
        self.memory.clear()
        logger.info("会话历史已清空")

    def chat_stream(self, user_input: str) -> Generator[str | object, None, None]:
        """流式处理用户输入，逐 token 返回回复文本。

        这是流式输出的主入口，返回一个同步生成器。
        每次 yield 一个 token 片段（str），前端可逐步拼接显示。

        流式完成后，可通过 self._last_stream_meta 获取本次调用的元信息：
            - rag_chunks: 检索到的原文片段（供前端溯源展示）
            - rag_confident: RAG 是否可信

        Args:
            user_input: 用户输入文本

        Yields:
            str: 逐个 token 片段
        """
        try:
            yield from self._chat_stream_internal(user_input)
        except Exception as e:
            logger.error(f"chat_stream 异常: {e}", exc_info=True)
            yield FALLBACK_REPLY

    def _chat_stream_internal(self, user_input: str) -> Generator[str | object, None, None]:
        """流式输出的内部实现。

        处理流程与 _chat_internal 相同（路由→RAG→Agent），
        但 Agent 执行阶段使用 astream_events 实现 token 级流式输出。

        关键技术：
        - 使用 LangChain 的 astream_events(version="v2") 获取流式事件
        - 过滤 "on_chat_model_stream" 事件获取 LLM 输出的 token
        - 通过 threading + queue 将 async generator 桥接为 sync generator
          （因为 Streamlit 运行在同步环境中）
        """
        logger.info(f"[stream] 收到用户输入: {user_input[:50]}...")

        # Step 1: 意图路由（同步，带超时）
        intent = self._route_with_timeout(user_input)
        logger.info(f"[stream] 意图识别: {intent}")

        # Step 2: 开始本轮会话
        self.memory.start_turn(user_input, intent)

        agent_input = user_input
        rag_chunks = []

        # Step 3: RAG 检索增强
        if intent == "rag" and self.rag_service is not None:
            rag_result = self._rag_with_timeout(user_input)
            if rag_result["context"] and rag_result["is_confident"]:
                self.memory.record_rag_contexts([rag_result["context"]])
                rag_chunks = rag_result["chunks"]
                numbered_chunks = "\n\n".join(
                    f"[{i+1}] {c['content']}" for i, c in enumerate(rag_chunks)
                )
                sources_str = "\n".join(
                    f"[{i+1}] {c['source']}（相关度: {c['score']:.2f}）"
                    for i, c in enumerate(rag_chunks)
                )
                agent_input = (f"参考资料（请在回答中用[1][2]等标注引用来源）：\n{numbered_chunks}\n\n"
                               f"参考来源：\n{sources_str}\n\n用户问题：{user_input}")
            elif rag_result["max_score"] > 0:
                self.rag_service.log_bad_case(user_input, rag_result["max_score"])
                agent_input = (f"（知识库中未找到足够可靠的信息，Rerank最高分={rag_result['max_score']:.2f}，"
                               f"低于阈值。请使用 web_search 工具搜索相关信息，"
                               f"并在回答中明确告知用户：'知识库暂无相关信息，以下内容来自通用搜索'）\n\n"
                               f"用户问题：{user_input}")
            else:
                self.rag_service.log_bad_case(user_input, 0.0)
                agent_input = (f"（知识库检索无结果。请使用 web_search 工具搜索相关信息，"
                               f"并在回答中明确告知用户：'知识库暂无相关信息，以下内容来自通用搜索'）\n\n"
                               f"用户问题：{user_input}")

        # Step 4: 流式调用 Agent（核心流式逻辑）
        # 保存元信息供调用方在流式结束后获取
        self._last_stream_meta = {"rag_chunks": rag_chunks, "rag_confident": bool(rag_chunks)}
        full_output = ""

        async def _astream():
            """异步生成器：通过 astream_events 获取 LLM 逐 token 输出。

            astream_events(version="v2") 会产出多种事件：
            - on_chain_start/end: Chain 级别的开始/结束
            - on_tool_start/end: 工具调用的开始/结束
            - on_chat_model_stream: LLM 输出的每个 token（我们只关心这个）
            """
            nonlocal full_output
            async for event in self.agent_executor.astream_events(
                {"input": agent_input, "chat_history": self.memory.history.messages},
                version="v2"
            ):
                # 只提取 LLM 流式输出的 token 事件
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        full_output += chunk.content
                        yield chunk.content

        # ========== async generator → sync generator 桥接 ==========
        # Streamlit 运行在同步环境中，无法直接消费 async generator。
        # 解决方案：在子线程中运行 asyncio 事件循环消费异步生成器，
        # 通过 queue.Queue 将 token 传递回主线程的同步生成器。
        loop = asyncio.new_event_loop()
        token_queue: queue.Queue[str | object] = queue.Queue()
        sentinel = object()  # 哨兵对象，标记流式结束

        def _run_async_in_thread():
            """子线程入口：运行异步事件循环，消费 _astream() 并将 token 放入队列。"""
            async def _consume():
                try:
                    async for token in _astream():
                        token_queue.put(token)
                except Exception as e:
                    logger.error(f"[stream] astream 异常: {e}")
                finally:
                    # 无论成功还是异常，都放入哨兵通知主线程结束
                    token_queue.put(sentinel)
            loop.run_until_complete(_consume())

        # 启动子线程运行异步流式消费
        t = threading.Thread(target=_run_async_in_thread, daemon=True)
        t.start()

        # 主线程：从队列中逐个取出 token 并 yield（同步生成器）
        while True:
            item = token_queue.get()
            if item is sentinel:
                break
            yield item

        # 等待子线程结束并清理事件循环
        t.join(timeout=5)
        loop.close()

        # Step 5: 结束本轮会话，将完整回复写入 memory
        if not full_output:
            full_output = FALLBACK_REPLY
            yield full_output
        self.memory.finish_turn(full_output)
