"""
Agent 主逻辑模块
负责创建和管理 ACG（动画/漫画/游戏）助手 Agent，包括：
- LLM 初始化（使用阿里云 DashScope）
- 工具集成
- 记忆管理（会话状态 + 对话历史）
- RAG 检索增强（两阶段检索 + Rerank + 可信度控制）
- 意图路由
- 超时机制与服务降级
"""
import logging
import concurrent.futures
from functools import wraps
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import DASHSCOPE_API_KEY, CHAT_MODEL_NAME, ROUTER_TIMEOUT, RAG_TIMEOUT, AGENT_TIMEOUT
from prompts import SYSTEM_PROMPT
from memory import SessionMemory  # 使用组合式会话 memory
from workflow import create_router, route_query
from tools import get_all_tools
from rag.rag_service import RAGService

# 配置日志
logger = logging.getLogger(__name__)

# ==================== 降级回复 ====================
FALLBACK_REPLY = "抱歉，我刚才遇到了一点问题，没能正常回答你的问题。你可以换个方式再问我一次，或者稍后再试哦~"


def with_timeout(timeout_seconds: int, fallback=None):
    """超时装饰器，超时后返回 fallback 值"""
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
    """
    ACG助手Agent
    
    核心职责：
    1. 接收用户输入，通过路由判断意图
    2. 根据意图决定是否使用 RAG 检索
    3. 结合对话历史，调用 LLM 生成回复
    """
    
    def __init__(self, session_id: str = "default"):
        """初始化 Agent 各组件"""
        logger.info(f"开始初始化 AnimeAgent, session_id={session_id}...")
        
        self.llm = ChatOpenAI(
            model=CHAT_MODEL_NAME,
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=60,
        )
        
        self.tools = get_all_tools()
        logger.info(f"已加载 {len(self.tools)} 个工具: {[t.name for t in self.tools]}")
        
        self.memory = SessionMemory(session_id=session_id)
        try:
            self.rag_service = RAGService()
        except Exception as e:
            logger.error(f"RAG 服务初始化失败，将禁用 RAG 功能: {e}")
            self.rag_service = None
        self.router = create_router(self.llm)
        self.agent_executor = self._create_agent()
        logger.info("AnimeAgent 初始化完成")
    
    def _create_agent(self) -> AgentExecutor:
        """创建 LangChain Agent 执行器"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    def _route_with_timeout(self, user_input: str) -> str:
        """带超时的意图路由，超时或异常时降级为 chat"""
        @with_timeout(ROUTER_TIMEOUT, fallback="chat")
        def _route(query):
            return route_query(self.router, query)
        return _route(user_input)

    def _rag_with_timeout(self, user_input: str) -> dict:
        """带超时的 RAG 检索，超时或异常时返回空结果"""
        @with_timeout(RAG_TIMEOUT, fallback=None)
        def _retrieve(query):
            return self.rag_service.retrieve(query)
        result = _retrieve(user_input)
        return result if result else {"context": "", "sources": [], "chunks": [], "max_score": 0.0, "is_confident": False}

    def _agent_invoke_with_timeout(self, agent_input: str) -> str:
        """带超时的 Agent 执行，超时时返回兜底回复"""
        @with_timeout(AGENT_TIMEOUT, fallback=None)
        def _invoke(inp, history):
            result = self.agent_executor.invoke({
                "input": inp,
                "chat_history": history
            })
            return result["output"]
        return _invoke(agent_input, self.memory.history.messages)

    def chat(self, user_input: str) -> dict:
        """
        处理用户输入并返回结构化回复（含超时机制与服务降级）
        
        返回 dict:
            - output: 模型生成的回复文本
            - rag_chunks: 检索原文片段列表（供前端高亮对比溯源）
            - rag_confident: RAG是否可信
        """
        try:
            return self._chat_internal(user_input)
        except Exception as e:
            logger.error(f"chat 方法异常: {e}", exc_info=True)
            return {"output": FALLBACK_REPLY, "rag_chunks": [], "rag_confident": False}

    def _chat_internal(self, user_input: str) -> dict:
        """内部实际处理逻辑"""
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
