"""
Agent 主逻辑模块
负责创建和管理 ACG（动画/漫画/游戏）助手 Agent，包括：
- LLM 初始化（使用阿里云 DashScope）
- 工具集成
- 记忆管理（会话状态 + 对话历史）
- RAG 检索增强
- 意图路由
"""
import logging
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import DASHSCOPE_API_KEY, CHAT_MODEL_NAME
from prompts import SYSTEM_PROMPT
from memory import SessionMemory  # 使用组合式会话 memory
from workflow import create_router, route_query
from tools import get_all_tools
from rag.rag_service import RAGService

# 配置日志
logger = logging.getLogger(__name__)


class AnimeAgent:
    """
    ACG助手Agent
    
    核心职责：
    1. 接收用户输入，通过路由判断意图
    2. 根据意图决定是否使用 RAG 检索
    3. 结合对话历史，调用 LLM 生成回复
    """
    
    def __init__(self, session_id: str = "default"):
        """初始化 Agent 各组件
        
        Args:
            session_id: 会话ID，用于隔离不同对话
        """
        logger.info(f"开始初始化 AnimeAgent, session_id={session_id}...")
        
        # 初始化 LLM（使用阿里云 DashScope 兼容 OpenAI 接口）
        logger.info(f"初始化 LLM，模型: {CHAT_MODEL_NAME}")
        self.llm = ChatOpenAI(
            model=CHAT_MODEL_NAME,
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 加载工具集
        self.tools = get_all_tools()
        logger.info(f"已加载 {len(self.tools)} 个工具: {[t.name for t in self.tools]}")
        
        # 初始化会话记忆（包含状态 + 对话历史）
        self.memory = SessionMemory(session_id=session_id)
        logger.info("会话记忆初始化完成")
        
        # 初始化 RAG 服务
        self.rag_service = RAGService()
        logger.info("RAG 服务初始化完成")
        
        # 初始化意图路由器
        self.router = create_router(self.llm)
        logger.info("意图路由器初始化完成")
        
        # 创建 Agent 执行器
        self.agent_executor = self._create_agent()
        logger.info("AnimeAgent 初始化完成")
    
    def _create_agent(self) -> AgentExecutor:
        """
        创建 LangChain Agent 执行器
        
        Prompt 结构：
        - system: 系统提示词，定义 Agent 角色和行为
        - chat_history: 对话历史，保持上下文连贯
        - human: 用户当前输入
        - agent_scratchpad: Agent 思考过程（工具调用中间结果）
        """
        logger.debug("创建 Agent Prompt 模板...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 创建支持工具调用的 Agent
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        logger.debug("Agent 执行器创建完成")
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    def chat(self, user_input: str) -> str:
        """
        处理用户输入并返回回复
        
        处理流程：
        1. 意图路由 - 判断是否需要 RAG 检索
        2. 开始本轮会话 - 更新状态并记录用户消息
        3. RAG 增强 - 如需要，检索相关文档作为上下文
        4. Agent 执行 - 调用 LLM 生成最终回复
        5. 结束本轮会话 - 记录助手回复
        
        Args:
            user_input: 用户输入的文本
            
        Returns:
            Agent 生成的回复文本
        """
        logger.info(f"收到用户输入: {user_input[:50]}...")
        
        # Step 1: 意图路由
        intent = route_query(self.router, user_input)
        logger.info(f"意图识别结果: {intent}")
        
        # Step 2: 开始本轮会话（更新状态 + 记录用户消息）
        self.memory.start_turn(user_input, intent)
        
        # 构建发送给 Agent 的输入（可能会被 RAG 增强）
        agent_input = user_input
        
        # Step 3: RAG 检索增强（如果意图为 rag）
        if intent == "rag":
            logger.info("触发 RAG 检索...")
            context = self.rag_service.retrieve(user_input)
            if context:
                logger.info(f"RAG 检索到 {len(context)} 字符的上下文")
                # 记录检索结果到会话状态
                self.memory.record_rag_contexts([context])
                agent_input = f"参考资料：{context}\n\n用户问题：{user_input}"
            else:
                logger.warning("RAG 检索未返回结果")
        
        # Step 4: 调用 Agent 执行
        logger.info("调用 Agent 执行...")
        logger.debug(f"chat_history 消息数: {len(self.memory.history.messages)}")
        result = self.agent_executor.invoke({
            "input": agent_input,
            "chat_history": self.memory.history.messages
        })
        
        output = result["output"]
        
        # Step 5: 结束本轮会话（记录助手回复）
        self.memory.finish_turn(output)
        
        logger.info(f"Agent 回复生成完成，长度: {len(output)} 字符")
        return output
    
    def clear_history(self):
        """清空会话历史"""
        self.memory.clear()
        logger.info("会话历史已清空")
