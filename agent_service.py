"""Agent 主逻辑"""
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import DASHSCOPE_API_KEY, CHAT_MODEL_NAME
from prompts import SYSTEM_PROMPT
from memory import ConversationMemory, UserProfileMemory
from workflow import create_router, route_query
from tools import get_all_tools
from rag.rag_service import RAGService


class AnimeAgent:
    """ACG助手Agent"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=CHAT_MODEL_NAME,
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.tools = get_all_tools()
        self.conversation_memory = ConversationMemory()
        self.user_profile = UserProfileMemory()
        self.rag_service = RAGService()
        self.router = create_router(self.llm)
        self.agent_executor = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    def chat(self, user_input: str) -> str:
        """处理用户输入"""
        intent = route_query(self.router, user_input)
        
        if intent == "rag":
            context = self.rag_service.retrieve(user_input)
            if context:
                user_input = f"参考资料：{context}\n\n用户问题：{user_input}"
        
        user_context = self.user_profile.get_context()
        if user_context:
            user_input = f"{user_context}\n{user_input}"
        
        result = self.agent_executor.invoke({
            "input": user_input,
            "chat_history": self.conversation_memory.messages
        })
        
        return result["output"]
