"""会话内 memory 设计。

设计顺序：
1. 先设计 State：明确当前这轮 Agent 运行到底需要保存什么状态。
2. 再处理会话内上下文：把消息历史、工具调用结果、当前意图等组织好。
3. 暂不考虑长期记忆：不落盘、不维护用户长期画像，避免过早复杂化。

结论：这个文件当前只负责“单次会话 / 单个 session 内”的状态与上下文管理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from typing_extensions import override

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config import DEFAULT_SESSION_ID, MAX_CONVERSATION_MESSAGES


# ==================== State 设计 ====================
# 这里先定义“当前会话真正需要的状态”，而不是一上来就设计长期记忆。
#
# 当前建议保留的 state：
# - session_id: 当前会话标识，方便多会话隔离
# - latest_intent: 记录最近一次路由决策的意图标签（如 chat / rag / tool）
# - latest_query: 最近一次用户输入
# - latest_response: 最近一次模型回复
# - retrieved_contexts: 最近一次 RAG 检索到的上下文
# - tool_traces: 当前会话里工具调用的简要记录
# - slots: 会话内临时槽位，用于存放“当前正在聊的番剧/角色/需求”等
#
# 这些都属于“会话内状态”，生命周期跟 session 绑定。
# 它们不是长期记忆，不需要落盘。


@dataclass
class SessionState:
    """单个会话的运行状态。

    这个类负责承载当前 session 在运行时需要保存的“结构化状态”。
    它不是对话文本本身，而是对当前会话进度、上下文结果和临时槽位的抽象。
    """

    session_id: str = DEFAULT_SESSION_ID
    latest_intent: str = "chat"
    latest_query: str = ""
    latest_response: str = ""
    retrieved_contexts: list[str] = field(default_factory=list)
    tool_traces: list[dict[str, Any]] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)

    def update_turn(self, query: str, intent: str) -> None:
        """更新当前轮次的核心状态。

        作用：
        - 在一轮新对话开始时，记录本轮的用户输入和当前意图。
        - 这是会话状态推进的起点，后续的 RAG 检索、工具调用、模型回复都围绕这轮输入展开。

        参数：
        - query: 用户本轮输入的原始文本。
        - intent: 路由阶段判断出的意图标签，例如 `chat`、`rag`、`tool`。
        """
        self.latest_query = query
        self.latest_intent = intent

    def set_response(self, response: str) -> None:
        """记录当前轮次的最终回复。

        作用：
        - 在模型完成回复后，把最终输出写入状态。
        - 方便调试、日志记录、前端展示或后续多轮逻辑复用。

        参数：
        - response: 模型或 Agent 产出的最终文本回复。
        """
        self.latest_response = response

    def set_retrieved_contexts(self, contexts: list[str]) -> None:
        """保存当前轮次的 RAG 检索结果。

        作用：
        - 在执行知识库检索后，把命中的上下文片段暂存在 state 中。
        - 这些内容通常只对当前轮次有价值，因此属于运行态而不是长期记忆。

        参数：
        - contexts: 检索得到的文本片段列表。
        """
        self.retrieved_contexts = contexts

    def add_tool_trace(self, tool_name: str, tool_input: Any, tool_output: Any) -> None:
        """追加一条工具调用轨迹。

        作用：
        - 记录某次工具调用的名称、输入和输出。
        - 便于调试 Agent 行为、排查工具调用链路，以及后续做简单可观测性。

        参数：
        - tool_name: 工具名称。
        - tool_input: 传给工具的输入参数，可以是字符串、字典或其他结构。
        - tool_output: 工具返回结果。
        """
        self.tool_traces.append(
            {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_output,
            }
        )

    def set_slot(self, key: str, value: Any) -> None:
        """写入一个会话级槽位。

        作用：对话浓缩
        - 将当前会话中的临时语义信息结构化存储下来。
        - 例如：当前讨论的番剧名、角色名、用户这一轮的需求类型等。
        - 这种设计比直接把所有东西都塞进聊天记录更容易维护和检索。

        参数：
        - key: 槽位名称，例如 `current_topic`。
        - value: 槽位对应的值。
        """
        self.slots[key] = value

    def get_slot(self, key: str, default: Any = None) -> Any:
        """读取一个会话级槽位。

        作用：
        - 从当前 session 的临时槽位中获取结构化上下文。
        - 当槽位不存在时，返回调用方提供的默认值，避免直接抛错。

        参数：
        - key: 槽位名称。
        - default: 槽位不存在时返回的默认值。

        返回：
        - 对应槽位的值；若不存在则返回 default。
        """
        return self.slots.get(key, default)

    def clear_runtime_data(self) -> None:
        """清理当前轮的易膨胀运行态数据，但保留必要槽位。

        作用：
        - 在新一轮开始前，清掉上一轮留下的临时大对象或调用轨迹。
        - 避免 `retrieved_contexts` 和 `tool_traces` 随对话轮次不断累积。
        - `slots` 不清空，是因为某些主题信息可能需要跨多轮沿用。
        """
        self.retrieved_contexts = []
        self.tool_traces = []


class ConversationMemory(BaseChatMessageHistory):
    """LangChain 兼容的会话内消息历史。

    这个类负责管理“消息序列”，也就是用户说了什么、助手回复了什么。
    它与 `SessionState` 的区别在于：
    - `SessionState` 关注结构化状态
    - `ConversationMemory` 关注原始对话历史
    """

    def __init__(self, max_messages: int = MAX_CONVERSATION_MESSAGES):
        """初始化消息历史容器。

        参数：
        - max_messages: 最多保留多少条消息。
          当消息数超过这个值时，会自动裁剪最旧的消息，防止上下文无限增长。
        """
        self.messages: list[BaseMessage] = []
        self.max_messages = max_messages

    def add_message(self, message: BaseMessage) -> None:
        """添加一条原始消息，并执行长度裁剪。

        作用：
        - 接收任意 LangChain 消息对象并加入历史。
        - 如果消息数超过上限，只保留最近的 `max_messages` 条。

        参数：
        - message: 一条 LangChain 消息对象，例如 `HumanMessage` 或 `AIMessage`。
        """
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    @override
    def add_user_message(self, message: HumanMessage | str) -> None:
        """添加一条用户消息。

        作用：
        - 将普通字符串包装成 `HumanMessage` 后写入历史。
        - 让上层调用不需要关心 LangChain 消息对象的构造细节。

        参数：
        - message: 用户输入文本或 HumanMessage 对象。
        """
        if isinstance(message, HumanMessage):
            self.add_message(message)
        else:
            self.add_message(HumanMessage(content=message))

    @override
    def add_ai_message(self, message: AIMessage | str) -> None:
        """添加一条助手消息。

        作用：
        - 将普通字符串包装成 `AIMessage` 后写入历史。
        - 用于在一轮处理结束后，把模型输出纳入后续上下文。

        参数：
        - message: 助手回复文本或 AIMessage 对象。
        """
        if isinstance(message, AIMessage):
            self.add_message(message)
        else:
            self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        """清空全部消息历史。

        作用：
        - 重置当前 session 的对话记录。
        - 适用于“新建会话”或“手动清空上下文”等场景。
        """
        self.messages = []

    def format_recent_context(self, limit: int = 6) -> str:
        """把最近几条消息整理为可读文本。

        作用：
        - 将最近的对话历史格式化成简单字符串，便于日志查看、调试或拼接附加上下文。
        - 这不是 LangChain 必需接口，而是一个辅助方法。

        参数：
        - limit: 最多取最近多少条消息。

        返回：
        - 形如“用户: ...\n助手: ...”的文本块。
        """
        recent_messages = self.messages[-limit:]
        lines: list[str] = []
        for msg in recent_messages:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)


class SessionMemory:
    """组合式会话 memory：状态 + 消息历史。

    这是对外使用的主入口，负责把：
    - `SessionState`（结构化状态）
    - `ConversationMemory`（消息历史）
    组合在一起，形成一个完整的会话内 memory 抽象。
    """

    def __init__(self, session_id: str = DEFAULT_SESSION_ID, max_messages: int = MAX_CONVERSATION_MESSAGES):
        """初始化一个会话 memory。

        参数：
        - session_id: 当前会话 ID，用于区分不同 session。
        - max_messages: 历史消息最大保留数量。
        """
        self.state = SessionState(session_id=session_id)
        self.history = ConversationMemory(max_messages=max_messages)

    def start_turn(self, user_input: str, intent: str) -> None:
        """开始一轮新的对话处理。

        作用：
        - 更新当前轮次的 query 和 intent。
        - 清掉上一轮累积的临时运行态数据。
        - 把当前用户输入写入消息历史。

        参数：
        - user_input: 用户本轮输入。
        - intent: 本轮路由得到的意图标签。
        """
        self.state.update_turn(query=user_input, intent=intent)
        self.state.clear_runtime_data()
        self.history.add_user_message(user_input)

    def finish_turn(self, assistant_output: str) -> None:
        """结束当前轮对话处理。

        作用：
        - 保存当前轮的最终回复。
        - 将助手回复写入消息历史，供下一轮继续使用。

        参数：
        - assistant_output: 助手最终输出文本。
        """
        self.state.set_response(assistant_output)
        self.history.add_ai_message(assistant_output)

    def record_rag_contexts(self, contexts: list[str]) -> None:
        """记录当前轮次的 RAG 检索结果。

        作用：
        - 把知识库检索出的上下文同步保存到 session state 中。
        - 方便后续调试、观测或做额外控制。

        参数：
        - contexts: 检索得到的上下文列表。
        """
        self.state.set_retrieved_contexts(contexts)

    def record_tool_call(self, tool_name: str, tool_input: Any, tool_output: Any) -> None:
        """记录一次工具调用。

        作用：
        - 将工具调用轨迹写入当前 session state。
        - 保持调用链路可追踪，而不必把这些信息直接混在用户聊天文本中。

        参数：
        - tool_name: 工具名称。
        - tool_input: 工具输入。
        - tool_output: 工具输出。
        """
        self.state.add_tool_trace(tool_name=tool_name, tool_input=tool_input, tool_output=tool_output)

    def set_current_topic(self, topic: str) -> None:
        """设置当前会话的主题槽位。

        作用：
        - 将当前讨论主题（例如某部番剧名）写入统一槽位 `current_topic`。
        - 这是 `set_slot` 的一个语义化封装，方便业务层直接使用。

        参数：
        - topic: 当前讨论主题。
        """
        self.state.set_slot("current_topic", topic)

    def get_current_topic(self) -> str | None:
        """获取当前会话的主题槽位。

        作用：
        - 读取 `current_topic`，用于多轮对话中延续讨论对象。

        返回：
        - 当前主题字符串；如果未设置则返回 `None`。
        """
        return self.state.get_slot("current_topic")

    def clear(self) -> None:
        """重置整个会话 memory。

        作用：
        - 清空消息历史。
        - 重置结构化状态。
        - 保留原 session_id，避免会话标识丢失。
        """
        self.history.clear()
        self.state = SessionState(session_id=self.state.session_id)


# 兼容当前工程里可能还存在的旧引用。
# 这里先保留一个空实现，明确表示：当前阶段不处理长期记忆。
class UserProfileMemory:
    """占位类：当前阶段不启用长期记忆。

    保留这个类的目的只有两个：
    1. 兼容旧代码中的 import，避免立即改动所有调用点。
    2. 明确告诉后续实现：长期记忆现在不是重点，先把 state 和会话上下文设计稳定。
    """

    def get_context(self) -> str:
        """返回长期记忆上下文。

        当前阶段不启用长期记忆，因此这里固定返回空字符串。
        这样上层代码仍可安全调用，而不会引入额外行为。
        """
        return ""
