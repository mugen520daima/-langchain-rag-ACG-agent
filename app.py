"""Streamlit 页面入口 — 含登录鉴权 + 历史会话管理 + 流式对话"""
# 必须在任何 transformers/sentence-transformers 相关导入之前设置
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TQDM_DISABLE"] = "1"

# 运行：source venv/bin/activate && streamlit run app.py
# 使用 HuggingFace 镜像站加速国内下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 用安全的 stderr 包装器替换 sys.stderr，避免 BrokenPipeError
import sys
class _SafeStderr:
    def __init__(self, original):
        self._original = original
    def write(self, s):
        try:
            return self._original.write(s)
        except (BrokenPipeError, OSError, ValueError):
            return len(s) if s else 0
    def flush(self):
        try:
            return self._original.flush()
        except (BrokenPipeError, OSError, ValueError):
            return None
    def __getattr__(self, name):
        return getattr(self._original, name)

if sys.stderr is not None:
    sys.stderr = _SafeStderr(sys.stderr)

import uuid
import streamlit as st
from pathlib import Path
from agent_service import AnimeAgent
from auth import authenticate, register, create_users_table, check_db_connection
from memory import chat_store

# 启动命令：source venv/bin/activate && streamlit run app.py
# http://localhost:8557/
st.set_page_config(page_title="Weller的ACG助手", page_icon="🎌", layout="centered")

# 确保 users 表存在
create_users_table()

# 检查数据库连接
db_ok, db_msg = check_db_connection()
if not db_ok:
    st.error(f"⚠️ {db_msg}")
    st.info("请在 config.py 中配置正确的 TIDB_PASSWORD 后重启应用")
    st.stop()

# 头像路径
AI_AVATAR = "img/巧克力.jpg"
USER_AVATAR = "img/香草.jpg"
BACKGROUND = "static/background.jpg"

# 自定义样式
st.markdown("""
<style>
/* 背景图片 + 顶部渐变遮罩 */
.stApp {
    background-image: url("app/static/background.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 180px;
    background: linear-gradient(to bottom, rgba(0,0,0,0.25) 0%, rgba(0,0,0,0.1) 50%, transparent 100%);
    pointer-events: none;
    z-index: 1;
}

/* 移除默认容器背景 */
.stMain, .block-container, [data-testid="stAppViewContainer"] {
    background: transparent !important;
}

/* 主内容区域 */
.block-container {
    max-width: 100% !important;
    padding-left: 20px !important;
    padding-right: 20px !important;
}

/* 右上角功能按钮 - 白色背景 */
[data-testid="stToolbar"] {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 8px !important;
    padding: 4px 8px !important;
    position: fixed !important;
    right: 12px !important;
    top: 12px !important;
    width: auto !important;
    height: auto !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
}

[data-testid="stToolbar"] button {
    color: #555 !important;
    padding: 4px !important;
}

/* 顶部默认 header 透明 */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: auto !important;
}

header[data-testid="stHeader"] > div {
    background: transparent !important;
}

/* 标题栏 - 毛玻璃悬浮效果 */
.fixed-header {
    position: fixed;
    top: 25px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 999;
    background: rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 18px 40px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.3);
}

.main-title {
    font-size: 2em;
    font-weight: bold;
    color: #fff;
    text-shadow: 0 2px 8px rgba(0,0,0,0.4), 0 0 20px rgba(255,182,193,0.5);
    margin: 0;
    letter-spacing: 2px;
}

.subtitle {
    color: rgba(255, 255, 255, 0.85);
    font-size: 0.85em;
    margin-top: 10px;
    text-shadow: 0 1px 4px rgba(0,0,0,0.3);
    letter-spacing: 1px;
}

.decorative-line {
    width: 120px;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.8), transparent);
    margin: 10px auto;
    border-radius: 2px;
}

/* 为固定标题留出空间 */
.block-container {
    padding-top: 140px !important;
}

/* 聊天容器 - 控制聊天区域宽度 */
[data-testid="stChatMessageContainer"] {
    background: transparent !important;
    box-shadow: none !important;
    padding: 15px 0 !important;
    max-width: 66vw !important;
    margin: 0 auto !important;
}

/* 消息气泡 - 用户消息 */
.stChatMessage {
    background: rgba(255, 255, 255, 0.88) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-radius: 16px !important;
    padding: 14px 18px !important;
    margin: 10px 0 !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.5);
    color: #333 !important;
}

.stChatMessage p, .stChatMessage span, .stChatMessage div,
.stChatMessage .stMarkdown, .stChatMessage .stMarkdown p,
.stChatMessage [data-testid="stMarkdownContainer"],
.stChatMessage [data-testid="stMarkdownContainer"] p {
    color: #333 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* 消息气泡 - AI回复（粉色底） */
.stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
.stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
    background: rgba(255, 228, 235, 0.9) !important;
    border: 1px solid rgba(255, 182, 193, 0.4);
}

/* 输入框区域 */
[data-testid="stBottom"] {
    background: transparent !important;
    padding: 0 20px !important;
}

[data-testid="stBottom"] > div {
    max-width: 700px !important;
    margin: 0 auto !important;
    background: transparent !important;
}

/* 输入框 - 精致现代风格 */
[data-testid="stChatInput"] {
    background: rgba(255, 255, 255, 0.85) !important;
    border-radius: 24px !important;
    padding: 6px 12px !important;
    border: 1px solid rgba(255, 182, 193, 0.5) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    width: 100% !important;
    max-width: 100% !important;
}

[data-testid="stChatInput"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] input {
    background: transparent !important;
    color: #333 !important;
}

[data-testid="stChatInput"] input::placeholder {
    color: #888 !important;
    opacity: 1 !important;
}

/* 发送按钮 */
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #ff9a9e, #ff6b9d) !important;
    border: none !important;
    border-radius: 50% !important;
    box-shadow: 0 2px 8px rgba(255, 105, 180, 0.3) !important;
}

[data-testid="stChatInput"] button:hover {
    background: linear-gradient(135deg, #ff6b9d, #c44569) !important;
    transform: scale(1.05);
}

[data-testid="stChatInput"] button svg {
    fill: white !important;
}

/* 作者信息 */
.author-info {
    position: fixed;
    bottom: 80px;
    right: 20px;
    color: rgba(255, 255, 255, 0.7);
    font-size: 0.75em;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3);
    z-index: 100;
}

/* 隐藏默认header背景 */
header[data-testid="stHeader"] {
    background: transparent !important;
    pointer-events: none;
}

header[data-testid="stHeader"] [data-testid="stToolbar"] {
    pointer-events: auto;
}

/* 侧边栏样式 */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.95) !important;
}

[data-testid="stSidebar"] .stButton button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


# ==================== 登录/注册页面 ====================
if "username" not in st.session_state:
    st.session_state["username"] = None

if not st.session_state["username"]:
    st.markdown("""
    <div class="fixed-header">
        <div class="main-title">🎀 巧克力の小窝 🎀</div>
        <div class="decorative-line"></div>
        <div class="subtitle">✨ 请登录后与巧克力聊天喵~ ✨</div>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["🔑 登录", "📝 注册"])

    with tab_login:
        with st.form("login_form"):
            login_user = st.text_input("用户名")
            login_pass = st.text_input("密码", type="password")
            submitted = st.form_submit_button("登录", use_container_width=True)
            if submitted:
                ok, msg = authenticate(login_user, login_pass)
                if ok:
                    st.session_state["username"] = login_user
                    st.rerun()
                else:
                    st.error(msg)

    with tab_register:
        with st.form("register_form"):
            reg_user = st.text_input("用户名")
            reg_nick = st.text_input("昵称（可选）")
            reg_pass = st.text_input("密码", type="password")
            reg_pass2 = st.text_input("确认密码", type="password")
            submitted = st.form_submit_button("注册", use_container_width=True)
            if submitted:
                if reg_pass != reg_pass2:
                    st.error("两次密码不一致")
                else:
                    ok, msg = register(reg_user, reg_pass, reg_nick or None)
                    if ok:
                        st.success("注册成功！请切换到登录标签页登录")
                    else:
                        st.error(msg)

    st.stop()


# ==================== 已登录：主界面 ====================
username = st.session_state["username"]

# 初始化会话相关 state
if "current_session_id" not in st.session_state:
    st.session_state["current_session_id"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []


def _new_session():
    """创建新会话"""
    session_id = str(uuid.uuid4())[:16]
    chat_store.create_session(session_id, user_id=username, title="新对话")
    st.session_state["current_session_id"] = session_id
    st.session_state["messages"] = []
    # 重新初始化 Agent 使用新 session_id
    st.session_state["agent"] = AnimeAgent(session_id=session_id)


def _switch_session(session_id: str):
    """切换到指定会话，从数据库加载历史消息"""
    st.session_state["current_session_id"] = session_id
    # 从 DB 加载历史消息
    db_messages = chat_store.get_messages(session_id, limit=100)
    st.session_state["messages"] = [
        {"role": msg["role"].replace("human", "user"), "content": msg["content"]}
        for msg in db_messages
    ]
    # 重新初始化 Agent 使用对应 session_id
    st.session_state["agent"] = AnimeAgent(session_id=session_id)


# ==================== 侧边栏：会话管理 ====================
with st.sidebar:
    st.markdown(f"### 👤 {username}")
    if st.button("🚪 登出", use_container_width=True):
        st.session_state["username"] = None
        st.session_state["current_session_id"] = None
        st.session_state["messages"] = []
        if "agent" in st.session_state:
            del st.session_state["agent"]
        st.rerun()

    st.divider()

    if st.button("➕ 新建对话", use_container_width=True):
        _new_session()
        st.rerun()

    st.divider()
    st.markdown("#### 💬 历史会话")

    sessions = chat_store.get_session_list(user_id=username, limit=20)
    for s in sessions:
        col1, col2 = st.columns([4, 1])
        title = s.get("title") or "新对话"
        session_id = s["session_id"]
        is_current = session_id == st.session_state.get("current_session_id")

        with col1:
            label = f"**▶ {title}**" if is_current else title
            if st.button(label, key=f"switch_{session_id}", use_container_width=True):
                _switch_session(session_id)
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{session_id}"):
                chat_store.delete_session(session_id)
                if is_current:
                    st.session_state["current_session_id"] = None
                    st.session_state["messages"] = []
                st.rerun()

# 如果没有当前会话，自动创建一个
if not st.session_state["current_session_id"]:
    # 尝试加载最近一个会话
    sessions = chat_store.get_session_list(user_id=username, limit=1)
    if sessions:
        _switch_session(sessions[0]["session_id"])
    else:
        _new_session()


# ==================== Agent 初始化 ====================
if "agent" not in st.session_state or not hasattr(st.session_state.agent, "chat_stream"):
    with st.spinner("正在初始化巧克力...首次加载模型可能需要一点时间喵~"):
        st.session_state.agent = AnimeAgent(session_id=st.session_state["current_session_id"])
        if st.session_state.agent.rag_service:
            st.session_state.agent.rag_service.retrieve("测试查询")


# ==================== 固定标题 ====================
st.markdown("""
<div class="fixed-header">
    <div class="main-title">🎀 巧克力の小窝 🎀</div>
    <div class="decorative-line"></div>
    <div class="subtitle">✨ Weller的专属ACG助手 ~ 查番剧 · 找周边 · 正版渠道 ✨</div>
</div>
<div class="author-info">作者：Weller</div>
""", unsafe_allow_html=True)


# ==================== 聊天区域 ====================
for msg in st.session_state.messages:
    avatar = AI_AVATAR if msg["role"] == "assistant" else USER_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("和巧克力聊聊吧~ 喵"):
    # 用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    # 持久化用户消息到 DB
    chat_store.save_message(st.session_state["current_session_id"], "human", prompt)

    # AI 流式回复
    with st.chat_message("assistant", avatar=AI_AVATAR):
        placeholder = st.empty()
        full_response = ""
        for chunk in st.session_state.agent.chat_stream(prompt):
            full_response += chunk
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    # 持久化 AI 回复到 DB
    chat_store.save_message(st.session_state["current_session_id"], "ai", full_response)

    # 用第一条消息作为会话标题
    if len(st.session_state.messages) == 2:
        title = prompt[:20] + ("..." if len(prompt) > 20 else "")
        conn = chat_store._get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE chat_sessions SET title = %s WHERE session_id = %s",
                    (title, st.session_state["current_session_id"]),
                )
                conn.commit()
        finally:
            conn.close()
