"""Streamlit 页面入口"""
# 必须在任何 transformers/sentence-transformers 相关导入之前设置
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TQDM_DISABLE"] = "1"

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

import streamlit as st
from pathlib import Path
from agent_service import AnimeAgent

#启动命令：source venv/bin/activate && streamlit run app.py
# http://localhost:8557/
st.set_page_config(page_title="Weller的ACG助手", page_icon="🎌", layout="centered")

# 头像路径（相对路径）
AI_AVATAR = "img/巧克力.jpg"
USER_AVATAR = "img/香草.jpg"
BACKGROUND = "static/background.jpg"

# 检查图片文件是否存在
for img_path in [AI_AVATAR, USER_AVATAR, BACKGROUND]:
    if not Path(img_path).exists():
        st.warning(f"图片文件不存在: {img_path}")

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
/* 调整方法: 修改 max-width 值，66vw = 屏幕宽度的2/3 */
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
</style>
""", unsafe_allow_html=True)

# 固定标题
st.markdown("""
<div class="fixed-header">
    <div class="main-title">🎀 巧克力の小窝 🎀</div>
    <div class="decorative-line"></div>
    <div class="subtitle">✨ Weller的专属ACG助手 ~ 查番剧 · 找周边 · 正版渠道 ✨</div>
</div>
<div class="author-info">作者：Weller</div>
""", unsafe_allow_html=True)

if "agent" not in st.session_state:
    with st.spinner("正在初始化巧克力...首次加载模型可能需要一点时间喵~"):
        st.session_state.agent = AnimeAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = AI_AVATAR if msg["role"] == "assistant" else USER_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

if prompt := st.chat_input("和巧克力聊聊吧~ 喵"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(prompt)
    
    with st.chat_message("assistant", avatar=AI_AVATAR):
        with st.spinner("巧克力正在思考中...喵~"):
            response = st.session_state.agent.chat(prompt)
        st.write(response["output"])
    st.session_state.messages.append({"role": "assistant", "content": response["output"]})
