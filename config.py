import os
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "qwen3.7-plus")

KNOWLEDGE_DIR = "data/knowledge"
MEMORY_FILE = "data/memory/user_profile.json"

# ==================== Memory 配置 ====================
MAX_CONVERSATION_MESSAGES = 20  # 会话历史最大保留消息数
DEFAULT_SESSION_ID = "default"  # 默认会话ID