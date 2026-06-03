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

# ==================== TiDB 数据库配置 ====================
TIDB_HOST = os.getenv("TIDB_HOST", "gateway01.ap-northeast-1.prod.aws.tidbcloud.com")
TIDB_PORT = int(os.getenv("TIDB_PORT", "4000"))
TIDB_USER = os.getenv("TIDB_USER", "383cB4AL4Dq7Cwj.root")
TIDB_PASSWORD = os.getenv("TIDB_PASSWORD", "95qsF8hHh1Ng5guK")
TIDB_DATABASE = os.getenv("TIDB_DATABASE", "test")