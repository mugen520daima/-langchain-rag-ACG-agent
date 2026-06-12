import os
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

DASHSCOPE_API_KEY = SecretStr(os.getenv("DASHSCOPE_API_KEY", ""))
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "qwen3.7-plus")

KNOWLEDGE_DIR = "data/knowledge"
MEMORY_FILE = "data/memory/user_profile.json"

# ==================== RAG 配置 ====================
RAG_DEFAULT_K = 3  # 默认检索返回的文档数量
MIN_RAG_SCORE = 0.6  # RAG 可信度阈值（Rerank 分数低于此值视为不可信）

# ==================== Rerank 配置 ====================
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"  # Cross-Encoder 重排序模型

# ==================== Bad Case 记录 ====================
BAD_CASE_LOG_PATH = "data/bad_cases.jsonl"  # 检索失败的问题记录

# ==================== 向量数据库配置 ====================
VECTORSTORE_PERSIST_DIR = "data/vectorstore"  # 向量数据库持久化目录
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # Embedding 模型名称（中文优化）

# ==================== Memory 配置 ====================
MAX_CONVERSATION_MESSAGES = 20  # 会话历史最大保留消息数
DEFAULT_SESSION_ID = "default"  # 默认会话ID

# ==================== TiDB 数据库配置 ====================  写你自己的
TIDB_HOST = os.getenv("TIDB_HOST", "gateway01.ap-northeast-1.prod.aws.tidbcloud.com")
TIDB_PORT = int(os.getenv("TIDB_PORT", "4000"))
TIDB_USER = os.getenv("TIDB_USER", "383cB4AL4Dq7Cwj.root")
TIDB_PASSWORD = os.getenv("TIDB_PASSWORD", "kJPCLXvbCXz9NhfV")
TIDB_DATABASE = os.getenv("TIDB_DATABASE", "test")


# ==================== 超时与降级配置 ====================
ROUTER_TIMEOUT = int(os.getenv("ROUTER_TIMEOUT", "10"))      # 意图路由超时（秒）
RAG_TIMEOUT = int(os.getenv("RAG_TIMEOUT", "120"))           # RAG 检索超时（秒）- 首次加载模型需要下载，预留足够时间
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "120"))       # Agent 执行超时（秒）

# ==================== 登录态 JWT 配置 ====================
# 生产环境务必通过环境变量 JWT_SECRET 覆盖此默认值，不要使用代码里的默认密钥
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-prod-acg-agent-secret")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "168"))  # 登录态有效期（小时），默认 7 天
