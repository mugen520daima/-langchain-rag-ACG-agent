# 🐱 weller的猫娘 ACG 助手

基于 LangChain 构建的 ACG（动画/漫画/游戏）领域智能助手，采用 RAG + Rerank + Agent 架构，前端使用 Streamlit。

## ✨ 特性

- **角色扮演**：傲娇猫娘"巧克力"人设，拒绝冷冰冰的 AI 语气
- **两阶段 RAG 检索**：向量召回 + Cross-Encoder Rerank 精排，提升知识检索准确度
- **意图路由**：LLM 自动判断走 RAG / 工具调用 / 闲聊三条路径
- **多工具调用**：番剧查询、周边搜索、正版渠道推荐、网络搜索、时间查询
- **会话记忆**：内存滑动窗口 + TiDB Cloud 持久化双层存储
- **超时降级**：分级超时保护，服务不可用时自动降级

## 🎭 角色设定

```
你是巧克力 (Chocolat)，一只傲娇猫娘，是用户的专属 ACG 助手。

【性格】嘴硬心软，高兴时咕噜噜，惊吓时炸毛
【语气】句尾带"喵/喵呜"，称呼用户为"主人"或"笨蛋主人"
【动作】在 * * 中描写猫耳/猫尾状态和心理活动
【原则】推荐正版渠道，不推荐盗版资源
```

## 🛠️ 工具能力

| 工具 | 功能 |
|------|------|
| `anime_info` | 查询番剧基本信息 |
| `merch_search` | 搜索周边商品 |
| `legal_site` | 推荐正版观看/购买渠道 |
| `web_search` | 网络搜索获取最新信息 |
| `time_tool` | 获取当前日期/时间 |

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key

# 运行
streamlit run app.py
```

## 📁 项目结构

```
├── app.py             # Streamlit 前端入口
├── agent_service.py   # Agent 主控逻辑（意图路由 + RAG + 工具调用）
├── memory.py          # 会话记忆（SessionState + ConversationMemory + TiDB 持久化）
├── prompts.py         # Prompt 模板（角色/RAG/路由）
├── workflow.py        # 意图路由（rag / tool / chat）
├── config.py          # 全局配置
├── rag/
│   ├── rag_service.py     # RAG 检索流程协调
│   ├── reranker.py        # Cross-Encoder Rerank 重排序
│   ├── vector_store.py    # Chroma 向量库管理
│   ├── document_loader.py # 知识库文档加载
│   ├── splitter.py        # 文本分块
│   └── retriever.py       # 向量检索器
├── tools/             # 工具集（5 个 LangChain Tool）
└── data/knowledge/    # 知识库 Markdown 文件
```

## 🔧 技术栈

- **框架**：LangChain + Streamlit
- **LLM**：阿里云 DashScope（Qwen 系列）
- **向量模型**：`BAAI/bge-small-zh-v1.5`（Embedding）
- **重排序模型**：`BAAI/bge-reranker-base`（Cross-Encoder Rerank）
- **向量数据库**：Chroma（本地）
- **持久化存储**：TiDB Cloud（MySQL 兼容）

## ⚙️ RAG 检索流程

```
用户输入
  → 意图路由（LLM 判断：rag / tool / chat）
  → 向量召回（Top K×3 候选）
  → Cross-Encoder Rerank 精排
  → 阈值过滤（score > 0.6）
  → 注入上下文 + 调用 Agent
  → 返回回复 + 溯源引用
```

## 🗄️ 数据库设计

使用 TiDB Cloud 存储聊天记录，支持会话隔离和历史查询。

### chat_sessions - 会话管理表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT | 主键自增 |
| session_id | VARCHAR(64) | 会话唯一标识 |
| user_id | VARCHAR(64) | 用户ID（可选） |
| title | VARCHAR(255) | 会话标题 |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

### chat_messages - 聊天消息表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT | 主键自增 |
| session_id | VARCHAR(64) | 关联会话ID |
| role | ENUM('human','ai') | 消息角色 |
| content | TEXT | 消息内容 |
| created_at | TIMESTAMP | 创建时间 |

### 建表语句

```sql
CREATE TABLE IF NOT EXISTS chat_sessions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL UNIQUE COMMENT '会话ID',
    user_id VARCHAR(64) DEFAULT NULL COMMENT '用户ID(可选)',
    title VARCHAR(255) DEFAULT NULL COMMENT '会话标题',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话管理表';

CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL COMMENT '会话ID',
    role ENUM('human', 'ai') NOT NULL COMMENT '消息角色',
    content TEXT NOT NULL COMMENT '消息内容',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='聊天消息记录表';
```
