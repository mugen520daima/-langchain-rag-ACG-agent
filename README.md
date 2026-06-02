# 🐱 巧克力猫娘 ACG 助手

基于 LangChain 构建的 ACG（动画/漫画/游戏）领域智能助手，采用 RAG + Agent 架构。

## ✨ 特性

- **角色扮演**：傲娇猫娘"巧克力"人设，拒绝冷冰冰的 AI 语气
- **RAG 知识检索**：内置 ACG 术语、正版渠道、周边商店等知识库
- **多工具调用**：番剧查询、周边搜索、正版渠道推荐、网络搜索
- **会话记忆**：支持多轮对话上下文管理

## 🎭 角色设定 (Prompt)

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

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key

# 运行
python app.py
```

## 📁 项目结构

```
├── agent_service.py   # Agent 主逻辑
├── memory.py          # 会话记忆管理
├── prompts.py         # Prompt 模板
├── workflow.py        # 意图路由
├── rag/               # RAG 检索模块
├── tools/             # 工具集
└── data/              # 知识库数据
```

## 🔧 技术栈

- LangChain + LangChain OpenAI
- 阿里云 DashScope (通义千问)
- RAG 向量检索
