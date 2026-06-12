"""
FastAPI HTTP 接口服务
====================
为 Java 后端提供 REST API，支持对话、流式对话、会话管理等功能。

启动方式：
    uvicorn api_server:app --host 0.0.0.0 --port 8600 --reload
"""
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from agent_service import AnimeAgent
from auth import authenticate, register, create_users_table, check_db_connection
from memory import chat_store

logger = logging.getLogger(__name__)

# 会话级 Agent 实例缓存
_agents: dict[str, AnimeAgent] = {}


def _get_agent(session_id: str) -> AnimeAgent:
    if session_id not in _agents:
        _agents[session_id] = AnimeAgent(session_id=session_id)
    return _agents[session_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_users_table()
    db_ok, db_msg = check_db_connection()
    if not db_ok:
        logger.error(f"数据库连接失败: {db_msg}")
    yield
    _agents.clear()


app = FastAPI(title="ACG Agent API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """兜底异常处理：记录完整异常到日志，只向客户端返回通用错误，
    避免堆栈/原始异常信息泄露到响应体。"""
    logger.error(f"未处理异常 {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "服务器内部错误，请稍后重试"})


# ==================== 请求/响应模型 ====================

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    nickname: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: str
    message: str

class NewSessionRequest(BaseModel):
    user_id: str
    title: Optional[str] = "新对话"


# ==================== 认证接口 ====================

@app.post("/api/auth/login")
def login(req: LoginRequest):
    ok, msg = authenticate(req.username, req.password)
    if not ok:
        raise HTTPException(status_code=401, detail=msg)
    return {"success": True, "username": req.username}


@app.post("/api/auth/register")
def register_user(req: RegisterRequest):
    ok, msg = register(req.username, req.password, req.nickname)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True}


# ==================== 会话管理接口 ====================

@app.post("/api/session/create")
def create_session(req: NewSessionRequest):
    session_id = str(uuid.uuid4())[:16]
    chat_store.create_session(session_id, user_id=req.user_id, title=req.title)
    return {"session_id": session_id}


@app.get("/api/session/list")
def list_sessions(user_id: str, limit: int = 20):
    sessions = chat_store.get_session_list(user_id=user_id, limit=limit)
    return {"sessions": sessions}


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    chat_store.delete_session(session_id)
    _agents.pop(session_id, None)
    return {"success": True}


@app.get("/api/session/{session_id}/messages")
def get_messages(session_id: str, limit: int = 100):
    messages = chat_store.get_messages(session_id, limit=limit)
    return {"messages": messages}


# ==================== 对话接口 ====================

@app.post("/api/chat")
def chat(req: ChatRequest):
    """同步对话接口，返回完整回复"""
    agent = _get_agent(req.session_id)
    chat_store.save_message(req.session_id, "human", req.message)
    try:
        result = agent.chat(req.message)
    except Exception as e:
        logger.error(f"对话处理失败 session={req.session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="对话处理失败，请稍后重试")
    chat_store.save_message(req.session_id, "ai", result["output"])
    return {
        "reply": result["output"],
        "rag_chunks": result["rag_chunks"],
        "rag_confident": result["rag_confident"],
    }


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """流式对话接口，SSE 格式返回"""
    agent = _get_agent(req.session_id)
    chat_store.save_message(req.session_id, "human", req.message)

    def event_generator():
        full_response = ""
        try:
            for chunk in agent.chat_stream(req.message):
                text = str(chunk)
                full_response += text
                yield f"data: {text}\n\n"
        except Exception as e:
            logger.error(f"流式对话失败 session={req.session_id}: {e}", exc_info=True)
            yield "data: [对话处理出错，请稍后重试]\n\n"
        yield "data: [DONE]\n\n"
        if full_response:
            chat_store.save_message(req.session_id, "ai", full_response)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ==================== 健康检查 ====================

@app.get("/api/health")
def health():
    db_ok, db_msg = check_db_connection()
    return {"status": "ok" if db_ok else "degraded", "database": db_msg}
