"""登录态 JWT 工具。

签发 / 校验用户登录令牌。令牌内只放 username + 过期时间，
不放任何敏感信息；密钥由 config.JWT_SECRET 提供（生产用环境变量覆盖）。
"""
import logging
from datetime import datetime, timedelta, timezone

import jwt

from config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRE_HOURS

logger = logging.getLogger(__name__)


def create_token(username: str) -> str:
    """为指定用户签发 JWT，有效期 config.JWT_EXPIRE_HOURS 小时。"""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "iat": now,
        "exp": now + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> str | None:
    """校验 JWT，通过则返回 username，过期或无效返回 None。"""
    if not token:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        logger.info("登录令牌已过期")
        return None
    except jwt.InvalidTokenError as e:
        logger.info(f"登录令牌无效: {e}")
        return None
