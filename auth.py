"""用户认证模块。

提供用户注册和登录功能，密码使用 bcrypt 哈希存储。
数据持久化到 TiDB Cloud 的 users 表中。
"""
import logging

import bcrypt
import pymysql

from config import TIDB_HOST, TIDB_PORT, TIDB_USER, TIDB_PASSWORD, TIDB_DATABASE

logger = logging.getLogger(__name__)

_db_config = {
    "host": TIDB_HOST,
    "port": TIDB_PORT,
    "user": TIDB_USER,
    "password": TIDB_PASSWORD,
    "database": TIDB_DATABASE,
    "ssl": {"ssl": {}},
    "charset": "utf8mb4",
}


def _get_connection() -> pymysql.Connection:
    if not TIDB_HOST or not TIDB_PASSWORD:
        raise ConnectionError("TiDB 数据库未配置（请在 config.py 中填写 TIDB_HOST 和 TIDB_PASSWORD）")
    return pymysql.connect(**_db_config)


def check_db_connection() -> tuple[bool, str]:
    """测试数据库连通性，返回 (是否成功, 消息)。

    失败时只向调用方返回通用提示，原始异常写入日志，避免泄露到前端。
    """
    try:
        conn = _get_connection()
        conn.close()
        return True, "数据库连接正常"
    except ConnectionError as e:
        # 配置缺失类错误，提示信息本身不含敏感细节，可直接返回
        logger.warning(f"数据库未配置: {e}")
        return False, "数据库未配置，请联系管理员"
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        return False, "数据库连接失败，请稍后重试"


def create_users_table():
    """创建 users 表（如不存在）。应在应用启动时调用一次。"""
    try:
        conn = _get_connection()
    except Exception as e:
        logger.warning(f"数据库连接失败，跳过建表: {e}")
        return
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(64) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    nickname VARCHAR(64) DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户表'
            """)
            conn.commit()
    finally:
        conn.close()


def register(username: str, password: str, nickname: str | None = None) -> tuple[bool, str]:
    """注册新用户。

    Returns:
        (成功与否, 提示消息)
    """
    if not username or not password:
        return False, "用户名和密码不能为空"
    if len(username) < 2 or len(username) > 32:
        return False, "用户名长度需在 2-32 字符之间"
    if len(password) < 4:
        return False, "密码长度不能少于 4 位"

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    try:
        conn = _get_connection()
    except (ConnectionError, Exception) as e:
        logger.error(f"注册时数据库连接失败: {e}")
        return False, "服务暂时不可用，请稍后重试"
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (username, password_hash, nickname) VALUES (%s, %s, %s)",
                (username, password_hash, nickname or username),
            )
            conn.commit()
            logger.info(f"[注册成功] 用户: {username}")
            return True, "注册成功"
    except pymysql.err.IntegrityError:
        return False, "用户名已存在"
    except Exception as e:
        logger.error(f"注册失败: {e}")
        return False, "注册失败，请稍后重试"
    finally:
        conn.close()


def authenticate(username: str, password: str) -> tuple[bool, str]:
    """验证用户登录。

    Returns:
        (验证通过与否, 提示消息)
    """
    if not username or not password:
        return False, "请输入用户名和密码"

    try:
        conn = _get_connection()
    except (ConnectionError, Exception) as e:
        logger.error(f"登录时数据库连接失败: {e}")
        return False, "服务暂时不可用，请稍后重试"
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT password_hash FROM users WHERE username = %s", (username,))
            row = cursor.fetchone()
            if not row:
                return False, "用户名或密码错误"
            if bcrypt.checkpw(password.encode("utf-8"), row[0].encode("utf-8")):
                logger.info(f"[登录成功] 用户: {username}")
                return True, "登录成功"
            return False, "用户名或密码错误"
    except Exception as e:
        logger.error(f"登录验证异常: {e}")
        return False, "登录失败，请稍后重试"
    finally:
        conn.close()
