"""网络搜索工具（预留MCP接入）"""
from langchain_core.tools import tool


@tool
def web_search_tool(query: str) -> str:
    """搜索网络获取最新动漫资讯，可后续接入MCP搜索服务"""
    # TODO: 接入MCP搜索服务或其他搜索API
    return f"[模拟搜索结果] 搜索'{query}'：建议访问B站、萌娘百科、AniList等获取最新信息"
