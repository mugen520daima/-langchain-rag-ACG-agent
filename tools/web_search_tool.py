"""网络搜索工具（使用 DuckDuckGo）"""
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


# 初始化 DuckDuckGo 搜索工具
_ddg_search = DuckDuckGoSearchRun()


@tool
def web_search_tool(query: str) -> str:
    """搜索网络获取最新动漫资讯，返回实时搜索结果"""
    try:
        result = _ddg_search.invoke(query)
        return result
    except Exception as e:
        return f"[搜索失败] 无法获取搜索结果: {str(e)}。建议访问B站、萌娘百科、AniList等获取最新信息"
