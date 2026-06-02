"""查询番剧资料工具"""
from langchain_core.tools import tool

ANIME_DB = {
    "进击的巨人": {"name": "进击的巨人", "episodes": 87, "genre": "热血/奇幻", "status": "完结", "rating": 9.1},
    "咒术回战": {"name": "咒术回战", "episodes": 47, "genre": "热血/奇幻", "status": "连载中", "rating": 8.8},
    "间谍过家家": {"name": "间谍过家家", "episodes": 37, "genre": "喜剧/动作", "status": "连载中", "rating": 9.0},
    "鬼灭之刃": {"name": "鬼灭之刃", "episodes": 55, "genre": "热血/奇幻", "status": "连载中", "rating": 8.9},
    "葬送的芙莉莲": {"name": "葬送的芙莉莲", "episodes": 28, "genre": "奇幻/冒险", "status": "完结", "rating": 9.3},
}


@tool
def anime_info_tool(anime_name: str) -> str:
    """查询番剧基本信息，包括集数、类型、状态、评分等"""
    for key, info in ANIME_DB.items():
        if key in anime_name or anime_name in key:
            return f"《{info['name']}》：{info['genre']}，共{info['episodes']}集，{info['status']}，评分{info['rating']}"
    return f"未找到'{anime_name}'的信息，建议使用web_search搜索"
