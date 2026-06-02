"""查询周边信息工具"""
from langchain_core.tools import tool

MERCH_DB = {
    "手办": ["GSC（Good Smile Company）", "Alter", "Kotobukiya", "Bandai"],
    "周边": ["animate", "GAMERS", "Melonbooks", "虎之穴"],
    "服饰": ["COSPA", "二次元服饰专营店"],
}


@tool
def merch_search_tool(query: str) -> str:
    """搜索动漫周边商品信息，包括手办、周边、服饰等"""
    results = []
    for category, shops in MERCH_DB.items():
        if category in query or any(s.lower() in query.lower() for s in shops):
            results.append(f"{category}推荐店铺：{', '.join(shops)}")
    
    if not results:
        results = [f"手办推荐：{', '.join(MERCH_DB['手办'])}",
                   f"周边店铺：{', '.join(MERCH_DB['周边'])}"]
    
    return "\n".join(results) + "\n提示：建议通过官方渠道购买正版周边"
