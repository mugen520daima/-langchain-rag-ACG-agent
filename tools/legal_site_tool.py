"""查询正版观看/购买渠道工具"""
from langchain_core.tools import tool

LEGAL_SITES = {
    "视频平台": {
        "哔哩哔哩": "https://www.bilibili.com - 国内最大弹幕视频网站，正版新番",
        "爱奇艺": "https://www.iqiyi.com - 部分独播番剧",
        "腾讯视频": "https://v.qq.com - 部分独播番剧",
        "优酷": "https://www.youku.com - 部分番剧资源",
    },
    "漫画平台": {
        "哔哩哔哩漫画": "https://manga.bilibili.com - 正版漫画阅读",
        "快看漫画": "https://www.kuaikanmanhua.com - 国漫为主",
        "腾讯动漫": "https://ac.qq.com - 正版漫画平台",
    }
}


@tool
def legal_site_tool(content_type: str) -> str:
    """查询正版动漫观看或漫画阅读渠道，参数为'视频'或'漫画'"""
    if "漫画" in content_type:
        sites = LEGAL_SITES["漫画平台"]
        return "正版漫画平台：\n" + "\n".join(f"- {k}: {v}" for k, v in sites.items())
    else:
        sites = LEGAL_SITES["视频平台"]
        return "正版视频平台：\n" + "\n".join(f"- {k}: {v}" for k, v in sites.items())
