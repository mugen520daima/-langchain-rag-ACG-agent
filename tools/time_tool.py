"""时间工具 - 获取当前日期和时间"""
from datetime import datetime
from langchain_core.tools import tool


@tool
def time_tool() -> str:
    """获取当前的日期和时间。当用户询问现在几点、今天日期、星期几等时间相关问题时使用。"""
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')} {weekdays[now.weekday()]}"
