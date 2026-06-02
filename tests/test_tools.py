"""工具测试"""
import pytest
from tools.anime_info_tool import anime_info_tool
from tools.merch_tool import merch_search_tool
from tools.legal_site_tool import legal_site_tool
from tools.web_search_tool import web_search_tool


def test_anime_info_tool():
    result = anime_info_tool.invoke({"anime_name": "进击的巨人"})
    assert "进击的巨人" in result
    assert "热血" in result


def test_anime_info_tool_not_found():
    result = anime_info_tool.invoke({"anime_name": "不存在的番"})
    assert "未找到" in result


def test_merch_search_tool():
    result = merch_search_tool.invoke({"query": "手办"})
    assert "GSC" in result or "手办" in result


def test_legal_site_tool_video():
    result = legal_site_tool.invoke({"content_type": "视频"})
    assert "哔哩哔哩" in result


def test_legal_site_tool_manga():
    result = legal_site_tool.invoke({"content_type": "漫画"})
    assert "漫画" in result


def test_web_search_tool():
    result = web_search_tool.invoke({"query": "新番推荐"})
    assert "搜索" in result
