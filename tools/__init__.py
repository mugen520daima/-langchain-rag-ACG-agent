"""工具模块"""
from tools.anime_info_tool import anime_info_tool
from tools.merch_tool import merch_search_tool
from tools.legal_site_tool import legal_site_tool
from tools.web_search_tool import web_search_tool


def get_all_tools():
    return [anime_info_tool, merch_search_tool, legal_site_tool, web_search_tool]
