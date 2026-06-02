"""任务规划与路由逻辑"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from prompts import ROUTER_PROMPT


def create_router(llm):
    """创建意图路由器"""
    prompt = PromptTemplate.from_template(ROUTER_PROMPT)
    return prompt | llm | StrOutputParser()


def route_query(router, query: str) -> str:
    """路由用户查询到合适的处理流程"""
    result = router.invoke({"input": query}).strip().lower()
    if "rag" in result:
        return "rag"
    elif "tool" in result:
        return "tool"
    return "chat"
