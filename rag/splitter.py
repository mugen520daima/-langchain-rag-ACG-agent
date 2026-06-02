"""文本切分"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
    """将文档切分为小块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " "]
    )
    return splitter.split_documents(documents)
