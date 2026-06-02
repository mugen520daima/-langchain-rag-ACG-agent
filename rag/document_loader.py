"""加载知识库文档"""
from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document


def load_documents(knowledge_dir: str = "data/knowledge") -> list[Document]:
    """加载知识库目录下的所有md文件"""
    path = Path(knowledge_dir)
    if not path.exists():
        return []
    
    docs = []
    for file in path.glob("*.md"):
        content = file.read_text(encoding="utf-8")
        docs.append(Document(page_content=content, metadata={"source": str(file)}))
    return docs
