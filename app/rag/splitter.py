"""
Text Splitter - Chunk documents for optimal retrieval
"""

from typing import Optional
from langchain_core.documents import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)


class DocumentSplitter:
    """
    Smart document splitter with multiple strategies
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = "recursive"
    ):
        """
        Initialize splitter
        
        Args:
            chunk_size: Max characters per chunk
            chunk_overlap: Overlap between chunks (for context continuity)
            strategy: 'recursive', 'character', or 'token'
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.splitter = self._create_splitter()
    
    def _create_splitter(self):
        """Create appropriate splitter based on strategy"""
        if self.strategy == "recursive":
            # Best for most cases - splits by paragraphs, then sentences
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
        elif self.strategy == "character":
            # Simple character-based splitting
            return CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n"
            )
        elif self.strategy == "token":
            # Token-based splitting (more accurate for LLMs)
            return TokenTextSplitter(
                chunk_size=self.chunk_size // 4,  # Rough char to token
                chunk_overlap=self.chunk_overlap // 4
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks"""
        return self.splitter.split_documents(documents)
    
    def split_text(self, text: str) -> list[str]:
        """Split raw text into chunks"""
        return self.splitter.split_text(text)
    
    def split_with_metadata(
        self, 
        text: str, 
        base_metadata: Optional[dict] = None
    ) -> list[Document]:
        """Split text and create documents with metadata"""
        chunks = self.split_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(base_metadata or {})
            }
            documents.append(Document(page_content=chunk, metadata=metadata))
        
        return documents
    
    @staticmethod
    def get_optimal_settings(doc_type: str) -> dict:
        """Get recommended settings for different document types"""
        settings = {
            "pdf": {"chunk_size": 1000, "chunk_overlap": 200, "strategy": "recursive"},
            "code": {"chunk_size": 1500, "chunk_overlap": 100, "strategy": "recursive"},
            "chat": {"chunk_size": 500, "chunk_overlap": 50, "strategy": "character"},
            "article": {"chunk_size": 1200, "chunk_overlap": 200, "strategy": "recursive"},
            "qa": {"chunk_size": 800, "chunk_overlap": 100, "strategy": "recursive"}
        }
        return settings.get(doc_type, settings["pdf"])


def create_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = "recursive"
) -> DocumentSplitter:
    """Factory function to create splitter"""
    return DocumentSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy
    )


# Default splitter
default_splitter = DocumentSplitter()


if __name__ == "__main__":
    # Test splitter
    splitter = DocumentSplitter(chunk_size=200, chunk_overlap=50)
    
    sample_text = """
    LangChain is a framework for developing applications powered by language models.
    It provides tools for prompt management, memory, and chains.
    
    ChromaDB is an open-source embedding database. It stores vector embeddings
    and allows for efficient similarity search. This is crucial for RAG systems.
    
    FastAPI is a modern, fast web framework for building APIs with Python.
    It's based on standard Python type hints and provides automatic documentation.
    """
    
    chunks = splitter.split_text(sample_text)
    print(f"Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
