"""
Document Loader - Load various document types
Supports: PDF, TXT, DOCX, Markdown
"""

from pathlib import Path
from typing import Union
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    Docx2txtLoader
)


class DocumentLoader:
    """
    Universal document loader with multiple format support
    """
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'pdf',
        '.txt': 'text',
        '.md': 'text',
        '.docx': 'docx',
    }
    
    def __init__(self):
        self.loaded_docs: list[Document] = []
    
    def load_file(self, file_path: Union[str, Path]) -> list[Document]:
        """Load a single file"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")
        
        file_type = self.SUPPORTED_EXTENSIONS[ext]
        
        if file_type == 'pdf':
            loader = PyPDFLoader(str(path))
        elif file_type == 'text':
            loader = TextLoader(str(path))
        elif file_type == 'docx':
            loader = Docx2txtLoader(str(path))
        else:
            raise ValueError(f"No loader for type: {file_type}")
        
        docs = loader.load()
        
        # Add metadata
        for doc in docs:
            doc.metadata['source_file'] = path.name
            doc.metadata['file_type'] = file_type
        
        self.loaded_docs.extend(docs)
        return docs
    
    def load_directory(
        self, 
        dir_path: Union[str, Path],
        glob_pattern: str = "**/*.*",
        recursive: bool = True
    ) -> list[Document]:
        """Load all supported documents from a directory"""
        path = Path(dir_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        all_docs = []
        
        # Load each supported file type
        for ext in self.SUPPORTED_EXTENSIONS.keys():
            pattern = f"**/*{ext}" if recursive else f"*{ext}"
            
            for file_path in path.glob(pattern):
                try:
                    docs = self.load_file(file_path)
                    all_docs.extend(docs)
                    print(f"✓ Loaded: {file_path.name}")
                except Exception as e:
                    print(f"✗ Failed to load {file_path.name}: {e}")
        
        self.loaded_docs.extend(all_docs)
        return all_docs
    
    def load_text(self, text: str, metadata: dict = None) -> Document:
        """Create document from raw text"""
        doc = Document(
            page_content=text,
            metadata=metadata or {"source": "direct_input"}
        )
        self.loaded_docs.append(doc)
        return doc
    
    def load_texts(self, texts: list[str], metadata_list: list[dict] = None) -> list[Document]:
        """Create documents from multiple texts"""
        if metadata_list and len(metadata_list) != len(texts):
            raise ValueError("metadata_list must match texts length")
        
        docs = []
        for i, text in enumerate(texts):
            metadata = metadata_list[i] if metadata_list else {"source": f"text_{i}"}
            doc = self.load_text(text, metadata)
            docs.append(doc)
        
        return docs
    
    def get_all_docs(self) -> list[Document]:
        """Get all loaded documents"""
        return self.loaded_docs
    
    def clear(self):
        """Clear loaded documents"""
        self.loaded_docs = []
    
    def get_stats(self) -> dict:
        """Get statistics about loaded documents"""
        if not self.loaded_docs:
            return {"total_docs": 0}
        
        total_chars = sum(len(doc.page_content) for doc in self.loaded_docs)
        sources = set(doc.metadata.get('source_file', 'unknown') for doc in self.loaded_docs)
        
        return {
            "total_docs": len(self.loaded_docs),
            "total_characters": total_chars,
            "estimated_tokens": total_chars // 4,
            "unique_sources": list(sources)
        }


# Global instance
doc_loader = DocumentLoader()


if __name__ == "__main__":
    loader = DocumentLoader()
    
    # Test with sample text
    loader.load_texts([
        "LangChain is a framework for building LLM applications.",
        "ChromaDB is a vector database for embeddings.",
        "FastAPI is a modern Python web framework."
    ])
    
    print("Stats:", loader.get_stats())
    print("Docs:", [doc.page_content[:50] for doc in loader.get_all_docs()])
