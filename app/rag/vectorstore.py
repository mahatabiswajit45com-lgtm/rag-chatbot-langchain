"""
Vector Store - ChromaDB for document storage and retrieval
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from .embeddings import get_embeddings

load_dotenv()


class VectorStoreManager:
    """
    Manage ChromaDB vector store for RAG
    """
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[str] = None,
        embeddings: Optional[Embeddings] = None
    ):
        """
        Initialize vector store
        
        Args:
            collection_name: Name for the Chroma collection
            persist_directory: Where to save the DB (None = in-memory)
            embeddings: Embedding model to use
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.embeddings = embeddings or get_embeddings()
        self.vectorstore: Optional[Chroma] = None
        
        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
    
    def create_from_documents(self, documents: list[Document]) -> Chroma:
        """Create new vector store from documents"""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        return self.vectorstore
    
    def create_from_texts(
        self, 
        texts: list[str], 
        metadatas: Optional[list[dict]] = None
    ) -> Chroma:
        """Create vector store from raw texts"""
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        return self.vectorstore
    
    def load_existing(self) -> Chroma:
        """Load existing vector store from disk"""
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        return self.vectorstore
    
    def get_or_create(self, documents: Optional[list[Document]] = None) -> Chroma:
        """Load existing or create new vector store"""
        try:
            self.load_existing()
            # Check if collection has documents
            if self.vectorstore._collection.count() > 0:
                print(f"✓ Loaded existing collection: {self.collection_name}")
                return self.vectorstore
        except Exception:
            pass
        
        if documents:
            print(f"✓ Creating new collection: {self.collection_name}")
            return self.create_from_documents(documents)
        else:
            # Create empty collection
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            return self.vectorstore
    
    def add_documents(self, documents: list[Document]):
        """Add documents to existing store"""
        if not self.vectorstore:
            self.load_existing()
        
        self.vectorstore.add_documents(documents)
        print(f"✓ Added {len(documents)} documents")
    
    def add_texts(self, texts: list[str], metadatas: Optional[list[dict]] = None):
        """Add texts to existing store"""
        if not self.vectorstore:
            self.load_existing()
        
        self.vectorstore.add_texts(texts, metadatas=metadatas)
        print(f"✓ Added {len(texts)} texts")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[dict] = None
    ) -> list[Document]:
        """Search for similar documents"""
        if not self.vectorstore:
            self.load_existing()
        
        return self.vectorstore.similarity_search(query, k=k, filter=filter)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> list[tuple[Document, float]]:
        """Search with relevance scores"""
        if not self.vectorstore:
            self.load_existing()
        
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """Get as LangChain retriever"""
        if not self.vectorstore:
            self.load_existing()
        
        return self.vectorstore.as_retriever(
            search_kwargs=search_kwargs or {"k": 4}
        )
    
    def delete_collection(self):
        """Delete the entire collection"""
        if self.vectorstore:
            self.vectorstore.delete_collection()
            self.vectorstore = None
            print(f"✓ Deleted collection: {self.collection_name}")
    
    def get_stats(self) -> dict:
        """Get collection statistics"""
        if not self.vectorstore:
            self.load_existing()
        
        count = self.vectorstore._collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory
        }


# ===========================================
# CONVENIENCE FUNCTIONS
# ===========================================

def create_vectorstore(
    documents: list[Document],
    collection_name: str = "documents"
) -> VectorStoreManager:
    """Quick create vector store from documents"""
    manager = VectorStoreManager(collection_name=collection_name)
    manager.create_from_documents(documents)
    return manager


def load_vectorstore(collection_name: str = "documents") -> VectorStoreManager:
    """Quick load existing vector store"""
    manager = VectorStoreManager(collection_name=collection_name)
    manager.load_existing()
    return manager


# Global default instance
default_vectorstore = VectorStoreManager()


if __name__ == "__main__":
    # Test vector store
    manager = VectorStoreManager(collection_name="test_collection")
    
    # Create from texts
    texts = [
        "LangChain is a framework for building LLM applications.",
        "ChromaDB is an open-source vector database.",
        "FastAPI is a modern Python web framework.",
        "RAG combines retrieval with generation for better answers.",
        "Embeddings convert text into numerical vectors."
    ]
    
    manager.create_from_texts(texts)
    print("Stats:", manager.get_stats())
    
    # Search
    results = manager.similarity_search("What is LangChain?", k=2)
    print("\nSearch results for 'What is LangChain?':")
    for doc in results:
        print(f"  - {doc.page_content}")
    
    # Search with scores
    results_with_scores = manager.similarity_search_with_score("vector database", k=2)
    print("\nSearch with scores for 'vector database':")
    for doc, score in results_with_scores:
        print(f"  - {doc.page_content} (score: {score:.4f})")
