"""
Embeddings - Convert text to vectors
Supports: OpenAI, HuggingFace (free), and local models
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()


class EmbeddingsConfig:
    """
    Embeddings factory with multiple provider support
    """
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
    
    def get_openai(
        self,
        model: str = "text-embedding-3-small"
    ) -> OpenAIEmbeddings:
        """
        Get OpenAI embeddings
        
        Models:
        - text-embedding-3-small: Fast, cheap, good quality
        - text-embedding-3-large: Best quality, more expensive
        - text-embedding-ada-002: Legacy, still works
        """
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        return OpenAIEmbeddings(
            model=model,
            api_key=self.openai_key
        )
    
    def get_default(self) -> Embeddings:
        """Get default embeddings (OpenAI small model)"""
        return self.get_openai()
    
    def get_embeddings(
        self,
        provider: str = "openai",
        model: Optional[str] = None
    ) -> Embeddings:
        """
        Get embeddings by provider
        
        Args:
            provider: 'openai' or 'huggingface'
            model: Specific model name
        """
        if provider == "openai":
            return self.get_openai(model or "text-embedding-3-small")
        else:
            return self.get_openai()  # Fallback to OpenAI


# Global instance
embeddings_config = EmbeddingsConfig()


def get_embeddings(provider: str = "openai", model: str = None) -> Embeddings:
    """Convenience function to get embeddings"""
    return embeddings_config.get_embeddings(provider, model)


if __name__ == "__main__":
    # Test embeddings
    embeddings = get_embeddings()
    
    # Embed single text
    vector = embeddings.embed_query("Hello, world!")
    print(f"Vector dimension: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
    
    # Embed multiple texts
    texts = ["Hello", "World", "LangChain"]
    vectors = embeddings.embed_documents(texts)
    print(f"Embedded {len(vectors)} texts")
