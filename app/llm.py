"""
LLM Configuration - Multi-Provider Support
Supports: OpenAI, Anthropic with automatic fallback
"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()


class LLMConfig:
    """LLM Factory with fallback support"""
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    
    def get_openai(
        self, 
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        streaming: bool = True
    ) -> ChatOpenAI:
        """Get OpenAI Chat Model"""
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            api_key=self.openai_key
        )
    
    def get_anthropic(
        self,
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        streaming: bool = True
    ) -> ChatAnthropic:
        """Get Anthropic Chat Model"""
        if not self.anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env")
        
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            streaming=streaming,
            api_key=self.anthropic_key
        )
    
    def get_llm(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7,
        streaming: bool = True
    ) -> BaseChatModel:
        """
        Get LLM with automatic provider selection
        
        Args:
            provider: 'openai' or 'anthropic'
            model: Model name (optional, uses default)
            temperature: Creativity level 0-1
            streaming: Enable streaming responses
        """
        if provider == "anthropic":
            return self.get_anthropic(
                model=model or "claude-3-sonnet-20240229",
                temperature=temperature,
                streaming=streaming
            )
        else:
            return self.get_openai(
                model=model or self.default_model,
                temperature=temperature,
                streaming=streaming
            )
    
    def get_llm_with_fallback(
        self,
        temperature: float = 0.7,
        streaming: bool = True
    ) -> BaseChatModel:
        """Get LLM with automatic fallback - tries OpenAI first, then Anthropic"""
        try:
            return self.get_openai(temperature=temperature, streaming=streaming)
        except ValueError:
            try:
                return self.get_anthropic(temperature=temperature, streaming=streaming)
            except ValueError:
                raise ValueError("No valid API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")


# Global instance
llm_config = LLMConfig()


def get_llm(**kwargs) -> BaseChatModel:
    """Convenience function to get LLM"""
    return llm_config.get_llm(**kwargs)


def get_llm_with_fallback(**kwargs) -> BaseChatModel:
    """Convenience function to get LLM with fallback"""
    return llm_config.get_llm_with_fallback(**kwargs)


# Quick test
if __name__ == "__main__":
    llm = get_llm_with_fallback()
    response = llm.invoke("Say hello in Bengali!")
    print(response.content)
