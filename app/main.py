"""
Main Application - FastAPI Server
"""

import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

from .api import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events
    """
    # Startup
    print("🚀 Starting AI Chatbot Server...")
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Check API keys
    if os.getenv("OPENAI_API_KEY"):
        print("✓ OpenAI API key found")
    else:
        print("⚠ OpenAI API key not found")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        print("✓ Anthropic API key found")
    else:
        print("⚠ Anthropic API key not found")
    
    yield
    
    # Shutdown
    print("👋 Shutting down AI Chatbot Server...")


# Create FastAPI app
app = FastAPI(
    title="AI Chatbot API",
    description="""
    Production-ready AI Chatbot with:
    - 💬 Chat with memory
    - 📄 RAG (Document Q&A)
    - 🤖 AI Agent with tools
    - 🔄 Multiple AI providers (OpenAI, Anthropic)
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to AI Chatbot API",
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "chat": "/api/v1/chat",
            "rag_chat": "/api/v1/chat/rag",
            "agent_chat": "/api/v1/chat/agent",
            "upload_document": "/api/v1/documents/upload",
            "sessions": "/api/v1/sessions"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug
    )
