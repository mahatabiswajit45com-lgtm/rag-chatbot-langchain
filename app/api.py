"""
API Endpoints - FastAPI routes for chatbot
"""

import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from langchain.chains import ConversationChain, RetrievalQA
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .llm import get_llm_with_fallback
from .prompt import get_chat_prompt, get_rag_prompt, GENERAL_ASSISTANT_PROMPT, RAG_ASSISTANT_PROMPT
from .memory import get_memory, ConversationMemoryManager
from .rag import DocumentLoader, DocumentSplitter, VectorStoreManager
from .tools import get_tools

router = APIRouter()

# ===========================================
# REQUEST/RESPONSE MODELS
# ===========================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = 0.7


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tokens_used: Optional[int] = None


class RAGQueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    collection_name: Optional[str] = "documents"
    k: Optional[int] = 4


class AgentRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    message_count: int
    messages: list[dict]


# ===========================================
# MEMORY INSTANCES
# ===========================================

memory_manager = get_memory(persist=True)
conversation_manager = ConversationMemoryManager(memory_manager)


# ===========================================
# CHAT ENDPOINTS
# ===========================================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Simple chat endpoint with memory
    """
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Get LLM
        llm = get_llm_with_fallback(temperature=request.temperature)
        
        # Get chat history
        history = conversation_manager.get_windowed_history(session_id)
        
        # Create prompt
        prompt = get_chat_prompt(GENERAL_ASSISTANT_PROMPT)
        
        # Build chain manually for more control
        chain = prompt | llm
        
        # Invoke
        response = chain.invoke({
            "input": request.message,
            "chat_history": history
        })
        
        # Save to memory
        memory_manager.add_message(session_id, "human", request.message)
        memory_manager.add_message(session_id, "ai", response.content)
        
        return ChatResponse(
            response=response.content,
            session_id=session_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/rag", response_model=ChatResponse)
async def chat_with_rag(request: RAGQueryRequest):
    """
    Chat with RAG (document context)
    """
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Load vector store
        vectorstore_manager = VectorStoreManager(collection_name=request.collection_name)
        vectorstore = vectorstore_manager.get_or_create()
        
        # Check if documents exist
        stats = vectorstore_manager.get_stats()
        if stats["document_count"] == 0:
            raise HTTPException(
                status_code=400, 
                detail="No documents in collection. Upload documents first."
            )
        
        # Get retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": request.k})
        
        # Get relevant documents
        docs = retriever.invoke(request.query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Get LLM
        llm = get_llm_with_fallback()
        
        # Get history
        history = conversation_manager.get_windowed_history(session_id)
        
        # Create RAG prompt with context
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_ASSISTANT_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Invoke
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "input": request.query,
            "chat_history": history
        })
        
        # Save to memory
        memory_manager.add_message(session_id, "human", request.query)
        memory_manager.add_message(session_id, "ai", response.content)
        
        return ChatResponse(
            response=response.content,
            session_id=session_id
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/agent", response_model=ChatResponse)
async def chat_with_agent(request: AgentRequest):
    """
    Chat with AI agent (can use tools)
    """
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Get LLM
        llm = get_llm_with_fallback()
        
        # Get tools
        tools = get_tools()
        
        # Create agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with access to various tools.
Use tools when they would help answer the question.
Always explain what you're doing and show the results clearly."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Get history
        history = conversation_manager.get_windowed_history(session_id)
        
        # Run agent
        result = agent_executor.invoke({
            "input": request.message,
            "chat_history": history
        })
        
        # Save to memory
        memory_manager.add_message(session_id, "human", request.message)
        memory_manager.add_message(session_id, "ai", result["output"])
        
        return ChatResponse(
            response=result["output"],
            session_id=session_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================
# DOCUMENT ENDPOINTS
# ===========================================

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form(default="documents")
):
    """
    Upload a document to the RAG system
    """
    try:
        # Save uploaded file temporarily
        import tempfile
        import os
        
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load document
        loader = DocumentLoader()
        docs = loader.load_file(tmp_path)
        
        # Split documents
        splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        # Add to vector store
        vectorstore_manager = VectorStoreManager(collection_name=collection_name)
        vectorstore_manager.get_or_create()
        vectorstore_manager.add_documents(chunks)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "collection": collection_name
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/add-text")
async def add_text_to_documents(
    texts: list[str],
    collection_name: str = "documents",
    metadata: Optional[list[dict]] = None
):
    """
    Add text directly to the RAG system
    """
    try:
        # Split texts
        splitter = DocumentSplitter(chunk_size=1000, chunk_overlap=200)
        all_chunks = []
        
        for i, text in enumerate(texts):
            meta = metadata[i] if metadata and i < len(metadata) else {"source": f"text_{i}"}
            chunks = splitter.split_with_metadata(text, meta)
            all_chunks.extend(chunks)
        
        # Add to vector store
        vectorstore_manager = VectorStoreManager(collection_name=collection_name)
        vectorstore_manager.get_or_create()
        vectorstore_manager.add_documents(all_chunks)
        
        return {
            "status": "success",
            "texts_processed": len(texts),
            "chunks_created": len(all_chunks),
            "collection": collection_name
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/stats/{collection_name}")
async def get_document_stats(collection_name: str = "documents"):
    """
    Get statistics about a document collection
    """
    try:
        vectorstore_manager = VectorStoreManager(collection_name=collection_name)
        stats = vectorstore_manager.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{collection_name}")
async def delete_collection(collection_name: str):
    """
    Delete a document collection
    """
    try:
        vectorstore_manager = VectorStoreManager(collection_name=collection_name)
        vectorstore_manager.delete_collection()
        return {"status": "success", "deleted": collection_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===========================================
# SESSION ENDPOINTS
# ===========================================

@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """
    Get session history
    """
    messages = memory_manager.get_messages_as_dict(session_id)
    return SessionResponse(
        session_id=session_id,
        message_count=len(messages),
        messages=messages
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session
    """
    memory_manager.delete_session(session_id)
    return {"status": "success", "deleted": session_id}


@router.get("/sessions")
async def list_sessions():
    """
    List all active sessions
    """
    sessions = memory_manager.list_sessions()
    return {
        "count": len(sessions),
        "sessions": [memory_manager.get_session_summary(s) for s in sessions]
    }


# ===========================================
# HEALTH CHECK
# ===========================================

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "AI Chatbot API",
        "version": "1.0.0"
    }
