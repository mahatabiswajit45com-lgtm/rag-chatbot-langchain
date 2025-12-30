"""
Memory Management - Chat History & Session Management
Supports: In-memory, File-based persistence
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


class SessionMemory:
    """
    Manages chat memory for multiple sessions
    Each session has its own conversation history
    """
    
    def __init__(self, persist_dir: Optional[str] = None):
        self.sessions: dict[str, ChatMessageHistory] = {}
        self.persist_dir = Path(persist_dir) if persist_dir else None
        
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_all_sessions()
    
    def get_session(self, session_id: str) -> ChatMessageHistory:
        """Get or create a session's chat history"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMessageHistory()
            # Try to load from disk if persistence enabled
            if self.persist_dir:
                self._load_session(session_id)
        
        return self.sessions[session_id]
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to session history"""
        history = self.get_session(session_id)
        
        if role == "human":
            history.add_user_message(content)
        elif role == "ai":
            history.add_ai_message(content)
        
        # Auto-save if persistence enabled
        if self.persist_dir:
            self._save_session(session_id)
    
    def get_messages(self, session_id: str) -> list[BaseMessage]:
        """Get all messages from a session"""
        return self.get_session(session_id).messages
    
    def get_messages_as_dict(self, session_id: str) -> list[dict]:
        """Get messages as list of dicts (for API responses)"""
        messages = self.get_messages(session_id)
        return [
            {
                "role": "human" if isinstance(m, HumanMessage) else "ai",
                "content": m.content
            }
            for m in messages
        ]
    
    def clear_session(self, session_id: str):
        """Clear a session's history"""
        if session_id in self.sessions:
            self.sessions[session_id].clear()
            
            if self.persist_dir:
                file_path = self.persist_dir / f"{session_id}.json"
                if file_path.exists():
                    file_path.unlink()
    
    def delete_session(self, session_id: str):
        """Completely remove a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            
            if self.persist_dir:
                file_path = self.persist_dir / f"{session_id}.json"
                if file_path.exists():
                    file_path.unlink()
    
    def list_sessions(self) -> list[str]:
        """List all active session IDs"""
        return list(self.sessions.keys())
    
    def get_session_summary(self, session_id: str) -> dict:
        """Get session metadata"""
        messages = self.get_messages(session_id)
        return {
            "session_id": session_id,
            "message_count": len(messages),
            "human_messages": sum(1 for m in messages if isinstance(m, HumanMessage)),
            "ai_messages": sum(1 for m in messages if isinstance(m, AIMessage))
        }
    
    # ===========================================
    # PERSISTENCE METHODS
    # ===========================================
    
    def _save_session(self, session_id: str):
        """Save session to disk"""
        if not self.persist_dir:
            return
        
        messages = self.get_messages_as_dict(session_id)
        data = {
            "session_id": session_id,
            "updated_at": datetime.now().isoformat(),
            "messages": messages
        }
        
        file_path = self.persist_dir / f"{session_id}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_session(self, session_id: str):
        """Load session from disk"""
        if not self.persist_dir:
            return
        
        file_path = self.persist_dir / f"{session_id}.json"
        if not file_path.exists():
            return
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        history = self.sessions[session_id]
        for msg in data.get("messages", []):
            if msg["role"] == "human":
                history.add_user_message(msg["content"])
            else:
                history.add_ai_message(msg["content"])
    
    def _load_all_sessions(self):
        """Load all sessions from disk on startup"""
        if not self.persist_dir:
            return
        
        for file_path in self.persist_dir.glob("*.json"):
            session_id = file_path.stem
            self.sessions[session_id] = ChatMessageHistory()
            self._load_session(session_id)


class ConversationMemoryManager:
    """
    High-level memory manager with windowing support
    Prevents context overflow by limiting history size
    """
    
    def __init__(
        self,
        session_memory: SessionMemory,
        max_messages: int = 20,
        max_tokens: int = 4000
    ):
        self.session_memory = session_memory
        self.max_messages = max_messages
        self.max_tokens = max_tokens
    
    def get_windowed_history(self, session_id: str) -> list[BaseMessage]:
        """Get recent messages within window limit"""
        messages = self.session_memory.get_messages(session_id)
        
        # Simple windowing: keep last N messages
        if len(messages) > self.max_messages:
            messages = messages[-self.max_messages:]
        
        return messages
    
    def estimate_tokens(self, messages: list[BaseMessage]) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        total_chars = sum(len(m.content) for m in messages)
        return total_chars // 4
    
    def get_context_for_llm(self, session_id: str) -> list[BaseMessage]:
        """Get optimized context for LLM call"""
        messages = self.get_windowed_history(session_id)
        
        # Further trim if token limit exceeded
        while self.estimate_tokens(messages) > self.max_tokens and len(messages) > 2:
            messages = messages[2:]  # Remove oldest pair
        
        return messages


# ===========================================
# GLOBAL INSTANCES
# ===========================================

# In-memory session manager (no persistence)
memory_manager = SessionMemory()

# Persistent session manager
persistent_memory = SessionMemory(persist_dir="./chat_sessions")


def get_memory(persist: bool = False) -> SessionMemory:
    """Get appropriate memory manager"""
    return persistent_memory if persist else memory_manager


if __name__ == "__main__":
    # Test memory
    mem = SessionMemory(persist_dir="./test_sessions")
    
    # Add messages
    mem.add_message("test-123", "human", "Hello!")
    mem.add_message("test-123", "ai", "Hi there! How can I help?")
    mem.add_message("test-123", "human", "What's LangChain?")
    mem.add_message("test-123", "ai", "LangChain is a framework for building AI applications.")
    
    # Check
    print("Messages:", mem.get_messages_as_dict("test-123"))
    print("Summary:", mem.get_session_summary("test-123"))
    print("Sessions:", mem.list_sessions())
