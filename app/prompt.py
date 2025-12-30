"""
Prompt Templates - Customizable System Prompts
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ===========================================
# SYSTEM PROMPTS
# ===========================================

GENERAL_ASSISTANT_PROMPT = """You are a helpful AI assistant named BotAI, created by Biswajit.

Your capabilities:
- Answer questions on any topic
- Help with coding and technical problems
- Provide explanations and tutorials
- Assist with writing and editing

Guidelines:
- Be concise but thorough
- Use examples when helpful
- Admit when you don't know something
- Be friendly and professional

Current conversation context will be provided in chat history."""


RAG_ASSISTANT_PROMPT = """You are a knowledgeable AI assistant with access to a document knowledge base.

Your task:
- Answer questions based on the provided context
- If the context doesn't contain the answer, say so clearly
- Always cite which document/source your information comes from
- Be accurate and don't make up information

Context from documents:
{context}

Guidelines:
- Only use information from the provided context
- If unsure, ask for clarification
- Provide specific quotes when relevant"""


CODING_ASSISTANT_PROMPT = """You are an expert programming assistant specializing in:
- Python, JavaScript, and other languages
- AI/ML development with LangChain, OpenAI, etc.
- API development with FastAPI
- Database design and queries

Guidelines:
- Provide working code examples
- Explain your code with comments
- Suggest best practices
- Handle errors gracefully"""


TOOL_AGENT_PROMPT = """You are an AI agent with access to various tools.

Available tools:
{tools}

You can use these tools to help answer questions and complete tasks.
Always use the most appropriate tool for the task.
If no tool is needed, respond directly.

Think step by step:
1. Understand what the user wants
2. Decide if a tool would help
3. Use the tool if needed
4. Provide a helpful response"""


# ===========================================
# CHAT PROMPT TEMPLATES
# ===========================================

def get_chat_prompt(system_prompt: str = GENERAL_ASSISTANT_PROMPT) -> ChatPromptTemplate:
    """Get chat prompt with memory support"""
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])


def get_rag_prompt() -> ChatPromptTemplate:
    """Get RAG prompt with context and memory"""
    return ChatPromptTemplate.from_messages([
        ("system", RAG_ASSISTANT_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])


def get_simple_prompt(system_prompt: str = GENERAL_ASSISTANT_PROMPT) -> ChatPromptTemplate:
    """Get simple prompt without memory"""
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])


# ===========================================
# CUSTOM PROMPT BUILDER
# ===========================================

class PromptBuilder:
    """Build custom prompts dynamically"""
    
    def __init__(self):
        self.prompts = {
            "general": GENERAL_ASSISTANT_PROMPT,
            "rag": RAG_ASSISTANT_PROMPT,
            "coding": CODING_ASSISTANT_PROMPT,
            "agent": TOOL_AGENT_PROMPT
        }
    
    def get_prompt(self, prompt_type: str = "general") -> str:
        """Get system prompt by type"""
        return self.prompts.get(prompt_type, GENERAL_ASSISTANT_PROMPT)
    
    def create_custom_prompt(
        self,
        name: str,
        role: str,
        capabilities: list[str],
        guidelines: list[str]
    ) -> str:
        """Create a custom system prompt"""
        caps = "\n".join([f"- {c}" for c in capabilities])
        guides = "\n".join([f"- {g}" for g in guidelines])
        
        prompt = f"""You are {name}, {role}.

Your capabilities:
{caps}

Guidelines:
{guides}"""
        
        return prompt
    
    def add_prompt(self, name: str, prompt: str):
        """Add a new prompt template"""
        self.prompts[name] = prompt


# Global instance
prompt_builder = PromptBuilder()


if __name__ == "__main__":
    # Test custom prompt creation
    custom = prompt_builder.create_custom_prompt(
        name="TechBot",
        role="a technical support specialist",
        capabilities=["Debug code", "Explain errors", "Suggest fixes"],
        guidelines=["Be patient", "Use simple language", "Provide examples"]
    )
    print(custom)
