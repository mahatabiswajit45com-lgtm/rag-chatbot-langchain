# 🤖 AI Chatbot - Production Ready

A comprehensive AI chatbot with LangChain, ChromaDB, FastAPI, and Docker support.

## ✨ Features

- 💬 **Chat with Memory** - Remembers conversation history
- 📄 **RAG (Document Q&A)** - Answer questions from your documents
- 🤖 **AI Agent** - Uses tools (calculator, date, text processing)
- 🔄 **Multi-Provider** - OpenAI & Anthropic support with fallback
- 🐳 **Docker Ready** - One command deployment
- 📊 **Session Management** - Persistent chat sessions

## 🚀 Quick Start

### 1. Clone & Setup

```bash
cd chatbot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env` file:
```env
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here  # Optional
```

### 3. Run Server

```bash
uvicorn app.main:app --reload
```

Open: http://localhost:8000/docs

## 📖 API Endpoints

### Chat Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat` | POST | Simple chat with memory |
| `/api/v1/chat/rag` | POST | Chat with document context |
| `/api/v1/chat/agent` | POST | Chat with AI agent (tools) |

### Document Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/documents/upload` | POST | Upload PDF/DOCX/TXT |
| `/api/v1/documents/add-text` | POST | Add raw text |
| `/api/v1/documents/stats/{name}` | GET | Collection stats |
| `/api/v1/documents/{name}` | DELETE | Delete collection |

### Session Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/sessions` | GET | List all sessions |
| `/api/v1/sessions/{id}` | GET | Get session history |
| `/api/v1/sessions/{id}` | DELETE | Delete session |

## 💡 Usage Examples

### Simple Chat

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! What is LangChain?"}'
```

### RAG Chat (Document Q&A)

```bash
# First upload a document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@document.pdf" \
  -F "collection_name=my_docs"

# Then query it
curl -X POST http://localhost:8000/api/v1/chat/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the document say about AI?", "collection_name": "my_docs"}'
```

### Agent Chat (with Tools)

```bash
curl -X POST http://localhost:8000/api/v1/chat/agent \
  -H "Content-Type: application/json" \
  -d '{"message": "Calculate sqrt(144) + 25"}'
```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Using Docker directly

```bash
# Build
docker build -t ai-chatbot .

# Run
docker run -d -p 8000:8000 \
  -e OPENAI_API_KEY=sk-xxx \
  ai-chatbot
```

## 📁 Project Structure

```
chatbot/
├── app/
│   ├── main.py           # FastAPI app
│   ├── api.py            # API endpoints
│   ├── llm.py            # LLM configuration
│   ├── prompt.py         # Prompt templates
│   ├── memory.py         # Chat history
│   ├── rag/
│   │   ├── loader.py     # Document loading
│   │   ├── splitter.py   # Text chunking
│   │   ├── embeddings.py # Vector embeddings
│   │   └── vectorstore.py# ChromaDB
│   └── tools/
│       └── calculator.py # Agent tools
├── data/                 # PDF storage
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env
```

## 🛠️ Available Tools (Agent)

| Tool | Description |
|------|-------------|
| `calculator` | Math calculations |
| `get_current_datetime` | Current date/time |
| `calculate_date_difference` | Days between dates |
| `word_counter` | Text statistics |
| `text_transformer` | Text transformations |
| `json_formatter` | JSON formatting |
| `unit_converter` | Unit conversions |

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `ANTHROPIC_API_KEY` | Anthropic API key | Optional |
| `DEFAULT_MODEL` | Default LLM model | gpt-4o-mini |
| `CHROMA_PERSIST_DIR` | ChromaDB storage | ./chroma_db |
| `HOST` | Server host | 0.0.0.0 |
| `PORT` | Server port | 8000 |

## 📚 Key Concepts Learned

### LangChain
- LLM integration with multiple providers
- Prompt templates and management
- Conversation memory (buffer, window)
- Chains for complex workflows
- Agents with tool calling

### ChromaDB
- Vector embeddings storage
- Similarity search
- Document collections
- Persistence

### RAG (Retrieval Augmented Generation)
- Document loading (PDF, DOCX, TXT)
- Text chunking strategies
- Embedding generation
- Context retrieval for LLM

### FastAPI
- REST API design
- Request/Response models
- File uploads
- Error handling

### Docker
- Containerization
- Docker Compose
- Environment management
- Health checks

## 🚀 Next Steps

1. Add authentication (API keys)
2. Add rate limiting
3. Add Redis caching
4. Add streaming responses
5. Add more tools (web search, etc.)
6. Add frontend UI

## 📝 License

MIT License - Built by Biswajit

---

Made with ❤️ using LangChain, ChromaDB, FastAPI, and Docker
