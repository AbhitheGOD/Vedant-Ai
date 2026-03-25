# Vedic Knowledge Chatbot

A multilingual chatbot powered by RAG (Retrieval-Augmented Generation) and FastAPI, answering questions about Vedic/Sanskrit knowledge in multiple languages including English, Hindi, Sanskrit, and regional Indian languages.

## Architecture

```
User Query (any language)
    │
    ▼
[FastAPI: POST /chat]
    │
    ▼
[Language Detection]  ─── detect input language (langdetect + Devanagari heuristics)
    │
    ▼
[Query Translation]   ─── translate to English if needed (via Ollama LLM)
    │
    ▼
[RAG Pipeline]
    ├── Embed the query (sentence-transformers)
    ├── Search ChromaDB for top-k relevant chunks
    └── Return Vedic text passages with metadata
    │
    ▼
[LLM Call via Ollama]
    ├── System prompt: Vedic scholar persona
    ├── Context: retrieved Vedic text chunks
    └── Instruction: respond in {detected_language}
    │
    ▼
[Response] ── returned in user's original language
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

## Setup

### 1. Clone and install dependencies

```bash
cd vedic-chatbot
pip install -r requirements.txt
```

### 2. Install and start Ollama

```bash
# Install from https://ollama.ai, then:
ollama serve

# Pull the default model
ollama pull llama3:8b
```

### 3. Configure environment (optional)

```bash
cp .env.example .env
# Edit .env if you want to change model or settings
```

### 4. Ingest Vedic texts into ChromaDB

```bash
python scripts/ingest.py
```

To reset and re-ingest:
```bash
python scripts/ingest.py --reset
```

### 5. Start the server

```bash
python run.py
```

Server runs at `http://localhost:8000`
API docs at `http://localhost:8000/docs`

---

## API Reference

### POST /chat

Chat with the Vedic knowledge bot.

**Request:**
```json
{
  "message": "What is dharma?",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "response": "Dharma is a multifaceted concept...",
  "language": "en",
  "sources": [
    {
      "text": "Relevant passage...",
      "source": "bhagavad_gita",
      "chapter": "2",
      "verse": "47",
      "relevance_score": 0.85
    }
  ],
  "session_id": "abc123"
}
```

### POST /chat/stream

Same as `/chat` but returns Server-Sent Events (SSE) stream.

Events:
- `{"event": "meta", "session_id": "...", "language": "...", "sources": [...]}` — sent first
- `{"event": "token", "token": "..."}` — one per LLM token
- `{"event": "done"}` — stream complete

### GET /health

Check system status.

```json
{
  "status": "ok",
  "ollama_connected": true,
  "model_loaded": true,
  "documents_count": 142
}
```

### GET /sources

List all available knowledge sources.

```json
{
  "sources": ["bhagavad_gita", "upanishads_key_concepts", "vedic_concepts", "yoga_sutras"],
  "total_documents": 142
}
```

---

## Example Queries

```bash
# English
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is dharma?"}'

# Hindi
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "कर्म क्या है?"}'

# Sanskrit
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "धर्मस्य लक्षणं किम्?"}'

# With session (maintains conversation context)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about the Bhagavad Gita", "session_id": "my-session"}'

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What does it say about karma yoga?", "session_id": "my-session"}'

# Health check
curl http://localhost:8000/health

# List sources
curl http://localhost:8000/sources
```

---

## Project Structure

```
vedic-chatbot/
├── app/
│   ├── main.py                 # FastAPI app with lifespan (startup loads models)
│   ├── config.py               # Pydantic-settings config with .env support
│   ├── routers/
│   │   └── chat.py             # POST /chat, POST /chat/stream, GET /health, GET /sources
│   ├── services/
│   │   ├── llm_service.py      # Abstract LLMService + OllamaService (swappable)
│   │   ├── rag_service.py      # ChromaDB + sentence-transformers embeddings
│   │   ├── language_service.py # Language detection + LLM-based translation
│   │   └── ingestion_service.py# Text chunking + metadata parsing
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response models
│   └── prompts/
│       └── system_prompts.py   # Vedic scholar system prompt
├── data/
│   └── texts/                  # Vedic knowledge base (.txt files)
│       ├── bhagavad_gita_selected.txt
│       ├── upanishads_key_concepts.txt
│       ├── yoga_sutras.txt
│       └── vedic_concepts.txt
├── scripts/
│   └── ingest.py               # CLI ingestion script
├── requirements.txt
├── .env.example
├── run.py
└── README.md
```

## Adding More Vedic Texts

Place `.txt` files in `data/texts/` using this format:

```
[SOURCE: Text Name] [CHAPTER: 1] [VERSE: 1-5]
Your content here...

[SOURCE: Text Name] [CHAPTER: 1] [VERSE: 6]
Next section...
```

Then re-run: `python scripts/ingest.py`

## Swapping to a Different LLM

To swap from Ollama to Gemini or any other LLM, implement the `LLMService` abstract class in `app/services/llm_service.py` and update the `get_llm_service()` factory function. No other code changes required.
