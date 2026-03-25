import uuid
import asyncio
from typing import Dict, List, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json

try:
    from sarvamai import SarvamAI
    _sarvam_available = True
except ImportError:
    _sarvam_available = False

from app.models.schemas import ChatRequest, ChatResponse, HealthResponse, SourcesResponse, SourceInfo, RecommendationsRequest, RecommendationsResponse, TTSRequest
from app.config import settings
from app.services.rag_service import rag_service
from app.services.llm_service import get_llm_service
from app.services.language_service import (
    detect_language,
    translate_to_english,
    get_response_language_instruction,
)
from app.prompts.system_prompts import build_system_prompt

router = APIRouter()

# In-memory conversation history keyed by session_id
# Stores last MAX_HISTORY exchanges as list of {role, content} dicts
_conversation_memory: Dict[str, List[Dict]] = {}
MAX_HISTORY = 5


def _get_history(session_id: str) -> List[Dict]:
    return _conversation_memory.get(session_id, [])


def _update_history(session_id: str, user_msg: str, assistant_msg: str):
    history = _conversation_memory.get(session_id, [])
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    # Keep only last MAX_HISTORY exchanges (2 messages per exchange)
    if len(history) > MAX_HISTORY * 2:
        history = history[-(MAX_HISTORY * 2):]
    _conversation_memory[session_id] = history


def _build_rag_context(sources: List[SourceInfo]) -> str:
    if not sources:
        return "No specific passages retrieved. Answer from general Vedic knowledge."
    parts = ["Relevant passages from the Vedic knowledge base:\n"]
    for i, src in enumerate(sources, 1):
        ref = f"[{src.source}"
        if src.chapter:
            ref += f", Chapter {src.chapter}"
        if src.verse:
            ref += f", Verse {src.verse}"
        ref += "]"
        parts.append(f"{i}. {ref}\n{src.text}\n")
    return "\n".join(parts)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # 1. Detect language
    lang_code = detect_language(message)

    # 2. Translate to English for RAG retrieval if needed
    english_query = await translate_to_english(message, lang_code)

    # 3. RAG retrieval
    sources = rag_service.query(english_query)

    # 4. Build context + prompt
    rag_context = _build_rag_context(sources)
    lang_instruction = get_response_language_instruction(lang_code)
    system_prompt = build_system_prompt(lang_instruction)

    full_prompt = (
        f"Context from Vedic texts:\n{rag_context}\n\n"
        f"User question: {english_query}\n\n"
        f"Please answer based on the context above and your knowledge of Vedic texts. "
        f"{lang_instruction}"
    )

    # 5. Get conversation history
    history = _get_history(session_id)

    # 6. Call LLM
    llm = get_llm_service()
    try:
        response_text = await llm.generate(
            prompt=full_prompt,
            system_prompt=system_prompt,
            history=history,
        )
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 7. Update conversation history
    _update_history(session_id, message, response_text)

    return ChatResponse(
        response=response_text,
        language=lang_code,
        sources=sources,
        session_id=session_id,
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    lang_code = detect_language(message)
    translated = lang_code != "en"
    english_query = await translate_to_english(message, lang_code)
    sources = rag_service.query(english_query)
    rag_context = _build_rag_context(sources)
    lang_instruction = get_response_language_instruction(lang_code)
    system_prompt = build_system_prompt(lang_instruction)

    full_prompt = (
        f"Context from Vedic texts:\n{rag_context}\n\n"
        f"User question: {english_query}\n\n"
        f"Please answer based on the context above and your knowledge of Vedic texts. "
        f"{lang_instruction}"
    )

    history = _get_history(session_id)
    llm = get_llm_service()
    collected_response: List[str] = []

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            meta = {
                "event": "meta",
                "session_id": session_id,
                "language": lang_code,
                "translated": translated,
                "sources": [s.model_dump() for s in sources],
            }
            yield f"data: {json.dumps(meta)}\n\n"

            try:
                async for token in llm.generate_stream(
                    prompt=full_prompt,
                    system_prompt=system_prompt,
                    history=history,
                ):
                    collected_response.append(token)
                    yield f"data: {json.dumps({'event': 'token', 'token': token})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
                return

            full_response = "".join(collected_response)
            _update_history(session_id, message, full_response)
            yield f"data: {json.dumps({'event': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    llm = get_llm_service()
    from app.services.llm_service import OllamaService
    connected = False
    model_loaded = False
    if isinstance(llm, OllamaService):
        connected = await llm.is_connected()
        if connected:
            model_loaded = await llm.is_model_loaded()

    doc_count = rag_service.get_document_count()
    status = "ok" if connected else "degraded"

    return HealthResponse(
        status=status,
        ollama_connected=connected,
        model_loaded=model_loaded,
        documents_count=doc_count,
    )


@router.get("/sources", response_model=SourcesResponse)
async def list_sources():
    sources = rag_service.get_available_sources()
    return SourcesResponse(sources=sources, total_documents=rag_service.get_document_count())


@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    if not settings.SARVAM_API_KEY:
        raise HTTPException(status_code=503, detail="Sarvam API key not configured")
    if not _sarvam_available:
        raise HTTPException(status_code=503, detail="sarvamai SDK not installed. Run: pip install sarvamai")

    SARVAM_LANG_MAP = {
        "en": "en-IN", "hi": "hi-IN", "sa": "hi-IN", "bn": "bn-IN",
        "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN", "or": "od-IN",
        "pa": "pa-IN", "ta": "ta-IN", "te": "te-IN", "gu": "gu-IN",
    }
    lang_code = SARVAM_LANG_MAP.get(request.language, "en-IN")

    SPEAKER_MAP = {
        "hi-IN": "arvind", "en-IN": "arvind", "mr-IN": "arvind",
        "pa-IN": "arvind", "gu-IN": "arvind",
    }
    speaker = SPEAKER_MAP.get(lang_code, "meera")

    import re
    clean = request.text
    clean = re.sub(r'\|', ' ', clean)           # markdown table pipes
    clean = re.sub(r'[-]{3,}', ' ', clean)      # horizontal rules
    clean = re.sub(r'[#*`_~>]', '', clean)      # markdown symbols
    clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)  # links → text
    clean = re.sub(r'\s+', ' ', clean).strip()
    if not clean:
        raise HTTPException(status_code=400, detail="Text is empty after cleaning")
    clean = clean[:500]

    def _call_sarvam():
        client = SarvamAI(api_subscription_key=settings.SARVAM_API_KEY)
        return client.text_to_speech.convert(
            text=clean,
            target_language_code=lang_code,
            speaker=speaker,
            model="bulbul:v1",
            enable_preprocessing=True,
        )

    try:
        response = await asyncio.to_thread(_call_sarvam)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Sarvam TTS error: {e}")

    audios = getattr(response, "audios", None) or (response.get("audios") if isinstance(response, dict) else None)
    if not audios:
        raise HTTPException(status_code=502, detail="No audio returned from Sarvam")

    return {"audio": audios[0]}


@router.post("/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(request: RecommendationsRequest):
    llm = get_llm_service()
    prompt = (
        f"Based on this question about Vedic knowledge: \"{request.message}\"\n"
        f"Generate exactly 3 short, related follow-up questions a curious learner might ask next.\n"
        f"Return only the 3 questions, one per line, no numbering, no extra text."
    )
    system = "You are a Vedic knowledge assistant. Generate concise, relevant follow-up questions."
    try:
        result = await llm.generate(prompt=prompt, system_prompt=system)
        questions = [q.strip() for q in result.strip().split("\n") if q.strip()][:3]
        return RecommendationsResponse(questions=questions)
    except Exception:
        return RecommendationsResponse(questions=[])
