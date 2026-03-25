from pydantic import BaseModel, Field
from typing import Optional, List


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message in any language")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation continuity")


class SourceInfo(BaseModel):
    text: str
    source: str
    chapter: str = ""
    verse: str = ""
    relevance_score: float = 0.0


class ChatResponse(BaseModel):
    response: str
    language: str
    sources: List[SourceInfo]
    session_id: str


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    model_loaded: bool
    documents_count: int


class SourcesResponse(BaseModel):
    sources: List[str]
    total_documents: int


class RecommendationsRequest(BaseModel):
    message: str
    response: str = ""
    language: str = "en"


class RecommendationsResponse(BaseModel):
    questions: List[str]


class TTSRequest(BaseModel):
    text: str
    language: str = "en"
