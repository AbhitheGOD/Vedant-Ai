from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routers.chat import router as chat_router
from app.services.rag_service import rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load embedding model and initialize ChromaDB
    print("Starting Vedic Knowledge Chatbot...")
    rag_service.initialize()
    print("Ready. Visit http://localhost:8000/docs for API documentation.")
    yield
    # Shutdown (nothing to clean up for now)
    print("Shutting down...")


app = FastAPI(
    title="Vedic Knowledge Chatbot",
    description=(
        "A multilingual chatbot powered by RAG and LLM, "
        "providing answers about Vedic/Sanskrit knowledge across multiple languages."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="", tags=["Chat"])

# Serve frontend
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join(static_dir, "index.html"))
