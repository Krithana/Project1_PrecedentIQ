from fastapi import APIRouter
from typing import Dict, Any
from ...config import settings
from ...data.loader import DataLoader
from ...retrieval.qdrant_store import QdrantStore
from ...ai.ollama_client import OllamaClient

router = APIRouter(prefix="/health", tags=["System & Health"])


@router.get("", response_model=Dict[str, Any])
def health_check():
    """Returns application status, corpus metrics, and service availability."""
    loader = DataLoader.get_instance()
    stats = loader.get_corpus_statistics()
    qdrant = QdrantStore()
    ollama = OllamaClient()

    ollama_server_active = ollama.is_available()
    ollama_model_available = ollama.is_model_available() if ollama_server_active else False

    return {
        "status": "online",
        "service": "PrecedentIQ Legal RAG Backend",
        "version": "2.0.0",
        "corpus": {
            "total_chunks": stats["total_chunks"],
            "total_judgments": stats["total_judgments"],
            "courts": stats["courts"],
        },
        "services": {
            "qdrant_active": qdrant.is_available(),
            "ollama_active": ollama_server_active and ollama_model_available,
            "ollama_server_active": ollama_server_active,
            "ollama_model_available": ollama_model_available,
            "ollama_models": ollama.list_models() if ollama_server_active else [],
        },
        "embedding_model": settings.DENSE_EMBEDDING_MODEL,
        "llm_model": settings.OLLAMA_MODEL
    }
