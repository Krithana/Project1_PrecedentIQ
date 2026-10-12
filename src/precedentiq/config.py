from pathlib import Path
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data" / "processed"
    CHUNKS_PATH: Path = DATA_DIR / "chunks.jsonl"
    JUDGMENTS_PATH: Path = DATA_DIR / "judgments.jsonl"
    MANIFEST_PATH: Path = DATA_DIR / "qdrant_manifest.json"
    VECTOR_CACHE_DIR: Path = BASE_DIR / "data" / "cache"

    # API Settings
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    DEBUG: bool = False

    @field_validator("DEBUG", mode="before")
    @classmethod
    def normalize_debug_value(cls, value):
        """Accept common deployment labels used by local launch scripts."""
        if isinstance(value, str) and value.strip().lower() in {"release", "production", "prod"}:
            return False
        return value

    # Qdrant Vector DB Settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "precedentiq_chunks"
    USE_QDRANT: bool = False  # Falls back to in-memory/cache dense retriever if False

    # Dense Embedding Settings
    DENSE_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # LLM / Ollama Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"  # 2.0GB fast model (installed on user machine)
    OLLAMA_TIMEOUT_SECONDS: int = 25   # Fast failover to deterministic engine if CPU is busy
    USE_OLLAMA_FALLBACK: bool = True

    # Retrieval Scoring & Hierarchy Weights
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 20
    BM25_WEIGHT: float = 0.5
    DENSE_WEIGHT: float = 0.5
    RRF_K: int = 60
    SUPREME_COURT_WEIGHT: float = 1.2  # Article 141 stare decisis boost
    HIGH_COURT_WEIGHT: float = 1.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_directories(self) -> None:
        """Create necessary cache/data directories if they don't exist."""
        self.VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
