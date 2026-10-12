import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.health import router as health_router
from .routes.research import router as research_router
from ..config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("precedentiq.api")


def create_app() -> FastAPI:
    app = FastAPI(
        title="PrecedentIQ API",
        description="A Hybrid Retrieval-Augmented Framework for Precedent Discovery and Argument Synthesis over Indian Case Law",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Allow CORS for local dev and frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routers
    app.include_router(health_router)
    app.include_router(research_router)

    @app.get("/")
    def root():
        return {
            "message": "Welcome to PrecedentIQ API",
            "documentation": "/docs",
            "health": "/health",
            "jurisdiction": "Supreme Court & High Courts of India",
            "chunks_indexed": 11720
        }

    @app.on_event("startup")
    async def startup_event():
        logger.info("PrecedentIQ API initializing...")
        # Pre-load data loader in background
        from ..data.loader import DataLoader
        loader = DataLoader.get_instance()
        stats = loader.get_corpus_statistics()
        logger.info(f"Corpus initialized with {stats['total_chunks']} chunks from {stats['total_judgments']} judgments.")

    return app


app = create_app()
