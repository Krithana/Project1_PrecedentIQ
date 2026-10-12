import sys
import logging
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.precedentiq.config import settings
from src.precedentiq.data.loader import DataLoader
from src.precedentiq.retrieval.qdrant_store import QdrantStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("index_qdrant")


def index_chunks_to_qdrant(batch_size: int = 64):
    store = QdrantStore()
    client = store.get_client()
    if not client:
        logger.error("Could not connect to Qdrant. Make sure Qdrant is running (e.g., docker compose up -d qdrant).")
        return

    logger.info(f"Connecting to Qdrant at {store.host}:{store.port}...")
    store.init_collection(dimension=settings.EMBEDDING_DIMENSION)

    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client.http import models
    except ImportError as e:
        logger.error(f"Required package missing: {e}. Please pip install sentence-transformers qdrant-client.")
        return

    logger.info(f"Loading embedding model: {settings.DENSE_EMBEDDING_MODEL}...")
    model = SentenceTransformer(settings.DENSE_EMBEDDING_MODEL)

    loader = DataLoader.get_instance()
    chunks = loader.load_all_chunks()
    logger.info(f"Total chunks to index: {len(chunks)}")

    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing to Qdrant", total=total_batches):
        batch = chunks[i:i + batch_size]
        texts = [c.text for c in batch]
        vectors = model.encode(texts, normalize_embeddings=True).tolist()

        points = []
        for idx, (chunk, vec) in enumerate(zip(batch, vectors)):
            point_id = i + idx
            payload = {
                "chunk_id": chunk.chunk_id,
                "judgment_id": chunk.judgment_id,
                "citation": chunk.citation,
                "court": chunk.court,
                "section": chunk.section,
                "source_year": chunk.metadata.source_year,
                "court_bench": chunk.metadata.court_bench,
                "treatment_signals": chunk.metadata.treatment_signals,
                "text": chunk.text
            }
            points.append(models.PointStruct(id=point_id, vector=vec, payload=payload))

        client.upsert(
            collection_name=store.collection_name,
            points=points
        )

    col_info = client.get_collection(collection_name=store.collection_name)
    logger.info(f"Successfully indexed {col_info.points_count} points into '{store.collection_name}'.")


if __name__ == "__main__":
    index_chunks_to_qdrant()
