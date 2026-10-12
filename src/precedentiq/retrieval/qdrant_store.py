import logging
from typing import List, Tuple, Optional, Dict, Any
from ..config import settings
from ..schemas.judgment import JudgmentChunk
from ..schemas.retrieval import FilterParams

logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Wrapper for Qdrant Vector Database interaction, supporting payload
    indexing and metadata-filtered nearest neighbor search.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection: Optional[str] = None
    ):
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.collection_name = collection or settings.QDRANT_COLLECTION
        self._client = None

    def get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(host=self.host, port=self.port, timeout=5)
            except Exception as e:
                logger.debug(f"Qdrant client initialization error: {e}")
                self._client = None
        return self._client

    def is_available(self) -> bool:
        """Check if Qdrant server is reachable."""
        try:
            client = self.get_client()
            if client is None:
                return False
            client.get_collections()
            return True
        except Exception:
            return False

    def init_collection(self, dimension: int = 384) -> bool:
        """Ensure the target collection exists."""
        try:
            from qdrant_client.http import models
            client = self.get_client()
            if not client:
                return False

            collections = [c.name for c in client.get_collections().collections]
            if self.collection_name not in collections:
                logger.info(f"Creating Qdrant collection: {self.collection_name} (dim={dimension})...")
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=dimension,
                        distance=models.Distance.COSINE
                    )
                )
            return True
        except Exception as e:
            logger.warning(f"Error creating Qdrant collection: {e}")
            return False

    def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filters: Optional[FilterParams] = None,
        chunk_map: Optional[Dict[str, JudgmentChunk]] = None
    ) -> List[Tuple[JudgmentChunk, float, int]]:
        """Search Qdrant for top-K nearest chunks."""
        client = self.get_client()
        if not client or not chunk_map:
            return []

        try:
            from qdrant_client.http import models
            must_clauses = []
            if filters:
                if filters.court:
                    must_clauses.append(models.FieldCondition(
                        key="court",
                        match=models.MatchValue(value=filters.court.lower())
                    ))
                if filters.sections:
                    must_clauses.append(models.FieldCondition(
                        key="section",
                        match=models.MatchAny(any=[s.lower() for s in filters.sections])
                    ))

            q_filter = models.Filter(must=must_clauses) if must_clauses else None

            search_results = client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=q_filter,
                limit=top_k
            )

            results: List[Tuple[JudgmentChunk, float, int]] = []
            for rank, hit in enumerate(search_results, start=1):
                chunk_id = str(hit.payload.get("chunk_id"))
                if chunk_id in chunk_map:
                    results.append((chunk_map[chunk_id], float(hit.score), rank))

            return results
        except Exception as e:
            logger.warning(f"Qdrant search error: {e}")
            return []
