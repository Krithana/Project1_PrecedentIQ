import time
import logging
from typing import List, Dict, Optional, Any
from ..config import settings
from ..data.loader import DataLoader
from ..schemas.judgment import JudgmentChunk
from ..schemas.retrieval import SearchQuery, ScoredChunk, HybridSearchResult, FilterParams
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .qdrant_store import QdrantStore
from .reranker import LegalReranker

logger = logging.getLogger(__name__)


class HybridRetrievalEngine:
    """
    Orchestrates Dense + Sparse (BM25) hybrid legal retrieval
    with Reciprocal Rank Fusion (RRF), Court Hierarchy Weighting,
    and Contextual Reranking.
    """

    _instance: Optional["HybridRetrievalEngine"] = None

    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.loader = data_loader or DataLoader.get_instance()
        self.chunks = self.loader.load_all_chunks()
        self.bm25_retriever = BM25Retriever(self.chunks)
        self.dense_retriever = DenseRetriever(self.chunks)
        self.qdrant_store = QdrantStore()
        self.reranker = LegalReranker()

    @classmethod
    def get_instance(cls) -> "HybridRetrievalEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def search(self, search_query: SearchQuery) -> HybridSearchResult:
        """Execute hybrid search pipeline."""
        start_time = time.perf_counter()
        query = search_query.query
        top_k = search_query.top_k
        filters = search_query.filters

        # 1. Retrieve BM25 candidates
        candidate_pool_size = max(top_k * 4, 30)
        bm25_hits = self.bm25_retriever.search(query, top_k=candidate_pool_size, filters=filters)

        # 2. Retrieve Dense candidates (prefer Qdrant if running, else local dense)
        dense_hits = []
        if settings.USE_QDRANT and self.qdrant_store.is_available():
            try:
                self.dense_retriever._lazy_load_model()
                if self.dense_retriever.model:
                    q_vec = self.dense_retriever.model.encode([query], normalize_embeddings=True)[0].tolist()
                    dense_hits = self.qdrant_store.search(
                        q_vec,
                        top_k=candidate_pool_size,
                        filters=filters,
                        chunk_map=self.loader._chunk_map
                    )
            except Exception as e:
                logger.warning(f"Qdrant query failed: {e}. Using local dense cache.")

        if not dense_hits:
            dense_hits = self.dense_retriever.search(query, top_k=candidate_pool_size, filters=filters)

        # 3. Reciprocal Rank Fusion (RRF)
        scored_dict: Dict[str, ScoredChunk] = {}
        rrf_k = settings.RRF_K
        bm25_weight = search_query.bm25_weight
        dense_weight = search_query.dense_weight

        for chunk, score, rank in bm25_hits:
            cid = chunk.chunk_id
            if cid not in scored_dict:
                court_boost = (
                    settings.SUPREME_COURT_WEIGHT
                    if chunk.court == "supreme_court"
                    else settings.HIGH_COURT_WEIGHT
                )
                scored_dict[cid] = ScoredChunk(
                    chunk=chunk,
                    court_boost=court_boost
                )
            scored_dict[cid].bm25_rank = rank
            scored_dict[cid].bm25_score = score
            scored_dict[cid].rrf_score += bm25_weight * (1.0 / (rrf_k + rank))

        for chunk, score, rank in dense_hits:
            cid = chunk.chunk_id
            if cid not in scored_dict:
                court_boost = (
                    settings.SUPREME_COURT_WEIGHT
                    if chunk.court == "supreme_court"
                    else settings.HIGH_COURT_WEIGHT
                )
                scored_dict[cid] = ScoredChunk(
                    chunk=chunk,
                    court_boost=court_boost
                )
            scored_dict[cid].dense_rank = rank
            scored_dict[cid].dense_score = score
            scored_dict[cid].rrf_score += dense_weight * (1.0 / (rrf_k + rank))

        # Initial sort by RRF * court boost
        for item in scored_dict.values():
            item.final_score = item.rrf_score * item.court_boost

        fused_candidates = sorted(scored_dict.values(), key=lambda x: x.final_score, reverse=True)

        # 4. Contextual Reranking of top candidates
        top_candidates = fused_candidates[:candidate_pool_size]
        reranked_chunks = self.reranker.rerank(query, top_candidates, top_k=top_k)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return HybridSearchResult(
            query=query,
            scored_chunks=reranked_chunks,
            total_candidates_evaluated=len(scored_dict),
            execution_time_ms=round(elapsed_ms, 2)
        )
