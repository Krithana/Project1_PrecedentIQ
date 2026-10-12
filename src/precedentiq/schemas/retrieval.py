from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .judgment import JudgmentChunk


class FilterParams(BaseModel):
    court: Optional[str] = None  # supreme_court, high_court, or None for all
    court_bench: Optional[str] = None  # e.g., mphc_db_gwl
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    sections: Optional[List[str]] = None  # e.g., ["ratio_decidendi", "order", "reasoning"]


class SearchQuery(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[FilterParams] = None
    bm25_weight: float = 0.5
    dense_weight: float = 0.5


class ScoredChunk(BaseModel):
    chunk: JudgmentChunk
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    rrf_score: float = 0.0
    court_boost: float = 1.0
    final_score: float = 0.0
    matched_terms: List[str] = Field(default_factory=list)


class HybridSearchResult(BaseModel):
    query: str
    scored_chunks: List[ScoredChunk]
    total_candidates_evaluated: int = 0
    execution_time_ms: float = 0.0
