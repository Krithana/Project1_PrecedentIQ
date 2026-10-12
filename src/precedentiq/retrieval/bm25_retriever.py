import re
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from rank_bm25 import BM25Okapi
from ..config import settings
from ..schemas.judgment import JudgmentChunk
from ..schemas.retrieval import FilterParams

logger = logging.getLogger(__name__)


class LegalTokenizer:
    """Legal-specific tokenizer for Indian case law matching."""

    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "from", "up", "down", "in", "out", "on", "off",
        "over", "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
    }

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        # Normalize statutory references
        t = re.sub(r"(?i)\b(?:section|sec\.?|s\.)\s*(\d+[a-z]?)", r"sec_\1", text)
        t = re.sub(r"(?i)\b(?:article|art\.?)\s*(\d+[a-z]?)", r"art_\1", t)
        t = re.sub(r"(?i)\border\s*([ivxlcdm\d]+)\s*rule\s*(\d+)", r"order_\1_rule_\2", t)

        tokens = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", t.lower())
        return [tok for tok in tokens if tok not in cls.STOPWORDS]


class BM25Retriever:
    """BM25 Sparse Keyword Retriever over pre-processed legal chunks."""

    def __init__(self, chunks: Optional[List[JudgmentChunk]] = None):
        self.chunks: List[JudgmentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.cache_path = settings.VECTOR_CACHE_DIR / "bm25_corpus.pkl"

        if chunks:
            self.build_index(chunks)

    def build_index(self, chunks: List[JudgmentChunk], use_cache: bool = True) -> None:
        """Build or load BM25Okapi index."""
        self.chunks = chunks
        if use_cache and self.cache_path.exists():
            try:
                logger.info(f"Loading cached BM25 tokens from {self.cache_path}...")
                with open(self.cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                    if len(cached_data) == len(chunks):
                        self.tokenized_corpus = cached_data
                        self.bm25 = BM25Okapi(self.tokenized_corpus)
                        logger.info("BM25 index restored from cache.")
                        return
            except Exception as e:
                logger.warning(f"Could not load BM25 cache: {e}. Rebuilding index...")

        logger.info(f"Tokenizing {len(chunks)} chunks for BM25...")
        self.tokenized_corpus = [LegalTokenizer.tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        try:
            settings.VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.tokenized_corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("BM25 token cache saved.")
        except Exception as e:
            logger.warning(f"Failed to write BM25 cache: {e}")

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[FilterParams] = None
    ) -> List[Tuple[JudgmentChunk, float, int]]:
        """
        Search corpus using BM25.
        Returns List of (chunk, score, rank).
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = LegalTokenizer.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        
        # Candidate pairs (index, score)
        query_token_set = set(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: (
                len(query_token_set.intersection(self.tokenized_corpus[i]))
                if scores[i] <= 0 else 0,
                float(scores[i]),
            ),
            reverse=True,
        )

        results: List[Tuple[JudgmentChunk, float, int]] = []
        rank = 1

        # For very small corpora BM25's IDF can be zero/negative even when a
        # document has the requested terms. Keep lexical-overlap candidates so
        # search remains useful in tests and small deployments.
        for idx in ranked_indices:
            score = float(scores[idx])
            chunk = self.chunks[idx]
            overlap = len(query_token_set.intersection(self.tokenized_corpus[idx]))
            if score <= 0 and overlap == 0:
                continue

            # Apply metadata filters
            if filters:
                if filters.court and chunk.court.lower() != filters.court.lower():
                    continue
                if filters.sections and chunk.section.lower() not in [s.lower() for s in filters.sections]:
                    continue
                if filters.court_bench and chunk.metadata.court_bench:
                    if filters.court_bench.lower() not in chunk.metadata.court_bench.lower():
                        continue
                if filters.year_from or filters.year_to:
                    try:
                        chunk_year = int(chunk.metadata.source_year) if chunk.metadata.source_year else None
                        if chunk_year:
                            if filters.year_from and chunk_year < filters.year_from:
                                continue
                            if filters.year_to and chunk_year > filters.year_to:
                                continue
                    except (ValueError, TypeError):
                        pass

            results.append((chunk, score, rank))
            rank += 1
            if len(results) >= top_k:
                break

        return results
