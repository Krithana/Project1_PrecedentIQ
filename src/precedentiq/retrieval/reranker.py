import logging
import re
from typing import List, Tuple, Optional
from ..schemas.judgment import JudgmentChunk
from ..schemas.retrieval import ScoredChunk

logger = logging.getLogger(__name__)


class LegalReranker:
    """
    Reranks candidate legal precedents using either Cross-Encoder
    or high-precision domain-specific lexical-semantic scoring.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None

    def _lazy_load(self):
        if self.model is None:
            try:
                import os
                import ssl
                try:
                    ssl._create_default_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass
                os.environ["CURL_CA_BUNDLE"] = ""
                os.environ["PYTHONHTTPSVERIFY"] = "0"
                os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

                from sentence_transformers import CrossEncoder
                try:
                    self.model = CrossEncoder(self.model_name, local_files_only=True, max_length=512)
                    logger.info(f"Loaded Cross-Encoder model from local cache: {self.model_name}")
                    return
                except Exception:
                    pass

                self.model = CrossEncoder(self.model_name, max_length=512)
                logger.info(f"Loaded Cross-Encoder model: {self.model_name}")
            except Exception as e:
                logger.debug(f"Cross-Encoder model unavailable ({e}). Using heuristic legal reranker.")
                self.model = None

    def rerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        top_k: int = 10
    ) -> List[ScoredChunk]:
        """Rerank candidates based on contextual relevance and judicial weight."""
        if not candidates:
            return []

        self._lazy_load()

        if self.model is not None:
            try:
                pairs = [[query, c.chunk.text[:1000]] for c in candidates]
                scores = self.model.predict(pairs)
                for i, score in enumerate(scores):
                    # Combine original fused score with cross-encoder score
                    candidates[i].final_score = float(score)
                candidates.sort(key=lambda x: x.final_score, reverse=True)
                return candidates[:top_k]
            except Exception as e:
                logger.warning(f"Cross-Encoder inference error: {e}. Falling back to domain reranker.")

        # Domain-specific heuristic reranking
        query_terms = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", query.lower()))

        for item in candidates:
            c = item.chunk
            text_lower = c.text.lower()

            # 1. Section importance boost
            section_boost = 1.0
            if c.section in ["ratio_decidendi", "reasoning"]:
                section_boost = 1.3
            elif c.section == "order":
                section_boost = 1.15
            elif c.section == "arguments":
                section_boost = 0.95

            # 2. Treatment signals boost
            treatment_boost = 1.0
            signals = [s.lower() for s in c.metadata.treatment_signals]
            if "followed" in signals or "approved" in signals:
                treatment_boost = 1.2
            elif "reversed" in signals or "overruled" in signals:
                treatment_boost = 0.85

            # 3. Exact query phrase matching
            phrase_boost = 1.0
            if query.lower() in text_lower:
                phrase_boost = 1.25

            # Term overlap count
            matched = [t for t in query_terms if t in text_lower]
            item.matched_terms = matched
            overlap_ratio = len(matched) / max(len(query_terms), 1)

            # Re-scale final score
            item.final_score = (
                item.rrf_score * item.court_boost * section_boost * treatment_boost * phrase_boost * (1.0 + 0.5 * overlap_ratio)
            )

        candidates.sort(key=lambda x: x.final_score, reverse=True)
        return candidates[:top_k]
