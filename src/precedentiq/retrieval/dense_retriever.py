import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from ..config import settings
from ..schemas.judgment import JudgmentChunk
from ..schemas.retrieval import FilterParams

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Dense Semantic Embedding Retriever using Sentence-Transformers
    with local disk caching for fast cold-starts.
    """

    def __init__(self, chunks: Optional[List[JudgmentChunk]] = None):
        self.model_name = settings.DENSE_EMBEDDING_MODEL
        self.model = None
        self.chunks: List[JudgmentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.cache_dir = settings.VECTOR_CACHE_DIR
        self.embeddings_cache_path = self.cache_dir / "dense_embeddings.npy"
        self.ids_cache_path = self.cache_dir / "dense_chunk_ids.pkl"

        if chunks:
            self.build_index(chunks)

    def _lazy_load_model(self):
        """Lazy load SentenceTransformer model to avoid startup delays."""
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

                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading dense embedding model: {self.model_name}...")
                
                # 1. Try loading from local Hugging Face cache first (instant, zero network calls)
                try:
                    self.model = SentenceTransformer(self.model_name, local_files_only=True)
                    logger.info("Dense model loaded instantly from local cache.")
                    return
                except Exception:
                    pass

                # 2. If not in local cache, download with unverified SSL
                self.model = SentenceTransformer(self.model_name)
                logger.info("Dense model loaded successfully.")
            except ImportError:
                logger.warning("sentence-transformers not installed. Dense retrieval disabled.")
            except Exception as e:
                logger.warning(f"Error loading embedding model {self.model_name}: {e}. Dense retrieval will gracefully fallback to BM25.")
                self.model = None

    def build_index(self, chunks: List[JudgmentChunk], use_cache: bool = True) -> None:
        """Build or restore dense embeddings index."""
        self.chunks = chunks
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if use_cache and self.embeddings_cache_path.exists() and self.ids_cache_path.exists():
            try:
                logger.info("Loading precomputed dense embeddings from cache...")
                with open(self.ids_cache_path, "rb") as f:
                    cached_ids = pickle.load(f)
                loaded_embeddings = np.load(self.embeddings_cache_path)

                if len(cached_ids) == len(chunks):
                    self.embeddings = loaded_embeddings
                    logger.info(f"Loaded {len(self.embeddings)} dense vectors from cache.")
                    return
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}. Computing embeddings...")

        self._lazy_load_model()
        if self.model is None:
            logger.warning("Dense retriever initialized without active model or cached embeddings.")
            return

        logger.info(f"Encoding {len(chunks)} chunks using {self.model_name}...")
        texts = [c.text for c in chunks]
        # Encode in batches to keep memory bounded
        self.embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        try:
            np.save(self.embeddings_cache_path, self.embeddings)
            chunk_ids = [c.chunk_id for c in chunks]
            with open(self.ids_cache_path, "wb") as f:
                pickle.dump(chunk_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Saved computed embeddings to local cache.")
        except Exception as e:
            logger.warning(f"Could not write embedding cache: {e}")

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[FilterParams] = None
    ) -> List[Tuple[JudgmentChunk, float, int]]:
        """
        Cosine similarity search over dense chunk embeddings.
        Returns List of (chunk, score, rank).
        """
        if self.embeddings is None:
            return []

        self._lazy_load_model()
        if self.model is None:
            return []

        # Encode query
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        # Cosine similarity is dot product when normalized
        scores = np.dot(self.embeddings, query_vec)

        ranked_indices = np.argsort(-scores)

        results: List[Tuple[JudgmentChunk, float, int]] = []
        rank = 1

        for idx in ranked_indices:
            score = float(scores[idx])
            chunk = self.chunks[idx]

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
