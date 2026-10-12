import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Any
from ..config import settings
from ..schemas.judgment import JudgmentChunk, JudgmentRecord, ChunkMetadata

logger = logging.getLogger(__name__)


class DataLoader:
    """
    High-performance data loader for PrecedentIQ processed corpus:
    - 1,989 judgments (judgments.jsonl)
    - 11,720 legal chunks (chunks.jsonl)
    """

    _instance: Optional["DataLoader"] = None

    def __init__(
        self,
        chunks_path: Optional[Path] = None,
        judgments_path: Optional[Path] = None
    ):
        self.chunks_path = chunks_path or settings.CHUNKS_PATH
        self.judgments_path = judgments_path or settings.JUDGMENTS_PATH
        self._chunks: Optional[List[JudgmentChunk]] = None
        self._chunk_map: Dict[str, JudgmentChunk] = {}
        self._judgment_map: Dict[str, JudgmentRecord] = {}
        self._judgment_chunks_map: Dict[str, List[JudgmentChunk]] = {}

    @classmethod
    def get_instance(cls) -> "DataLoader":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def iter_chunks(self) -> Iterator[JudgmentChunk]:
        """Stream chunks from disk one by one to preserve memory."""
        if not self.chunks_path.exists():
            logger.error(f"Chunks file not found at {self.chunks_path}")
            return

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    meta = data.get("metadata", {})
                    metadata_obj = ChunkMetadata(
                        source_file=meta.get("source_file"),
                        source_year=str(meta.get("source_year")) if meta.get("source_year") else None,
                        treatment_signals=meta.get("treatment_signals", []),
                        chunk_method=meta.get("chunk_method", "heading"),
                        court_bench=meta.get("court_bench")
                    )
                    yield JudgmentChunk(
                        chunk_id=str(data["chunk_id"]),
                        judgment_id=str(data["judgment_id"]),
                        citation=str(data.get("citation", data["judgment_id"])),
                        court=str(data.get("court", "high_court")),
                        section=str(data.get("section", "full_text")),
                        text=str(data.get("text", "")),
                        metadata=metadata_obj
                    )
                except Exception as e:
                    logger.warning(f"Error parsing chunk row: {e}")

    def load_all_chunks(self, force_reload: bool = False) -> List[JudgmentChunk]:
        """Load all chunks into memory and build lookup indices."""
        if self._chunks is not None and not force_reload:
            return self._chunks

        logger.info(f"Loading chunks from {self.chunks_path}...")
        chunks = []
        chunk_map = {}
        judgment_chunks_map = {}

        for chunk in self.iter_chunks():
            chunks.append(chunk)
            chunk_map[chunk.chunk_id] = chunk
            if chunk.judgment_id not in judgment_chunks_map:
                judgment_chunks_map[chunk.judgment_id] = []
            judgment_chunks_map[chunk.judgment_id].append(chunk)

        self._chunks = chunks
        self._chunk_map = chunk_map
        self._judgment_chunks_map = judgment_chunks_map
        logger.info(f"Loaded {len(chunks)} chunks across {len(judgment_chunks_map)} judgments.")
        return self._chunks

    def get_chunk(self, chunk_id: str) -> Optional[JudgmentChunk]:
        """Get a single chunk by chunk_id."""
        if not self._chunk_map:
            self.load_all_chunks()
        return self._chunk_map.get(chunk_id)

    def get_chunks_for_judgment(self, judgment_id: str) -> List[JudgmentChunk]:
        """Get all chunks associated with a specific judgment."""
        if not self._judgment_chunks_map:
            self.load_all_chunks()
        return self._judgment_chunks_map.get(judgment_id, [])

    def load_judgment(self, judgment_id: str) -> Optional[JudgmentRecord]:
        """Look up a judgment by judgment_id."""
        if judgment_id in self._judgment_map:
            return self._judgment_map[judgment_id]

        if not self.judgments_path.exists():
            return None

        with open(self.judgments_path, "r", encoding="utf-8") as f:
            for line in f:
                if judgment_id in line:
                    try:
                        data = json.loads(line)
                        if data.get("judgment_id") == judgment_id:
                            record = JudgmentRecord(
                                judgment_id=data["judgment_id"],
                                court=data.get("court", "high_court"),
                                source_group=data.get("source_group"),
                                source_bench=data.get("source_bench"),
                                source_year=str(data.get("source_year")) if data.get("source_year") else None,
                                judgment_date=data.get("judgment_date"),
                                source_file=data.get("source_file"),
                                page_count=data.get("page_count", 0),
                                character_count=data.get("character_count", 0),
                                text=data.get("text", "")
                            )
                            self._judgment_map[judgment_id] = record
                            return record
                    except Exception:
                        pass
        return None

    def get_corpus_statistics(self) -> Dict[str, Any]:
        """Compute corpus breakdown statistics."""
        chunks = self.load_all_chunks()
        court_counts: Dict[str, int] = {}
        section_counts: Dict[str, int] = {}
        treatment_counts: Dict[str, int] = {}

        for c in chunks:
            court_counts[c.court] = court_counts.get(c.court, 0) + 1
            section_counts[c.section] = section_counts.get(c.section, 0) + 1
            for sig in c.metadata.treatment_signals:
                treatment_counts[sig] = treatment_counts.get(sig, 0) + 1

        return {
            "total_chunks": len(chunks),
            "total_judgments": len(self._judgment_chunks_map),
            "courts": court_counts,
            "sections": section_counts,
            "treatment_signals": treatment_counts,
        }
