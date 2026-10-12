from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class LegalSectionEnum(str, Enum):
    FACTS = "facts"
    ISSUES = "issues"
    ARGUMENTS = "arguments"
    REASONING = "reasoning"
    RATIO_DECIDENDI = "ratio_decidendi"
    OBITER_DICTA = "obiter_dicta"
    ORDER = "order"
    FULL_TEXT = "full_text"
    FALLBACK = "fallback"


class CourtLevelEnum(str, Enum):
    SUPREME_COURT = "supreme_court"
    HIGH_COURT = "high_court"


class TreatmentSignalEnum(str, Enum):
    REVERSED = "reversed"
    FOLLOWED = "followed"
    APPROVED = "approved"
    OVERRULED = "overruled"
    DISTINGUISHED = "distinguished"


class ChunkMetadata(BaseModel):
    source_file: Optional[str] = None
    source_year: Optional[str] = None
    treatment_signals: List[str] = Field(default_factory=list)
    chunk_method: Optional[str] = "heading"
    court_bench: Optional[str] = None


class JudgmentChunk(BaseModel):
    chunk_id: str
    judgment_id: str
    citation: str
    court: str  # supreme_court or high_court
    section: str  # e.g., order, arguments, full_text, etc.
    text: str
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)


class JudgmentRecord(BaseModel):
    judgment_id: str
    court: str
    source_group: Optional[str] = None
    source_bench: Optional[str] = None
    source_year: Optional[str] = None
    judgment_date: Optional[str] = None
    source_file: Optional[str] = None
    page_count: Optional[int] = 0
    character_count: Optional[int] = 0
    text: str
