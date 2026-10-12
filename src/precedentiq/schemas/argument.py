from typing import List, Optional
from pydantic import BaseModel, Field
from .analysis import HoldingExtraction
from .retrieval import ScoredChunk


class CitationSource(BaseModel):
    chunk_id: str
    judgment_id: str
    citation: str
    court: str
    section: str
    excerpt: str = Field(
        ...,
        description="Exact verbatim excerpt from the judgment chunk supporting the claim"
    )


class ArgumentPoint(BaseModel):
    title: str
    legal_proposition: str
    strength: str = Field(default="Strong", description="'Strong', 'Moderate', or 'Persuasive'")
    supporting_citations: List[CitationSource] = Field(default_factory=list)
    applicability_reasoning: str


class CounterArgumentPoint(BaseModel):
    potential_objection: str
    adverse_citations: List[CitationSource] = Field(default_factory=list)
    rebuttal_or_distinction: str = Field(
        ...,
        description="Grounds on facts or law to distinguish this adverse precedent"
    )


class ProContraOutline(BaseModel):
    issue_summary: str
    client_context_summary: Optional[str] = None
    supporting_arguments: List[ArgumentPoint] = Field(
        default_factory=list,
        description="Affirmative legal arguments supporting the client's proposition"
    )
    counter_arguments: List[CounterArgumentPoint] = Field(
        default_factory=list,
        description="Anticipated opposing counsel contentions and factual distinctions"
    )
    strategic_advice: str = Field(
        default="",
        description="Practical procedural or litigation counsel for the advocate"
    )


class LegalResearchBrief(BaseModel):
    query: str
    case_file_name: Optional[str] = None
    timestamp: str
    total_precedents_analyzed: int
    top_precedents: List[ScoredChunk] = Field(default_factory=list)
    holdings: List[HoldingExtraction] = Field(default_factory=list)
    argument_outline: ProContraOutline
