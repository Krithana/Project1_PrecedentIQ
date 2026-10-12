from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HoldingExtraction(BaseModel):
    judgment_id: str
    case_title: Optional[str] = None
    court: str  # supreme_court or high_court
    bench: Optional[str] = None
    judgment_date: Optional[str] = None
    material_facts: List[str] = Field(
        default_factory=list,
        description="Key background facts established in the case"
    )
    legal_issues: List[str] = Field(
        default_factory=list,
        description="Core questions of law framed or addressed by the court"
    )
    ratio_decidendi: str = Field(
        ...,
        description="The binding legal principle or rule of law declared by the court"
    )
    ratio_evidence: List[str] = Field(default_factory=list, description="Exact source excerpts supporting the candidate ratio")
    ratio_paragraphs: List[str] = Field(default_factory=list, description="Paragraph references supporting the candidate ratio")
    obiter_dicta: Optional[str] = Field(
        default=None,
        description="Non-binding persuasive judicial observations or commentary"
    )
    classification_note: Optional[str] = None
    final_order: Optional[str] = Field(
        default=None,
        description="The operative disposition (allowed, dismissed, remanded, quashed)"
    )
    order_evidence: List[str] = Field(default_factory=list, description="Exact source excerpts supporting the operative order")
    decision_status: Optional[str] = Field(default=None, description="e.g. majority, dissent, referred to larger bench")
    review_required: bool = True
    statutes_considered: List[str] = Field(
        default_factory=list,
        description="Statutory provisions and articles interpreted (e.g. Art 226, Sec 138)"
    )
    precedents_cited: List[str] = Field(
        default_factory=list,
        description="Earlier landmark authorities referenced by the bench"
    )
    confidence_score: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Extraction reliability confidence"
    )
