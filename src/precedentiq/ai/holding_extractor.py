import json
import logging
from typing import List, Optional
from ..config import settings
from ..schemas.judgment import JudgmentChunk
from ..schemas.analysis import HoldingExtraction
from .ollama_client import OllamaClient
from .fallbacks import HeuristicLegalSynthesizer

logger = logging.getLogger(__name__)


class HoldingExtractor:
    """
    Extracts structured holding, separating binding Ratio Decidendi
    from non-binding Obiter Dicta and procedural facts.
    """

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.client = ollama_client or OllamaClient()

    def extract(self, chunks: List[JudgmentChunk], judgment_id: str) -> HoldingExtraction:
        """Extract holding using Ollama (if available) or deterministic fallback."""
        if not chunks:
            raise ValueError(f"No chunks provided for judgment_id: {judgment_id}")

        # Check if Ollama is available
        if self.client.is_model_available():
            try:
                system_prompt = (
                    "You are an expert Indian Supreme Court & High Court legal analyst. "
                    "Analyze the provided judgment text and extract the exact legal holding. "
                    "CRITICAL: Separate the binding 'ratio_decidendi' (the legal rule deciding the case) "
                    "from 'obiter_dicta' (judicial comments made in passing). "
                    "Respond with ONLY a valid JSON object matching this schema:\n"
                    "{\n"
                    '  "judgment_id": "string",\n'
                    '  "case_title": "string",\n'
                    '  "court": "supreme_court or high_court",\n'
                    '  "bench": "string or null",\n'
                    '  "judgment_date": "string or null",\n'
                    '  "material_facts": ["fact 1", "fact 2"],\n'
                    '  "legal_issues": ["issue 1", "issue 2"],\n'
                    '  "ratio_decidendi": "core binding legal principle",\n'
                    '  "obiter_dicta": "persuasive observation or null",\n'
                    '  "final_order": "operative disposal",\n'
                    '  "statutes_considered": ["statute 1"],\n'
                    '  "confidence_score": 0.95\n'
                    "}"
                )

                chunk_context = "\n---\n".join([
                    f"[{c.section.upper()} | {c.citation}]: {c.text[:400]}"
                    for c in chunks[:3]
                ])

                user_prompt = f"Extract holding for Judgment {judgment_id}:\n\n{chunk_context}"

                data = self.client.generate_json(user_prompt, system=system_prompt)
                if data and "ratio_decidendi" in data:
                    return HoldingExtraction(
                        judgment_id=str(data.get("judgment_id", judgment_id)),
                        case_title=data.get("case_title", chunks[0].citation),
                        court=str(data.get("court", chunks[0].court)),
                        bench=data.get("bench", chunks[0].metadata.court_bench),
                        judgment_date=str(data.get("judgment_date", chunks[0].metadata.source_year)),
                        material_facts=data.get("material_facts", []),
                        legal_issues=data.get("legal_issues", []),
                        ratio_decidendi=str(data["ratio_decidendi"]),
                        ratio_evidence=data.get("ratio_evidence", []),
                        ratio_paragraphs=data.get("ratio_paragraphs", []),
                        obiter_dicta=data.get("obiter_dicta"),
                        final_order=data.get("final_order"),
                        order_evidence=data.get("order_evidence", []),
                        decision_status=data.get("decision_status"),
                        classification_note="LLM candidate extraction; verify every claim against the original judgment.",
                        statutes_considered=data.get("statutes_considered", []),
                        confidence_score=float(data.get("confidence_score", 0.7)),
                        review_required=True
                    )
            except Exception as e:
                logger.warning(f"Ollama holding extraction failed: {e}. Using deterministic fallback.")

        # Heuristic fallback
        return HeuristicLegalSynthesizer.extract_holding(chunks, judgment_id)
