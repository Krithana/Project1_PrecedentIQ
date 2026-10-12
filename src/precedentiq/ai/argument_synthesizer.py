import logging
from typing import List, Optional, Dict, Any
from ..config import settings
from ..schemas.judgment import JudgmentChunk
from ..schemas.retrieval import ScoredChunk
from ..schemas.argument import (
    CitationSource,
    ArgumentPoint,
    CounterArgumentPoint,
    ProContraOutline
)
from .ollama_client import OllamaClient
from .fallbacks import HeuristicLegalSynthesizer

logger = logging.getLogger(__name__)


class ArgumentSynthesizer:
    """
    Synthesizes two-sided legal briefs (Pro vs Contra) grounded
    strictly in retrieved Indian case law chunks.
    """

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.client = ollama_client or OllamaClient()

    def synthesize(
        self,
        query: str,
        scored_chunks: List[ScoredChunk],
        client_case_context: Optional[str] = None
    ) -> ProContraOutline:
        """Synthesize balanced legal argument outline with exact citations."""
        if not scored_chunks:
            return ProContraOutline(
                issue_summary=f"No precedents retrieved for query: {query}",
                strategic_advice="Broaden query parameters or remove court filters."
            )

        raw_chunks = [sc.chunk for sc in scored_chunks]

        if self.client.is_model_available():
            try:
                system_prompt = (
                    "You are a Senior Advocate drafting a legal argument outline for an Indian court. "
                    "Construct two-sided legal arguments: Supporting (Pro) and Counter-Arguments (Contra) with Factual Distinctions. "
                    "CRITICAL SAFETY RULE: You must NEVER invent citations or facts. "
                    "Every claim MUST be backed by an exact chunk_id and verbatim excerpt from the provided text. "
                    "Respond ONLY with a JSON object conforming to this schema:\n"
                    "{\n"
                    '  "issue_summary": "Summary of legal issue",\n'
                    '  "supporting_arguments": [\n'
                    '    {\n'
                    '      "title": "Short title",\n'
                    '      "legal_proposition": "Core argument proposition",\n'
                    '      "strength": "Strong or Moderate",\n'
                    '      "supporting_citations": [\n'
                    '        {\n'
                    '          "chunk_id": "id from text",\n'
                    '          "judgment_id": "judgment id",\n'
                    '          "citation": "citation",\n'
                    '          "court": "supreme_court or high_court",\n'
                    '          "section": "section name",\n'
                    '          "excerpt": "exact short quote from chunk"\n'
                    '        }\n'
                    '      ],\n'
                    '      "applicability_reasoning": "how it applies"\n'
                    '    }\n'
                    '  ],\n'
                    '  "counter_arguments": [\n'
                    '    {\n'
                    '      "potential_objection": "opposing counsel argument",\n'
                    '      "adverse_citations": [ ... citation object ... ],\n'
                    '      "rebuttal_or_distinction": "factual or legal grounds to distinguish"\n'
                    '    }\n'
                    '  ],\n'
                    '  "strategic_advice": "Practical litigation strategy for the advocate"\n'
                    "}"
                )

                precedent_context = "\n\n".join([
                    f"[Chunk ID: {sc.chunk.chunk_id} | {sc.chunk.citation} | {sc.chunk.court} | {sc.chunk.section}]\n{sc.chunk.text[:400]}"
                    for sc in scored_chunks[:4]
                ])

                context_note = f"\nClient Case File Summary:\n{client_case_context[:300]}" if client_case_context else ""

                user_prompt = (
                    f"Legal Question / Objective: {query}\n"
                    f"{context_note}\n\n"
                    f"Retrieved Precedent Chunks:\n{precedent_context}\n\n"
                    "Generate the structured Pro/Contra legal brief outline now."
                )

                data = self.client.generate_json(user_prompt, system=system_prompt)
                if data and "supporting_arguments" in data:
                    supporting = []
                    for sa in data.get("supporting_arguments", []):
                        citations = [
                            CitationSource(**c) for c in sa.get("supporting_citations", [])
                            if "chunk_id" in c and "excerpt" in c
                        ]
                        supporting.append(ArgumentPoint(
                            title=sa.get("title", "Supporting Ground"),
                            legal_proposition=sa.get("legal_proposition", ""),
                            strength=sa.get("strength", "Strong"),
                            supporting_citations=citations,
                            applicability_reasoning=sa.get("applicability_reasoning", "")
                        ))

                    counters = []
                    for ca in data.get("counter_arguments", []):
                        adv_cits = [
                            CitationSource(**c) for c in ca.get("adverse_citations", [])
                            if "chunk_id" in c and "excerpt" in c
                        ]
                        counters.append(CounterArgumentPoint(
                            potential_objection=ca.get("potential_objection", ""),
                            adverse_citations=adv_cits,
                            rebuttal_or_distinction=ca.get("rebuttal_or_distinction", "")
                        ))

                    return ProContraOutline(
                        issue_summary=data.get("issue_summary", f"Legal analysis of {query}"),
                        client_context_summary=client_case_context[:300] if client_case_context else None,
                        supporting_arguments=supporting,
                        counter_arguments=counters,
                        strategic_advice=data.get("strategic_advice", "")
                    )
            except Exception as e:
                logger.warning(f"Ollama argument synthesis failed: {e}. Using deterministic fallback.")

        # Fallback to high-accuracy heuristic synthesis
        return HeuristicLegalSynthesizer.synthesize_arguments(
            query=query,
            chunks=raw_chunks,
            case_context=client_case_context
        )
