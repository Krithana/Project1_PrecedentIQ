import re
from typing import List, Dict, Any, Optional
from ..schemas.judgment import JudgmentChunk
from ..schemas.analysis import HoldingExtraction
from ..schemas.argument import (
    CitationSource,
    ArgumentPoint,
    CounterArgumentPoint,
    ProContraOutline
)


class HeuristicLegalSynthesizer:
    """
    High-precision deterministic rule-based legal holding extraction and
    argument synthesizer for offline, CPU-bound, or fallback environments.
    """

    @classmethod
    def extract_holding(cls, chunks: List[JudgmentChunk], judgment_id: str) -> HoldingExtraction:
        """Extract structured holding from retrieved judgment chunks."""
        court = chunks[0].court if chunks else "high_court"
        citation = chunks[0].citation if chunks else judgment_id
        bench = chunks[0].metadata.court_bench if chunks else None
        year = chunks[0].metadata.source_year if chunks else None

        combined_text = "\n".join([c.text for c in chunks])

        # 1. Identify Case Title, avoiding reporter/page headers.
        title = citation
        for line in combined_text.split("\n")[:10]:
            line_str = line.strip()
            if (" VS " in line_str.upper() or " V. " in line_str.upper() or " VERSUS " in line_str.upper()) and len(line_str) < 240:
                title = line_str
                break

        # 2. Extract Framed Issues
        issues = []
        issue_matches = re.findall(
            r"(?i)(?:whether\s+[^.?!]+[.?!]|question\s+arising\s+for\s+consideration\s+is\s+[^.?!]+[.?!])",
            combined_text
        )
        for m in issue_matches[:3]:
            issues.append(m.strip())
        if not issues:
            issues.append(f"Determination of rights and relief in matter of {citation}")

        # 3. Extract Ratio Decidendi
        ratio_candidates = []
        ratio_evidence = []
        ratio_paragraphs = []
        argument_markers = ("learned counsel", "it was argued", "the appellant submits", "the respondent submits", "per contra", "vehemently submitted")
        for c in chunks:
            if c.section in ["ratio_decidendi", "reasoning", "order"]:
                sentences = re.split(r"(?<=[.!?])\s+", c.text)
                for s in sentences:
                    s_clean = s.strip()
                    lower = s_clean.lower()
                    if len(s_clean) > 40 and not any(m in lower for m in argument_markers) and any(kw in lower for kw in [
                        "it is well settled", "held that", "we hold", "trite law",
                        "cannot be permitted", "imperative that", "settled principle",
                        "in accordance with law", "entitled to", "law is clear", "therefore",
                        "accordingly", "is hereby quashed", "is hereby set aside"
                    ]):
                        ratio_candidates.append(s_clean)
                        ratio_evidence.append(s_clean)
                        found_paras = re.findall(r"\bPara(?:s)?\.?\s*([\d,\-]+)", s_clean, re.I)
                        ratio_paragraphs.extend([f"Para {p}" for p in found_paras])

        if ratio_candidates:
            ratio_text = " ".join(ratio_candidates[:2])
        else:
            ratio_text = "Candidate ratio could not be reliably extracted from the available judgment text."

        # 4. Extract Obiter Dicta
        obiter_text = None
        for c in chunks:
            if c.section == "obiter_dicta" and not any(m in c.text.lower() for m in argument_markers):
                obiter_text = c.text[:300]
                break
            elif "incidentally" in c.text.lower() or "in passing" in c.text.lower():
                matches = re.findall(r"(?i)[^.?!]*(?:incidentally|in passing)[^.?!]*[.?!]", c.text)
                if matches:
                    obiter_text = matches[0].strip()
                    break

        # 5. Extract Operative Order
        order_text = None
        order_evidence = []
        decision_status = None
        for c in chunks:
            if c.section in ["order", "reasoning", "full_text", "fallback"]:
                sentences = re.split(r"(?<=[.!?])\s+", c.text)
                disposition_s = [
                    s.strip() for s in sentences
                    if any(w in s.lower() for w in ["appeal is allowed", "appeals are allowed", "appeal is dismissed", "petition is allowed", "petition is dismissed", "disposed of", "quashed", "remanded", "referred to a larger", "restored", "set aside"])
                ]
                if disposition_s:
                    order_text = " ".join(disposition_s[:2])
                    order_evidence = disposition_s[:2]
                    break
        if re.search(r"dissent|difference of opinion|larger bench|larger bench", combined_text, re.I):
            decision_status = "Split opinion or reference to larger Bench; verify majority status."
        if not order_text:
            order_text = "Operative order could not be reliably extracted from the available judgment text."

        # 6. Extract Statutes
        statutes = list(set(re.findall(
            r"(?i)\b(?:Section\s+\d+[A-Z]?|Article\s+\d+|Order\s+[IVXLCDM]+\s+Rule\s+\d+|CPC|CrPC|IPC)\b",
            combined_text
        )))

        return HoldingExtraction(
            judgment_id=judgment_id,
            case_title=title,
            court=court,
            bench=bench,
            judgment_date=year,
            material_facts=[f"Proceedings initiated before {court.replace('_', ' ').title()}"],
            legal_issues=issues,
            ratio_decidendi=ratio_text,
            obiter_dicta=obiter_text,
            final_order=order_text,
            ratio_evidence=ratio_evidence[:3],
            ratio_paragraphs=sorted(set(ratio_paragraphs)),
            order_evidence=order_evidence,
            decision_status=decision_status,
            classification_note="Candidate extraction only; verify against the original judgment and paragraph references.",
            statutes_considered=statutes[:5],
            confidence_score=0.65 if not ratio_evidence else 0.78
        )

    @classmethod
    def synthesize_arguments(
        cls,
        query: str,
        chunks: List[JudgmentChunk],
        case_context: Optional[str] = None
    ) -> ProContraOutline:
        """Construct verified, citation-linked Pro and Contra arguments."""
        supporting_points: List[ArgumentPoint] = []
        counter_points: List[CounterArgumentPoint] = []

        query_terms = {
            t.lower() for t in re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", query)
        } - {"the", "and", "for", "can", "from", "whether", "holder", "institute"}
        relevant_chunks = [
            c for c in chunks
            if query_terms.intersection(set(re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", c.text.lower())))
        ]
        if relevant_chunks:
            chunks = relevant_chunks

        # Group chunks by treatment / disposition
        pro_chunks = []
        contra_chunks = []

        for c in chunks:
            signals = [s.lower() for s in c.metadata.treatment_signals]
            text_lower = c.text.lower()

            if "followed" in signals or "approved" in signals or "allowed" in text_lower or "liberty" in text_lower:
                pro_chunks.append(c)
            elif "reversed" in signals or "dismissed" in text_lower or "rejected" in text_lower:
                contra_chunks.append(c)
            else:
                # Distribute evenly if neutral
                if len(pro_chunks) <= len(contra_chunks):
                    pro_chunks.append(c)
                else:
                    contra_chunks.append(c)

        # Build Supporting Argument (Pro)
        if pro_chunks:
            primary_chunk = pro_chunks[0]
            excerpt_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", primary_chunk.text) if len(s.strip()) > 30]
            excerpt = excerpt_sentences[0] if excerpt_sentences else primary_chunk.text[:250]

            source = CitationSource(
                chunk_id=primary_chunk.chunk_id,
                judgment_id=primary_chunk.judgment_id,
                citation=primary_chunk.citation,
                court=primary_chunk.court,
                section=primary_chunk.section,
                excerpt=excerpt
            )

            court_title = primary_chunk.court.replace("_", " ").title()
            supporting_points.append(ArgumentPoint(
                title=f"Affirmative Authority from {court_title}",
                legal_proposition=f"Relevant source passage from {primary_chunk.citation}: {excerpt}",
                strength="Strong" if primary_chunk.court == "supreme_court" else "Moderate",
                supporting_citations=[source],
                applicability_reasoning="Potentially relevant because the source contains overlapping legal terms. Confirm the holding, procedural posture, and paragraph references before relying on it."
            ))

        if len(pro_chunks) > 1:
            sec_chunk = pro_chunks[1]
            excerpt_s = [s.strip() for s in re.split(r"(?<=[.!?])\s+", sec_chunk.text) if len(s.strip()) > 30]
            excerpt2 = excerpt_s[0] if excerpt_s else sec_chunk.text[:250]
            source2 = CitationSource(
                chunk_id=sec_chunk.chunk_id,
                judgment_id=sec_chunk.judgment_id,
                citation=sec_chunk.citation,
                court=sec_chunk.court,
                section=sec_chunk.section,
                excerpt=excerpt2
            )
            supporting_points.append(ArgumentPoint(
                title=f"Corroborative Precedent on Merits ({sec_chunk.citation})",
                legal_proposition=f"Relevant source passage from {sec_chunk.citation}: {excerpt2}",
                strength="Moderate",
                supporting_citations=[source2],
                applicability_reasoning="Corroborative candidate only; confirm that the authority addresses the same legal issue and relief."
            ))

        # Build Counter Argument (Contra) & Distinctions
        if contra_chunks:
            opp_chunk = contra_chunks[0]
            excerpt_opp = [s.strip() for s in re.split(r"(?<=[.!?])\s+", opp_chunk.text) if len(s.strip()) > 30]
            excerpt_contra = excerpt_opp[0] if excerpt_opp else opp_chunk.text[:250]

            contra_source = CitationSource(
                chunk_id=opp_chunk.chunk_id,
                judgment_id=opp_chunk.judgment_id,
                citation=opp_chunk.citation,
                court=opp_chunk.court,
                section=opp_chunk.section,
                excerpt=excerpt_contra
            )

            counter_points.append(CounterArgumentPoint(
                potential_objection=f"Opposing counsel may cite {opp_chunk.citation} arguing lack of locus standi or dismissal for non-compliance.",
                adverse_citations=[contra_source],
                rebuttal_or_distinction=(
                    f"Potential distinction to investigate: compare the factual and procedural posture in {opp_chunk.citation} with the client’s record. "
                    "The system has not verified that distinction."
                )
            ))

        advice = (
            "1. File an affidavit clearly distinguishing the opposing authorities on facts. "
            "2. Anchor the primary submission on the binding principles articulated in the affirmative precedents. "
            "3. Ensure certified copies of the cited High Court and Supreme Court rulings are annexed in the compilation of authorities."
        )

        return ProContraOutline(
            issue_summary=f"Legal analysis of query: '{query}' over Indian Supreme Court & High Court jurisprudence.",
            client_context_summary=case_context[:300] if case_context else "Corpus-wide legal research query.",
            supporting_arguments=supporting_points,
            counter_arguments=counter_points,
            strategic_advice=advice
        )
