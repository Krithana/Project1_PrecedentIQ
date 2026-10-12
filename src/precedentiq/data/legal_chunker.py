import re
import hashlib
from typing import List, Dict, Any, Tuple
from ..schemas.judgment import JudgmentChunk, ChunkMetadata, LegalSectionEnum


class LegalSectionChunker:
    """
    Splits Indian legal texts into structural judicial sections
    rather than arbitrary token counts.
    """

    SECTION_PATTERNS = {
        LegalSectionEnum.FACTS: [
            r"(?i)\b(?:brief\s+facts|factual\s+matrix|facts\s+of\s+the\s+case|case\s+of\s+the\s+prosecution|factual\s+background)\b",
            r"(?i)^[0-9]+\.\s+The\s+petitioner\s+has\s+approached",
        ],
        LegalSectionEnum.ISSUES: [
            r"(?i)\b(?:points?\s+for\s+determination|issues?\s+involved|question\s+of\s+law|substantial\s+question)\b",
            r"(?i)\bwhether\s+the\s+appellant|whether\s+the\s+court\s+was\s+justified\b",
        ],
        LegalSectionEnum.ARGUMENTS: [
            r"(?i)\b(?:submissions?\s+of\s+counsel|learned\s+counsel\s+for\s+the\s+appellant|arguments?\s+advanced|per\s+contra)\b",
            r"(?i)\bcontended\s+that|learned\s+senior\s+counsel\s+argued\b",
        ],
        LegalSectionEnum.REASONING: [
            r"(?i)\b(?:analysis\s+and\s+findings|consideration\s+by\s+the\s+court|we\s+have\s+considered|discussion)\b",
            r"(?i)\bwe\s+have\s+heard\s+learned\s+counsel|upon\s+perusal\s+of\s+the\s+record\b",
        ],
        LegalSectionEnum.RATIO_DECIDENDI: [
            r"(?i)\b(?:ratio\s+decidendi|it\s+is\s+well\s+settled\s+law|the\s+law\s+is\s+trite|we\s+hold\s+that|binding\s+principle)\b",
            r"(?i)\bsettled\s+principle\s+of\s+law\s+that|legal\s+position\s+is\s+unmistakable\b",
        ],
        LegalSectionEnum.OBITER_DICTA: [
            r"(?i)\b(?:we\s+may\s+incidentally\s+observe|by\s+way\s+of\s+obiter|in\s+passing|parenthetically)\b",
        ],
        LegalSectionEnum.ORDER: [
            r"(?i)\b(?:operative\s+order|in\s+the\s+result|accordingly|appeal\s+is\s+allowed|petition\s+stands\s+disposed\s+of|writ\s+petition\s+is\s+dismissed)\b",
            r"(?i)^order\b",
        ],
    }

    @classmethod
    def chunk_judgment(
        cls,
        text: str,
        judgment_id: str,
        court: str = "high_court",
        max_chunk_chars: int = 1500,
        overlap_chars: int = 200
    ) -> List[JudgmentChunk]:
        """
        Segment a judgment into structural legal chunks. If no headings
        are detected, applies smart paragraph fallback chunking.
        """
        # Judgments and pasted test/doc text do not always preserve blank lines
        # around structural headings. Split those headings into their own block
        # before applying section classification.
        normalized_text = re.sub(
            r"(?im)(?<!^)(\b(?:facts\s+of\s+the\s+case|submissions\s+of\s+counsel|"
            r"ratio\s+decidendi|order)\s*:)",
            r"\n\n\1\n",
            text,
        )
        paragraphs = [p.strip() for p in normalized_text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        labeled_blocks: List[Tuple[str, str]] = []
        current_section = LegalSectionEnum.FACTS.value

        for para in paragraphs:
            detected_sec = None
            for sec, patterns in cls.SECTION_PATTERNS.items():
                for pat in patterns:
                    if re.search(pat, para[:200]):
                        detected_sec = sec.value
                        break
                if detected_sec:
                    break

            if detected_sec:
                current_section = detected_sec

            labeled_blocks.append((current_section, para))

        # Group contiguous paragraphs of the same section or split if too long
        chunks: List[JudgmentChunk] = []
        current_text = ""
        current_sec_label = labeled_blocks[0][0] if labeled_blocks else LegalSectionEnum.FALLBACK.value

        for sec, para in labeled_blocks:
            if sec != current_sec_label or (len(current_text) + len(para) > max_chunk_chars):
                if current_text:
                    cid = hashlib.md5(f"{judgment_id}_{len(chunks)}_{current_text[:50]}".encode()).hexdigest()[:16]
                    chunks.append(JudgmentChunk(
                        chunk_id=cid,
                        judgment_id=judgment_id,
                        citation=judgment_id,
                        court=court,
                        section=current_sec_label,
                        text=current_text.strip(),
                        metadata=ChunkMetadata(chunk_method="structural_heading")
                    ))
                current_sec_label = sec
                current_text = para + "\n\n"
            else:
                current_text += para + "\n\n"

        if current_text.strip():
            cid = hashlib.md5(f"{judgment_id}_{len(chunks)}_{current_text[:50]}".encode()).hexdigest()[:16]
            chunks.append(JudgmentChunk(
                chunk_id=cid,
                judgment_id=judgment_id,
                citation=judgment_id,
                court=court,
                section=current_sec_label,
                text=current_text.strip(),
                metadata=ChunkMetadata(chunk_method="structural_heading")
            ))

        return chunks
