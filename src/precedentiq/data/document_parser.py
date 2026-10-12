import re
import io
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class DocumentParser:
    """
    Parses client-uploaded case files (PDF, DOCX, TXT, MD) into
    clean structured text and extracted legal signals.
    """

    STATUTE_PATTERNS = [
        r"(?:Section|Sec\.?|S\.)\s*\d+[A-Z]?(?:\s*(?:of|read with)\s*[^,\n\.]+)?",
        r"(?:Article|Art\.?)\s*\d+[A-Z]?",
        r"(?:Order|O\.)\s*[IVXLCDM]+\s*(?:Rule|R\.)\s*\d+[A-Z]?",
        r"(?:IPC|CrPC|CPC|NI Act|Constitution of India|Arbitration Act|Evidence Act)"
    ]

    @classmethod
    def parse_bytes(cls, content: bytes, filename: str) -> Dict[str, Any]:
        """Parse raw file bytes based on file extension."""
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            return cls._parse_pdf(content, filename)
        elif ext in [".docx", ".doc"]:
            return cls._parse_docx(content, filename)
        elif ext in [".txt", ".md", ".rtf"]:
            return cls._parse_text(content, filename)
        else:
            # Fallback text decode
            try:
                text = content.decode("utf-8", errors="ignore")
                return cls._post_process(text, filename)
            except Exception as e:
                raise ValueError(f"Unsupported file format: {filename} ({e})")

    @classmethod
    def _parse_pdf(cls, content: bytes, filename: str) -> Dict[str, Any]:
        text_parts = []
        # Attempt 1: PyMuPDF
        try:
            import pymupdf
            doc = pymupdf.open(stream=content, filetype="pdf")
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
        except ImportError:
            # Attempt 2: pdfplumber
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text_parts.append(extracted)
            except ImportError:
                # Attempt 3: pypdf
                try:
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(content))
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text_parts.append(extracted)
                except Exception as e:
                    logger.error(f"Failed to parse PDF with available engines: {e}")
                    raise RuntimeError("No suitable PDF extraction library available.")

        full_text = "\n".join(text_parts)
        return cls._post_process(full_text, filename)

    @classmethod
    def _parse_docx(cls, content: bytes, filename: str) -> Dict[str, Any]:
        try:
            from docx import Document
            doc = Document(io.BytesIO(content))
            text_parts = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    text_parts.append(" | ".join([c.text.strip() for c in row.cells if c.text.strip()]))
            full_text = "\n".join(text_parts)
            return cls._post_process(full_text, filename)
        except Exception as e:
            logger.error(f"Error reading docx: {e}")
            raise ValueError(f"Could not read DOCX document: {e}")

    @classmethod
    def _parse_text(cls, content: bytes, filename: str) -> Dict[str, Any]:
        text = content.decode("utf-8", errors="replace")
        return cls._post_process(text, filename)

    @classmethod
    def _post_process(cls, text: str, filename: str) -> Dict[str, Any]:
        # Normalize whitespace
        clean_text = re.sub(r"\r\n", "\n", text)
        clean_text = re.sub(r"[ \t]+", " ", clean_text)
        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()

        # Extract statutes
        statutes = set()
        for pat in cls.STATUTE_PATTERNS:
            matches = re.findall(pat, clean_text, re.IGNORECASE)
            for m in matches:
                statutes.add(m.strip())

        # Extract potential cause title (first 5 lines)
        lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
        title_candidate = lines[0] if lines else filename
        for line in lines[:8]:
            if "VERSUS" in line.upper() or " VS " in line.upper() or " V. " in line.upper() or "PETITION" in line.upper():
                title_candidate = line
                break

        # Distill core context summary (first ~800 words)
        words = clean_text.split()
        summary_sample = " ".join(words[:400])

        return {
            "filename": filename,
            "char_count": len(clean_text),
            "word_count": len(words),
            "text": clean_text,
            "probable_title": title_candidate,
            "statutes": sorted(list(statutes)),
            "context_summary": summary_sample,
        }
