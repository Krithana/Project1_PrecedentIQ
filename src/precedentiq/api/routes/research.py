import datetime
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from ...schemas.retrieval import SearchQuery, FilterParams
from ...schemas.argument import LegalResearchBrief
from ...retrieval.hybrid_engine import HybridRetrievalEngine
from ...ai.holding_extractor import HoldingExtractor
from ...ai.argument_synthesizer import ArgumentSynthesizer
from ...data.document_parser import DocumentParser
from ...export.word_exporter import WordBriefExporter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["Legal Research & Precedent Synthesis"])


@router.post("", response_model=LegalResearchBrief)
def execute_research(query_request: SearchQuery) -> LegalResearchBrief:
    """
    Execute hybrid retrieval, holding extraction, and citation-linked
    argument synthesis over Indian case law.
    """
    engine = HybridRetrievalEngine.get_instance()
    search_result = engine.search(query_request)

    holding_extractor = HoldingExtractor()
    argument_synthesizer = ArgumentSynthesizer()

    # Extract holdings for top unique judgments
    holdings = []
    seen_judgments = set()
    for sc in search_result.scored_chunks:
        jid = sc.chunk.judgment_id
        if jid not in seen_judgments:
            seen_judgments.add(jid)
            related_chunks = engine.loader.get_chunks_for_judgment(jid)
            if not related_chunks:
                related_chunks = [sc.chunk]
            holding = holding_extractor.extract(related_chunks, jid)
            holdings.append(holding)
        if len(holdings) >= 4:
            break

    # Synthesize argument outline
    outline = argument_synthesizer.synthesize(
        query=query_request.query,
        scored_chunks=search_result.scored_chunks
    )

    now_str = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")

    return LegalResearchBrief(
        query=query_request.query,
        timestamp=now_str,
        total_precedents_analyzed=len(search_result.scored_chunks),
        top_precedents=search_result.scored_chunks,
        holdings=holdings,
        argument_outline=outline
    )


@router.post("/upload", response_model=LegalResearchBrief)
async def upload_case_file_and_research(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None),
    court_filter: Optional[str] = Form(None),
    top_k: int = Form(5)
) -> LegalResearchBrief:
    """
    Upload a client case document (.pdf, .docx, .txt), parse factual matrix
    and legal issues, and ground hybrid precedent discovery against it.
    """
    try:
        content = await file.read()
        parsed_doc = DocumentParser.parse_bytes(content, file.filename)
    except Exception as e:
        logger.error(f"Error parsing uploaded document: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process case file: {e}")

    # Formulate effective search query
    effective_query = query.strip() if query and query.strip() else ""
    if not effective_query:
        # Synthesize query from probable title and statutes
        statute_str = " ".join(parsed_doc.get("statutes", [])[:3])
        effective_query = f"{parsed_doc.get('probable_title', '')} {statute_str}".strip()
        if not effective_query:
            effective_query = parsed_doc.get("context_summary", "")[:200]

    filters = FilterParams(court=court_filter) if court_filter else None
    search_req = SearchQuery(query=effective_query, top_k=top_k, filters=filters)

    engine = HybridRetrievalEngine.get_instance()
    search_result = engine.search(search_req)

    holding_extractor = HoldingExtractor()
    argument_synthesizer = ArgumentSynthesizer()

    # Extract holdings
    holdings = []
    seen_judgments = set()
    for sc in search_result.scored_chunks:
        jid = sc.chunk.judgment_id
        if jid not in seen_judgments:
            seen_judgments.add(jid)
            related_chunks = engine.loader.get_chunks_for_judgment(jid)
            if not related_chunks:
                related_chunks = [sc.chunk]
            holding = holding_extractor.extract(related_chunks, jid)
            holdings.append(holding)
        if len(holdings) >= 4:
            break

    # Synthesize pro/contra outline grounded in uploaded document
    outline = argument_synthesizer.synthesize(
        query=effective_query,
        scored_chunks=search_result.scored_chunks,
        client_case_context=parsed_doc.get("context_summary")
    )

    now_str = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")

    return LegalResearchBrief(
        query=effective_query,
        case_file_name=file.filename,
        timestamp=now_str,
        total_precedents_analyzed=len(search_result.scored_chunks),
        top_precedents=search_result.scored_chunks,
        holdings=holdings,
        argument_outline=outline
    )


@router.post("/export/docx")
def export_chamber_brief(brief: LegalResearchBrief):
    """Generate and stream a formatted Word (.docx) chamber brief."""
    try:
        doc_stream = WordBriefExporter.create_brief_document(brief)
        filename = f"PrecedentIQ_Brief_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        return StreamingResponse(
            doc_stream,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"Error generating Word brief: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate brief: {e}")
