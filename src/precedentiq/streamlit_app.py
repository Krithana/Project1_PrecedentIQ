import os
import sys
import ssl
import io
import time
import datetime
from pathlib import Path

# Fix Windows Python SSL certificate verification for Hugging Face downloads
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import urllib3
    urllib3.disable_warnings()
    if hasattr(urllib3.util, "ssl_"):
        orig_ctx = urllib3.util.ssl_.create_urllib3_context
        def _safe_ctx(*args, **kwargs):
            ctx = orig_ctx(*args, **kwargs)
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        urllib3.util.ssl_.create_urllib3_context = _safe_ctx
        urllib3.util.create_urllib3_context = _safe_ctx
except Exception:
    pass

import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.precedentiq.config import settings
from src.precedentiq.data.loader import DataLoader
from src.precedentiq.data.document_parser import DocumentParser
from src.precedentiq.schemas.retrieval import SearchQuery, FilterParams
from src.precedentiq.schemas.argument import LegalResearchBrief
from src.precedentiq.retrieval.hybrid_engine import HybridRetrievalEngine
from src.precedentiq.ai.holding_extractor import HoldingExtractor
from src.precedentiq.ai.argument_synthesizer import ArgumentSynthesizer
from src.precedentiq.export.word_exporter import WordBriefExporter

# Configure page settings
st.set_page_config(
    page_title="PrecedentIQ — Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive legal typography and polished design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Merriweather:ital,wght@0,300;0,400;0,700;1,300&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    .legal-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .legal-title {
        font-family: 'Merriweather', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    
    .legal-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
    }
    
    .badge-sc {
        background: #f59e0b;
        color: #0f172a;
        font-weight: 600;
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 6px;
        text-transform: uppercase;
    }
    
    .badge-hc {
        background: #3b82f6;
        color: #ffffff;
        font-weight: 600;
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 6px;
        text-transform: uppercase;
    }
    
    .badge-section {
        background: #334155;
        color: #cbd5e1;
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 6px;
    }
    
    .precedent-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 16px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    
    .precedent-card:hover {
        border-color: #64748b;
    }
    
    .ratio-box {
        background: rgba(16, 185, 129, 0.08);
        border-left: 4px solid #10b981;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }
    
    .obiter-box {
        background: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }
    
    .order-box {
        background: rgba(59, 130, 246, 0.08);
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }
    
    .arg-pro {
        background: rgba(16, 185, 129, 0.05);
        border: 1px solid #065f46;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
    }
    
    .arg-contra {
        background: rgba(239, 68, 68, 0.05);
        border: 1px solid #991b1b;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
    }
    
    .citation-tag {
        font-family: monospace;
        font-size: 0.8rem;
        background: #0f172a;
        color: #38bdf8;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_retrieval_engine():
    """Cache the retrieval engine in memory across interactions."""
    loader = DataLoader.get_instance()
    engine = HybridRetrievalEngine.get_instance()
    return engine, loader


def main():
    engine, loader = get_retrieval_engine()
    stats = loader.get_corpus_statistics()

    # Sidebar: Repository Stats & Config
    with st.sidebar:
        st.markdown("### 🏛️ PrecedentIQ Engine")
        st.caption("Hybrid RAG Platform for Indian Case Law")

        st.divider()
        st.markdown("**Corpus Repositories**")
        st.metric("Searchable Chunks", f"{stats['total_chunks']:,}")
        st.metric("Indexed Judgments", f"{stats['total_judgments']:,}")

        sc_count = stats["courts"].get("supreme_court", 0)
        hc_count = stats["courts"].get("high_court", 0)
        st.caption(f"• Supreme Court: **{sc_count:,}** chunks")
        st.caption(f"• High Courts: **{hc_count:,}** chunks")

        st.divider()
        st.markdown("**Retrieval Controls**")
        court_filter = st.selectbox(
            "Target Jurisdiction",
            options=["All Jurisdictions", "Supreme Court", "High Court"],
            index=0
        )
        top_k = st.slider("Precedents to Retrieve", min_value=1, max_value=15, value=5)

        with st.expander("⚙️ Advanced Weights"):
            bm25_weight = st.slider("BM25 Keyword Weight", 0.0, 1.0, 0.5, 0.05)
            dense_weight = st.slider("Dense Semantic Weight", 0.0, 1.0, 0.5, 0.05)

        st.divider()
        st.markdown("**AI Intelligence Engine**")
        ai_engine_choice = st.selectbox(
            "Generation Engine",
            options=[
                "Llama 3.2 (3B - Fast CPU)",
                "Deterministic Legal Engine (Instant)",
                "Llama 3.1 (8B - Detailed)"
            ],
            index=0,
            help="Llama 3.2 3B is 4x faster on CPU. Deterministic engine provides instant 0.1s output without Ollama."
        )

        st.caption(f"Active: **{ai_engine_choice}**")
        st.caption("Embedding: all-MiniLM-L6-v2 (384d)")

    # Main Header
    st.markdown("""
    <div class="legal-header">
        <div class="legal-title">PrecedentIQ</div>
        <div class="legal-subtitle">A Hybrid Retrieval-Augmented Framework for Precedent Discovery & Argument Synthesis</div>
    </div>
    """, unsafe_allow_html=True)

    # Input Area: Query + Case Document Upload
    col_input, col_upload = st.columns([3, 2])

    with col_input:
        query_input = st.text_area(
            "Enter Legal Query or Question of Law:",
            value="Whether a power of attorney holder can institute proceedings on behalf of the principal",
            height=95,
            help="Enter a natural language question or statutory provision to retrieve relevant precedents."
        )

        st.caption("Sample Queries:")
        sample_cols = st.columns(3)
        if sample_cols[0].button("Power of Attorney", use_container_width=True):
            query_input = "Whether a power of attorney holder can institute proceedings on behalf of the principal"
        if sample_cols[1].button("Section 482 CrPC", use_container_width=True):
            query_input = "Quashing of criminal proceedings under Section 482 CrPC for commercial dispute"
        if sample_cols[2].button("Retrospective Rules", use_container_width=True):
            query_input = "Retrospective application of amended recruitment rules for Panchayat Karmi"

    with col_upload:
        uploaded_file = st.file_uploader(
            "Ground on Client Case Document (Optional):",
            type=["pdf", "docx", "txt", "md"],
            help="Upload a petition, plaint, or FIR summary to contextualize the precedent search against your client's facts."
        )
        if uploaded_file:
            st.success(f"📎 Grounded on: **{uploaded_file.name}** ({uploaded_file.size // 1024} KB)")

    court_filter_val = None
    if court_filter == "Supreme Court":
        court_filter_val = "supreme_court"
    elif court_filter == "High Court":
        court_filter_val = "high_court"

    filter_obj = FilterParams(court=court_filter_val) if court_filter_val else None

    # Research Execution Button
    run_research = st.button("🚀 Research Precedents & Synthesize Arguments", type="primary", use_container_width=True)

    if run_research and query_input:
        with st.spinner("Executing hybrid retrieval, holding extraction, and argument synthesis..."):
            client_context = None
            if uploaded_file:
                try:
                    file_bytes = uploaded_file.read()
                    parsed = DocumentParser.parse_bytes(file_bytes, uploaded_file.name)
                    client_context = parsed.get("context_summary")
                except Exception as e:
                    st.warning(f"Could not parse uploaded document: {e}")

            # 1. Hybrid Retrieval
            search_req = SearchQuery(
                query=query_input,
                top_k=top_k,
                filters=filter_obj,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight
            )
            search_result = engine.search(search_req)

            # Configure AI Client based on sidebar selection
            from src.precedentiq.ai.ollama_client import OllamaClient
            if "3.2" in ai_engine_choice:
                ollama_client = OllamaClient(model="llama3.2:3b", timeout=25)
                max_holdings = 2  # Keep CPU latency under 10 seconds
            elif "3.1" in ai_engine_choice:
                ollama_client = OllamaClient(model="llama3.1:8b", timeout=60)
                max_holdings = 2
            else:
                ollama_client = None  # Pure instantaneous heuristic engine
                max_holdings = 4

            # 2. Holding Extraction for top unique judgments
            holding_extractor = HoldingExtractor(ollama_client=ollama_client)
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
                if len(holdings) >= max_holdings:
                    break

            # 3. Argument Synthesis
            argument_synthesizer = ArgumentSynthesizer(ollama_client=ollama_client)
            outline = argument_synthesizer.synthesize(
                query=query_input,
                scored_chunks=search_result.scored_chunks,
                client_case_context=client_context
            )

            now_str = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")
            brief = LegalResearchBrief(
                query=query_input,
                case_file_name=uploaded_file.name if uploaded_file else None,
                timestamp=now_str,
                total_precedents_analyzed=len(search_result.scored_chunks),
                top_precedents=search_result.scored_chunks,
                holdings=holdings,
                argument_outline=outline
            )
            st.session_state["active_brief"] = brief

    # Display Results if Available
    if "active_brief" in st.session_state:
        brief: LegalResearchBrief = st.session_state["active_brief"]

        st.divider()

        # Top Bar: Result Summary & Word Export
        res_col1, res_col2 = st.columns([3, 1])
        with res_col1:
            st.markdown(f"### 📋 Research Findings ({len(brief.top_precedents)} Authorities Identified)")
            st.caption(f"Query: *\"{brief.query}\"* | Evaluated {brief.total_precedents_analyzed} ranked candidate chunks")
        with res_col2:
            # Word brief export
            docx_stream = WordBriefExporter.create_brief_document(brief)
            st.download_button(
                label="📥 Export Chamber Brief (.docx)",
                data=docx_stream,
                file_name=f"PrecedentIQ_Brief_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )

        # Tabbed Layout
        tab_prec, tab_holdings, tab_args, tab_evidence = st.tabs([
            "🏛️ Precedent Discovery",
            "⚖️ Holding Analysis (Ratio vs Obiter)",
            "⚔️ Argument Outline (Pro & Contra)",
            "🔍 Evidence & Citations"
        ])

        # TAB 1: Precedents
        with tab_prec:
            if not brief.top_precedents:
                st.info("No matching precedents found. Adjust your query or filters.")
            for idx, sc in enumerate(brief.top_precedents, start=1):
                chunk = sc.chunk
                court_badge = '<span class="badge-sc">Supreme Court</span>' if chunk.court == "supreme_court" else '<span class="badge-hc">High Court</span>'
                section_badge = f'<span class="badge-section">{chunk.section.replace("_", " ").title()}</span>'
                
                signals_html = ""
                for sig in chunk.metadata.treatment_signals:
                    signals_html += f' <span style="background:#475569; color:#f1f5f9; font-size:0.7rem; padding:2px 6px; border-radius:4px;">{sig.upper()}</span>'

                st.markdown(f"""
                <div class="precedent-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <div>
                            <strong>#{idx}. {chunk.citation}</strong> &nbsp; {court_badge} &nbsp; {section_badge} {signals_html}
                        </div>
                        <div style="color:#38bdf8; font-weight:600; font-size:0.9rem;">
                            Score: {sc.final_score:.3f}
                        </div>
                    </div>
                    <div style="color:#cbd5e1; font-size:0.95rem; line-height:1.5;">
                        {chunk.text[:400]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"Inspect Full Text Chunk ({chunk.chunk_id})"):
                    st.text(chunk.text)

        # TAB 2: Holdings (Ratio Decidendi vs Obiter Dicta)
        with tab_holdings:
            st.markdown("#### Structured Holding Extraction")
            st.caption("Separating binding precedent (*ratio decidendi*) from judicial observations (*obiter dicta*) under Article 141.")

            for h in brief.holdings:
                court_label = "Supreme Court of India" if h.court == "supreme_court" else "High Court"
                st.markdown(f"### 📑 {h.case_title or h.judgment_id}")
                st.caption(f"{court_label} | Bench: {h.bench or 'Standard'} | Year: {h.judgment_date or 'N/A'}")

                st.markdown(f"""
                <div class="ratio-box">
                    <strong style="color:#10b981;">⚖️ Ratio Decidendi (Binding Legal Principle):</strong><br>
                    {h.ratio_decidendi}
                </div>
                """, unsafe_allow_html=True)

                if h.obiter_dicta:
                    st.markdown(f"""
                    <div class="obiter-box">
                        <strong style="color:#f59e0b;">💬 Obiter Dicta (Persuasive Judicial Remark):</strong><br>
                        {h.obiter_dicta}
                    </div>
                    """, unsafe_allow_html=True)

                if h.final_order:
                    st.markdown(f"""
                    <div class="order-box">
                        <strong style="color:#3b82f6;">📜 Operative Order / Disposition:</strong><br>
                        {h.final_order}
                    </div>
                    """, unsafe_allow_html=True)

                if h.statutes_considered:
                    st.markdown("**Statutes Interpreted:** " + ", ".join([f"`{s}`" for s in h.statutes_considered]))

                st.divider()

        # TAB 3: Arguments (Pro vs Contra)
        with tab_args:
            outline = brief.argument_outline
            st.markdown(f"#### {outline.issue_summary}")

            col_pro, col_contra = st.columns(2)

            with col_pro:
                st.markdown("### 🟢 Affirmative Grounds (Pro)")
                if outline.supporting_arguments:
                    for arg in outline.supporting_arguments:
                        st.markdown(f"""
                        <div class="arg-pro">
                            <strong style="font-size:1.05rem; color:#10b981;">[{arg.strength}] {arg.title}</strong><br>
                            <div style="margin-top:6px;">{arg.legal_proposition}</div>
                            <div style="margin-top:8px; font-size:0.85rem; color:#94a3b8;">
                                <em>Application:</em> {arg.applicability_reasoning}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        for cit in arg.supporting_citations:
                            st.caption(f"📌 Citation: `{cit.citation}` ({cit.court}) | Excerpt: *\"{cit.excerpt}\"*")
                else:
                    st.info("No supporting arguments identified.")

            with col_contra:
                st.markdown("### 🔴 Opposing Arguments & Distinctions (Contra)")
                if outline.counter_arguments:
                    for carg in outline.counter_arguments:
                        st.markdown(f"""
                        <div class="arg-contra">
                            <strong style="font-size:1.05rem; color:#ef4444;">Opposing Objection:</strong><br>
                            <div style="margin-top:6px;">{carg.potential_objection}</div>
                            <div style="margin-top:8px; font-size:0.85rem; color:#fca5a5;">
                                <strong>Grounds to Distinguish:</strong> {carg.rebuttal_or_distinction}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        for cit in carg.adverse_citations:
                            st.caption(f"⚠️ Adverse Authority: `{cit.citation}` | Excerpt: *\"{cit.excerpt}\"*")
                else:
                    st.info("No substantial adverse authorities found.")

            if outline.strategic_advice:
                st.markdown("### 💡 Practical Strategic Advice for Counsel")
                st.info(outline.strategic_advice)

        # TAB 4: Evidence & Chunk Inspection
        with tab_evidence:
            st.markdown("#### Evidence Verification Appendix")
            st.caption("Hard evidence constraint: Every proposition links directly to verified chunk IDs and source texts.")

            for sc in brief.top_precedents:
                with st.expander(f"Chunk `{sc.chunk.chunk_id}` — {sc.chunk.citation} ({sc.chunk.court})"):
                    st.markdown(f"**Section:** `{sc.chunk.section}` | **Score:** `{sc.final_score:.4f}`")
                    st.markdown(f"**Source File:** `{sc.chunk.metadata.source_file}`")
                    st.text_area("Chunk Content", sc.chunk.text, height=180, key=f"text_{sc.chunk.chunk_id}")


if __name__ == "__main__":
    main()
