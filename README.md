# PrecedentIQ: A Hybrid Retrieval-Augmented Framework for Precedent Discovery and Argument Synthesis over Indian Case Law

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC2626.svg?logo=qdrant&logoColor=white)](https://qdrant.tech)
[![Jurisdiction](https://img.shields.io/badge/Jurisdiction-Supreme%20Court%20%26%20High%20Courts%20of%20India-blue.svg)](#)

---

## ⚖️ Executive Summary

Junior advocates and small/medium law firms in India spend significant portions of their workweek manually scanning thousands of Supreme Court and High Court judgments to discover binding precedents. Even after identifying relevant cases, advocates must spend hours parsing lengthy judgments to separate binding principles (**ratio decidendi**) from judicial observations (**obiter dicta**) and manually drafting structured arguments.

**PrecedentIQ** bridges this gap with an end-to-end Legal RAG framework that automates the complete chain from discovery to draft:
1. **Hybrid Retrieval**: Combines semantic dense embeddings with BM25 sparse keyword search via Reciprocal Rank Fusion (RRF) and court hierarchy weighting (Supreme Court Article 141 priority).
2. **Holding Extraction**: Separates the binding *ratio decidendi* from persuasive *obiter dicta* and operative orders.
3. **Citation-Linked Argument Synthesis**: Constructs two-sided (**Pro vs. Contra**) argument outlines with factual distinctions and hard evidence constraints (zero hallucination).
4. **Session Case Grounding**: Upload a client petition, plaint, or case summary (`.pdf`, `.docx`, `.txt`) to contextualize precedent search against specific facts.
5. **Chamber Brief Export**: One-click download of a formatted Word (`.docx`) legal brief.

---

## 🏗️ System Architecture

```
[ Advocate / Browser ]
          │
    ┌─────┴────────────────┐
    │  Streamlit Terminal  │  (UI with Precedent Cards, Ratio vs Obiter, Pro/Contra Deck)
    └─────┬────────────────┘
          │ HTTP / REST
    ┌─────▼────────────────┐
    │   FastAPI Backend    │  (The Brain: /research, /research/upload, /export/docx)
    └─────┬────────────────┘
          │
   ┌──────┴───────────────┬────────────────────────┬──────────────────────┐
   ▼                      ▼                        ▼                      ▼
┌──────────────┐   ┌──────────────┐         ┌──────────────┐       ┌──────────────┐
│ BM25 Sparse  │   │ Dense Vector │         │ Reciprocal   │       │ Contextual   │
│ Index        │   │ (MiniLM /    │ ──────> │ Rank Fusion  │ ────> │ Legal        │
│ (Statutes)   │   │  Qdrant)     │         │ (RRF)        │       │ Reranker     │
└──────────────┘   └──────────────┘         └──────────────┘       └──────┬───────┘
                                                                          │
                                                                   Top Precedents
                                                                          │
                                           ┌──────────────────────────────┴──────────────────────────┐
                                           ▼                                                         ▼
                               ┌────────────────────────┐                                ┌────────────────────────┐
                               │   Holding Extractor    │                                │  Argument Synthesizer  │
                               │ (Ratio vs Obiter Dicta)│                                │ (Pro vs Contra Brief)  │
                               └────────────────────────┘                                └───────────┬────────────┘
                                                                                                     │
                                                                                          ┌──────────▼────────────┐
                                                                                          │ .docx Chamber Brief   │
                                                                                          └───────────────────────┘
```

---

## 📁 Repository & Data Layout

```
precedentiq_v2/
├── data/
│   ├── processed/
│   │   ├── chunks.jsonl            # 11,720 pre-segmented searchable legal chunks
│   │   ├── judgments.jsonl         # 1,989 complete judgments (SC & HC benches)
│   │   ├── qdrant_manifest.json    # Collection definition & embedding spec
│   │   └── extracted/              # Extracted plain text judgment archives
│   ├── validation/                 # Academic gold benchmarks
│   │   ├── il_tur_cjpe/            # Expert salient sentences & outcomes
│   │   ├── il_pcsr_queries/        # Ground-truth legal queries & precedents
│   │   └── il_pcsr_precedents/     # Candidate precedent texts
│   └── raw/                        # Original Supreme Court & High Court archives
├── src/
│   └── precedentiq/
│       ├── config.py               # Pydantic settings & weights
│       ├── schemas/                # Data contracts (Judgment, Retrieval, Analysis, Argument)
│       ├── data/                   # DataLoader, DocumentParser, LegalSectionChunker
│       ├── retrieval/              # BM25Retriever, DenseRetriever, QdrantStore, HybridEngine
│       ├── ai/                     # OllamaClient, HoldingExtractor, ArgumentSynthesizer, Fallbacks
│       ├── export/                 # WordBriefExporter (.docx generator)
│       ├── api/                    # FastAPI routes (/health, /research, /export/docx)
│       └── streamlit_app.py        # Modern Legal Research Terminal UI
├── scripts/
│   ├── run_api.py                  # Launch FastAPI server
│   ├── run_ui.py                   # Launch Streamlit frontend
│   ├── index_qdrant.py             # Vector store batch indexing CLI
│   └── evaluate_retrieval.py       # Benchmark evaluation script (nDCG@10, MRR, Recall)
├── tests/                          # Unit and integration test suite
├── docker-compose.yml              # Qdrant, Redis, Ollama local orchestration
├── requirements.txt                # Python dependencies
└── .env.example                    # Environment template
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch FastAPI Backend
In terminal 1:
```bash
python scripts/run_api.py
```
*API will run at:* `http://localhost:8000`  
*Interactive Swagger docs:* `http://localhost:8000/docs`

### 3. Launch Streamlit Legal Terminal
In terminal 2:
```bash
streamlit run src/precedentiq/streamlit_app.py
```
*UI will open at:* `http://localhost:8501`

### 4. Optional: Run Vector DB & Local LLM (Docker Compose)
To enable Qdrant and Ollama with Llama 3.1:
```bash
docker compose up -d
ollama pull llama3.1:8b
python scripts/index_qdrant.py
```
> **Resilient Architecture Note**: If Docker, Qdrant, or Ollama are not running, PrecedentIQ automatically and seamlessly operates using its high-speed in-memory BM25 + local vector search + deterministic legal reasoning fallback engine with **zero crashes**!

### 4A. Full Docker Deployment

The full Compose deployment runs the API, Streamlit UI, Qdrant, Redis, and Ollama:

```powershell
docker compose build
docker compose up -d qdrant redis ollama
docker compose exec ollama ollama pull llama3.2:3b
docker compose up -d api ui
```

Open `http://localhost:8502` for the UI, `http://localhost:8000/docs` for the API, and verify services with:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:11434/api/tags
```

The Docker Ollama model is stored in the persistent `ollama_storage` volume. The current configuration uses the local/cache dense retriever (`USE_QDRANT=false`); enable Qdrant only after running `scripts/index_qdrant.py` inside an application container.

To stop the deployment while preserving models and indexes:

```powershell
docker compose down
```

### 5. Run Benchmark Evaluation
To measure retrieval quality (Precision@5, MRR, nDCG@5):
```bash
python scripts/evaluate_retrieval.py
```

### 6. Run Automated Test Suite
```bash
python -m unittest discover tests/
```

---

## 🎯 Usage Flow

1. **Enter Query**: Ask a legal question (e.g., *"Whether a power of attorney holder can institute proceedings on behalf of the principal"*).
2. **Ground on Client Case (Optional)**: Drag-and-drop your client's petition or plaint (`.pdf`, `.docx`, `.txt`).
3. **Adjust Filters**: Select target court (Supreme Court / High Court) and precedent depth.
4. **Click Research**:
   - Inspect top precedents ranked by RRF score.
   - Review the extracted **Ratio Decidendi** vs. **Obiter Dicta**.
   - Review affirmative **Pro** arguments and opposing **Contra** points with factual distinctions.
5. **Export**: Click **"Export Chamber Brief (.docx)"** for an immediate, court-ready research brief.

---


