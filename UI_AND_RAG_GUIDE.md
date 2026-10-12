# PrecedentIQ UI and RAG Guide

This document explains the current Streamlit interface, the numbers shown in it, and how a user query moves through the Retrieval-Augmented Generation (RAG) workflow. It describes the visible UI and its current implementation. Extracted legal holdings and generated arguments require advocate verification.

## 1. Starting the UI

Local launch:

```powershell
streamlit run src\precedentiq\streamlit_app.py --server.port 8502
```

Docker URL:

```text
http://localhost:8502
```

The page is a wide, dark-themed legal research terminal with an expanded sidebar.

## 2. Visible UI components

The interface contains:

1. A sidebar with corpus statistics and controls.
2. A PrecedentIQ header.
3. A legal query input.
4. Three sample-query buttons.
5. An optional client document uploader.
6. A research execution button.
7. Result tabs for discovery, holdings, arguments, and evidence.
8. A DOCX chamber-brief export button after results are available.

## 3. Sidebar corpus statistics

The current UI corpus displays:

| UI metric | Current value | Meaning |
|---|---:|---|
| Searchable Chunks | 11,720 | Searchable sections of judgment text |
| Indexed Judgments | 1,988 | Distinct judgments represented in the index |
| Supreme Court | 11,486 chunks | Supreme Court chunk count |
| High Courts | 234 chunks | High Court chunk count |

The court values are chunk counts, not judgment counts. A single judgment can produce multiple chunks.

## 4. Retrieval controls

### Target Jurisdiction

Available choices:

- **All Jurisdictions**: searches both court groups.
- **Supreme Court**: filters to `supreme_court` chunks.
- **High Court**: filters to `high_court` chunks.

### Precedents to Retrieve

The `top_k` slider controls the final number of ranked chunks.

| Setting | Value |
|---|---:|
| Minimum | 1 |
| Default | 5 |
| Maximum | 15 |

This is a chunk count, not necessarily a unique-case count.

### BM25 Keyword Weight

| Setting | Value |
|---|---:|
| Minimum | 0.00 |
| Default | 0.50 |
| Maximum | 1.00 |
| Step | 0.05 |

BM25 represents sparse keyword retrieval. It is useful for exact legal phrases, section numbers, article numbers, rules, statutes, and procedural terms.

### Dense Semantic Weight

| Setting | Value |
|---|---:|
| Minimum | 0.00 |
| Default | 0.50 |
| Maximum | 1.00 |
| Step | 0.05 |

Dense retrieval represents semantic similarity. It is useful when the query and judgment express a similar legal concept with different words.

The UI does not require the two weights to add up to `1.00`. The default is `0.50 + 0.50`.

### Embedding value

The sidebar displays:

```text
Embedding: all-MiniLM-L6-v2 (384d)
```

`384d` is the embedding vector dimension. It is not a relevance percentage or score.

## 5. AI engine selector

The UI offers three choices:

### Llama 3.2 (3B - Fast CPU)

Uses Ollama model `llama3.2:3b`. The UI limits holding extraction to 2 unique judgments. If the model is unavailable, the deterministic engine is used.

### Deterministic Legal Engine (Instant)

Uses the local rule-based engine without Ollama. The UI permits up to 4 unique judgments for holding extraction. Its results are candidate classifications and require verification.

### Llama 3.1 (8B - Detailed)

Uses Ollama model `llama3.1:8b`. The UI limits holding extraction to 2 unique judgments and may require more CPU time and memory.

## 6. Query and upload area

The main text area accepts a natural-language legal question, issue, statutory provision, or short fact pattern.

Default query:

```text
Whether a power of attorney holder can institute proceedings on behalf of the principal
```

Sample buttons:

- Power of Attorney
- Section 482 CrPC
- Retrospective Rules

Supported upload types:

```text
.pdf, .docx, .txt, .md
```

An uploaded petition, plaint, FIR summary, or other client document is parsed into a context summary. Uploading alone does not start research; the user must press the research button.

## 7. UI RAG workflow

The visible research workflow is:

```text
Legal query + jurisdiction + weights
              |
              v
BM25 keyword retrieval + dense semantic retrieval
              |
              v
Reciprocal Rank Fusion and court weighting
              |
              v
Contextual reranking
              |
              v
Holding extraction for unique judgments
              |
              v
Pro/Contra argument synthesis
              |
              v
Precedent, holding, argument, evidence tabs
              |
              v
Optional DOCX export
```

RAG means that retrieved judgment chunks are supplied as context to the holding and argument components. The generation stage is not intended to answer from an empty prompt.

## 8. Retrieval and score values

### Candidate pool

Before final results are shown, the system uses:

```text
candidate_pool_size = max(top_k * 4, 30)
```

Examples:

| `top_k` | Candidate pool |
|---:|---:|
| 1 | 30 |
| 5 | 30 |
| 10 | 40 |
| 15 | 60 |

### Reciprocal Rank Fusion

The current RRF constant is:

```text
RRF_K = 60
```

The initial fused score is:

```text
RRF score = BM25 weight / (60 + BM25 rank)
          + Dense weight / (60 + Dense rank)
```

If a chunk appears in only one retrieval list, it receives only that method's contribution. If it appears in both, both contributions are added.

### Court hierarchy values

| Court | Multiplier |
|---|---:|
| Supreme Court | 1.20 |
| High Court | 1.00 |

The initial score uses:

```text
initial final score = RRF score * court multiplier
```

This is a ranking preference, not proof of legal applicability.

### Heuristic reranking values

When the cross-encoder is unavailable, the visible result ranking uses these implementation values:

| Section | Multiplier |
|---|---:|
| Ratio decidendi | 1.30 |
| Reasoning | 1.30 |
| Order | 1.15 |
| Arguments | 0.95 |
| Other sections | 1.00 |

Treatment signals:

| Signal | Multiplier |
|---|---:|
| Followed or approved | 1.20 |
| Reversed or overruled | 0.85 |
| No signal | 1.00 |

Query matching:

| Match | Multiplier |
|---|---:|
| Exact full query phrase | 1.25 |
| No exact phrase | 1.00 |

Term overlap adds:

```text
1.0 + 0.5 * overlap_ratio
```

where `overlap_ratio` is matched query terms divided by total query terms.

The heuristic formula is:

```text
final score = RRF score
            * court multiplier
            * section multiplier
            * treatment multiplier
            * phrase multiplier
            * overlap multiplier
```

If the cross-encoder is available, its prediction replaces the heuristic final score. Therefore, the UI's `Score: 0.xxx` is a ranking score, not a probability, percentage, or legal-confidence guarantee.

## 9. Result tabs

### Precedent Discovery

Each result card can show:

- rank number;
- case citation;
- Supreme Court or High Court badge;
- section badge;
- treatment signals;
- score formatted to three decimal places;
- short text preview;
- expandable full chunk text.

The card represents a retrieved chunk. Several cards may originate from the same judgment.

### Holding Analysis

The holding tab can show:

- case title or judgment ID;
- court;
- bench;
- year/date;
- candidate ratio decidendi;
- candidate non-ratio observation;
- operative order;
- statutes considered.

The words "candidate" and "verify" are important. A heading, party submission, dissent, reference to a larger Bench, or procedural passage must not automatically be treated as binding ratio.

### Argument Outline

The Pro column may contain:

- strength label;
- legal proposition;
- application reasoning;
- supporting citation;
- source excerpt.

The Contra column may contain:

- possible opposing objection;
- adverse authority;
- source excerpt;
- possible factual or legal distinction.

The fallback engine uses cautious wording when it cannot verify that an authority directly decides the query.

### Evidence and Citations

The evidence area exposes:

- chunk ID;
- citation;
- court;
- section;
- final score;
- source file;
- full chunk content.

An internal chunk ID is a repository trace reference. It is not a formal court citation by itself.

## 10. DOCX export

After results exist, the UI provides **Export Chamber Brief (.docx)**. The export can contain:

- query and generated time;
- retrieved authority matrix;
- candidate holding analysis;
- Pro/Contra arguments;
- source excerpts;
- verification notice.

The export is a research draft. Read the original judgment and verify every legal proposition before court use.

## 11. Meaning of important UI numbers

| Number | Meaning |
|---:|---|
| 11,720 | Current searchable chunks |
| 1,988 | Current distinct judgments |
| 11,486 | Supreme Court chunks |
| 234 | High Court chunks |
| 384 | Dense embedding dimensions |
| 1 to 15 | Final result slider range |
| 0.00 to 1.00 | BM25 and dense weight ranges |
| 0.05 | Weight slider increment |
| 60 | RRF rank constant |
| 1.20 | Supreme Court multiplier |
| 1.00 | High Court multiplier |
| 1.30 | Ratio/reasoning section multiplier |
| 1.15 | Order section multiplier |
| 0.95 | Arguments section multiplier |
| 1.25 | Exact-query phrase multiplier |

## 12. Recommended user procedure

1. Start with **All Jurisdictions**.
2. Keep `top_k` at `5` for the first search.
3. Keep BM25 and dense weights at `0.50` each for broad questions.
4. Increase BM25 for exact sections, articles, rules, or statutory phrases.
5. Increase dense weight for conceptual questions expressed in ordinary language.
6. Inspect the Evidence tab before relying on a result.
7. Read the original judgment before treating a candidate ratio as binding.
8. Treat the DOCX as a research draft, not an unreviewed filing.

## 13. Current UI limitations

- A high score means ranking relevance, not legal correctness.
- Multiple chunks can belong to one judgment.
- Retrieved authorities may be factually or procedurally distinguishable.
- Ratio and obiter extraction require legal review.
- Missing Ollama models trigger deterministic fallback behavior.
- Docker certificate or model-cache problems can disable dense and cross-encoder retrieval.
- The UI does not independently certify every generated legal proposition against the original judgment.

