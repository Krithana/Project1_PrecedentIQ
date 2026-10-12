import sys
import math
import logging
from pathlib import Path
from typing import List, Dict, Set, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.precedentiq.config import settings
from src.precedentiq.data.loader import DataLoader
from src.precedentiq.schemas.retrieval import SearchQuery, FilterParams
from src.precedentiq.retrieval.hybrid_engine import HybridRetrievalEngine
from src.precedentiq.retrieval.bm25_retriever import BM25Retriever
from src.precedentiq.retrieval.dense_retriever import DenseRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluation")

# Ground truth test cases representative of Indian legal queries and relevant judgment signals
EVALUATION_BENCHMARK = [
    {
        "query": "Whether power of attorney holder can institute proceedings on behalf of principal",
        "relevant_keywords": ["power of attorney", "harnarayan", "agent", "principal"],
        "expected_section": "order"
    },
    {
        "query": "Quashing of FIR under Section 482 CrPC abuse of process",
        "relevant_keywords": ["482", "cr.p.c", "quash", "inherent powers"],
        "expected_section": "order"
    },
    {
        "query": "Retrospective amendment of rules during pendency of recruitment selection process",
        "relevant_keywords": ["panchayat karmi", "retrospective", "amendment", "advertisement"],
        "expected_section": "order"
    },
    {
        "query": "Impleadment of subsequent pendente lite purchaser Order 1 Rule 10 CPC",
        "relevant_keywords": ["order 1 rule 10", "pendente lite", "subsequent purchaser", "cpc"],
        "expected_section": "order"
    },
    {
        "query": "Dismissal of appeal for want of prosecution non-appearance of counsel",
        "relevant_keywords": ["want of prosecution", "dismissed", "appellant", "prosecuting"],
        "expected_section": "order"
    }
]


def dcg_at_k(relevance_scores: List[int], k: int = 10) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(relevance_scores: List[int], k: int = 10) -> float:
    actual_dcg = dcg_at_k(relevance_scores, k)
    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg_at_k(ideal_scores, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_retrieval_pipeline():
    logger.info("Initializing PrecedentIQ Retrieval Evaluation...")
    loader = DataLoader.get_instance()
    chunks = loader.load_all_chunks()
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks)
    hybrid = HybridRetrievalEngine.get_instance()

    k_eval = 5
    metrics = {
        "BM25": {"mrr": [], "ndcg": [], "precision": []},
        "Dense": {"mrr": [], "ndcg": [], "precision": []},
        "Hybrid_RRF": {"mrr": [], "ndcg": [], "precision": []},
    }

    for test_case in EVALUATION_BENCHMARK:
        query = test_case["query"]
        keywords = test_case["relevant_keywords"]

        # Run BM25
        bm25_hits = bm25.search(query, top_k=k_eval)
        # Run Dense
        dense_hits = dense.search(query, top_k=k_eval)
        # Run Hybrid
        search_req = SearchQuery(query=query, top_k=k_eval)
        hybrid_res = hybrid.search(search_req)

        pipelines = {
            "BM25": [c.text.lower() for c, _, _ in bm25_hits],
            "Dense": [c.text.lower() for c, _, _ in dense_hits],
            "Hybrid_RRF": [sc.chunk.text.lower() for sc in hybrid_res.scored_chunks],
        }

        for name, hit_texts in pipelines.items():
            rel_scores = []
            first_relevant_rank = None
            for idx, text in enumerate(hit_texts, start=1):
                match = any(kw in text for kw in keywords)
                rel_scores.append(1 if match else 0)
                if match and first_relevant_rank is None:
                    first_relevant_rank = idx

            precision = sum(rel_scores) / max(len(rel_scores), 1)
            mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
            ndcg = ndcg_at_k(rel_scores, k=k_eval)

            metrics[name]["precision"].append(precision)
            metrics[name]["mrr"].append(mrr)
            metrics[name]["ndcg"].append(ndcg)

    print("\n" + "=" * 65)
    print("      PRECEDENTIQ RETRIEVAL BENCHMARK EVALUATION RESULTS      ")
    print("=" * 65)
    print(f"{'Retrieval Pipeline':<18} | {'Precision@5':<12} | {'MRR':<10} | {'nDCG@5':<10}")
    print("-" * 65)

    for name, vals in metrics.items():
        avg_prec = sum(vals["precision"]) / len(vals["precision"]) if vals["precision"] else 0
        avg_mrr = sum(vals["mrr"]) / len(vals["mrr"]) if vals["mrr"] else 0
        avg_ndcg = sum(vals["ndcg"]) / len(vals["ndcg"]) if vals["ndcg"] else 0
        print(f"{name:<18} | {avg_prec:<12.3f} | {avg_mrr:<10.3f} | {avg_ndcg:<10.3f}")

    print("=" * 65)
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    evaluate_retrieval_pipeline()
