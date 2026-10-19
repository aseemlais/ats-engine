import time
import math
import sys
from pathlib import Path
from typing import List, Dict, Set
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.storage import MetadataStorage
from app.services.jd_parser import JDParserService
from app.services.embedding_service import EmbeddingService
from app.services.bm25_search import BM25SearchService
from app.services.hybrid_ranker import HybridRankerService

def compute_ndcg_at_k(ranked_labels: List[int], k: int) -> float:
    ranked_labels = ranked_labels[:k]
    dcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(ranked_labels))
    ideal_labels = sorted(ranked_labels, reverse=True)
    idcg = sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(ideal_labels))
    return dcg / idcg if idcg > 0 else 0.0

def compute_mrr(ranked_labels: List[int]) -> float:
    for idx, rel in enumerate(ranked_labels):
        if rel > 0:
            return 1.0 / (idx + 1)
    return 0.0

def run_evaluation():
    print("=" * 75)
    print("   RETRIEVAL ACCURACY & QUALITY EVALUATION (P@K, R@K, NDCG, MRR)")
    print("=" * 75)

    storage = MetadataStorage()
    jd_parser = JDParserService()
    embedding_service = EmbeddingService()
    bm25_service = BM25SearchService()
    ranker = HybridRankerService()

    resumes = storage.get_all_resumes()
    if len(resumes) < 10:
        print("Need at least 10 resumes indexed for meaningful evaluation.")
        return

    # Define Evaluation Query (Ground Truth: NLP/ML profiles are Relevant (2), Data Science is Partially Relevant (1), Others Not Relevant (0))
    test_jd_text = """Job Title: Senior NLP Engineer
Requirements:
- Strong experience in Python, PyTorch, Transformers, NLP, Machine Learning
- 3+ years experience in Natural Language Processing and Deep Learning
Preferred Skills:
- HuggingFace, FastAPI, Docker, BERT"""

    parsed_jd = jd_parser.parse_job_description(test_jd_text)

    # Establish Ground Truth Labels based on candidate skills
    target_skills = {"python", "nlp", "transformers", "pytorch", "machine learning", "deep learning"}
    ground_truth: Dict[str, int] = {}
    total_relevant = 0

    for r in resumes:
        cand_skills = set(r.skills)
        overlap = len(cand_skills & target_skills)
        if overlap >= 3:
            relevance = 2 # Highly relevant
            total_relevant += 1
        elif overlap >= 1:
            relevance = 1 # Partially relevant
            total_relevant += 1
        else:
            relevance = 0 # Not relevant
        ground_truth[r.candidate_id] = relevance

    print(f"Evaluation Dataset: {len(resumes)} resumes | Relevant/Partially Relevant pool: {total_relevant}")

    # Precompute vectors
    vec_map = {r.candidate_id: embedding_service.compute_resume_vectors(r) for r in resumes}
    jd_vectors = embedding_service.compute_jd_vectors(parsed_jd)
    bm25_scores = bm25_service.search(parsed_jd.raw_text, top_k=len(resumes))

    methods = [
        ("Keyword Matching", {"semantic": 0.0, "bm25": 0.0, "skill": 1.0, "experience": 0.0}, False),
        ("BM25 Lexical", {"semantic": 0.0, "bm25": 1.0, "skill": 0.0, "experience": 0.0}, False),
        ("Semantic Search", {"semantic": 1.0, "bm25": 0.0, "skill": 0.0, "experience": 0.0}, False),
        ("Hybrid (BM25+Sem)", {"semantic": 0.60, "bm25": 0.40, "skill": 0.0, "experience": 0.0}, False),
        ("Full ATS Pipeline", {"semantic": 0.45, "bm25": 0.30, "skill": 0.15, "experience": 0.10}, True),
    ]

    print(f"\n{'Method':<20} | {'P@10':<7} | {'P@50':<7} | {'R@10':<7} | {'R@50':<7} | {'NDCG@50':<8} | {'MRR':<6} | {'Latency'}")
    print("-" * 80)

    for name, weights, use_hard_filter in methods:
        ranker.hard_filter_service.enabled = use_hard_filter
        t0 = time.perf_counter()
        resp = ranker.rank_candidates(
            resumes=resumes,
            resume_vectors_map=vec_map,
            jd=parsed_jd,
            top_k=min(50, len(resumes)),
            weights=weights
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        ranked_labels = [ground_truth.get(c.candidate_id, 0) for c in resp.results]

        # Precision@K
        p_10 = sum(1 for l in ranked_labels[:10] if l > 0) / min(10, len(ranked_labels)) if ranked_labels else 0
        p_50 = sum(1 for l in ranked_labels[:50] if l > 0) / min(50, len(ranked_labels)) if ranked_labels else 0

        # Recall@K
        r_10 = sum(1 for l in ranked_labels[:10] if l > 0) / total_relevant if total_relevant > 0 else 0
        r_50 = sum(1 for l in ranked_labels[:50] if l > 0) / total_relevant if total_relevant > 0 else 0

        # NDCG@50 & MRR
        ndcg_50 = compute_ndcg_at_k(ranked_labels, 50)
        mrr = compute_mrr(ranked_labels)

        print(f"{name:<20} | {p_10:<7.2f} | {p_50:<7.2f} | {r_10:<7.2f} | {r_50:<7.2f} | {ndcg_50:<8.3f} | {mrr:<6.3f} | {latency_ms:.1f}ms")

    print("=" * 80)

if __name__ == "__main__":
    run_evaluation()
