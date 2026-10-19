import pytest
from app.services.bm25_search import BM25SearchService

def test_bm25_lexical_scoring(tmp_path):
    # temp index path: the test must never overwrite data/bm25_index.pkl
    service = BM25SearchService(index_path=tmp_path / "bm25_test.pkl")
    candidate_ids = ["CV_001", "CV_002", "CV_003"]
    corpus = [
        "Senior Python engineer working with FastAPI, Docker, and PostgreSQL databases",
        "Frontend React developer building web interfaces with TypeScript and CSS",
        "DevOps engineer managing Kubernetes clusters and Terraform on AWS"
    ]
    service.build_index(candidate_ids, corpus)

    query = "Looking for Senior Python engineer with FastAPI and Docker"
    scores = service.search(query, top_k=3)

    assert "CV_001" in scores
    # CV_001 should score higher than CV_002 on Python query
    assert scores["CV_001"] > scores.get("CV_002", 0.0)
