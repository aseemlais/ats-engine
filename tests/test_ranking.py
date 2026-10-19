import pytest
from tests._helpers import require_real_model
from app.models.resume import ParsedResume
from app.models.job import JobDescription
from app.services.hybrid_ranker import HybridRankerService
from app.services.embedding_service import EmbeddingService

def test_hybrid_ranking_end_to_end():
    require_real_model()   # REAL-MODEL INTEGRATION TEST: skipped when the model cannot load
    ranker = HybridRankerService()
    embedding_service = EmbeddingService()

    # Candidate 1: Perfect match
    r1 = ParsedResume(
        candidate_id="CV_001",
        source_file="cv1.pdf",
        file_hash="h1",
        text_hash="th1",
        name="Alice NLP",
        cleaned_text="Senior NLP Engineer with Python, PyTorch, Transformers, NLP. 4 years experience.",
        raw_text="...",
        skills=["python", "pytorch", "transformers", "nlp"],
        experience_years=4.0,
        work_experience_years=4.0,
        job_titles=["Senior NLP Engineer"],
        certifications=["AWS Certified Machine Learning Specialist", "DeepLearning.AI NLP Specialization"],
        section_texts={
            "experience": "Senior NLP Engineer with Python, PyTorch, Transformers, NLP. 4 years experience.",
            "certifications": "AWS Certified Machine Learning Specialist, DeepLearning.AI NLP Specialization"
        }
    )

    # Candidate 2: Partial match
    r2 = ParsedResume(
        candidate_id="CV_002",
        source_file="cv2.pdf",
        file_hash="h2",
        text_hash="th2",
        name="Bob Dev",
        cleaned_text="Junior Developer with basic Python and Git. 1 year experience.",
        raw_text="...",
        skills=["python", "git"],
        experience_years=1.0,
        degrees=["bachelor"]
    )

    # Candidate 3: Non-tech
    r3 = ParsedResume(
        candidate_id="CV_003",
        source_file="cv3.pdf",
        file_hash="h3",
        text_hash="th3",
        name="Charlie Chef",
        cleaned_text="Executive Chef with cooking and catering management experience.",
        raw_text="...",
        skills=["leadership"],
        experience_years=5.0,
        degrees=["none"]
    )

    resumes = [r1, r2, r3]
    vec_map = {r.candidate_id: embedding_service.compute_resume_vectors(r) for r in resumes}

    # Index BM25
    ranker.bm25_service.build_index([r.candidate_id for r in resumes], [r.cleaned_text for r in resumes], persist=False)

    jd = JobDescription(
        job_id="JD_TEST",
        title="Senior NLP Engineer",
        raw_text="Senior NLP Engineer with Python, PyTorch, Transformers, NLP. Minimum 3 years experience.",
        required_skills=["python", "pytorch", "transformers", "nlp"],
        minimum_experience_years=3.0,
        mandatory_skills=["python", "nlp"]
    )

    response = ranker.rank_candidates(
        resumes=resumes,
        resume_vectors_map=vec_map,
        jd=jd,
        top_k=3
    )

    # Alice must be Rank 1
    assert len(response.results) > 0
    top_cand = response.results[0]
    assert top_cand.candidate_id == "CV_001"
    assert top_cand.rank == 1
    assert top_cand.final_score > 70.0
    assert "Alice" in top_cand.name
    assert "✓" in top_cand.explanation
