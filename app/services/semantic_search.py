from typing import Dict, List, Optional
import numpy as np

from app.config import config
from app.models.resume import ParsedResume
from app.models.job import JobDescription
from app.services.embedding_service import EmbeddingService
from app.database.qdrant import QdrantVectorDB

class SemanticSearchService:
    def __init__(self, embedding_service: Optional[EmbeddingService] = None, qdrant_db: Optional[QdrantVectorDB] = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.qdrant_db = qdrant_db or QdrantVectorDB()

    def compute_cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Computes cosine similarity between two vectors."""
        if vec_a is None or vec_b is None:
            raise ValueError("missing vector (neither the section nor the 'full' vector is available)")
        if np.shape(vec_a) != np.shape(vec_b):
            raise ValueError(f"Vector dimension mismatch: {np.shape(vec_a)} vs {np.shape(vec_b)}")
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        # Clip to [0.0, 1.0]
        return min(max(sim, 0.0), 1.0)

    def compute_section_similarity(
        self,
        resume_vectors: Dict[str, np.ndarray],
        jd_vectors: Dict[str, np.ndarray]
    ) -> float:
        """
        Computes section-weighted semantic similarity score.
        Semantic Score =
            40% Skills similarity +
            30% Experience similarity +
            20% Project similarity +
            10% Education similarity
        """
        sim_skills = self.compute_cosine_similarity(
            resume_vectors.get("skills", resume_vectors.get("full")),
            jd_vectors.get("skills", jd_vectors.get("full"))
        )
        sim_exp = self.compute_cosine_similarity(
            resume_vectors.get("experience", resume_vectors.get("full")),
            jd_vectors.get("experience", jd_vectors.get("full"))
        )
        sim_proj = self.compute_cosine_similarity(
            resume_vectors.get("projects", resume_vectors.get("full")),
            jd_vectors.get("projects", jd_vectors.get("full"))
        )
        sim_edu = self.compute_cosine_similarity(
            resume_vectors.get("education", resume_vectors.get("full")),
            jd_vectors.get("education", jd_vectors.get("full"))
        )

        w_skills = config.section_skills_weight
        w_exp = config.section_experience_weight
        w_proj = config.section_projects_weight
        w_edu = config.section_education_weight

        w_total = w_skills + w_exp + w_proj + w_edu
        if w_total <= 0:
            raise ValueError("section weights must have a positive sum")
        weighted_sim = (w_skills * sim_skills + w_exp * sim_exp + w_proj * sim_proj + w_edu * sim_edu) / w_total
        # Heuristic calibration of raw cosine to 0-1 (floor/ceiling configurable via SEMANTIC_COSINE_FLOOR/CEIL).
        # NOT validated against labelled data: dense bi-encoders typically give ~0.15 for unrelated text and
        # ~0.70 for a strong match, which is where the defaults come from. Treat as an ordinal score.
        floor, ceil = config.semantic_cosine_floor, config.semantic_cosine_ceil
        return min(max((weighted_sim - floor) / (ceil - floor), 0.0), 1.0)

    def search_candidates(self, jd_full_vector: np.ndarray, top_k: int = 500) -> List[Dict]:
        """Queries Qdrant vector database for initial candidate set."""
        return self.qdrant_db.search_similar(jd_full_vector, limit=top_k)
