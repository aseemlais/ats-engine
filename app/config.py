import os
from pathlib import Path
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent

# Configuration precedence (highest first): per-request overrides (API) > process environment variables >
# <project>/.env file > code defaults. A .env file never overrides variables already set in the environment.
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=False)
except ImportError:  # python-dotenv is optional; without it only real environment variables are read
    pass

class ATSConfig(BaseModel):
    # Base paths
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    resumes_dir: Path = BASE_DIR / "data" / "resumes"
    jobs_dir: Path = BASE_DIR / "data" / "jobs"
    checkpoint_file: Path = BASE_DIR / "data" / "ingestion_checkpoint.json"
    database_file: Path = BASE_DIR / "data" / "structured_resumes.json"
    bm25_index_file: Path = BASE_DIR / "data" / "bm25_index.pkl"

    # Embedding settings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

    # Qdrant settings
    qdrant_url: str | None = os.getenv("QDRANT_URL", None) or None
    qdrant_path: str = os.getenv("QDRANT_PATH", str(BASE_DIR / "data" / "qdrant_storage"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "resumes_collection")

    # Bounded benefit (fraction of the 0-1 score) for relevant certifications when the JD does NOT mention certifications.
    # 0 disables it. Never a penalty; final score is capped at 100.
    cert_bonus_none: float = float(os.getenv("CERT_BONUS_NONE", "0.03"))

    # Scoring pipeline weights (default sum = 1.0)
    semantic_weight: float = float(os.getenv("SEMANTIC_WEIGHT", "0.35"))
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.20"))
    skill_weight: float = float(os.getenv("SKILL_WEIGHT", "0.27"))
    experience_weight: float = float(os.getenv("EXPERIENCE_WEIGHT", "0.10"))
    certification_weight: float = float(os.getenv("CERTIFICATION_WEIGHT", "0.08"))

    # Semantic degraded mode. Default OFF: if the embedding model cannot be loaded the request FAILS LOUDLY.
    # When ON, the semantic component is excluded (active weights are renormalised) and every result is
    # flagged degraded=True / scoring_mode="degraded_no_semantic". Degraded scores are NOT comparable with
    # full-model scores.
    allow_degraded_semantic: bool = os.getenv("ALLOW_DEGRADED_SEMANTIC", "false").lower() in ("true", "1", "yes")

    # Long-text chunking for embeddings (all-MiniLM-L6-v2 truncates at ~256 word-pieces, i.e. ~180 words)
    chunk_max_words: int = int(os.getenv("CHUNK_MAX_WORDS", "180"))
    chunk_overlap_words: int = int(os.getenv("CHUNK_OVERLAP_WORDS", "30"))
    max_chunks_per_section: int = int(os.getenv("MAX_CHUNKS_PER_SECTION", "8"))

    # Section-level semantic weights
    section_skills_weight: float = float(os.getenv("SECTION_WEIGHT_SKILLS", "0.40"))
    section_experience_weight: float = float(os.getenv("SECTION_WEIGHT_EXPERIENCE", "0.30"))
    section_projects_weight: float = float(os.getenv("SECTION_WEIGHT_PROJECTS", "0.20"))
    section_education_weight: float = float(os.getenv("SECTION_WEIGHT_EDUCATION", "0.10"))

    # Heuristic raw-cosine -> 0..1 calibration (unvalidated defaults; see semantic_search.compute_section_similarity)
    semantic_cosine_floor: float = float(os.getenv("SEMANTIC_COSINE_FLOOR", "0.15"))
    semantic_cosine_ceil: float = float(os.getenv("SEMANTIC_COSINE_CEIL", "0.70"))

    # Filtering & search controls
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "100"))
    hard_filter_enabled: bool = os.getenv("HARD_FILTER_ENABLED", "true").lower() in ("true", "1", "yes")
    strict_hard_filter: bool = os.getenv("STRICT_HARD_FILTER", "false").lower() in ("true", "1", "yes")
    shortlist_threshold: float = float(os.getenv("SHORTLIST_THRESHOLD", "60.0"))
    review_threshold: float = float(os.getenv("REVIEW_THRESHOLD", "45.0"))

    # Missing/not-applicable criteria (certifications, skills): "neutral_imputation" (default) or "legacy" (old 0.60 / 0.0 behaviour)
    na_policy: str = os.getenv("NA_POLICY", "neutral_imputation").lower()
    best_available_k: int = int(os.getenv("BEST_AVAILABLE_K", "5"))

    # BM25 normalisation: theoretical_bound (default) | reference_percentile (experimental) | legacy_ceiling (old, for comparison)
    bm25_normalization: str = os.getenv("BM25_NORMALIZATION", "theoretical_bound").lower()
    # BM25 query construction: structured (default, baseline) | filtered (also drops generic boilerplate words from free JD text)
    bm25_query_mode: str = os.getenv("BM25_QUERY_MODE", "structured").lower()
    # Reference corpus smaller than this is flagged unreliable and keeps unseen query terms in the denominator
    bm25_min_reference_docs: int = int(os.getenv("BM25_MIN_REFERENCE_DOCS", "30"))

    # Ingestion settings
    ingestion_batch_size: int = int(os.getenv("INGESTION_BATCH_SIZE", "256"))
    max_resume_size_mb: int = int(os.getenv("MAX_RESUME_SIZE_MB", "15"))

    def scoring_weights(self, overrides: dict | None = None) -> dict:
        """Nominal component weights (env/defaults, then per-request overrides). NOT normalised here: the score
        composer divides by the sum of the ACTIVE components' weights, so any positive scale works."""
        w = {"semantic": self.semantic_weight, "bm25": self.bm25_weight, "skill": self.skill_weight,
             "experience": self.experience_weight, "certification": self.certification_weight}
        for k, v in (overrides or {}).items():
            if v is not None and k in w:
                w[k] = float(v)
        return w

    def ensure_directories(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.resumes_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        Path(self.qdrant_path).mkdir(parents=True, exist_ok=True)


config = ATSConfig()
if not (0.0 <= config.review_threshold <= config.shortlist_threshold <= 100.0):
    raise ValueError(f"Need 0 <= REVIEW_THRESHOLD ({config.review_threshold}) <= SHORTLIST_THRESHOLD ({config.shortlist_threshold}) <= 100")
_bad = {k: v for k, v in config.scoring_weights().items() if v < 0}
if _bad or sum(config.scoring_weights().values()) <= 0:
    raise ValueError(f"Scoring weights must be non-negative with a positive sum: {config.scoring_weights()}")
if config.chunk_max_words < 20 or not (0 <= config.chunk_overlap_words < config.chunk_max_words) or config.max_chunks_per_section < 1:
    raise ValueError("Need CHUNK_MAX_WORDS >= 20, 0 <= CHUNK_OVERLAP_WORDS < CHUNK_MAX_WORDS, MAX_CHUNKS_PER_SECTION >= 1")
if not (0.0 <= config.semantic_cosine_floor < config.semantic_cosine_ceil <= 1.0):
    raise ValueError("Need 0 <= SEMANTIC_COSINE_FLOOR < SEMANTIC_COSINE_CEIL <= 1")
if config.na_policy not in ("neutral_imputation", "legacy"):
    raise ValueError("NA_POLICY must be 'neutral_imputation' or 'legacy'")
if config.bm25_query_mode not in ("structured", "filtered"):
    raise ValueError("BM25_QUERY_MODE must be 'structured' or 'filtered'")
config.ensure_directories()
