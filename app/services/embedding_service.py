import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Ensure SSL certificate bundle is valid and points to certifi if missing or broken
try:
    import certifi
    ca_bundle = certifi.where()
    if os.path.exists(ca_bundle):
        os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
        os.environ.setdefault("CURL_CA_BUNDLE", ca_bundle)
except Exception:
    pass

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # package missing: model-dependent calls raise EmbeddingUnavailableError, imports still work
    SentenceTransformer = None  # type: ignore

from app.config import config
from app.models.resume import ParsedResume
from app.models.job import JobDescription
from app.services.text_chunking import aggregate_vectors, chunk_text

# Semantic sections and the resume sections that feed each one. Internship text is merged into "experience"
# (so it is no longer invisible to semantic matching) and is matched against the JD's experience query.
SECTION_SOURCES: Dict[str, Tuple[str, ...]] = {
    "skills": ("skills",),
    "experience": ("experience", "internship"),
    "projects": ("projects",),
    "education": ("education",),
}
SEMANTIC_SECTIONS = tuple(SECTION_SOURCES)


class EmbeddingUnavailableError(RuntimeError):
    """The embedding model cannot be loaded / used (missing package, no cache and no network, ...)."""


def section_text(resume: ParsedResume, section: str) -> str:
    """Text used for a semantic section ('' when the resume has none of the source sections)."""
    parts = [(resume.section_texts or {}).get(src, "").strip() for src in SECTION_SOURCES[section]]
    return "\n".join(p for p in parts if p)


class EmbeddingService:
    """Sentence-embedding wrapper. Long texts are chunked (bounded) and mean-pooled into ONE unit vector per
    text, so the stored/consumed representation (one vector per section) is unchanged."""

    def __init__(self, model_name: str | None = None, batch_size: int | None = None,
                 model=None, dimension: int | None = None):
        self.model_name = model_name or config.embedding_model
        self.batch_size = batch_size or config.embedding_batch_size
        self.dimension = dimension or config.embedding_dimension
        self._model = model            # injectable (tests use a mock; production loads lazily)
        self._load_error: str | None = None

    @property
    def model(self):
        if self._model is None:
            if self._load_error:                       # fail fast: never retry a blocked download
                raise EmbeddingUnavailableError(self._load_error)
            try:
                if SentenceTransformer is None or SentenceTransformer is object:
                    raise ImportError("sentence-transformers is not installed")
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:                    # noqa: BLE001 - any load failure means "unavailable"
                self._load_error = f"Embedding model '{self.model_name}' unavailable: {type(e).__name__}: {str(e)[:200]}"
                raise EmbeddingUnavailableError(self._load_error) from e
        return self._model

    def _zero(self) -> np.ndarray:
        return np.zeros(self.dimension, dtype=np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Raw batch encode (no chunking). Validates the vector dimension."""
        clean_texts = [t if t.strip() else " " for t in texts]
        vecs = np.asarray(self.model.encode(clean_texts, batch_size=self.batch_size,
                                            normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: model returned {vecs.shape}, expected (*, {self.dimension}); "
                             f"check EMBEDDING_MODEL / EMBEDDING_DIMENSION")
        return vecs

    def _encode_many(self, texts: List[str]) -> List[np.ndarray]:
        """Chunk each text, encode ALL chunks in one batched call, mean-pool per text. Empty text -> zero vector."""
        chunk_lists = [chunk_text(t, config.chunk_max_words, config.chunk_overlap_words, config.max_chunks_per_section)[0]
                       for t in texts]
        flat = [c for cl in chunk_lists for c in cl]
        vecs = self.encode_batch(flat) if flat else np.zeros((0, self.dimension), dtype=np.float32)
        out: List[np.ndarray] = []
        pos = 0
        for cl in chunk_lists:
            if not cl:
                out.append(self._zero())
                continue
            out.append(aggregate_vectors(vecs[pos:pos + len(cl)], [len(c.split()) for c in cl]))
            pos += len(cl)
        return out

    def encode_text(self, text: str) -> np.ndarray:
        """Normalised embedding of one text (chunked + pooled when long; zero vector when empty)."""
        return self._encode_many([text])[0]

    def compute_resume_vectors(self, resume: ParsedResume) -> Dict[str, np.ndarray]:
        """Vectors for 'full' plus each semantic section; a section the resume lacks falls back to the full vector."""
        texts = [resume.cleaned_text] + [section_text(resume, s) for s in SEMANTIC_SECTIONS]
        vecs = self._encode_many(texts)
        out = {"full": vecs[0]}
        for s, t, v in zip(SEMANTIC_SECTIONS, texts[1:], vecs[1:]):
            out[s] = v if t.strip() else out["full"]
        return out

    def compute_corpus_vectors_batched(self, resumes: List[ParsedResume]) -> Dict[str, Dict[str, np.ndarray]]:
        """Batched version of compute_resume_vectors over a corpus (all chunks of all texts in shared batches)."""
        if not resumes:
            return {}
        texts: List[str] = []
        for r in resumes:
            texts.append(r.cleaned_text)
            texts.extend(section_text(r, s) for s in SEMANTIC_SECTIONS)
        vecs = self._encode_many(texts)
        stride = 1 + len(SEMANTIC_SECTIONS)
        corpus: Dict[str, Dict[str, np.ndarray]] = {}
        for i, r in enumerate(resumes):
            base = i * stride
            d = {"full": vecs[base]}
            for j, s in enumerate(SEMANTIC_SECTIONS, start=1):
                d[s] = vecs[base + j] if texts[base + j].strip() else d["full"]
            corpus[r.candidate_id] = d
        return corpus

    def compute_jd_vectors(self, jd: JobDescription) -> Dict[str, np.ndarray]:
        """Section queries for the JD. Wording adapts to what the JD actually states:
        no experience minimum -> a role-experience/internship query (not '0.0 years'); supporting (trainable)
        skills are included in the skills query; an empty skills query falls back to the full JD vector."""
        skills = list(dict.fromkeys(jd.required_skills + jd.preferred_skills + jd.supporting_skills))
        if jd.has_experience_requirement and jd.minimum_experience_years > 0:
            exp_q = f"{jd.title} with {jd.minimum_experience_years:g} years experience"
        else:
            exp_q = f"{jd.title} role: relevant work experience, internships and hands-on practice"
        resp = jd.responsibilities.strip() or ("using " + ", ".join(skills) if skills else "")
        edu_q = " ".join(jd.education_requirements) if jd.education_requirements else f"{jd.title} degree education background"
        vecs = self._encode_many([jd.raw_text, " ".join(skills), exp_q,
                                  f"{jd.title} responsibilities and projects: {resp}", edu_q])
        out = dict(zip(("full", "skills", "experience", "projects", "education"), vecs))
        if not skills:
            out["skills"] = out["full"]
        return out
