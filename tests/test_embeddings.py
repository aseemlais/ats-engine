import pytest
from tests._helpers import require_real_model
import numpy as np
from app.services.embedding_service import EmbeddingService

def test_embedding_generation():
    require_real_model()   # REAL-MODEL INTEGRATION TEST: skipped when the model cannot load
    service = EmbeddingService()
    vec = service.encode_text("Machine Learning Engineer with Python experience")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    # Check normalized (L2 norm should be approximately 1.0)
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-3

def test_batch_embeddings():
    require_real_model()   # REAL-MODEL INTEGRATION TEST: skipped when the model cannot load
    service = EmbeddingService()
    texts = ["FastAPI developer", "Data scientist", "DevOps engineer"]
    vecs = service.encode_batch(texts)
    assert vecs.shape == (3, 384)
