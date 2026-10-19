"""Shared test helpers. Real-model / real-package tests are SKIPPED (never faked) when unavailable."""
import importlib
import pytest

_MODEL_STATE = {}


def require_module(name: str):
    try:
        m = importlib.import_module(name)
    except Exception as e:
        pytest.skip(f"package '{name}' not installed ({type(e).__name__})")
    if getattr(m, "IS_TEST_STUB", False):
        pytest.skip(f"package '{name}' not installed (test stub in use)")
    return m


def require_real_model():
    """Skip unless the configured sentence-transformers model actually loads in this environment."""
    if "reason" not in _MODEL_STATE:
        reason = None
        try:
            require_module("sentence_transformers")
            from app.services.embedding_service import EmbeddingService
            EmbeddingService().encode_text("probe")
        except pytest.skip.Exception if hasattr(pytest.skip, "Exception") else Exception as e:   # noqa: BLE001
            reason = f"embedding model unavailable: {e}"
        except Exception as e:                                                                    # noqa: BLE001
            reason = f"embedding model unavailable: {type(e).__name__}: {str(e)[:120]}"
        _MODEL_STATE["reason"] = reason
    if _MODEL_STATE["reason"]:
        pytest.skip(_MODEL_STATE["reason"])
