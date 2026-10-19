"""Test bootstrap.

Heavy third-party packages are stubbed ONLY when they cannot be imported (air-gapped sandbox), so the pure-Python
scoring logic stays testable. Stubs never provide real behaviour: anything needing the real package
(embedding model inference, Qdrant storage, PDF parsing) must be skipped/mocked explicitly in the test.
"""
import sys, types


def _stub(name, **attrs):
    try:
        __import__(name)
        return False
    except Exception:
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        m.IS_TEST_STUB = True
        sys.modules[name] = m
        return True


SENTENCE_TRANSFORMERS_STUBBED = _stub("sentence_transformers", SentenceTransformer=object)
if _stub("qdrant_client", QdrantClient=object):
    _stub("qdrant_client.models", Distance=object, VectorParams=object, PointStruct=object)
    sys.modules["qdrant_client"].models = sys.modules["qdrant_client.models"]
_stub("fitz")
