# Expose ASGI FastAPI application so `app:app` works seamlessly.
# Lazy import so that importing sub-modules (e.g. app.services.*) in unit tests
# does NOT eagerly initialise ATSScreeningAdapter / EmbeddingService, which
# would hang in sandboxed environments with no network access.
def __getattr__(name: str):
    if name == "app":
        from app.server import app as _app
        return _app
    raise AttributeError(f"module 'app' has no attribute {name!r}")
