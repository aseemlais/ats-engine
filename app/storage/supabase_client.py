"""
Supabase client initialization and management.
Provides authenticated access to Supabase Postgres and Storage using the service_role key on the backend.
Safely validates configuration, sanitizes Project URLs, and guards against exposing secrets.
"""
import os
import logging
from typing import Optional, Any, List, Tuple

logger = logging.getLogger(__name__)

_supabase_client: Optional[Any] = None


def get_supabase_url() -> Optional[str]:
    """
    Returns the normalized Supabase project base URL.
    Strips trailing slashes and ensures subpaths like '/rest/v1' are excluded.
    """
    raw_url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
    if not raw_url:
        return None
    url = raw_url.strip().rstrip("/")
    if url.endswith("/rest/v1"):
        url = url[:-len("/rest/v1")].rstrip("/")
    return url if url else None


def get_supabase_service_key() -> Optional[str]:
    """
    Returns the backend service_role secret key without leading/trailing whitespace.
    Never logged or returned in user-facing error messages.
    """
    raw_key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    return raw_key.strip() if raw_key else None


def get_missing_supabase_env_vars() -> List[str]:
    """
    Identifies which required environment variables are absent.
    Returns variable names only (never values).
    """
    missing = []
    if not get_supabase_url():
        missing.append("SUPABASE_URL")
    if not get_supabase_service_key():
        missing.append("SUPABASE_SERVICE_KEY")
    return missing


def is_supabase_configured() -> bool:
    """Returns True if both Supabase URL and service key are present in the environment."""
    return len(get_missing_supabase_env_vars()) == 0


def get_supabase_client():
    """
    Returns a singleton Supabase Client instance initialized with the service key.
    Returns None if environment variables are not configured.
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    missing = get_missing_supabase_env_vars()
    if missing:
        logger.warning(
            "[Storage] Supabase persistent storage is not configured. Missing variables: %s",
            ", ".join(missing),
        )
        return None

    url = get_supabase_url()
    key = get_supabase_service_key()

    try:
        from supabase import create_client
        _supabase_client = create_client(url, key)
        logger.info(f"[Storage] Initialized Supabase client for project base: {url}")
        return _supabase_client
    except Exception as exc:
        logger.error(f"[Storage] Failed to initialize Supabase client: {exc}")
        return None


def reset_supabase_client(client: Optional[Any] = None) -> None:
    """Allows testing harnesses to inject or reset the client instance."""
    global _supabase_client
    _supabase_client = client
