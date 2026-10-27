"""
Supabase client initialization and management.
Provides authenticated access to Supabase Postgres and Storage using the service_role key on the backend.
"""
import os
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

_supabase_client: Optional[Any] = None


def get_supabase_url() -> Optional[str]:
    return os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")


def get_supabase_service_key() -> Optional[str]:
    return os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")


def is_supabase_configured() -> bool:
    """Returns True if Supabase URL and service key are present in the environment."""
    url = get_supabase_url()
    key = get_supabase_service_key()
    return bool(url and key)


def get_supabase_client():
    """
    Returns a singleton Supabase Client instance initialized with the service key.
    Returns None if environment variables are not configured.
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    url = get_supabase_url()
    key = get_supabase_service_key()

    if not url or not key:
        return None

    try:
        from supabase import create_client
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as exc:
        logger.error(f"Failed to initialize Supabase client: {exc}")
        return None


def reset_supabase_client(client: Optional[Any] = None) -> None:
    """Allows testing harnesses to inject or reset the client instance."""
    global _supabase_client
    _supabase_client = client
