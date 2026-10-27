"""Supabase storage and persistence integration for ATS candidate resumes."""
from app.storage.supabase_client import get_supabase_client, is_supabase_configured
from app.storage.resume_storage import store_resume, check_duplicate_by_hash

__all__ = [
    "get_supabase_client",
    "is_supabase_configured",
    "store_resume",
    "check_duplicate_by_hash",
]
