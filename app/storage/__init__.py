"""Supabase storage and persistence integration for ATS candidate resumes."""
from app.storage.supabase_client import (
    get_supabase_client,
    is_supabase_configured,
    get_missing_supabase_env_vars,
)
from app.storage.resume_storage import store_resume, check_duplicate_by_hash
from app.storage.candidate_repository import (
    list_candidates,
    search_candidates,
    get_candidate,
    get_resume_download_url,
    delete_candidate,
    CandidateNotFoundError,
    RepositoryUnavailableError,
    PartialDeletionError,
)

__all__ = [
    "get_supabase_client",
    "is_supabase_configured",
    "get_missing_supabase_env_vars",
    "store_resume",
    "check_duplicate_by_hash",
    "list_candidates",
    "search_candidates",
    "get_candidate",
    "get_resume_download_url",
    "delete_candidate",
    "CandidateNotFoundError",
    "RepositoryUnavailableError",
    "PartialDeletionError",
]
