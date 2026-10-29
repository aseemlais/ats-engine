"""
Candidate Repository: read operations against ai_chatbot."CandidateResume".

Provides list, search, get-by-ID, and delete operations.
All DB access uses the shared Supabase client and the _get_table_ref helper so
the ai_chatbot schema is consistently targeted.

Security notes:
  - The service-role key stays in the backend process; no credentials are returned
    to callers.
  - Signed URLs for file downloads are created on demand with a short TTL
    (default 300 s) so permanent public links are never issued.
  - resumeText is intentionally excluded from list / search responses and only
    included when the caller explicitly requests a single candidate detail.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from app.storage.supabase_client import get_supabase_client
from app.storage.resume_storage import _get_table_ref, DatabaseError, StorageError

logger = logging.getLogger(__name__)

# Columns returned in list / search responses (no full resumeText, no secrets)
_LIST_COLUMNS = (
    "id, candidateName, email, skills, jobRole, originalFilename, createdAt, storageObjectPath"
)
# Full columns used when a single candidate is retrieved
_DETAIL_COLUMNS = (
    "id, candidateName, email, skills, jobRole, originalFilename, "
    "resumeText, createdAt, updatedAt, storageObjectPath, fileHash"
)

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
SIGNED_URL_TTL_SECONDS = 300  # 5 minutes


class CandidateNotFoundError(Exception):
    """Raised when the requested candidate ID does not exist."""


class RepositoryUnavailableError(Exception):
    """Raised when Supabase is not configured or unreachable."""


class PartialDeletionError(Exception):
    """
    Raised when Storage was successfully deleted but the DB row could not be removed.
    Carries the storage_object_path so the caller knows the file is already gone;
    a retry should skip Storage and attempt DB cleanup only.
    """
    def __init__(self, message: str, storage_object_path: str):
        super().__init__(message)
        self.storage_object_path = storage_object_path


def _require_client(client: Optional[Any] = None) -> Any:
    """Return a Supabase client or raise RepositoryUnavailableError."""
    supabase = client or get_supabase_client()
    if not supabase:
        raise RepositoryUnavailableError(
            "Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY."
        )
    return supabase


def _bucket_name() -> str:
    return os.environ.get("SUPABASE_STORAGE_BUCKET", "resumes")


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def list_candidates(
    limit: int = DEFAULT_PAGE_SIZE,
    offset: int = 0,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Return a paginated list of candidates sorted by createdAt DESC.

    Returns:
        {
            "candidates": [...],
            "total": int,
            "limit": int,
            "offset": int,
        }
    Raises:
        RepositoryUnavailableError if Supabase is not configured.
        DatabaseError on DB query failure.
    """
    limit = min(max(1, limit), MAX_PAGE_SIZE)
    offset = max(0, offset)

    supabase = _require_client(client)
    table = _get_table_ref(supabase, "CandidateResume")

    try:
        # Count query
        count_res = table.select("id", count="exact").execute()
        total = count_res.count if hasattr(count_res, "count") and count_res.count is not None else 0

        # Data query with pagination and sort
        data_res = (
            table.select(_LIST_COLUMNS)
            .order("createdAt", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        candidates = data_res.data or []
        return {
            "candidates": candidates,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    except Exception as exc:
        logger.error("[Repository] list_candidates failed: %s", exc)
        raise DatabaseError(f"Failed to list candidates: {exc}") from exc


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_candidates(
    query: str,
    limit: int = DEFAULT_PAGE_SIZE,
    offset: int = 0,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Case-insensitive search across candidateName, email, jobRole, and skills.

    Uses PostgREST ilike operator for text fields and the `cs` (contains)
    operator for the skills TEXT[] array so all searches are parameterised
    (no raw SQL interpolation).

    Returns same shape as list_candidates.
    Raises:
        RepositoryUnavailableError if Supabase is not configured.
        DatabaseError on DB query failure.
    """
    limit = min(max(1, limit), MAX_PAGE_SIZE)
    offset = max(0, offset)
    query = (query or "").strip()

    if not query:
        return list_candidates(limit=limit, offset=offset, client=client)

    supabase = _require_client(client)
    table = _get_table_ref(supabase, "CandidateResume")

    pattern = f"%{query}%"

    try:
        # PostgREST OR filter: ilike across text columns, cs for array
        # We build an OR across all searchable columns.
        data_res = (
            table.select(_LIST_COLUMNS)
            .or_(
                f'candidateName.ilike.{pattern},'
                f'email.ilike.{pattern},'
                f'jobRole.ilike.{pattern},'
                f'skills.cs.{{"{query}"}}'
            )
            .order("createdAt", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        candidates = data_res.data or []

        # Separate count query with the same filter
        count_res = (
            table.select("id", count="exact")
            .or_(
                f'candidateName.ilike.{pattern},'
                f'email.ilike.{pattern},'
                f'jobRole.ilike.{pattern},'
                f'skills.cs.{{"{query}"}}'
            )
            .execute()
        )
        total = count_res.count if hasattr(count_res, "count") and count_res.count is not None else len(candidates)

        return {
            "candidates": candidates,
            "total": total,
            "limit": limit,
            "offset": offset,
            "query": query,
        }
    except Exception as exc:
        logger.error("[Repository] search_candidates failed (query=%r): %s", query, exc)
        raise DatabaseError(f"Failed to search candidates: {exc}") from exc


# ---------------------------------------------------------------------------
# Get single candidate
# ---------------------------------------------------------------------------

def get_candidate(
    candidate_id: str,
    include_resume_text: bool = False,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Retrieve full metadata for a single candidate by UUID.

    Args:
        candidate_id: UUID string.
        include_resume_text: When True, the resumeText field is included.

    Returns:
        Candidate metadata dict.

    Raises:
        CandidateNotFoundError if no matching record exists.
        RepositoryUnavailableError if Supabase is not configured.
        DatabaseError on query failure.
    """
    _validate_candidate_id(candidate_id)
    supabase = _require_client(client)
    table = _get_table_ref(supabase, "CandidateResume")

    columns = _DETAIL_COLUMNS if include_resume_text else _LIST_COLUMNS

    try:
        res = table.select(columns).eq("id", candidate_id).limit(1).execute()
    except Exception as exc:
        logger.error("[Repository] get_candidate(%s) failed: %s", candidate_id, exc)
        raise DatabaseError(f"Failed to retrieve candidate: {exc}") from exc

    if not res.data:
        raise CandidateNotFoundError(f"Candidate '{candidate_id}' not found.")

    return res.data[0]


# ---------------------------------------------------------------------------
# Signed download URL
# ---------------------------------------------------------------------------

def get_resume_download_url(
    candidate_id: str,
    ttl_seconds: int = SIGNED_URL_TTL_SECONDS,
    client: Optional[Any] = None,
) -> str:
    """
    Create a short-lived signed URL for downloading the candidate's resume file.

    The URL is valid for `ttl_seconds` (default 300 s / 5 min) and does not
    expose the service-role key or any permanent public link.

    Raises:
        CandidateNotFoundError if candidate or file is not found.
        RepositoryUnavailableError if Supabase is not configured.
        StorageError on signed-URL creation failure.
    """
    candidate = get_candidate(candidate_id, include_resume_text=False, client=client)
    storage_path: str = candidate.get("storageObjectPath", "")

    if not storage_path:
        raise CandidateNotFoundError(
            f"Candidate '{candidate_id}' has no associated storage object."
        )

    bucket = _bucket_name()
    # storageObjectPath is stored as "resumes/<uuid>/<filename>"
    # Storage API needs only the object path inside the bucket
    if storage_path.startswith(f"{bucket}/"):
        object_path = storage_path[len(f"{bucket}/"):]
    else:
        object_path = storage_path

    supabase = _require_client(client)
    try:
        result = supabase.storage.from_(bucket).create_signed_url(object_path, ttl_seconds)
        signed_url = (
            result.get("signedURL")
            or result.get("signedUrl")
            or (result.get("data") or {}).get("signedURL")
            or (result.get("data") or {}).get("signedUrl")
        )
        if not signed_url:
            raise StorageError(f"Signed URL creation returned unexpected payload: {result}")
        return signed_url
    except StorageError:
        raise
    except Exception as exc:
        logger.error(
            "[Repository] Failed to create signed URL for candidate %s: %s", candidate_id, exc
        )
        raise StorageError(f"Failed to generate download URL: {exc}") from exc


# ---------------------------------------------------------------------------
# Delete candidate
# ---------------------------------------------------------------------------


def delete_candidate(
    candidate_id: str,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Delete a candidate's storage object and metadata row.

    Deletion order:
      1. Fetch the candidate record (raises CandidateNotFoundError if absent).
      2. Attempt to delete the file from Supabase Storage.
         - If the object is already absent (HTTP 404 / "not found" in message),
           treat that as a clean success and continue.
         - If deletion fails for any other reason, raise StorageError immediately.
           The DB row is NOT touched, preserving a recoverable state.
      3. Delete the DB row.
         - If this fails, raise PartialDeletionError, which records that
           Storage was already deleted so retries know not to re-attempt it.

    Repeated requests:
      - A second call on the same ID after full success raises CandidateNotFoundError
        (idempotent 404), not a spurious error.

    Returns:
        {
            "deleted": True,
            "candidateId": str,
            "storageDeleted": bool,        # False when path was absent / empty
            "storageAlreadyAbsent": bool,  # True when object was already gone
        }

    Raises:
        CandidateNotFoundError    – candidate record absent (safe to retry).
        RepositoryUnavailableError – Supabase not configured.
        StorageError               – Storage deletion failed; DB row is intact.
        PartialDeletionError       – Storage deleted but DB row removal failed.
        DatabaseError              – other DB failure (DB row may still exist).
    """
    _validate_candidate_id(candidate_id)
    supabase = _require_client(client)

    # 1. Fetch record — propagates CandidateNotFoundError if absent
    candidate = get_candidate(candidate_id, include_resume_text=False, client=supabase)
    storage_path: str = candidate.get("storageObjectPath", "")

    bucket = _bucket_name()
    storage_deleted = False
    storage_already_absent = False

    # 2. Delete from Storage — abort on real failures; skip if file already gone
    if storage_path:
        if storage_path.startswith(f"{bucket}/"):
            object_path = storage_path[len(f"{bucket}/"):]
        else:
            object_path = storage_path

        try:
            supabase.storage.from_(bucket).remove([object_path])
            storage_deleted = True
            logger.info(
                "[Repository] Deleted storage object %s for candidate %s",
                object_path, candidate_id,
            )
        except Exception as exc:
            err_lower = str(exc).lower()
            if "not found" in err_lower or "404" in err_lower or "does not exist" in err_lower:
                # File already absent — treat as clean; continue to DB deletion
                storage_already_absent = True
                logger.info(
                    "[Repository] Storage object %s already absent for candidate %s; continuing.",
                    object_path, candidate_id,
                )
            else:
                # Real storage failure — DB row is still intact; caller can retry
                logger.error(
                    "[Repository] Storage deletion failed for candidate %s (object %s): %s",
                    candidate_id, object_path, exc,
                )
                raise StorageError(
                    f"Storage deletion failed; candidate metadata is unchanged and safe to retry. "
                    f"Error: {exc}"
                ) from exc
    else:
        logger.info(
            "[Repository] No storage path recorded for candidate %s; skipping Storage deletion.",
            candidate_id,
        )

    # 3. Delete DB row — storage is now gone (or was already absent)
    full_storage_path = storage_path  # preserved for PartialDeletionError
    table = _get_table_ref(supabase, "CandidateResume")
    try:
        table.delete().eq("id", candidate_id).execute()
        logger.info("[Repository] Deleted DB record for candidate %s", candidate_id)
    except Exception as exc:
        logger.error(
            "[Repository] DB row deletion failed for candidate %s after Storage was deleted: %s",
            candidate_id, exc,
        )
        raise PartialDeletionError(
            f"Storage object deleted but database row removal failed. "
            f"The storage object at '{full_storage_path}' is already gone. "
            f"Re-running delete will skip storage and attempt DB cleanup again. "
            f"DB error: {exc}",
            storage_object_path=full_storage_path,
        ) from exc

    return {
        "deleted": True,
        "candidateId": candidate_id,
        "storageDeleted": storage_deleted,
        "storageAlreadyAbsent": storage_already_absent,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_candidate_id(candidate_id: str) -> None:
    """Raise ValueError for clearly invalid IDs (empty or non-UUID-shaped strings)."""
    import re as _re
    if not candidate_id or not isinstance(candidate_id, str):
        raise ValueError("candidate_id must be a non-empty string.")
    uuid_re = _re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", _re.I
    )
    if not uuid_re.match(candidate_id.strip()):
        raise ValueError(f"Invalid candidate ID format: {candidate_id!r}")
