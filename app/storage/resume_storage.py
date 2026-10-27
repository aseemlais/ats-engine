"""
Persistent resume storage service integrating Supabase Storage and Postgres.
Handles file persistence, SHA-256 duplicate checking, metadata storage, and orphaned object cleanup.
"""
import os
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from app.storage.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class ResumePersistenceError(Exception):
    """Base exception for resume storage and persistence failures."""
    pass


class StorageError(ResumePersistenceError):
    """Raised when uploading or deleting files in object storage fails."""
    pass


class DatabaseError(ResumePersistenceError):
    """Raised when querying or persisting candidate metadata fails."""
    pass


def _get_table_ref(client: Any, table_name: str = "CandidateResume"):
    """
    Returns table reference, targeting the 'ai_chatbot' schema where CandidateResume resides.
    Falls back to default schema if schema switching is not available.
    """
    if hasattr(client, "schema"):
        try:
            return client.schema("ai_chatbot").table(table_name)
        except Exception as exc:
            logger.warning(f"[Storage] Could not switch to schema 'ai_chatbot': {exc}")
    return client.table(table_name)


def check_duplicate_by_hash(file_hash: str, client: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """
    Checks if a resume with the exact SHA-256 file_hash has already been persisted.
    Returns the existing record dict if found, or None.
    """
    supabase = client or get_supabase_client()
    if not supabase:
        return None

    try:
        table = _get_table_ref(supabase, "CandidateResume")
        res = table.select(
            "id, fileHash, originalFilename, storageObjectPath, candidateName, email, jobRole, createdAt"
        ).eq("fileHash", file_hash).limit(1).execute()

        if res and hasattr(res, "data") and res.data and len(res.data) > 0:
            return res.data[0]
        return None
    except Exception as exc:
        logger.warning(f"[Storage] Error querying duplicate candidate resume by hash {file_hash[:12]}...: {exc}")
        return None


def store_resume(
    file_bytes: bytes,
    filename: str,
    file_hash: str,
    candidate_name: str,
    email: Optional[str],
    resume_text: str,
    skills: List[str],
    job_role: Optional[str],
    content_type: str = "application/pdf",
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Persists a resume file in Supabase Storage and candidate metadata in Supabase Postgres.
    
    1. Checks SHA-256 duplicate hash. If found, returns existing record without re-uploading.
    2. Uploads file to private Supabase Storage bucket ('resumes').
    3. Persists structured candidate metadata to ai_chatbot."CandidateResume".
    4. Cleans up storage object if metadata insertion fails.
    """
    supabase = client or get_supabase_client()
    if not supabase:
        raise ResumePersistenceError("Supabase client is not configured or unavailable.")

    # 1. Duplicate check
    existing = check_duplicate_by_hash(file_hash, client=supabase)
    if existing:
        logger.info(f"[Storage] Duplicate resume detected for hash {file_hash[:12]}... (Candidate ID: {existing.get('id')})")
        return {
            "candidateId": str(existing.get("id")),
            "isDuplicate": True,
            "storageObjectPath": existing.get("storageObjectPath"),
            "candidateName": existing.get("candidateName") or "Unknown",
            "jobRole": existing.get("jobRole") or "Role not identified",
            "email": existing.get("email"),
            "status": "existing_record_reused",
            "persistenceStatus": "duplicate_reused",
        }

    # 2. Upload file to Supabase Storage
    candidate_id = str(uuid.uuid4())
    safe_filename = Path(filename).name
    storage_subpath = f"{candidate_id}/{safe_filename}"
    bucket_name = os.environ.get("SUPABASE_STORAGE_BUCKET", "resumes")
    full_storage_path = f"{bucket_name}/{storage_subpath}"

    try:
        supabase.storage.from_(bucket_name).upload(
            path=storage_subpath,
            file=file_bytes,
            file_options={"content-type": content_type},
        )
    except Exception as exc:
        err_msg = str(exc)
        logger.error(f"[Storage] Failed to upload resume to storage path {storage_subpath}: {err_msg}")
        if "bucket not found" in err_msg.lower():
            raise StorageError(f"Storage bucket '{bucket_name}' not found in Supabase: {err_msg}") from exc
        raise StorageError(f"Storage upload failed: {err_msg}") from exc

    # 3. Persist metadata record in Postgres
    record = {
        "id": candidate_id,
        "fileHash": file_hash,
        "originalFilename": safe_filename,
        "storageObjectPath": full_storage_path,
        "candidateName": candidate_name or "Unknown",
        "email": email,
        "resumeText": resume_text,
        "skills": skills or [],
        "jobRole": job_role or "Role not identified",
    }

    try:
        table = _get_table_ref(supabase, "CandidateResume")
        table.insert(record).execute()
    except Exception as exc:
        err_msg = str(exc)
        # Storage reconciliation: clean up orphaned file object
        logger.error(
            f"[Storage] Failed to insert candidate record: {err_msg}. Rolling back storage object {storage_subpath}"
        )
        try:
            supabase.storage.from_(bucket_name).remove([storage_subpath])
        except Exception as cleanup_exc:
            logger.warning(f"[Storage] Failed to remove orphaned object {storage_subpath}: {cleanup_exc}")

        # Check for concurrent duplicate race condition
        existing_after_race = check_duplicate_by_hash(file_hash, client=supabase)
        if existing_after_race:
            return {
                "candidateId": str(existing_after_race.get("id")),
                "isDuplicate": True,
                "storageObjectPath": existing_after_race.get("storageObjectPath"),
                "candidateName": existing_after_race.get("candidateName") or "Unknown",
                "jobRole": existing_after_race.get("jobRole") or "Role not identified",
                "email": existing_after_race.get("email"),
                "status": "existing_record_reused_after_race_condition",
                "persistenceStatus": "duplicate_reused",
            }

        raise DatabaseError(f"Database insertion failed: {err_msg}") from exc

    return {
        "candidateId": candidate_id,
        "isDuplicate": False,
        "storageObjectPath": full_storage_path,
        "candidateName": candidate_name or "Unknown",
        "jobRole": job_role or "Role not identified",
        "email": email,
        "status": "new_candidate_stored",
        "persistenceStatus": "persisted",
    }
