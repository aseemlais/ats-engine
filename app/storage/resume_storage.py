"""
Persistent resume storage service integrating Supabase Storage and Postgres.
Handles file persistence, SHA-256 duplicate checking, metadata storage, and orphaned object cleanup.
"""
import os
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from app.storage.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


def is_supabase_configured() -> bool:
    """Returns True if Supabase URL and service role key are present."""
    return bool(os.getenv("SUPABASE_URL") and (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")))


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


def upload_resume_file_to_storage(
    file_bytes: bytes,
    filename: str,
    content_type: str = "application/pdf",
    client: Optional[Any] = None,
) -> Tuple[str, str]:
    """
    Uploads raw resume file bytes to Supabase Storage before job queueing.
    Returns (full_storage_path, storage_subpath).
    """
    supabase = client or get_supabase_client()
    if not supabase:
        raise ResumePersistenceError("Supabase client is not configured or unavailable.")

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
        return full_storage_path, storage_subpath
    except Exception as exc:
        err_msg = str(exc)
        logger.error(f"[Storage] Failed to upload resume to storage path {storage_subpath}: {err_msg}")
        if "bucket not found" in err_msg.lower():
            raise StorageError(f"Storage bucket '{bucket_name}' not found in Supabase: {err_msg}") from exc
        raise StorageError(f"Storage upload failed: {err_msg}") from exc


def download_resume_file_from_storage(
    storage_path: str,
    client: Optional[Any] = None,
) -> bytes:
    """
    Downloads raw resume file bytes from Supabase Storage for recovery processing.
    """
    supabase = client or get_supabase_client()
    if not supabase:
        raise ResumePersistenceError("Supabase client is not configured or unavailable.")

    bucket_name = os.environ.get("SUPABASE_STORAGE_BUCKET", "resumes")
    # Strip bucket prefix if present
    subpath = storage_path
    if subpath.startswith(f"{bucket_name}/"):
        subpath = subpath[len(bucket_name) + 1:]

    try:
        data = supabase.storage.from_(bucket_name).download(subpath)
        return data
    except Exception as exc:
        raise StorageError(f"Failed to download resume from storage '{subpath}': {exc}") from exc


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
    extracted_data: Optional[Dict[str, Any]] = None,
    extractor_version: Optional[str] = None,
    extraction_source: Optional[str] = None,
    storage_path: Optional[str] = None,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Persists a resume file in Supabase Storage and candidate metadata in Supabase Postgres.
    Supports version-aware upsert on fileHash.
    """
    supabase = client or get_supabase_client()
    if not supabase:
        raise ResumePersistenceError("Supabase client is not configured or unavailable.")

    # 1. Duplicate check (only reuse if versions match and extractedData is present)
    existing = check_duplicate_by_hash(file_hash, client=supabase)
    existing_version = existing.get("extractorVersion") if existing else None
    existing_has_data = bool(existing.get("extractedData")) if existing else False

    if existing and existing_has_data and (extractor_version is None or existing_version == extractor_version):
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
            "extractedData": existing.get("extractedData"),
        }

    bucket_name = os.environ.get("SUPABASE_STORAGE_BUCKET", "resumes")
    safe_filename = Path(filename).name

    # 2. Upload file if storage_path not already provided
    if storage_path:
        full_storage_path = storage_path
        storage_subpath = storage_path.replace(f"{bucket_name}/", "")
    else:
        candidate_id = str(uuid.uuid4())
        storage_subpath = f"{candidate_id}/{safe_filename}"
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
            # If already exists in storage, ignore conflict
            if "duplicate" not in err_msg.lower() and "already exists" not in err_msg.lower():
                raise StorageError(f"Storage upload failed: {err_msg}") from exc

    # 3. Persist / Upsert metadata record in Postgres
    record = {
        "fileHash": file_hash,
        "originalFilename": safe_filename,
        "storageObjectPath": full_storage_path,
        "candidateName": candidate_name or "Unknown",
        "email": email,
        "resumeText": resume_text,
        "skills": skills or [],
        "jobRole": job_role or "Role not identified",
        "extractedData": extracted_data,
        "extractorVersion": extractor_version,
        "extractionSource": extraction_source,
    }

    try:
        table = _get_table_ref(supabase, "CandidateResume")
        if existing:
            # Upsert on fileHash to refresh version / extractedData
            table.update(record).eq("fileHash", file_hash).execute()
            candidate_id = str(existing.get("id"))
        else:
            record["id"] = str(uuid.uuid4())
            res = table.insert(record).execute()
            candidate_id = record["id"]
    except Exception as exc:
        err_msg = str(exc)
        logger.error(f"[Storage] Failed to insert/upsert candidate record: {err_msg}")
        raise DatabaseError(f"Database insertion failed: {err_msg}") from exc

    return {
        "candidateId": candidate_id,
        "isDuplicate": bool(existing),
        "storageObjectPath": full_storage_path,
        "candidateName": candidate_name or "Unknown",
        "jobRole": job_role or "Role not identified",
        "email": email,
        "status": "candidate_stored",
        "persistenceStatus": "persisted",
        "extractedData": extracted_data,
    }
