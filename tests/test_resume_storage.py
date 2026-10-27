"""
Unit and integration tests for persistent resume storage in Supabase (Batch 1).
Verifies:
1. First-time resume storage (storage object + database metadata persistence)
2. Duplicate detection via SHA-256 content hash
3. Handling different resume files for the same candidate name
4. Persistence and structure of candidate metadata (name, email, skills, job role, hash)
5. Failure handling and storage rollback (no orphaned storage objects)
6. Concurrent duplicate upload race condition handling
7. Backward compatibility of upload endpoint when Supabase is not configured
8. Full upload endpoint integration with Supabase enabled
"""
import io
import asyncio
import hashlib
from unittest.mock import MagicMock, patch
import pytest
from starlette.datastructures import UploadFile

from app.storage.supabase_client import (
    is_supabase_configured,
    get_supabase_client,
    reset_supabase_client,
)
from app.storage.resume_storage import (
    store_resume,
    check_duplicate_by_hash,
    StorageError,
    DatabaseError,
)
from app.server import process_resume_upload


@pytest.fixture(autouse=True)
def reset_client():
    """Ensure clean client state before each test."""
    reset_supabase_client(None)
    yield
    reset_supabase_client(None)


def create_mock_supabase(existing_records=None):
    """Creates a mock Supabase client mimicking storage and database query builders."""
    existing_records = existing_records or {}
    mock_client = MagicMock()

    # Mock storage
    mock_storage_bucket = MagicMock()
    mock_client.storage.from_.return_value = mock_storage_bucket
    mock_storage_bucket.upload.return_value = {"Key": "resumes/mock-key"}
    mock_storage_bucket.remove.return_value = [{"name": "removed"}]

    # Mock table query builder
    mock_table = MagicMock()
    mock_client.schema.return_value.table.return_value = mock_table
    mock_client.table.return_value = mock_table

    # Mock select & eq chain
    def mock_select(*args, **kwargs):
        select_builder = MagicMock()

        def mock_eq(column, value):
            eq_builder = MagicMock()

            def mock_limit(n):
                limit_builder = MagicMock()

                def mock_execute():
                    res = MagicMock()
                    if column == "fileHash" and value in existing_records:
                        res.data = [existing_records[value]]
                    else:
                        res.data = []
                    return res

                limit_builder.execute = mock_execute
                return limit_builder

            eq_builder.limit = mock_limit
            eq_builder.execute = lambda: MagicMock(
                data=[existing_records[value]] if value in existing_records else []
            )
            return eq_builder

        select_builder.eq = mock_eq
        return select_builder

    mock_table.select.side_effect = mock_select

    # Mock insert builder
    insert_builder = MagicMock()
    insert_builder.execute.return_value = MagicMock(data=[{}])
    mock_table.insert.return_value = insert_builder

    return mock_client, mock_storage_bucket, mock_table


def test_first_time_resume_storage():
    """First-time upload: file is saved to storage and candidate metadata stored in DB."""
    mock_client, mock_storage_bucket, mock_table = create_mock_supabase()

    file_bytes = b"%PDF-1.4 Mock resume content for Jane Doe"
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    result = store_resume(
        file_bytes=file_bytes,
        filename="jane_doe_cv.pdf",
        file_hash=file_hash,
        candidate_name="Jane Doe",
        email="jane.doe@example.com",
        resume_text="Experienced Python and AI Developer...",
        skills=["Python", "FastAPI", "PyTorch"],
        job_role="Senior AI Engineer",
        content_type="application/pdf",
        client=mock_client,
    )

    assert result["isDuplicate"] is False
    assert result["candidateId"] is not None
    assert result["candidateName"] == "Jane Doe"
    assert result["jobRole"] == "Senior AI Engineer"
    assert result["email"] == "jane.doe@example.com"
    assert result["storageObjectPath"].startswith("resumes/")
    assert "jane_doe_cv.pdf" in result["storageObjectPath"]

    # Verify storage upload called
    assert mock_storage_bucket.upload.called
    upload_kwargs = mock_storage_bucket.upload.call_args[1]
    assert upload_kwargs["file"] == file_bytes

    # Verify DB insert called with expected columns
    assert mock_table.insert.called
    inserted_record = mock_table.insert.call_args[0][0]
    assert inserted_record["fileHash"] == file_hash
    assert inserted_record["candidateName"] == "Jane Doe"
    assert inserted_record["skills"] == ["Python", "FastAPI", "PyTorch"]
    assert inserted_record["jobRole"] == "Senior AI Engineer"


def test_duplicate_upload_detection():
    """Duplicate upload: same SHA-256 hash returns existing record without re-uploading."""
    file_bytes = b"%PDF-1.4 Identical resume content"
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    existing_record = {
        "id": "existing-uuid-1234",
        "fileHash": file_hash,
        "originalFilename": "existing_resume.pdf",
        "storageObjectPath": "resumes/existing-uuid-1234/existing_resume.pdf",
        "candidateName": "Alex Smith",
        "email": "alex.smith@example.com",
        "jobRole": "Lead Cloud Architect",
        "createdAt": "2026-01-01T00:00:00Z",
    }

    mock_client, mock_storage_bucket, mock_table = create_mock_supabase(
        existing_records={file_hash: existing_record}
    )

    result = store_resume(
        file_bytes=file_bytes,
        filename="alex_duplicate_upload.pdf",
        file_hash=file_hash,
        candidate_name="Alex Smith",
        email="alex.smith@example.com",
        resume_text="Cloud Architect experience...",
        skills=["AWS", "Terraform"],
        job_role="Lead Cloud Architect",
        client=mock_client,
    )

    assert result["isDuplicate"] is True
    assert result["candidateId"] == "existing-uuid-1234"
    assert result["storageObjectPath"] == "resumes/existing-uuid-1234/existing_resume.pdf"
    assert result["candidateName"] == "Alex Smith"
    assert result["status"] == "existing_record_reused"

    # Crucial: No storage upload or DB insert should occur for duplicates
    mock_storage_bucket.upload.assert_not_called()
    mock_table.insert.assert_not_called()


def test_different_resume_files_with_same_candidate_name():
    """Different files for the same candidate name must NOT be treated as duplicates."""
    mock_client, mock_storage_bucket, mock_table = create_mock_supabase()

    file1 = b"%PDF-1.4 Version 1 of resume for John Doe"
    file2 = b"%PDF-1.4 Version 2 of updated resume for John Doe"
    hash1 = hashlib.sha256(file1).hexdigest()
    hash2 = hashlib.sha256(file2).hexdigest()

    res1 = store_resume(
        file_bytes=file1,
        filename="john_v1.pdf",
        file_hash=hash1,
        candidate_name="John Doe",
        email="john@example.com",
        resume_text="V1 text",
        skills=["Python"],
        job_role="Junior Developer",
        client=mock_client,
    )

    res2 = store_resume(
        file_bytes=file2,
        filename="john_v2.pdf",
        file_hash=hash2,
        candidate_name="John Doe",
        email="john@example.com",
        resume_text="V2 updated text with more experience",
        skills=["Python", "Docker"],
        job_role="Software Engineer",
        client=mock_client,
    )

    # Both must be created independently
    assert res1["isDuplicate"] is False
    assert res2["isDuplicate"] is False
    assert res1["candidateId"] != res2["candidateId"]
    assert res1["storageObjectPath"] != res2["storageObjectPath"]


def test_storage_failure_aborts_before_db():
    """If object storage fails, no database record is created and StorageError is raised."""
    mock_client, mock_storage_bucket, mock_table = create_mock_supabase()
    mock_storage_bucket.upload.side_effect = Exception("S3 bucket storage timeout")

    file_bytes = b"%PDF-1.4 Sample bytes"
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    with pytest.raises(StorageError) as exc_info:
        store_resume(
            file_bytes=file_bytes,
            filename="sample.pdf",
            file_hash=file_hash,
            candidate_name="Sam",
            email=None,
            resume_text="Text",
            skills=[],
            job_role=None,
            client=mock_client,
        )

    assert "Storage upload failed" in str(exc_info.value)
    mock_table.insert.assert_not_called()


def test_db_failure_cleans_up_orphaned_storage_object():
    """If database insert fails, the newly uploaded storage object is deleted (rollback)."""
    mock_client, mock_storage_bucket, mock_table = create_mock_supabase()

    # Simulate Postgres failure on insert
    mock_table.insert.return_value.execute.side_effect = Exception("Postgres connection terminated")

    file_bytes = b"%PDF-1.4 Broken DB test"
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    with pytest.raises(DatabaseError) as exc_info:
        store_resume(
            file_bytes=file_bytes,
            filename="orphan_test.pdf",
            file_hash=file_hash,
            candidate_name="Test Candidate",
            email=None,
            resume_text="Text",
            skills=[],
            job_role=None,
            client=mock_client,
        )

    assert "Database insertion failed" in str(exc_info.value)
    # Rollback must have called remove on storage
    assert mock_storage_bucket.remove.called
    removed_path = mock_storage_bucket.remove.call_args[0][0][0]
    assert removed_path.endswith("orphan_test.pdf")


def test_concurrent_duplicate_race_condition():
    """If a concurrent insert raises a unique conflict, system reconciles to existing record."""
    file_bytes = b"%PDF-1.4 Concurrent race test"
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    existing_record = {
        "id": "concurrent-winner-uuid",
        "fileHash": file_hash,
        "originalFilename": "winner.pdf",
        "storageObjectPath": "resumes/concurrent-winner-uuid/winner.pdf",
        "candidateName": "Race Winner",
        "email": "race@example.com",
        "jobRole": "Concurrency Specialist",
    }

    mock_client, mock_storage_bucket, mock_table = create_mock_supabase()

    # Insert fails due to unique constraint:
    mock_table.insert.return_value.execute.side_effect = Exception(
        "duplicate key value violates unique constraint CandidateResume_fileHash_unique"
    )

    # First check: returns None. Second check: returns the winner's record
    call_count = 0
    def mock_select_dynamic(*args, **kwargs):
        builder = MagicMock()
        def mock_eq(col, val):
            eq_b = MagicMock()
            def mock_limit(n):
                lim_b = MagicMock()
                def mock_exec():
                    nonlocal call_count
                    call_count += 1
                    res = MagicMock()
                    if call_count > 1:
                        res.data = [existing_record]
                    else:
                        res.data = []
                    return res
                lim_b.execute = mock_exec
                return lim_b
            eq_b.limit = mock_limit
            return eq_b
        builder.eq = mock_eq
        return builder

    mock_table.select.side_effect = mock_select_dynamic

    res = store_resume(
        file_bytes=file_bytes,
        filename="second_runner.pdf",
        file_hash=file_hash,
        candidate_name="Race Runner",
        email="runner@example.com",
        resume_text="Race text",
        skills=[],
        job_role="Engineer",
        client=mock_client,
    )

    assert res["isDuplicate"] is True
    assert res["candidateId"] == "concurrent-winner-uuid"
    assert res["status"] == "existing_record_reused_after_race_condition"
    # Ensure orphaned file was rolled back
    assert mock_storage_bucket.remove.called


def test_upload_endpoint_backwards_compatibility_without_supabase(monkeypatch):
    """When Supabase is not configured, upload endpoint returns standard payload seamlessly."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    reset_supabase_client(None)

    txt_content = b"John Doe\njohn@example.com\nPython Developer with 5 years experience."
    resp = process_resume_upload("resume.txt", txt_content, "text/plain")

    assert resp["success"] is True
    assert resp["pathname"] == "resume.txt"
    assert "John Doe" in resp["resumeText"]
    # Persistent storage fields are not attached when unconfigured
    assert "candidateId" not in resp or resp.get("candidateId") is None


def test_upload_endpoint_with_supabase_configured(monkeypatch):
    """When Supabase is configured, upload endpoint stores file and augments response."""
    monkeypatch.setenv("SUPABASE_URL", "https://mockproject.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "mock-service-role-key-backend")

    mock_client, mock_storage_bucket, mock_table = create_mock_supabase()
    reset_supabase_client(mock_client)

    resume_text = (
        "Sarah Connor\n"
        "sarah@skynet-defense.com\n"
        "WORK EXPERIENCE\n"
        "Security Architect at Defense Systems\n"
        "2020 - Present\n"
        "- Hardened distributed infrastructure and AI defenses\n"
        "SKILLS\n"
        "Cybersecurity, Python, Kubernetes\n"
    )
    resp = process_resume_upload(
        "sarah_connor_cv.txt",
        resume_text.encode("utf-8"),
        "text/plain"
    )

    # Legacy fields
    assert resp["success"] is True
    assert resp["pathname"] == "sarah_connor_cv.txt"
    assert "Sarah Connor" in resp["resumeText"]

    # Augmented persistent storage fields
    assert resp["candidateId"] is not None
    assert resp["isDuplicate"] is False
    assert resp["storageObjectPath"].startswith("resumes/")
    assert resp["candidateName"] == "Sarah Connor"
    assert "Security" in resp["jobRole"]
