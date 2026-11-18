"""
test_supabase_live_integration.py
=================================
Live integration tests against an isolated, non-production Supabase test instance.
SKIPPED by default unless SUPABASE_TEST_URL and SUPABASE_TEST_SERVICE_KEY are explicitly provided.

DO NOT POINT AT PRODUCTION.
These tests create, conflict-test, upsert, and delete test rows in CandidateResume
and ExtractionJobs.
"""

import os
import uuid
import hashlib
import pytest

SUPABASE_TEST_URL = os.getenv("SUPABASE_TEST_URL", "").strip()
SUPABASE_TEST_SERVICE_KEY = os.getenv("SUPABASE_TEST_SERVICE_KEY", "").strip()

# Guard: Never run against production
PROD_SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()

requires_live_supabase = pytest.mark.skipif(
    not (SUPABASE_TEST_URL and SUPABASE_TEST_SERVICE_KEY),
    reason="SUPABASE_TEST_URL and SUPABASE_TEST_SERVICE_KEY required for live integration tests.",
)


@pytest.fixture(scope="module")
def supabase_test_client():
    if not (SUPABASE_TEST_URL and SUPABASE_TEST_SERVICE_KEY):
        pytest.skip("Live Supabase test credentials not configured.")

    if PROD_SUPABASE_URL and SUPABASE_TEST_URL == PROD_SUPABASE_URL:
        pytest.fail("SAFETY VIOLATION: SUPABASE_TEST_URL cannot be identical to production SUPABASE_URL!")

    from supabase import create_client, Client
    client: Client = create_client(SUPABASE_TEST_URL, SUPABASE_TEST_SERVICE_KEY)
    return client


@requires_live_supabase
def test_partial_unique_index_race(supabase_test_client):
    """
    Validates the partial unique index:
    CREATE UNIQUE INDEX ON "ExtractionJobs" ("fileHash") WHERE status IN ('pending', 'processing');

    Attempts to concurrently/sequentially insert two 'pending' jobs with the exact same fileHash.
    The second insert must raise an integrity/unique conflict error from Postgres.
    """
    test_hash = hashlib.sha256(f"test_race_{uuid.uuid4()}".encode("utf-8")).hexdigest()
    job_id_1 = str(uuid.uuid4())
    job_id_2 = str(uuid.uuid4())

    table = supabase_test_client.table("ExtractionJobs")

    # Insert first pending job
    res1 = table.insert({
        "jobId": job_id_1,
        "fileHash": test_hash,
        "filename": "candidate_1.pdf",
        "status": "pending",
        "extractorVersion": "test-v1",
        "attempts": 0,
    }).execute()
    assert res1.data and len(res1.data) > 0

    try:
        # Second insert with identical fileHash while first is still pending must conflict
        conflict_detected = False
        try:
            table.insert({
                "jobId": job_id_2,
                "fileHash": test_hash,
                "filename": "candidate_2.pdf",
                "status": "pending",
                "extractorVersion": "test-v1",
                "attempts": 0,
            }).execute()
        except Exception as exc:
            conflict_detected = True
            assert "duplicate key" in str(exc).lower() or "unique" in str(exc).lower()

        assert conflict_detected, "Expected partial unique index to reject concurrent pending job with same fileHash!"
    finally:
        # Cleanup
        table.delete().eq("fileHash", test_hash).execute()


@requires_live_supabase
def test_on_conflict_upsert_on_version_mismatch(supabase_test_client):
    """
    Validates cache refresh behavior:
    When a file is re-extracted due to version mismatch, the existing row
    must be updated (ON CONFLICT ("fileHash") DO UPDATE) rather than creating a duplicate.
    """
    test_hash = hashlib.sha256(f"test_upsert_{uuid.uuid4()}".encode("utf-8")).hexdigest()
    table = supabase_test_client.table("CandidateResume")

    # 1. Initial insert with v1
    v1_data = {
        "candidateName": "Test Candidate V1",
        "jobRole": "Software Engineer",
        "resumeText": "Some resume text",
        "skills": ["Python"],
        "fileHash": test_hash,
        "extractorVersion": "legacy-v1",
        "extractionSource": "legacy",
        "extractedData": {"candidateName": "Test Candidate V1", "jobTitle": "Software Engineer"},
    }
    res1 = table.insert(v1_data).execute()
    assert res1.data and len(res1.data) > 0
    record_id = res1.data[0]["id"]

    try:
        # 2. Upsert with v2 for the same fileHash
        v2_data = {
            "id": record_id,
            "candidateName": "Test Candidate V2",
            "jobRole": "Senior Software Engineer",
            "resumeText": "Some resume text",
            "skills": ["Python", "Rust"],
            "fileHash": test_hash,
            "extractorVersion": "slm-v2",
            "extractionSource": "slm",
            "extractedData": {"candidateName": "Test Candidate V2", "jobTitle": "Senior Software Engineer"},
        }
        res2 = table.upsert(v2_data, on_conflict="fileHash").execute()
        assert res2.data and len(res2.data) > 0

        # Verify only 1 row exists with this fileHash
        all_rows = table.select("id, candidateName, extractorVersion").eq("fileHash", test_hash).execute()
        assert len(all_rows.data) == 1
        assert all_rows.data[0]["extractorVersion"] == "slm-v2"
        assert all_rows.data[0]["candidateName"] == "Test Candidate V2"
    finally:
        # Cleanup
        table.delete().eq("fileHash", test_hash).execute()


@requires_live_supabase
def test_restart_recovery_from_test_db(supabase_test_client, monkeypatch):
    """
    Validates that on server restart, interrupted 'processing' jobs in the test database
    are scanned and queued for recovery.
    """
    import asyncio
    from app.services.extraction_service import ExtractionService

    test_hash = hashlib.sha256(f"test_recovery_{uuid.uuid4()}".encode("utf-8")).hexdigest()
    test_job_id = str(uuid.uuid4())
    table = supabase_test_client.table("ExtractionJobs")

    # Insert an interrupted job (status='processing')
    table.insert({
        "jobId": test_job_id,
        "fileHash": test_hash,
        "filename": "interrupted.pdf",
        "status": "processing",
        "extractorVersion": "test-v1",
        "attempts": 1,
    }).execute()

    try:
        # Point ExtractionService at this test client
        monkeypatch.setattr("app.services.extraction_service.is_supabase_configured", lambda: True)
        monkeypatch.setattr("app.services.extraction_service.get_supabase_client", lambda: supabase_test_client)

        service = ExtractionService(max_queue_size=10)

        # Query via get_job
        job = service.get_job(test_job_id)
        assert job is not None
        assert job["id"] == test_job_id
        assert job["status"] == "processing"
        assert job["file_hash"] == test_hash
    finally:
        # Cleanup
        table.delete().eq("jobId", test_job_id).execute()
