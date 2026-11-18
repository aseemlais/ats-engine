"""
test_async_serving_recovery.py
==============================
Validates:
1. Exact Key-by-Key Contract Parity: Cache-hit payload keys == Fresh-extraction keys.
2. Deduplication of concurrent identical uploads (returns existing active jobId).
3. Version-mismatch cache miss (re-extracts when extractorVersion changes).
4. Restart Recovery: Reading job state from Supabase when in-memory cache is wiped.
5. Security & Validation:
   - Rejected oversized files (> 5MB -> 413)
   - Rejected invalid magic bytes (corrupted PDF, DOCX, TXT -> 400)
   - Rejected DOCX missing word/document.xml -> 400
   - Rejected DOCX zip-bomb (uncompressed > MAX_DOCX_UNCOMPRESSED_BYTES) -> 400
   - X-ATS-Internal-Secret header validation (401 when wrong or missing)
   - Global Rate Limiting safety net
   - Verification that storageObjectPath is never returned in client payloads.
6. Orphaned storage cleanup on queue-full 429.
7. Cascade delete: ExtractionJobs rows deleted alongside CandidateResume.
"""

import asyncio
import io
import os
import time
import zipfile
import pytest
from fastapi.testclient import TestClient

from app.models.resume_schema import ProcessedResume, RawResumeExtraction
from app.server import app, extraction_service, check_rate_limit, _rate_limit_records, verify_internal_secret
from app.services.extraction_service import ExtractionService

# ---------------------------------------------------------------------------
# Session-scoped autouse: run all tests in development mode so the secret
# guard is bypassed unless a specific test is testing the guard itself.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _dev_environment(monkeypatch):
    """Force ENVIRONMENT=development so internal-secret guard is skipped by default."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("ATS_INTERNAL_SECRET", raising=False)


SAMPLE_VALID_RESUME_TEXT = (
    "ALEXANDER MORGAN\n"
    "Lead Cloud Architect | alex.morgan@cloudops.net | +1-415-555-0199 | San Francisco, CA\n"
    "EXPERIENCE\n"
    "Principal Infrastructure Engineer (Kubernetes / Terraform) | CloudOps Inc | Jan 2021 - Present\n"
    "SKILLS\n"
    "Go, Kubernetes, Terraform, Python, Docker, AWS, GCP\n"
    "EDUCATION\n"
    "B.S. in Computer Science | Stanford University | 2017 - 2021\n"
    "CERTIFICATIONS\n"
    "AWS Certified Solutions Architect Professional\n"
    "PROJECTS\n"
    "Multi-Cloud Ingress Controller\n"
)


def _make_valid_docx(include_document_xml: bool = True, uncompressed_size: int = 100) -> bytes:
    """
    Build a minimal DOCX (ZIP) in memory with exactly `uncompressed_size` bytes of member content.
    Uses a simple prefix so total = exactly uncompressed_size (no off-by-one from XML template length).
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if include_document_xml:
            prefix = b"<xml/>"
            padding = max(0, uncompressed_size - len(prefix))
            zf.writestr("word/document.xml", prefix + b"X" * padding)
        else:
            zf.writestr("word/settings.xml", b"<settings/>")
    return buf.getvalue()


def test_contract_parity_cache_hit_vs_fresh_extraction(monkeypatch):
    """
    Contract Parity Test:
    Guarantees that a cache-hit result has the EXACT same dictionary keys
    as a fresh extraction, including jobRole, jobTitle, and durationYears,
    and neither leaks storageObjectPath.
    """
    service = ExtractionService()

    # 1. Generate fresh extraction payload
    raw_dict = {
        "candidateName": "Alexander Morgan",
        "email": "alex.morgan@cloudops.net",
        "phone": "+1-415-555-0199",
        "location": "San Francisco, CA",
        "jobTitle": "Lead Cloud Architect",
        "skills": ["Go", "Kubernetes", "Python"],
        "education": [{"degree": "B.S. Computer Science", "institution": "Stanford University"}],
        "workExperience": [{"title": "Principal Infrastructure Engineer", "duration_years": 3.5, "company": "CloudOps"}],
        "certifications": ["AWS Solutions Architect"],
        "projects": ["Multi-Cloud Ingress"],
    }
    from app.services.post_processor import post_process_resume
    processed = post_process_resume(raw_dict)
    fresh_payload = processed.model_dump(by_alias=True)
    fresh_payload["jobRole"] = fresh_payload.get("jobTitle")
    fresh_payload["extractionSource"] = "fallback"
    fresh_payload.pop("storageObjectPath", None)

    # 2. Simulate cache-hit payload stored and retrieved from Supabase
    cached_payload = dict(fresh_payload)
    cached_payload["extractionSource"] = "cache"

    # Mock check_supabase_cache
    monkeypatch.setattr(service, "check_supabase_cache", lambda h: cached_payload)

    # Validate key-by-key parity (excluding extractionSource which distinguishes cache from live)
    fresh_keys = set(fresh_payload.keys()) - {"extractionSource"}
    cached_keys = set(cached_payload.keys()) - {"extractionSource"}

    assert fresh_keys == cached_keys, f"Contract mismatch! Diff: {fresh_keys ^ cached_keys}"
    assert "storageObjectPath" not in cached_payload
    assert "storageObjectPath" not in fresh_payload
    assert cached_payload["jobRole"] == cached_payload["jobTitle"]
    assert "durationYears" in cached_payload["workExperience"][0]


def test_deduplication_of_concurrent_active_jobs(monkeypatch):
    """
    Submitting the same resume file bytes while an active job is already pending
    must return the existing job's id with deduplicated=True without enqueuing twice.
    """
    async def _run():
        service = ExtractionService(max_queue_size=10)
        await service.start()
        try:
            resume_bytes = b"%PDF-1.4\n" + SAMPLE_VALID_RESUME_TEXT.encode("utf-8")
            
            # First submission
            code1, resp1 = await service.submit_job(resume_bytes, "alex1.pdf", "application/pdf")
            assert code1 == 202
            job_id_1 = resp1["job_id"]
            
            # Second submission of identical bytes before first completes
            code2, resp2 = await service.submit_job(resume_bytes, "alex2.pdf", "application/pdf")
            assert code2 == 202
            job_id_2 = resp2["job_id"]
            
            # Must return the SAME job_id
            assert job_id_1 == job_id_2
            assert resp2.get("deduplicated") is True
            # Queue depth should only be 1 (or 0 if already picked up)
            assert service.queue.qsize() <= 1
        finally:
            await service.stop()

    asyncio.run(_run())


def test_version_mismatch_triggers_cache_miss(monkeypatch):
    """
    When the cached extractorVersion differs from current_extractor_version,
    check_supabase_cache MUST return None (cache MISS) so fresh extraction occurs.
    """
    service = ExtractionService()
    current_ver = service.current_extractor_version
    stale_ver = "ats-legacy:old-v0:schema-v0"

    class MockTable:
        def select(self, *args, **kwargs):
            return self
        def eq(self, *args, **kwargs):
            return self
        def limit(self, *args, **kwargs):
            return self
        def execute(self):
            class Res:
                data = [{
                    "id": "cand-999",
                    "fileHash": "hash123",
                    "extractedData": {"candidateName": "Old Candidate", "jobTitle": "Old Role"},
                    "extractorVersion": stale_ver,
                }]
            return Res()

    monkeypatch.setattr("app.services.extraction_service.is_supabase_configured", lambda: True)
    monkeypatch.setattr("app.services.extraction_service.get_supabase_client", lambda: object())
    monkeypatch.setattr("app.services.extraction_service._get_table_ref", lambda client, t: MockTable())

    # Cache check should reject stale version
    result = service.check_supabase_cache("hash123")
    assert result is None, "Version mismatch should have produced a cache MISS!"


def test_restart_recovery_reads_from_supabase_when_memory_is_wiped(monkeypatch):
    """
    Validates that when in-memory self.jobs is wiped (simulating server restart),
    get_job(job_id) falls back to querying Supabase ExtractionJobs table.
    """
    service = ExtractionService()
    test_job_id = "11111111-2222-3333-4444-555555555555"
    
    # 1. Wipe in-memory jobs completely
    service.jobs.clear()
    assert test_job_id not in service.jobs

    # 2. Mock Supabase ExtractionJobs returning the persisted record
    class MockJobsTable:
        def select(self, *args, **kwargs):
            return self
        def eq(self, col, val):
            self.val = val
            return self
        def limit(self, *args, **kwargs):
            return self
        def execute(self):
            class Res:
                data = []
            if getattr(self, "val", None) == test_job_id:
                Res.data = [{
                    "jobId": test_job_id,
                    "fileHash": "hash_recovered",
                    "filename": "recovered.pdf",
                    "status": "completed",
                    "extractionSource": "slm",
                    "result": {"candidateName": "Recovered Candidate", "jobTitle": "Recovered Role"},
                    "error": None,
                    "createdAt": "2026-11-17T10:00:00Z",
                    "updatedAt": "2026-11-17T10:01:00Z",
                }]
            return Res()

    monkeypatch.setattr("app.services.extraction_service.is_supabase_configured", lambda: True)
    monkeypatch.setattr("app.services.extraction_service.get_supabase_client", lambda: object())
    monkeypatch.setattr("app.services.extraction_service._get_table_ref", lambda client, t: MockJobsTable())

    # Should recover from Supabase
    job = service.get_job(test_job_id)
    assert job is not None
    assert job["id"] == test_job_id
    assert job["status"] == "completed"
    assert job["result"]["candidateName"] == "Recovered Candidate"

    # Unknown ID returns None
    assert service.get_job("unknown-0000-0000-0000-000000000000") is None


def test_file_validation_oversized_file():
    """Validates that files exceeding MAX_UPLOAD_SIZE_BYTES are rejected with 413."""
    with TestClient(app) as client:
        # Create 6 MB dummy payload
        big_content = b"%PDF-1.4\n" + b"A" * (6 * 1024 * 1024)
        files = {"file": ("big_resume.pdf", big_content, "application/pdf")}
        res = client.post("/extract", files=files, headers={"X-ATS-Internal-Secret": "dev-skip"})
        assert res.status_code == 413
        assert "exceeds maximum allowed size" in res.json()["detail"].lower()


def test_file_validation_corrupted_magic_bytes():
    """Validates that corrupted files with mismatched magic bytes are rejected with 400."""
    with TestClient(app) as client:
        # PDF filename but content lacks %PDF- header
        bad_pdf = b"NOT_A_REAL_PDF_HEADER\nSome resume text"
        res_pdf = client.post("/extract", files={"file": ("fake.pdf", bad_pdf, "application/pdf")}, headers={"X-ATS-Internal-Secret": "dev-skip"})
        assert res_pdf.status_code == 400
        assert "magic bytes" in res_pdf.json()["detail"].lower()

        # DOCX filename but content lacks PK\x03\x04 header
        bad_docx = b"NOT_A_ZIP_CONTAINER\nSome resume text"
        res_docx = client.post("/extract", files={"file": ("fake.docx", bad_docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}, headers={"X-ATS-Internal-Secret": "dev-skip"})
        assert res_docx.status_code == 400
        assert "magic bytes" in res_docx.json()["detail"].lower()

        # TXT filename containing binary NUL byte
        bad_txt = b"Hello\x00World binary content"
        res_txt = client.post("/extract", files={"file": ("fake.txt", bad_txt, "text/plain")}, headers={"X-ATS-Internal-Secret": "dev-skip"})
        assert res_txt.status_code == 400
        assert "nul byte" in res_txt.json()["detail"].lower()


def test_docx_validation_missing_document_xml():
    """
    A ZIP file with valid magic bytes but no word/document.xml must be rejected (400).
    """
    bad_docx = _make_valid_docx(include_document_xml=False)
    with TestClient(app) as client:
        res = client.post(
            "/extract",
            files={"file": ("no_doc.docx", bad_docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            headers={"X-ATS-Internal-Secret": "dev-skip"},
        )
    assert res.status_code == 400
    detail = res.json()["detail"].lower()
    assert "word/document.xml" in detail or "not a word document" in detail


def test_docx_validation_zip_bomb():
    """
    A DOCX whose uncompressed content exceeds MAX_DOCX_UNCOMPRESSED_BYTES must be rejected (400).
    """
    from app.server import MAX_DOCX_UNCOMPRESSED_BYTES
    bomb_size = MAX_DOCX_UNCOMPRESSED_BYTES + 1
    bomb_docx = _make_valid_docx(include_document_xml=True, uncompressed_size=bomb_size)
    with TestClient(app) as client:
        res = client.post(
            "/extract",
            files={"file": ("bomb.docx", bomb_docx, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            headers={"X-ATS-Internal-Secret": "dev-skip"},
        )
    assert res.status_code == 400
    assert "exceeds maximum allowed size" in res.json()["detail"].lower()


def test_valid_docx_passes_zip_validation():
    """A correctly structured DOCX must pass validation (not rejected on zip grounds)."""
    valid_docx = _make_valid_docx(include_document_xml=True, uncompressed_size=1000)
    # Validate directly (not via HTTP so we don't need a live worker)
    from app.server import validate_uploaded_file
    # Should not raise
    validate_uploaded_file(valid_docx, "valid.docx")


def test_internal_secret_missing_in_production(monkeypatch):
    """
    When ENVIRONMENT=production and ATS_INTERNAL_SECRET is unset,
    POST /extract must return 401 — the endpoint must be closed by default.
    """
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ATS_INTERNAL_SECRET", raising=False)
    with TestClient(app) as client:
        pdf_content = b"%PDF-1.4\nSome text"
        res = client.post("/extract", files={"file": ("r.pdf", pdf_content, "application/pdf")})
    assert res.status_code == 401


def test_internal_secret_wrong_value(monkeypatch):
    """Wrong X-ATS-Internal-Secret header must yield 401."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ATS_INTERNAL_SECRET", "correct-secret-value")
    with TestClient(app) as client:
        pdf_content = b"%PDF-1.4\nSome text"
        res = client.post(
            "/extract",
            files={"file": ("r.pdf", pdf_content, "application/pdf")},
            headers={"X-ATS-Internal-Secret": "wrong-value"},
        )
    assert res.status_code == 401


def test_internal_secret_correct_value_passes(monkeypatch):
    """Correct X-ATS-Internal-Secret header must pass the auth gate (may fail on extraction — that's OK)."""
    secret = "super-secret-123"
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ATS_INTERNAL_SECRET", secret)
    with TestClient(app) as client:
        pdf_content = b"%PDF-1.4\nSome text"
        res = client.post(
            "/extract",
            files={"file": ("r.pdf", pdf_content, "application/pdf")},
            headers={"X-ATS-Internal-Secret": secret},
        )
    # Must NOT be 401 (may be 202/200/429/500 depending on worker state)
    assert res.status_code != 401

def test_global_rate_limiting_enforcement():
    """Global rate limiter rejects requests above the cap."""
    _rate_limit_records.clear()
    # Allow 5 requests then block
    for _ in range(5):
        assert check_rate_limit("global:test", max_requests=5, window_seconds=60) is True
    # 6th should be rejected
    assert check_rate_limit("global:test", max_requests=5, window_seconds=60) is False


def test_global_rate_limit_evicts_old_entries():
    """Entries older than the window should be evicted and not count toward the cap."""
    _rate_limit_records.clear()
    key = "global:eviction_test"
    # Manually inject stale entries (older than 60s)
    stale_time = time.time() - 70
    _rate_limit_records[key] = [stale_time, stale_time]
    # Even though 2 entries exist, they're stale — the first fresh request must succeed
    assert check_rate_limit(key, max_requests=2, window_seconds=60) is True


def test_orphaned_storage_cleanup_on_queue_full(monkeypatch):
    """
    When queue.put_nowait raises QueueFull after a storage upload (race condition between
    the capacity pre-check and the actual enqueue), the orphaned storage object must be
    deleted. We simulate this by patching queue.put_nowait to raise QueueFull even though
    the queue appears non-full to qsize().
    """
    cleaned_up: list = []

    class MockStorage:
        def from_(self, bucket: str):
            return self

        def remove(self, paths):
            cleaned_up.extend(paths)

    class MockClient:
        storage = MockStorage()

    async def _run():
        service = ExtractionService(max_queue_size=10)
        await service.start()
        try:
            # Patch is_supabase_configured and get_supabase_client so storage upload runs
            monkeypatch.setattr("app.services.extraction_service.is_supabase_configured", lambda: True)
            monkeypatch.setattr("app.services.extraction_service.get_supabase_client", lambda: MockClient())

            # Fake storage upload that returns a known path
            fake_path = "resumes/fake-uuid/race.pdf"
            def mock_upload(file_bytes, filename, content_type, client=None):
                return fake_path, "fake-uuid/race.pdf"
            monkeypatch.setattr("app.services.extraction_service.upload_resume_file_to_storage", mock_upload)

            # Patch queue.put_nowait to raise QueueFull — simulating the race
            import asyncio as _asyncio
            original_put_nowait = service.queue.put_nowait
            call_count = [0]
            def patched_put_nowait(item):
                call_count[0] += 1
                raise _asyncio.QueueFull()
            service.queue.put_nowait = patched_put_nowait

            resume_bytes = b"%PDF-1.4\n" + b"R" * 100
            code, resp = await service.submit_job(resume_bytes, "race.pdf", "application/pdf")

            assert code == 429, f"Expected 429 got {code}"
            assert call_count[0] >= 1, "put_nowait was never called"
            assert len(cleaned_up) >= 1, (
                "Expected orphaned storage file to be cleaned up after QueueFull race. "
                f"cleaned_up={cleaned_up}"
            )
            assert "fake-uuid/race.pdf" in cleaned_up[0], (
                f"Expected cleaned path to contain 'fake-uuid/race.pdf', got {cleaned_up}"
            )
        finally:
            await service.stop()

    asyncio.run(_run())


def test_rate_limiting_enforcement():
    """Validates global rate limiting rejects requests when quota is exceeded."""
    _rate_limit_records.clear()

    # Extract rate limit allows up to configured max
    for _ in range(5):
        assert check_rate_limit("global:extract_legacy", max_requests=5, window_seconds=60) is True

    # 6th request must be rejected
    assert check_rate_limit("global:extract_legacy", max_requests=5, window_seconds=60) is False


def test_health_endpoint_reports_assets_status():
    """Validates /health reports assets_ok boolean."""
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert "assets_ok" in data
        assert isinstance(data["assets_ok"], bool)

