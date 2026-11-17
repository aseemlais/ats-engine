"""
test_async_serving_local.py
===========================
Full asynchronous serving test suite designed for local execution outside
the restricted Windows terminal sandbox.

Runs without skips:
  1. ExtractionService Bounded asyncio.Queue + Single Background Worker loop
  2. Queue Saturation HTTP 429 handling
  3. FastAPI /extract and /jobs/{id} HTTP polling flow
  4. Shadow Mode field-level diff calculation and background execution

How to run locally:
  pytest tests/test_async_serving_local.py -v
"""

import asyncio
import os
import sys
import time
import pytest
from fastapi.testclient import TestClient

from app.adapters.extraction_adapter import ExtractionAdapter
from app.server import app, extraction_service
from app.services.extraction_service import ExtractionService


def test_local_extraction_service_queue_and_worker():
    """
    Validates end-to-end async queue processing:
    - Service starts background worker
    - Job is submitted and receives 202 Accepted + job_id
    - Worker picks up job, executes extraction, and sets status to 'completed'
    - Verifies 'extractionSource' is recorded
    """
    async def _run():
        service = ExtractionService(max_queue_size=5)
        await service.start()
        try:
            sample_resume = (
                b"SARAH CONNOR\n"
                b"Lead Systems Engineer | sarah@cyberdyne.org | +1-415-555-0199\n"
                b"EXPERIENCE\n"
                b"Principal Infrastructure Architect (Kubernetes / Go) | Tech Corp | Jan 2021 - Present\n"
                b"SKILLS\n"
                b"Go, Kubernetes, Docker, Python, AWS\n"
            )

            code, resp = await service.submit_job(sample_resume, "sarah_resume.txt", "text/plain")
            assert code == 202
            assert "job_id" in resp
            job_id = resp["job_id"]

            # Poll for worker completion (up to 15 seconds)
            for _ in range(75):
                await asyncio.sleep(0.2)
                job = service.get_job(job_id)
                if job and job["status"] in ("completed", "failed"):
                    break

            job = service.get_job(job_id)
            assert job is not None
            assert job["status"] == "completed", f"Job failed: {job.get('error')}"
            assert job["result"] is not None
            assert "candidateName" in job["result"]
            assert job.get("extractionSource") in ("slm", "fallback")
            assert job["result"].get("extractionSource") in ("slm", "fallback")
        finally:
            await service.stop()

    asyncio.run(_run())


def test_local_extraction_service_queue_capacity_429():
    """
    Validates that when the bounded queue is filled to capacity,
    subsequent requests are rejected immediately with HTTP 429 Too Many Requests.
    """
    async def _run():
        small_service = ExtractionService(max_queue_size=2)
        # Intentionally do not start worker so jobs accumulate
        code1, _ = await small_service.submit_job(b"Resume 1", "res1.txt")
        code2, _ = await small_service.submit_job(b"Resume 2", "res2.txt")
        code3, resp3 = await small_service.submit_job(b"Resume 3", "res3.txt")

        assert code1 == 202
        assert code2 == 202
        assert code3 == 429
        assert "capacity" in resp3["error"].lower()

    asyncio.run(_run())


def test_local_fastapi_extract_and_poll_endpoints():
    """
    Validates full HTTP API flow using FastAPI TestClient:
    - POST /extract -> returns 200 (cache hit) or 202 (enqueued)
    - GET /jobs/{job_id} -> returns job status and payload
    - GET /jobs/{bad_id} -> returns 404
    """
    with TestClient(app) as client:
        files = {
            "file": ("dev_cv.txt", b"Jane Doe\nStaff Engineer\njane.doe@example.com", "text/plain")
        }
        res = client.post("/extract", files=files)
        assert res.status_code in (200, 202)
        data = res.json()
        assert "job_id" in data
        job_id = data["job_id"]

        poll_res = client.get(f"/jobs/{job_id}")
        assert poll_res.status_code == 200
        poll_data = poll_res.json()
        assert poll_data["job_id"] == job_id
        assert poll_data["status"] in ("pending", "processing", "completed")

        bad_res = client.get("/jobs/non_existent_uuid_99999")
        assert bad_res.status_code == 404


def test_local_shadow_mode_diff_logging(monkeypatch):
    """
    Validates Shadow Mode:
    - When EXTRACTION_SHADOW_MODE=true, alternate extractor runs in background
    - Computes field-level differences (skills Jaccard, experience delta, title match)
    - Emits structured diff logs without disrupting or delaying the primary output
    """
    monkeypatch.setenv("EXTRACTION_SHADOW_MODE", "true")
    monkeypatch.setenv("USE_NEW_EXTRACTOR", "false")

    class MockSLMService:
        def _infer_llm(self, text):
            return {
                "candidateName": "Jane Doe",
                "jobTitle": "Lead Engineer (Python / Distributed Systems)",
                "email": "jane.doe@example.com",
                "phone": "+1-415-555-0100",
                "skills": ["Python", "Distributed Systems", "Kubernetes"],
                "workExperience": [
                    {"title": "Lead Engineer", "dates": "Jan 2020 - Jan 2024", "duration_years": 4.0}
                ]
            }

    adapter = ExtractionAdapter(extraction_service=MockSLMService())
    assert adapter.is_shadow_mode_enabled is True

    resume_text = (
        "Jane Doe\n"
        "Lead Engineer | jane.doe@example.com | +1-415-555-0100\n"
        "Skills: Python, Go, Docker\n"
    )

    primary_res = adapter.extract(resume_text)
    assert primary_res["extractionSource"] == "legacy"

    # Allow daemon background thread to finish
    time.sleep(0.5)

    assert adapter.latest_shadow_diff is not None
    diff = adapter.latest_shadow_diff
    assert diff["primary"] == "legacy"
    assert diff["shadow"] == "slm"
    assert diff["nameMatch"] is True
    assert "skillsJaccard" in diff
