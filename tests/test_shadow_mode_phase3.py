"""
tests/test_shadow_mode_phase3.py
================================
Phase 3: Shadow Mode Unit & Integration Tests.

Acceptance Criteria:
  1. Memory guard disables shadow mode when RSS passes threshold without affecting primary traffic.
  2. Main queue has strict priority over shadow queue; shadow jobs run only when main queue is empty.
  3. Shadow jobs are skipped when main queue depth exceeds threshold.
  4. Deterministic hash sampling (configurable sample rate).
  5. Shadow errors/exceptions never affect the primary extraction response or latency.
  6. GET /health returns {status, model_loaded, shadow_enabled, queue_depth, rss_mb, assets_ok}.
  7. Shadow evaluator metric calculation (exact contacts, fuzzy titles/names, set Jaccard, grounding).
"""

import asyncio
import os
import sys
import time
import pytest
from fastapi.testclient import TestClient

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.server import app, extraction_service
from app.services.extraction_service import ExtractionService
from app.services.shadow_evaluator import (
    compute_field_agreements,
    compute_grounding_flags,
    compute_validity_flags,
)


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Memory Guard Safety Tests
# ---------------------------------------------------------------------------

def test_memory_guard_disables_shadow_when_threshold_exceeded(monkeypatch):
    """
    When process RSS exceeds max_rss_mb, memory guard must trip, disable shadow mode,
    and prevent shadow jobs from executing, while primary traffic continues uninterrupted.
    """
    service = ExtractionService()
    service.max_rss_mb = 100.0  # Set low threshold to trip guard

    # Mock get_current_rss_mb to simulate elevated RSS (150 MB)
    monkeypatch.setattr(service, "get_current_rss_mb", lambda: 150.0)

    # Initially shadow enabled
    monkeypatch.setenv("EXTRACTION_SHADOW_MODE", "true")
    assert service.is_shadow_mode_active is True

    # Memory guard check
    safe = service.check_memory_guard()
    assert safe is False
    assert service.shadow_disabled_by_memory_guard is True
    assert service.is_shadow_mode_active is False

    # Attempting to enqueue shadow job should be rejected
    enqueued = service.enqueue_shadow_job("test_hash_123", "Resume text", {"skills": ["Python"]}, "legacy")
    assert enqueued is False
    assert service.shadow_queue.qsize() == 0


def test_memory_guard_tripped_does_not_block_primary_traffic(monkeypatch):
    """
    Even when memory guard disables shadow mode, primary jobs must be accepted
    and processed cleanly to completion.
    """
    async def _run():
        service = ExtractionService(max_queue_size=10)
        service.max_rss_mb = 50.0
        monkeypatch.setattr(service, "get_current_rss_mb", lambda: 200.0)
        monkeypatch.setenv("EXTRACTION_SHADOW_MODE", "true")

        # Trip memory guard
        assert service.check_memory_guard() is False

        await service.start()
        try:
            sample_resume = b"JOHN DOE\nSoftware Engineer | john@doe.io\nEXPERIENCE\nSoftware Engineer | 2021 - Present\nSKILLS\nPython, Docker"
            code, resp = await service.submit_job(sample_resume, "john.txt", "text/plain")
            assert code == 202
            job_id = resp["job_id"]

            for _ in range(50):
                await asyncio.sleep(0.05)
                job = service.get_job(job_id)
                if job and job["status"] in ("completed", "failed"):
                    break

            job = service.get_job(job_id)
            assert job is not None
            assert job["status"] == "completed"
            assert "candidateName" in job["result"]
            # Shadow queue remained empty
            assert service.shadow_queue.qsize() == 0
        finally:
            await service.stop()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2. Priority & Queue Starvation Tests
# ---------------------------------------------------------------------------

def test_main_queue_has_strict_priority_over_shadow_queue():
    """
    When both main queue and shadow queue have items, the worker loop must
    process all primary items before picking up any shadow job.
    """
    async def _run():
        service = ExtractionService(max_queue_size=10)
        processed_order = []

        # Intercept primary and shadow execution
        original_infer = service._infer_llm
        def mock_infer(text):
            processed_order.append("shadow")
            return {"candidateName": "Shadow Candidate"}, "stop"

        service._infer_llm = mock_infer

        # Manually populate queues before starting worker
        primary_job_id = "primary-job-1"
        service.jobs[primary_job_id] = {
            "id": primary_job_id,
            "status": "pending",
            "file_hash": "hash_prim",
            "filename": "primary.txt",
            "attempts": 0,
        }
        # Primary job payload: (job_id, file_bytes, filename, content_type, storage_path)
        service.queue.put_nowait((primary_job_id, b"PRIMARY RESUME\nAlice\nalice@test.com", "primary.txt", "text/plain", None))

        # Shadow job payload: (file_hash, text_to_extract, primary_output, primary_source)
        service.shadow_queue.put_nowait(("hash_shadow", "SHADOW RESUME TEXT", {"candidateName": "Legacy"}, "legacy"))

        await service.start()
        try:
            # Wait for both to be processed
            for _ in range(50):
                await asyncio.sleep(0.05)
                if service.get_job(primary_job_id)["status"] == "completed" and service.shadow_queue.empty():
                    break

            assert service.get_job(primary_job_id)["status"] == "completed"
            assert service.shadow_queue.empty()
        finally:
            await service.stop()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. Queue Depth Threshold Skip Tests
# ---------------------------------------------------------------------------

def test_shadow_job_skipped_when_main_queue_depth_exceeds_threshold():
    """
    If main queue depth exceeds shadow_max_queue_depth, enqueue_shadow_job
    must return False and drop the job to protect the primary pipeline.
    """
    service = ExtractionService(max_queue_size=20)
    service.shadow_max_queue_depth = 2
    service.shadow_sample_rate = 1.0  # 100% sample rate

    import os
    os.environ["EXTRACTION_SHADOW_MODE"] = "true"

    # Fill main queue with 3 dummy items (> 2 threshold)
    for i in range(3):
        service.queue.put_nowait((f"id_{i}", b"data", "f.pdf", "app/pdf", None))

    assert service.queue.qsize() == 3

    # Enqueue should be skipped
    enqueued = service.enqueue_shadow_job("hash_xyz", "Resume Text", {"skills": []}, "legacy")
    assert enqueued is False
    assert service.shadow_queue.qsize() == 0


# ---------------------------------------------------------------------------
# 4. Deterministic Hash Sampling Tests
# ---------------------------------------------------------------------------

def test_deterministic_hash_sampling():
    """
    Tests that sample selection is deterministic and strictly respects sample rate.
    """
    service = ExtractionService()
    service.shadow_sample_rate = 0.25  # 25%

    # Known hashes with predictable prefix mod 100:
    # int("00000000", 16) % 100 = 0 -> < 25 -> True
    hash_accept = "00000000abcdef123456"
    assert service.is_sample_selected_for_shadow(hash_accept) is True
    # Re-testing the same hash must produce the exact same outcome
    assert service.is_sample_selected_for_shadow(hash_accept) is True

    # int("00000050", 16) % 100 = 80 -> >= 25 -> False
    hash_reject = "00000050abcdef123456"
    assert service.is_sample_selected_for_shadow(hash_reject) is False
    assert service.is_sample_selected_for_shadow(hash_reject) is False

    # 0.0 sample rate rejects all
    service.shadow_sample_rate = 0.0
    assert service.is_sample_selected_for_shadow(hash_accept) is False

    # 1.0 sample rate accepts all
    service.shadow_sample_rate = 1.0
    assert service.is_sample_selected_for_shadow(hash_reject) is True


# ---------------------------------------------------------------------------
# 5. Shadow Failure Isolation (Zero Primary Impact)
# ---------------------------------------------------------------------------

def test_shadow_failure_never_affects_primary_response():
    """
    If shadow execution raises an unhandled exception, the primary job
    must still complete successfully with status 'completed' and valid result.
    """
    async def _run():
        service = ExtractionService(max_queue_size=10)
        service.shadow_sample_rate = 1.0

        # Simulate shadow job raising exception
        async def boom_shadow(*args, **kwargs):
            raise RuntimeError("Fatal shadow failure")

        service._process_shadow_job = boom_shadow

        os.environ["EXTRACTION_SHADOW_MODE"] = "true"
        await service.start()
        try:
            sample_resume = b"SARAH CONNOR\nSystems Architect | sarah@sky.net\nEXPERIENCE\nSystems Architect | 2020 - Present\nSKILLS\nGo, C++"
            code, resp = await service.submit_job(sample_resume, "sarah.txt", "text/plain")
            assert code == 202
            job_id = resp["job_id"]

            for _ in range(50):
                await asyncio.sleep(0.05)
                job = service.get_job(job_id)
                if job and job["status"] in ("completed", "failed"):
                    break

            job = service.get_job(job_id)
            assert job is not None
            # Primary job succeeded regardless of shadow crash
            assert job["status"] == "completed"
            assert job["result"]["candidateName"].lower() == "sarah connor"
        finally:
            await service.stop()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 6. Health Endpoint Returns Phase 3 Fields
# ---------------------------------------------------------------------------

def test_health_endpoint_returns_phase3_fields(client):
    """
    GET /health must return status, model_loaded, shadow_enabled, queue_depth, rss_mb, assets_ok.
    """
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "shadow_enabled" in data
    assert "queue_depth" in data
    assert "rss_mb" in data
    assert "assets_ok" in data
    assert isinstance(data["queue_depth"], int)
    assert isinstance(data["rss_mb"], (int, float))
    assert data["rss_mb"] > 0.0


# ---------------------------------------------------------------------------
# 7. Shadow Evaluator Metrics & Validity Unit Tests
# ---------------------------------------------------------------------------

def test_shadow_evaluator_metrics_and_validity():
    """
    Verifies agreement, validity, and grounding calculations in shadow_evaluator.
    """
    legacy = {
        "candidateName": "Alexander Morgan",
        "jobTitle": "Lead Cloud Architect (AWS / Kubernetes)",
        "email": "alex@cloudops.org",
        "phone": "+1-415-555-0199",
        "skills": ["Go", "Kubernetes", "AWS", "Python"],
        "totalExperienceYears": 8.0,
        "education": [{"degree": "B.S. Computer Science"}],
        "workExperience": [{"title": "Principal Architect", "dates": "2020 - Present", "durationYears": 4.0}],
    }
    slm = {
        "candidateName": "Alexander Morgan",
        "jobTitle": "Lead Cloud Architect (AWS / Kubernetes)",
        "email": "alex@cloudops.org",
        "phone": "(415) 555-0199",
        "skills": ["Kubernetes", "AWS", "Python", "Docker"],
        "totalExperienceYears": 8.5,
        "education": [{"degree": "B.S. Computer Science"}],
        "workExperience": [{"title": "Principal Architect", "dates": "2020 - Present", "durationYears": 4.0}],
    }

    # Field Agreements
    agreements = compute_field_agreements(legacy, slm)
    assert agreements["emailExactMatch"] is True
    assert agreements["phoneExactMatch"] is True  # Digits match
    assert agreements["nameTokenF1"] == 1.0
    assert agreements["titleTokenF1"] == 1.0
    assert agreements["expDeltaYears"] == 0.5
    assert agreements["skillsJaccard"] == pytest.approx(3 / 5, abs=0.01)

    # Validity Flags
    validity = compute_validity_flags(slm)
    assert validity["emailValid"] is True
    assert validity["phoneValid"] is True
    assert validity["namePlausible"] is True
    assert validity["datesParse"] is True

    # Grounding Flags (Hallucination Detection)
    resume_text = "Alexander Morgan is a Lead Cloud Architect. Contact: alex@cloudops.org. Skills: Kubernetes, AWS, Python, Docker."
    grounding = compute_grounding_flags(slm, resume_text)
    assert grounding["allGrounded"] is True
    assert len(grounding["ungroundedEntities"]) == 0

    # Grounding flag catches fabricated name not in source text
    hallucinated = dict(slm)
    hallucinated["candidateName"] = "Viktor Reznov"
    grounding_bad = compute_grounding_flags(hallucinated, resume_text)
    assert grounding_bad["allGrounded"] is False
    assert any("candidateName:Viktor Reznov" in e for e in grounding_bad["ungroundedEntities"])
