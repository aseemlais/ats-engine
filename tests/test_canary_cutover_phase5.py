"""
tests/test_canary_cutover_phase5.py
===================================
Acceptance & Unit Test Suite for Phase 5: Canary Cutover & Rollback Controls.

Validates:
  1. Deterministic file-hash routing consistency
  2. Statistical distribution across 0%, 10%, 50%, and 100% canary cutover
  3. Single-variable instant rollback (EXTRACTION_CUTOVER_PERCENTAGE=0)
  4. Worker canary execution with SLM success, fallback on error, and legacy routing
  5. GET /health reports cutover_percentage dynamically
  6. Re-extraction utility logic and dry-run execution
"""

import asyncio
import hashlib
import os
import pytest
from typing import Any, Dict
from fastapi.testclient import TestClient

from app.server import app
from app.services.extraction_service import ExtractionService
from scripts.reextract_resumes import run_reextraction


# ---------------------------------------------------------------------------
# 1. Deterministic Routing & Consistency Tests
# ---------------------------------------------------------------------------

def test_deterministic_hash_routing_consistency():
    service = ExtractionService()
    test_hash = "a1b2c3d4e5f67890abcdef1234567890"

    os.environ["EXTRACTION_CUTOVER_PERCENTAGE"] = "10.0"
    first_decision = service.should_route_to_slm(test_hash)

    # Must be 100% deterministic across multiple queries
    for _ in range(50):
        assert service.should_route_to_slm(test_hash) == first_decision


def test_cutover_distribution_proportions():
    service = ExtractionService()
    sample_hashes = [hashlib.sha256(f"resume_sample_{i}".encode()).hexdigest() for i in range(1000)]

    # 1. Test 0% Cutover (Instant Rollback)
    os.environ["EXTRACTION_CUTOVER_PERCENTAGE"] = "0.0"
    slm_count_0 = sum(1 for h in sample_hashes if service.should_route_to_slm(h))
    assert slm_count_0 == 0

    # 2. Test 10% Canary Cutover (Target ~100 / 1000, tolerance ±30)
    os.environ["EXTRACTION_CUTOVER_PERCENTAGE"] = "10.0"
    slm_count_10 = sum(1 for h in sample_hashes if service.should_route_to_slm(h))
    assert 70 <= slm_count_10 <= 130

    # 3. Test 50% Canary Cutover (Target ~500 / 1000, tolerance ±40)
    os.environ["EXTRACTION_CUTOVER_PERCENTAGE"] = "50.0"
    slm_count_50 = sum(1 for h in sample_hashes if service.should_route_to_slm(h))
    assert 460 <= slm_count_50 <= 540

    # 4. Test 100% Full Cutover
    os.environ["EXTRACTION_CUTOVER_PERCENTAGE"] = "100.0"
    slm_count_100 = sum(1 for h in sample_hashes if service.should_route_to_slm(h))
    assert slm_count_100 == 1000


def test_instant_rollback_zero_percentage():
    service = ExtractionService()
    os.environ["EXTRACTION_CUTOVER_PERCENTAGE"] = "0"
    os.environ["USE_NEW_EXTRACTOR"] = "false"

    assert service.current_cutover_percentage == 0.0
    for i in range(100):
        fhash = hashlib.sha256(f"candidate_{i}".encode()).hexdigest()
        assert service.should_route_to_slm(fhash) is False


def test_dynamic_env_override():
    service = ExtractionService()
    os.environ["EXTRACTION_CUTOVER_PERCENTAGE"] = "25.0"
    assert service.current_cutover_percentage == 25.0

    os.environ["EXTRACTION_CUTOVER_PERCENTAGE"] = "75.5"
    assert service.current_cutover_percentage == 75.5

    # USE_NEW_EXTRACTOR overrides to 100%
    os.environ["USE_NEW_EXTRACTOR"] = "true"
    assert service.current_cutover_percentage == 100.0
    os.environ["USE_NEW_EXTRACTOR"] = "false"


# ---------------------------------------------------------------------------
# 2. Worker Canary Execution & Fallback Tests
# ---------------------------------------------------------------------------

def test_worker_canary_routing_slm_success():
    """When inside cutover percentage and SLM succeeds, extractionSource is 'slm'."""
    async def _run():
        service = ExtractionService(max_queue_size=10)
        os.environ["EXTRACTION_CUTOVER_PERCENTAGE"] = "100.0"  # 100% canary

        # Mock successful SLM inference
        def mock_infer(text):
            return {
                "candidateName": "Elena Rostova",
                "email": "elena@quantum.io",
                "jobTitle": "Principal AI Engineer",
                "skills": ["PyTorch", "CUDA", "LLMs"],
                "totalExperienceYears": 7.0,
            }, "stop"

        service._infer_llm = mock_infer
        await service.start()
        try:
            resume_bytes = b"Elena Rostova\nPrincipal AI Engineer | elena@quantum.io\nPyTorch, CUDA, LLMs"
            code, resp = await service.submit_job(resume_bytes, "elena.txt", "text/plain")
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
            assert job["result"]["extractionSource"] == "slm"
            assert job["result"]["candidateName"].lower() == "elena rostova"
        finally:
            await service.stop()

    asyncio.run(_run())


def test_worker_canary_slm_failure_executes_safe_fallback():
    """When inside cutover and SLM inference fails or length truncates, safely falls back to 'fallback'."""
    async def _run():
        service = ExtractionService(max_queue_size=10)
        os.environ["EXTRACTION_CUTOVER_PERCENTAGE"] = "100.0"

        # Mock truncated SLM inference
        def mock_infer(text):
            return {"truncated": True}, "length"

        service._infer_llm = mock_infer
        await service.start()
        try:
            resume_bytes = b"Marcus Aurelius\nSenior Systems Architect | marcus@rome.org\nGo, C++"
            code, resp = await service.submit_job(resume_bytes, "marcus.txt", "text/plain")
            assert code == 202
            job_id = resp["job_id"]

            for _ in range(50):
                await asyncio.sleep(0.05)
                job = service.get_job(job_id)
                if job and job["status"] in ("completed", "failed"):
                    break

            job = service.get_job(job_id)
            assert job is not None
            # Must succeed via fallback without error
            assert job["status"] == "completed"
            assert job["result"]["extractionSource"] == "fallback"
            assert job["result"]["candidateName"].lower() == "marcus aurelius"
        finally:
            await service.stop()

    asyncio.run(_run())


def test_worker_canary_outside_percentage_routes_to_legacy():
    """When outside cutover percentage (0%), executes 'legacy' directly without calling SLM."""
    async def _run():
        service = ExtractionService(max_queue_size=10)
        os.environ["EXTRACTION_CUTOVER_PERCENTAGE"] = "0.0"  # 0% cutover

        # SLM should never be called
        def boom_infer(text):
            raise RuntimeError("SLM should not be called when cutover is 0%")

        service._infer_llm = boom_infer
        await service.start()
        try:
            resume_bytes = b"Ada Lovelace\nChief Algorithm Officer | ada@analytical.engine\nMath, Logic"
            code, resp = await service.submit_job(resume_bytes, "ada.txt", "text/plain")
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
            assert job["result"]["extractionSource"] == "legacy"
            assert job["result"]["candidateName"].lower() == "ada lovelace"
        finally:
            await service.stop()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. Health Endpoint Cutover Telemetry Test
# ---------------------------------------------------------------------------

def test_health_endpoint_reports_cutover_percentage():
    client = TestClient(app)
    os.environ["EXTRACTION_CUTOVER_PERCENTAGE"] = "35.5"

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()

    assert "cutover_percentage" in data
    assert data["cutover_percentage"] == 35.5
    assert data["status"] == "healthy"


# ---------------------------------------------------------------------------
# 4. Re-Extraction Script Test
# ---------------------------------------------------------------------------

def test_reextract_resumes_dry_run_and_execution():
    local_test_corpus = [
        {
            "candidate_id": "TEST_CAND_01",
            "file_hash": "hash_test_01",
            "raw_text": "Alex Mercer | alex@mercer.dev | Senior Rust Developer\nSkills: Rust, WebAssembly, Tokio\nExperience: 5 years building async engines.",
        },
        {
            "candidate_id": "TEST_CAND_02",
            "file_hash": "hash_test_02",
            "raw_text": "Jordan Lee | jordan@cloud.org | Staff Site Reliability Engineer\nSkills: Kubernetes, Terraform, AWS, Go\nExperience: 8 years in infra.",
        },
    ]

    res = run_reextraction(
        limit=10,
        force=True,
        dry_run=True,
        delay_seconds=0.0,
        target_version="ats-slm:qwen2.5-1.5b-instruct:chatml-v1:schema-v1",
        local_corpus=local_test_corpus,
    )

    assert res["processed_count"] == 2
    assert res["success_count"] == 2
    assert res["failure_count"] == 0
    assert res["dry_run"] is True
    assert len(res["summary"]) == 2
    assert res["summary"][0]["engine"] in ("slm", "fallback")
