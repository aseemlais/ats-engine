import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import ftfy

from app.models.resume_schema import ProcessedResume, RawResumeExtraction
from app.services.post_processor import post_process_resume
from app.services.prompt_templates import build_chatml_prompt
from app.services.resume_parser import ResumeParserService
from app.storage.resume_storage import (
    _get_table_ref,
    upload_resume_file_to_storage,
    download_resume_file_from_storage,
    store_resume,
)
from app.storage.supabase_client import get_supabase_client, is_supabase_configured

logger = logging.getLogger("extraction_service")

# Default queue capacity
DEFAULT_QUEUE_MAXSIZE = 50
MAX_RECOVERY_ATTEMPTS = 3
MAX_PARSING_TIMEOUT_SECONDS = 15.0


def get_current_rss_mb() -> float:
    """Returns current process Resident Set Size (RSS) in megabytes across platforms."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / (1024 * 1024)
        except Exception:
            pass
    else:
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return (usage / 1024.0) if sys.platform != "darwin" else (usage / (1024 * 1024))
        except Exception:
            pass
    return 0.0


class ExtractionService:
    """
    Asynchronous resume extraction service with:
    - Bounded asyncio.Queue (returns HTTP 429 when saturated)
    - Exactly ONE background worker task
    - Non-blocking CPU/GPU model inference offloaded to asyncio.to_thread
    - llama-cpp-python GGUF model integration with strict JSON grammar
    - In-memory job state backed by Supabase ExtractionJobs table
    - Version-aware Supabase file-hash deduplication cache
    - Safe fallback to deterministic extraction
    - Startup recovery of interrupted jobs
    - Phase 3 Shadow Mode: separate low-priority queue, memory guard, and deterministic sampling
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_queue_size: int = DEFAULT_QUEUE_MAXSIZE,
        fallback_extractor: Optional[Any] = None,
    ):
        self.model_path = model_path or os.getenv("GGUF_MODEL_PATH", "models/unsloth.Q4_K_M.gguf")
        self.max_queue_size = max_queue_size
        self._queue: Optional[asyncio.Queue] = None
        self._shadow_queue: Optional[asyncio.Queue] = None
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.llm = None
        self.model_loaded: bool = False
        self.worker_task: Optional[asyncio.Task] = None
        self.recovery_task: Optional[asyncio.Task] = None
        self.model_loader_task: Optional[asyncio.Task] = None
        self.running: bool = False
        self.parser = ResumeParserService()
        self._fallback_extractor = fallback_extractor
        self.shadow_disabled_by_memory_guard: bool = False
        self.max_rss_mb: float = float(os.getenv("SHADOW_MAX_RSS_MB", "6144.0"))
        self.shadow_sample_rate: float = float(os.getenv("SHADOW_SAMPLE_RATE", "0.25"))
        self.shadow_max_queue_depth: int = int(os.getenv("SHADOW_MAX_QUEUE_DEPTH", "5"))
        try:
            self.cutover_percentage: float = float(os.getenv("EXTRACTION_CUTOVER_PERCENTAGE", "0.0"))
        except ValueError:
            self.cutover_percentage: float = 0.0

    @property
    def current_cutover_percentage(self) -> float:
        """Dynamically reads cutover percentage from environment var (default: 0.0)."""
        if os.getenv("USE_NEW_EXTRACTOR", "").strip().lower() in ("true", "1", "yes", "on"):
            return 100.0
        val_str = os.getenv("EXTRACTION_CUTOVER_PERCENTAGE")
        if val_str is not None:
            try:
                return max(0.0, min(100.0, float(val_str.strip())))
            except ValueError:
                pass
        return self.cutover_percentage

    def should_route_to_slm(self, file_hash: str) -> bool:
        """
        Deterministically evaluates if the given resume file hash falls within
        the current canary cutover percentage.
        - If cutover == 0: immediately returns False (instant rollback)
        - If cutover == 100: returns True (full cutover)
        - Else: calculates SHA-256 hash integer modulo 10,000 against threshold.
        """
        pct = self.current_cutover_percentage
        if pct <= 0.0:
            return False
        if pct >= 100.0:
            return True

        h = hashlib.sha256(file_hash.encode("utf-8")).hexdigest()
        bucket = int(h, 16) % 10000
        threshold = int(pct * 100)
        return bucket < threshold

    @property
    def current_extractor_version(self) -> str:
        """Returns the canonical extractor version string based on the active feature flag or cutover."""
        pct = self.current_cutover_percentage
        if pct >= 100.0:
            return "ats-slm:qwen2.5-1.5b-instruct:chatml-v1:schema-v1"
        elif pct > 0.0:
            return f"ats-canary:slm-{int(pct)}pct:chatml-v1:schema-v1"
        return "ats-legacy:regex-spacy:prompt-none:schema-v1"

    @property
    def is_model_loaded(self) -> bool:
        return self.llm is not None and getattr(self, "model_loaded", False)

    @property
    def is_shadow_mode_active(self) -> bool:
        val = (
            os.getenv("EXTRACTION_SHADOW_MODE")
            or os.getenv("SHADOW_MODE", "false")
        ).strip().lower()
        enabled = val in ("true", "1", "yes", "on", "enabled")
        return enabled and not self.shadow_disabled_by_memory_guard

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize() if self._queue is not None else 0

    @property
    def shadow_queue(self) -> asyncio.Queue:
        if self._shadow_queue is None:
            self._shadow_queue = asyncio.Queue(maxsize=50)
        return self._shadow_queue

    def get_current_rss_mb(self) -> float:
        return get_current_rss_mb()

    def check_memory_guard(self) -> bool:
        """Disables shadow mode if resident memory exceeds configured safety threshold."""
        rss = self.get_current_rss_mb()
        if rss > self.max_rss_mb:
            if not self.shadow_disabled_by_memory_guard:
                logger.warning(
                    "[MemoryGuard] Process RSS (%.2f MB) exceeded threshold (%.2f MB). Disabling shadow mode.",
                    rss,
                    self.max_rss_mb,
                )
                self.shadow_disabled_by_memory_guard = True
            return False
        return True

    def is_sample_selected_for_shadow(self, file_hash: str) -> bool:
        """Deterministic sampling based on file hash modulo 100."""
        if not file_hash or self.shadow_sample_rate <= 0:
            return False
        if self.shadow_sample_rate >= 1.0:
            return True
        hash_val = int(file_hash[:8], 16) % 100
        return hash_val < int(self.shadow_sample_rate * 100)

    @property
    def queue(self) -> asyncio.Queue:
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.max_queue_size)
        return self._queue

    def load_model(self) -> None:
        """Loads GGUF model synchronously if called directly."""
        if not self.model_path or not os.path.exists(self.model_path):
            logger.info("[ExtractionService] GGUF model not found at %s. Running in fallback mode.", self.model_path)
            return

        try:
            import llama_cpp
            logger.info("[ExtractionService] Loading GGUF model from %s...", self.model_path)
            threads = int(os.getenv("LLAMA_THREADS", str(min(4, os.cpu_count() or 2))))
            self.llm = llama_cpp.Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_threads=threads,
                verbose=False,
            )
            self.model_loaded = True
            logger.info("[ExtractionService] GGUF model loaded successfully.")
        except Exception as exc:
            logger.warning("[ExtractionService] Failed loading llama_cpp model: %s. Using fallback mode.", exc)
            self.llm = None
            self.model_loaded = False

    async def _lazy_init_model(self) -> None:
        """
        Background startup task: downloads GGUF via huggingface_hub if repo id is specified,
        and initializes llama_cpp in a thread so /health passes immediately on boot.
        """
        repo_id = os.getenv("GGUF_REPO_ID", "Qwen/Qwen2.5-1.5B-Instruct-GGUF")
        if not repo_id or repo_id.strip() == "" or "unsloth" in repo_id.lower():
            repo_id = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"

        filename = os.getenv("GGUF_FILENAME", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
        if not filename or filename.strip() == "" or "unsloth" in filename.lower():
            filename = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
        if repo_id == "Qwen/Qwen2.5-1.5B-Instruct-GGUF":
            filename = filename.lower()

        token = os.getenv("HF_TOKEN")

        dest_dir = "models"
        target_path = os.path.join(dest_dir, filename)

        if not os.path.exists(target_path):
            try:
                logger.info("[ExtractionService] Downloading GGUF from Hugging Face Hub (%s/%s)...", repo_id, filename)
                from huggingface_hub import hf_hub_download
                os.makedirs(dest_dir, exist_ok=True)
                downloaded = await asyncio.to_thread(
                    hf_hub_download,
                    repo_id=repo_id,
                    filename=filename,
                    token=token,
                    local_dir=dest_dir,
                )
                self.model_path = downloaded
                logger.info("[ExtractionService] Downloaded GGUF to %s", self.model_path)
            except Exception as dl_err:
                logger.warning("[ExtractionService] Failed to download GGUF from %s: %s. Using fallback mode.", repo_id, dl_err)
        else:
            self.model_path = target_path
            logger.info("[ExtractionService] Found existing GGUF model at %s", self.model_path)

        if self.model_path and os.path.exists(self.model_path):
            try:
                import llama_cpp
                logger.info("[ExtractionService] Initializing llama_cpp in background from %s...", self.model_path)
                threads = int(os.getenv("LLAMA_THREADS", str(min(4, os.cpu_count() or 2))))
                def _init():
                    return llama_cpp.Llama(
                        model_path=self.model_path,
                        n_ctx=4096,
                        n_threads=threads,
                        verbose=False,
                    )
                self.llm = await asyncio.to_thread(_init)
                self.model_loaded = True
                logger.info("[ExtractionService] GGUF model initialized and ready.")
            except Exception as load_err:
                logger.warning("[ExtractionService] Failed loading llama_cpp: %s. Running in fallback mode.", load_err)
                self.llm = None
                self.model_loaded = False
        else:
            logger.info("[ExtractionService] GGUF model not found at %s. Operating in fallback mode.", self.model_path)
            self.model_loaded = False

    async def ensure_worker(self) -> None:
        """
        Guarantees that the queue is initialized and the worker task is active
        on the running event loop. Starts or restarts the worker if it was never
        started or ended unexpectedly.
        """
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.max_queue_size)
        if self._shadow_queue is None:
            self._shadow_queue = asyncio.Queue(maxsize=50)

        if not self.running or self.worker_task is None or self.worker_task.done():
            logger.info("[ExtractionService] Worker is inactive or terminated. Spawning background worker task...")
            self.running = True
            if not self.llm and not self.model_loader_task:
                self.model_loader_task = asyncio.create_task(self._lazy_init_model(), name="ats_lazy_model_loader")
            self.worker_task = asyncio.create_task(self._worker_loop(), name="ats_extraction_worker")
            logger.info("[ExtractionService] Background worker task spawned: %s", self.worker_task.get_name())

    async def start(self) -> None:
        """Starts the single background worker task, lazy model loader, and begins recovery."""
        if self.running and self.worker_task and not self.worker_task.done():
            logger.info("[ExtractionService] Async extraction worker is already running.")
            return
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.max_queue_size)
        if self._shadow_queue is None:
            self._shadow_queue = asyncio.Queue(maxsize=50)
        self.running = True

        # Non-blocking lazy model load in background
        self.model_loader_task = asyncio.create_task(self._lazy_init_model(), name="ats_lazy_model_loader")
        self.worker_task = asyncio.create_task(self._worker_loop(), name="ats_extraction_worker")
        logger.info("[ExtractionService] Async extraction worker started (queue capacity: %d).", self.max_queue_size)

        # Trigger startup recovery of interrupted/pending jobs in background
        if is_supabase_configured():
            self.recovery_task = asyncio.create_task(self._recover_interrupted_jobs(), name="ats_startup_recovery")

    async def stop(self) -> None:
        """Gracefully shuts down the background worker and recovery task."""
        self.running = False
        if self.model_loader_task:
            self.model_loader_task.cancel()
            try:
                await self.model_loader_task
            except asyncio.CancelledError:
                pass
            self.model_loader_task = None

        if self.recovery_task:
            self.recovery_task.cancel()
            try:
                await self.recovery_task
            except asyncio.CancelledError:
                pass
            self.recovery_task = None

        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            self.worker_task = None
        self._queue = None
        self._shadow_queue = None
        logger.info("[ExtractionService] Extraction worker stopped.")

    def check_supabase_cache(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Queries Supabase CandidateResume table for an existing parsed resume
        matching this file_hash AND the current extractorVersion.
        Rows without extractedData or with version mismatch are a cache MISS.
        Does NOT return storageObjectPath to the client.
        """
        if not is_supabase_configured():
            return None

        try:
            client = get_supabase_client()
            if not client:
                return None

            table = _get_table_ref(client, "CandidateResume")
            res = (
                table.select("id, candidateName, email, jobRole, skills, extractedData, extractorVersion, resumeText")
                .eq("fileHash", file_hash)
                .limit(1)
                .execute()
            )
            if res and hasattr(res, "data") and res.data and len(res.data) > 0:
                record = res.data[0]
                extracted_data = record.get("extractedData")
                cached_version = record.get("extractorVersion")

                # Cache hit requires extractedData present and matching version
                if extracted_data and isinstance(extracted_data, dict):
                    if cached_version == self.current_extractor_version:
                        result_copy = dict(extracted_data)
                        result_copy["extractionSource"] = "cache"
                        result_copy["candidateId"] = str(record.get("id"))
                        result_copy["resumeText"] = result_copy.get("resumeText") or record.get("resumeText") or ""
                        # Strip storageObjectPath if accidentally stored
                        result_copy.pop("storageObjectPath", None)
                        return result_copy
                    else:
                        logger.info(
                            "[ExtractionService] Cache version mismatch for %s (cached: '%s', required: '%s'). Re-extracting.",
                            file_hash[:12],
                            cached_version,
                            self.current_extractor_version,
                        )
        except Exception as exc:
            logger.warning("[ExtractionService] Cache lookup error: %s", exc)

        return None

    def find_active_job_by_hash(self, file_hash: str) -> Optional[str]:
        """
        Checks if a job with the same fileHash is already pending or processing
        either in memory or in Supabase ExtractionJobs table.
        """
        # 1. Check in-memory
        for job_id, job in self.jobs.items():
            if job.get("file_hash") == file_hash and job.get("status") in ("pending", "processing"):
                return job_id

        # 2. Check Supabase ExtractionJobs table
        if is_supabase_configured():
            try:
                client = get_supabase_client()
                if client:
                    table = _get_table_ref(client, "ExtractionJobs")
                    res = (
                        table.select('"jobId"')
                        .eq("fileHash", file_hash)
                        .in_("status", ["pending", "processing"])
                        .limit(1)
                        .execute()
                    )
                    if res and res.data and len(res.data) > 0:
                        return str(res.data[0]["jobId"])
            except Exception as exc:
                logger.warning("[ExtractionService] Active job dedupe query error: %s", exc)

        return None

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves current status and result of a job.
        Checks in-memory registry first; falls back to Supabase ExtractionJobs table.
        """
        # 1. In-memory check
        if job_id in self.jobs:
            return self.jobs[job_id]

        # 2. Supabase ExtractionJobs fallback
        if is_supabase_configured():
            try:
                client = get_supabase_client()
                if client:
                    table = _get_table_ref(client, "ExtractionJobs")
                    res = (
                        table.select('"jobId", "fileHash", filename, status, "extractionSource", result, error, "createdAt", "updatedAt"')
                        .eq("jobId", job_id)
                        .limit(1)
                        .execute()
                    )
                    if res and res.data and len(res.data) > 0:
                        row = res.data[0]
                        res_payload = row.get("result")
                        if res_payload and isinstance(res_payload, dict):
                            res_payload.pop("storageObjectPath", None)
                        job_record = {
                            "id": str(row["jobId"]),
                            "status": row.get("status"),
                            "file_hash": row.get("fileHash"),
                            "filename": row.get("filename"),
                            "extractionSource": row.get("extractionSource"),
                            "result": res_payload,
                            "error": row.get("error"),
                            "created_at": row.get("createdAt"),
                            "updated_at": row.get("updatedAt"),
                        }
                        # Cache into memory for fast subsequent polling
                        self.jobs[job_id] = job_record
                        return job_record
            except Exception as exc:
                logger.warning("[ExtractionService] Supabase get_job lookup error: %s", exc)

        return None

    async def submit_job(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str = "application/pdf",
        auto_start_worker: bool = True,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Enqueues an extraction job:
        - If already cached in Supabase with matching extractorVersion: returns (200, cached_payload).
        - If an active job for the same fileHash is already pending/processing: returns (202, existing_job_id).
        - If queue is saturated: returns (429, error_payload) WITHOUT creating an orphan DB record.
        - Otherwise: uploads file to Storage first, registers in DB & memory, enqueues, and returns (202, job_id).
        """
        if auto_start_worker:
            await self.ensure_worker()

        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # 1. Check Version-Aware Supabase cache
        cached_record = self.check_supabase_cache(file_hash)
        if cached_record:
            job_id = str(uuid.uuid4())
            cached_data = dict(cached_record)
            cached_data["extractionSource"] = "cache"
            cached_data.pop("storageObjectPath", None)
            self.jobs[job_id] = {
                "id": job_id,
                "status": "completed",
                "file_hash": file_hash,
                "filename": filename,
                "cached": True,
                "extractionSource": "cache",
                "result": cached_data,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            logger.info("[ExtractionService] Job %s served from cache (version: '%s').", job_id, self.current_extractor_version)
            return 200, {
                "job_id": job_id,
                "status": "completed",
                "cached": True,
                "extractionSource": "cache",
                "result": cached_data,
            }

        # 2. Check for existing active job with the same fileHash (Deduplication)
        existing_job_id = self.find_active_job_by_hash(file_hash)
        if existing_job_id:
            logger.info("[ExtractionService] Reusing active job %s for duplicate upload of %s", existing_job_id, file_hash[:12])
            return 202, {
                "job_id": existing_job_id,
                "status": "pending",
                "message": "Resume is already queued for processing.",
                "deduplicated": True,
            }

        # 3. Check Queue Capacity BEFORE creating any persistent state
        if self.queue.qsize() >= self.max_queue_size:
            logger.warning("[ExtractionService] Extraction queue at capacity (%d).", self.max_queue_size)
            return 429, {
                "error": "Extraction queue is at capacity. Please retry in a few moments.",
                "queue_size": self.queue.qsize(),
                "max_capacity": self.max_queue_size,
            }

        # 4. Upload file to Supabase Storage BEFORE enqueuing so recovery can reload it
        storage_path = None
        if is_supabase_configured():
            try:
                storage_path, _ = upload_resume_file_to_storage(
                    file_bytes=file_bytes,
                    filename=filename,
                    content_type=content_type,
                )
            except Exception as store_err:
                logger.warning("[ExtractionService] Pre-queue storage upload failed (non-fatal): %s", store_err)

        # 5. Generate unguessable random UUID jobId
        job_id = str(uuid.uuid4())
        job_record = {
            "id": job_id,
            "status": "pending",
            "file_hash": file_hash,
            "filename": filename,
            "content_type": content_type,
            "storage_path": storage_path,
            "extractor_version": self.current_extractor_version,
            "attempts": 0,
            "cached": False,
            "result": None,
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # 6. Enqueue job — if queue is now full, clean up the orphaned Storage object
        payload = (job_id, file_bytes, filename, content_type, storage_path)
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            # Queue filled up between our capacity check and here; delete orphaned storage object
            if storage_path and is_supabase_configured():
                try:
                    from app.storage.resume_storage import download_resume_file_from_storage
                    client = get_supabase_client()
                    if client:
                        import os as _os
                        bucket = _os.environ.get("SUPABASE_STORAGE_BUCKET", "resumes")
                        subpath = storage_path
                        if subpath.startswith(f"{bucket}/"):
                            subpath = subpath[len(bucket) + 1:]
                        client.storage.from_(bucket).remove([subpath])
                        logger.info("[ExtractionService] Cleaned up orphaned storage object %s after queue-full 429.", storage_path)
                except Exception as cleanup_err:
                    logger.warning("[ExtractionService] Failed to clean up orphaned storage on 429: %s", cleanup_err)
            return 429, {"error": "Extraction queue full."}

        # Only after queue accepts: register in memory and Supabase ExtractionJobs
        self.jobs[job_id] = job_record

        if is_supabase_configured():
            try:
                client = get_supabase_client()
                if client:
                    table = _get_table_ref(client, "ExtractionJobs")
                    table.insert({
                        "jobId": job_id,
                        "fileHash": file_hash,
                        "filename": filename,
                        "storageObjectPath": storage_path,
                        "extractorVersion": self.current_extractor_version,
                        "status": "pending",
                        "attempts": 0,
                    }).execute()
            except Exception as db_err:
                logger.warning("[ExtractionService] Failed to record pending job in ExtractionJobs table: %s", db_err)

        return 202, {
            "job_id": job_id,
            "status": "pending",
            "message": "Resume submitted for processing.",
        }

    async def _recover_interrupted_jobs(self) -> None:
        """
        Scans Supabase ExtractionJobs on boot for pending and interrupted processing jobs.
        Respects queue bound: feeds recovered jobs as queue slots become available.
        """
        logger.info("[ExtractionService] Starting startup job recovery scan...")
        await asyncio.sleep(1.0)  # Brief pause to allow main service to initialize

        try:
            client = get_supabase_client()
            if not client:
                return

            table = _get_table_ref(client, "ExtractionJobs")
            res = (
                table.select('"jobId", "fileHash", filename, "storageObjectPath", "extractorVersion", attempts, status')
                .in_("status", ["pending", "processing"])
                .order("createdAt")
                .execute()
            )

            if not res or not res.data:
                logger.info("[ExtractionService] No interrupted jobs found in Supabase.")
                return

            interrupted = res.data
            logger.info("[ExtractionService] Found %d interrupted/pending jobs to evaluate for recovery.", len(interrupted))

            for row in interrupted:
                job_id = str(row["jobId"])
                attempts = int(row.get("attempts") or 0)
                storage_path = row.get("storageObjectPath")
                filename = row.get("filename") or "recovered_resume.pdf"
                file_hash = row.get("fileHash")

                # If attempts >= 3, mark as failed permanently
                if attempts >= MAX_RECOVERY_ATTEMPTS:
                    logger.warning("[ExtractionService] Job %s exceeded max attempts (%d). Marking failed.", job_id, attempts)
                    table.update({
                        "status": "failed",
                        "error": f"Exceeded maximum recovery attempts ({MAX_RECOVERY_ATTEMPTS}).",
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                    }).eq("jobId", job_id).execute()
                    continue

                if not storage_path:
                    logger.warning("[ExtractionService] Job %s has no storageObjectPath. Cannot recover file bytes. Marking failed.", job_id)
                    table.update({
                        "status": "failed",
                        "error": "Cannot recover job: storageObjectPath is missing.",
                        "updatedAt": datetime.now(timezone.utc).isoformat(),
                    }).eq("jobId", job_id).execute()
                    continue

                # Wait for slot in bounded queue
                while self.running and self.queue.qsize() >= self.max_queue_size:
                    await asyncio.sleep(1.0)

                if not self.running:
                    break

                # Download file bytes from Storage
                try:
                    file_bytes = download_resume_file_from_storage(storage_path)
                except Exception as down_err:
                    logger.error("[ExtractionService] Failed to download storage file for job %s: %s", job_id, down_err)
                    continue

                job_record = {
                    "id": job_id,
                    "status": "pending",
                    "file_hash": file_hash,
                    "filename": filename,
                    "content_type": "application/pdf",
                    "storage_path": storage_path,
                    "attempts": attempts,
                    "cached": False,
                    "result": None,
                    "error": None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                self.jobs[job_id] = job_record

                # Reset to pending in DB
                table.update({
                    "status": "pending",
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                }).eq("jobId", job_id).execute()

                payload = (job_id, file_bytes, filename, "application/pdf", storage_path)
                await self.queue.put(payload)
                logger.info("[ExtractionService] Re-queued recovered job %s (attempt %d).", job_id, attempts + 1)

        except Exception as exc:
            logger.error("[ExtractionService] Startup recovery error: %s", exc, exc_info=True)

    def _infer_llm(self, raw_text: str) -> Optional[Tuple[Optional[Dict[str, Any]], str]]:
        """
        Synchronous model inference executed via asyncio.to_thread.
        Returns (parsed_dict, finish_reason).
        """
        if not self.llm:
            return None

        from app.services.prompt_templates import MAX_INPUT_CHARS_BUDGET
        prompt = build_chatml_prompt(raw_text, max_chars=MAX_INPUT_CHARS_BUDGET)
        max_output_tokens = int(os.getenv("LLAMA_MAX_OUTPUT_TOKENS", "2048"))
        try:
            output = self.llm(
                prompt,
                max_tokens=max_output_tokens,
                temperature=0.0,
                stop=["<|im_end|>", "\n\n\n"],
            )
            choice = output["choices"][0]
            finish_reason = choice.get("finish_reason", "stop")
            out_str = choice["text"].strip()

            if "```" in out_str:
                out_str = re.sub(r"^```(?:json)?\s*", "", out_str, flags=re.I)
                out_str = re.sub(r"\s*```.*$", "", out_str, flags=re.DOTALL).strip()

            parsed = None
            try:
                parsed = json.loads(out_str)
            except Exception:
                # If generation was cut off, attempt to close trailing brackets/braces
                try:
                    open_braces = out_str.count("{") - out_str.count("}")
                    open_brackets = out_str.count("[") - out_str.count("]")
                    fixed_str = out_str + ("]" * max(0, open_brackets)) + ("}" * max(0, open_braces))
                    parsed = json.loads(fixed_str)
                    logger.info("[ExtractionService] Salvaged truncated JSON output from LLM.")
                except Exception:
                    pass

            if not parsed:
                if finish_reason == "length":
                    logger.warning("[ExtractionService] LLM hit token length limit (finish_reason='length'). Rejecting output.")
                    return None, "length"
                logger.warning("[ExtractionService] LLM inference produced unparseable JSON.")
                return None, "invalid_json"

            return parsed, finish_reason
        except Exception as exc:
            logger.warning("[ExtractionService] LLM inference failed: %s", exc)
            return None

    async def _worker_loop(self) -> None:
        """Single background worker loop processing jobs sequentially with strict shadow prioritization."""
        logger.info("[ExtractionService] Worker loop running and waiting for jobs...")
        while self.running:
            try:
                job_item = None
                is_shadow_job = False

                # 1. Main Queue (High Priority)
                if not self.queue.empty():
                    job_item = await self.queue.get()
                    is_shadow_job = False
                elif not self.shadow_queue.empty() and self.queue.empty():
                    # Only consume shadow job when main queue is completely empty
                    if self.check_memory_guard():
                        job_item = await self.shadow_queue.get()
                        is_shadow_job = True
                    else:
                        # Drain shadow queue if memory guard tripped
                        while not self.shadow_queue.empty():
                            try:
                                self.shadow_queue.get_nowait()
                                self.shadow_queue.task_done()
                            except Exception:
                                break
                        await asyncio.sleep(0.1)
                        continue
                else:
                    # Both empty: wait up to 0.25s for main queue item
                    try:
                        job_item = await asyncio.wait_for(self.queue.get(), timeout=0.25)
                        is_shadow_job = False
                    except asyncio.TimeoutError:
                        if not self.shadow_queue.empty() and self.queue.empty() and self.check_memory_guard():
                            job_item = await self.shadow_queue.get()
                            is_shadow_job = True
                        else:
                            continue
                    except asyncio.CancelledError:
                        break

                if is_shadow_job:
                    file_hash, text_to_extract, primary_output, primary_source = job_item
                    try:
                        await self._process_shadow_job(file_hash, text_to_extract, primary_output, primary_source)
                    except Exception as s_exc:
                        logger.warning("[ExtractionService] [Shadow] Job failed: %s", s_exc)
                    finally:
                        self.shadow_queue.task_done()
                    continue

                job_id, file_bytes, filename, content_type, storage_path = job_item

                if job_id not in self.jobs:
                    logger.warning("[ExtractionService] Dequeued job_id %s not in jobs registry.", job_id)
                    self.queue.task_done()
                    continue

                job = self.jobs[job_id]
                current_attempts = job.get("attempts", 0) + 1
                job["attempts"] = current_attempts
                job["status"] = "processing"
                job["updated_at"] = datetime.now(timezone.utc).isoformat()

                # Persist processing status & attempts increment BEFORE processing starts
                if is_supabase_configured():
                    try:
                        client = get_supabase_client()
                        if client:
                            _get_table_ref(client, "ExtractionJobs").update({
                                "status": "processing",
                                "attempts": current_attempts,
                                "updatedAt": datetime.now(timezone.utc).isoformat(),
                            }).eq("jobId", job_id).execute()
                    except Exception as db_err:
                        logger.warning("[ExtractionService] Failed to update processing status in DB: %s", db_err)

                logger.info(
                    "[ExtractionService] [Queue Picked Up] Starting processing for job %s (file: '%s', attempt: %d). Remaining queue depth: %d",
                    job_id,
                    filename,
                    current_attempts,
                    self.queue.qsize(),
                )

                temp_file_path = None
                try:
                    # 1. Parse text from document bytes via temporary file
                    suffix = Path(filename).suffix if filename else ".pdf"
                    if not suffix:
                        suffix = ".pdf"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(file_bytes)
                        temp_file_path = tmp.name

                    logger.info("[ExtractionService] [Parsing] Parsing document text for job %s...", job_id)
                    success, raw_text, cleaned_text, file_hash, err = self.parser.parse_file(temp_file_path)
                    if not success or not (raw_text or cleaned_text):
                        logger.warning("[ExtractionService] [Failed] Job %s text extraction failed: %s", job_id, err)
                        job["status"] = "failed"
                        job["error"] = err or "Failed to extract text from document."
                        job["updated_at"] = datetime.now(timezone.utc).isoformat()
                        self._persist_job_terminal_state(job_id, "failed", error=job["error"])
                        continue

                    text_to_extract = raw_text or cleaned_text
                    logger.info(
                        "[ExtractionService] [Extracting] Extracted %d chars of text for job %s. Running structured extraction...",
                        len(text_to_extract),
                        job_id,
                    )

                    # 2. Canary Routing Check (Stage 5 Cutover)
                    use_slm = self.should_route_to_slm(file_hash)
                    raw_extracted = None
                    finish_reason = None
                    extraction_source = "legacy"

                    if use_slm:
                        logger.info(
                            "[ExtractionService] [Canary] Routing hash %s to fine-tuned SLM (cutover: %.1f%%).",
                            file_hash[:12],
                            self.current_cutover_percentage,
                        )
                        llm_result = await asyncio.to_thread(self._infer_llm, text_to_extract)
                        raw_extracted, finish_reason = (llm_result[0], llm_result[1]) if llm_result else (None, "failed")

                        if not raw_extracted and finish_reason == "length":
                            logger.warning("[ExtractionService] [Canary] LLM generation hit length limit and could not be parsed. Falling back to deterministic extractor.")

                        if raw_extracted:
                            processed = post_process_resume(raw_extracted)
                            extraction_source = "slm"
                        else:
                            logger.warning("[ExtractionService] [Canary] SLM inference unavailable/failed. Falling back to deterministic extractor.")
                            if self._fallback_extractor is None:
                                from app.services.structured_extractor import StructuredResumeExtractor
                                self._fallback_extractor = StructuredResumeExtractor()
                            fallback_data = self._fallback_extractor.get_safe_fallback_structure(text_to_extract)
                            processed = post_process_resume(fallback_data)
                            extraction_source = "fallback"
                    else:
                        logger.info(
                            "[ExtractionService] Routing hash %s to legacy extractor (cutover: %.1f%%).",
                            file_hash[:12],
                            self.current_cutover_percentage,
                        )
                        if self._fallback_extractor is None:
                            from app.services.structured_extractor import StructuredResumeExtractor
                            self._fallback_extractor = StructuredResumeExtractor()
                        legacy_data = self._fallback_extractor.get_safe_fallback_structure(text_to_extract)
                        processed = post_process_resume(legacy_data)
                        extraction_source = "legacy"

                    dumped_result = processed.model_dump(by_alias=True)
                    # Canonical field parity: provide both jobTitle and jobRole, durationYears, and resumeText
                    dumped_result["jobRole"] = dumped_result.get("jobTitle")
                    dumped_result["extractionSource"] = extraction_source
                    dumped_result["resumeText"] = text_to_extract
                    dumped_result.pop("storageObjectPath", None)  # Never leak storage path

                    # Trigger background shadow comparison if enabled and sampled
                    if self.is_shadow_mode_active and self.is_sample_selected_for_shadow(file_hash):
                        self.enqueue_shadow_job(
                            file_hash=file_hash,
                            text_to_extract=text_to_extract,
                            primary_output=dumped_result,
                            primary_source=extraction_source,
                        )

                    # 4. Upsert into Supabase CandidateResume
                    if is_supabase_configured():
                        try:
                            candidate_name = dumped_result.get("candidateName") or "Unknown"
                            candidate_role = dumped_result.get("jobTitle") or "Role not identified"
                            email = dumped_result.get("email")
                            skills = dumped_result.get("skills") or []
                            store_resume(
                                file_bytes=file_bytes,
                                filename=filename,
                                file_hash=file_hash,
                                candidate_name=candidate_name,
                                email=email,
                                resume_text=text_to_extract,
                                skills=skills,
                                job_role=candidate_role,
                                content_type=content_type,
                                extracted_data=dumped_result,
                                extractor_version=self.current_extractor_version,
                                extraction_source=extraction_source,
                                storage_path=storage_path,
                            )
                            logger.info("[ExtractionService] [Supabase] Upserted candidate resume to CandidateResume for job %s.", job_id)
                        except Exception as s_err:
                            logger.warning("[ExtractionService] Supabase persistence error for job %s: %s", job_id, s_err)

                    job["status"] = "completed"
                    job["result"] = dumped_result
                    job["extractionSource"] = extraction_source
                    job["updated_at"] = datetime.now(timezone.utc).isoformat()
                    self._persist_job_terminal_state(job_id, "completed", result=dumped_result, extraction_source=extraction_source)

                    logger.info(
                        "[ExtractionService] [Completed] Job %s completed via source='%s' (skills: %d, exp: %.1f yrs)",
                        job_id,
                        extraction_source,
                        len(dumped_result.get("skills") or []),
                        float(dumped_result.get("totalExperienceYears") or 0.0),
                    )

                    # Enqueue shadow evaluation (runs only when main queue is idle)
                    self.enqueue_shadow_job(
                        file_hash=file_hash,
                        text_to_extract=text_to_extract,
                        primary_output=dumped_result,
                        primary_source=extraction_source,
                    )

                except Exception as exc:
                    logger.error("[ExtractionService] [Failed] Job %s encountered error: %s", job_id, exc, exc_info=True)
                    job["status"] = "failed"
                    job["error"] = str(exc)
                    job["updated_at"] = datetime.now(timezone.utc).isoformat()
                    self._persist_job_terminal_state(job_id, "failed", error=str(exc))
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                        except OSError:
                            pass
                    self.queue.task_done()

            except Exception as outer_exc:
                logger.error("[ExtractionService] Unexpected error in worker loop: %s", outer_exc, exc_info=True)
                await asyncio.sleep(0.5)

        logger.info("[ExtractionService] Worker loop has terminated.")

    def enqueue_shadow_job(
        self,
        file_hash: str,
        text_to_extract: str,
        primary_output: Dict[str, Any],
        primary_source: str,
    ) -> bool:
        """
        Enqueues a shadow comparison job into the separate low-priority queue.
        Enforces:
          1. Shadow mode active and memory guard not tripped
          2. Main queue depth <= shadow_max_queue_depth
          3. Deterministic file_hash sampling rate
        Returns True if enqueued, False if skipped.
        """
        if not self.is_shadow_mode_active:
            return False
        if not self.check_memory_guard():
            return False
        if self.queue.qsize() > self.shadow_max_queue_depth:
            logger.info(
                "[ExtractionService] [Shadow] Main queue depth (%d) > threshold (%d). Skipping shadow job for %s",
                self.queue.qsize(),
                self.shadow_max_queue_depth,
                file_hash[:12],
            )
            return False
        if not self.is_sample_selected_for_shadow(file_hash):
            logger.info(
                "[ExtractionService] [Shadow] Skipped shadow job for %s due to sampling rate (%.0f%%)",
                file_hash[:12],
                self.shadow_sample_rate * 100,
            )
            return False

        try:
            payload = (file_hash, text_to_extract, primary_output, primary_source)
            self.shadow_queue.put_nowait(payload)
            logger.info(
                "[ExtractionService] [Shadow] Enqueued shadow job for %s (shadow queue depth: %d)",
                file_hash[:12],
                self.shadow_queue.qsize(),
            )
            return True
        except asyncio.QueueFull:
            logger.warning("[ExtractionService] [Shadow] Shadow queue is full. Dropping shadow job for %s", file_hash[:12])
            return False

    async def _process_shadow_job(
        self,
        file_hash: str,
        text_to_extract: str,
        primary_output: Dict[str, Any],
        primary_source: str,
    ) -> None:
        """
        Executes alternate model in shadow mode and evaluates diff/agreement.
        Writes result to extraction_shadow_logs without impacting client traffic.
        """
        from app.services.shadow_evaluator import (
            compute_field_agreements,
            compute_validity_flags,
            compute_grounding_flags,
            persist_shadow_log_to_supabase,
        )

        t0 = time.perf_counter()
        slm_output = None
        error_flags = []
        tokens_out = 0

        if primary_source in ("legacy", "fallback", "cache"):
            # Primary was legacy/fallback/cache: run SLM in shadow
            if self.llm:
                infer_res = await asyncio.to_thread(self._infer_llm, text_to_extract)
                if infer_res and infer_res[0]:
                    raw_data, finish_reason = infer_res
                    if finish_reason == "length":
                        error_flags.append("finish_reason_length")
                    processed = post_process_resume(raw_data)
                    slm_output = processed.model_dump(by_alias=True)
                    slm_output["jobRole"] = slm_output.get("jobTitle")
                    tokens_out = int(len(json.dumps(slm_output)) / 3.4)
                else:
                    error_flags.append("slm_inference_empty_or_failed")
            else:
                error_flags.append("model_not_loaded")

            legacy_output = primary_output
        else:
            # Primary was SLM: run legacy extractor in shadow
            slm_output = primary_output
            if self._fallback_extractor is None:
                from app.services.structured_extractor import StructuredResumeExtractor
                self._fallback_extractor = StructuredResumeExtractor()
            leg_data = self._fallback_extractor.extract(raw_text=text_to_extract)
            legacy_output = post_process_resume(leg_data).model_dump(by_alias=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if slm_output and legacy_output:
            agreements = compute_field_agreements(legacy_output, slm_output)
            validity = compute_validity_flags(slm_output)
            grounding = compute_grounding_flags(slm_output, text_to_extract)

            logger.info(
                "[ShadowEvaluator] Comparison for hash %s | Email match: %s | Phone match: %s | Title F1: %.2f | Skills Jaccard: %.2f | Latency: %.1fms",
                file_hash[:12],
                agreements.get("emailExactMatch"),
                agreements.get("phoneExactMatch"),
                agreements.get("titleTokenF1", 0.0),
                agreements.get("skillsJaccard", 0.0),
                elapsed_ms,
            )

            persist_shadow_log_to_supabase(
                file_hash=file_hash,
                legacy_version="ats-legacy:heuristic-v1",
                slm_version=self.current_extractor_version,
                legacy_output=legacy_output,
                slm_output=slm_output,
                field_agreements=agreements,
                validity_flags=validity,
                grounding_flags=grounding,
                slm_latency_ms=elapsed_ms,
                tokens_out=tokens_out,
                error_flags=error_flags,
                char_length=len(text_to_extract),
                rss_mb=self.get_current_rss_mb(),
            )

    def _persist_job_terminal_state(
        self,
        job_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        extraction_source: Optional[str] = None,
    ) -> None:
        """
        Writes terminal job status (completed/failed) to Supabase ExtractionJobs table.
        On permanent failure (status='failed' and attempts >= 3), cleans up the orphaned
        Supabase Storage object so we don't accumulate dead file objects.
        """
        if not is_supabase_configured():
            return
        try:
            client = get_supabase_client()
            if client:
                table = _get_table_ref(client, "ExtractionJobs")
                payload = {
                    "status": status,
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                }
                if result:
                    payload["result"] = result
                if error:
                    payload["error"] = error
                if extraction_source:
                    payload["extractionSource"] = extraction_source
                table.update(payload).eq("jobId", job_id).execute()

                # Clean up orphaned storage file on permanent failure
                if status == "failed":
                    job = self.jobs.get(job_id, {})
                    attempts = job.get("attempts", 0)
                    storage_path = job.get("storage_path")
                    if attempts >= 3 and storage_path:
                        try:
                            import os as _os
                            bucket = _os.environ.get("SUPABASE_STORAGE_BUCKET", "resumes")
                            subpath = storage_path
                            if subpath.startswith(f"{bucket}/"):
                                subpath = subpath[len(bucket) + 1:]
                            client.storage.from_(bucket).remove([subpath])
                            logger.info(
                                "[ExtractionService] Cleaned up orphaned storage %s for permanently failed job %s.",
                                storage_path, job_id,
                            )
                        except Exception as cleanup_err:
                            logger.warning(
                                "[ExtractionService] Failed to clean up storage for failed job %s: %s",
                                job_id, cleanup_err,
                            )
        except Exception as exc:
            logger.warning("[ExtractionService] Failed to persist terminal state for job %s: %s", job_id, exc)
