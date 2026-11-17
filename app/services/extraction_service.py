import asyncio
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import uuid
from typing import Any, Dict, Optional, Tuple

import ftfy

from app.models.resume_schema import ProcessedResume, RawResumeExtraction
from app.services.post_processor import post_process_resume
from app.services.prompt_templates import build_chatml_prompt
from app.services.resume_parser import ResumeParserService
from app.storage.resume_storage import _get_table_ref
from app.storage.supabase_client import get_supabase_client, is_supabase_configured

logger = logging.getLogger("extraction_service")

# Default queue capacity
DEFAULT_QUEUE_MAXSIZE = 50


class ExtractionService:
    """
    Asynchronous resume extraction service with:
    - Bounded asyncio.Queue (returns HTTP 429 when saturated)
    - Exactly ONE background worker task
    - Non-blocking CPU/GPU model inference offloaded to asyncio.to_thread
    - llama-cpp-python GGUF model integration with strict JSON grammar
    - In-memory job state and Supabase file-hash deduplication cache
    - Safe fallback to deterministic extraction
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
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.llm = None
        self.worker_task: Optional[asyncio.Task] = None
        self.running: bool = False
        self.parser = ResumeParserService()
        self._fallback_extractor = fallback_extractor

    @property
    def queue(self) -> asyncio.Queue:
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.max_queue_size)
        return self._queue

    def load_model(self) -> None:
        """Loads GGUF model once at startup if llama_cpp is installed and model exists."""
        if not self.model_path or not os.path.exists(self.model_path):
            logger.info("[ExtractionService] GGUF model not found at %s. Running in fallback mode.", self.model_path)
            return

        try:
            import llama_cpp
            logger.info("[ExtractionService] Loading GGUF model from %s...", self.model_path)
            self.llm = llama_cpp.Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_threads=int(os.getenv("LLAMA_THREADS", "4")),
                verbose=False,
            )
            logger.info("[ExtractionService] GGUF model loaded successfully.")
        except Exception as exc:
            logger.warning("[ExtractionService] Failed loading llama_cpp model: %s. Using fallback mode.", exc)
            self.llm = None

    async def start(self) -> None:
        """Starts the single background worker task and loads the model."""
        if self.running:
            return
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.running = True
        self.load_model()
        self.worker_task = asyncio.create_task(self._worker_loop(), name="ats_extraction_worker")
        logger.info("[ExtractionService] Async extraction worker started (queue capacity: %d).", self.max_queue_size)

    async def stop(self) -> None:
        """Gracefully shuts down the background worker."""
        self.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            self.worker_task = None
        self._queue = None
        logger.info("[ExtractionService] Extraction worker stopped.")

    def check_supabase_cache(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Queries Supabase CandidateResume table for an existing parsed resume
        matching this file_hash to avoid redundant re-extraction.
        """
        if not is_supabase_configured():
            return None

        try:
            client = get_supabase_client()
            if not client:
                return None

            table = _get_table_ref(client, "CandidateResume")
            res = (
                table.select("id, candidateName, email, jobRole, skills, resumeText, storageObjectPath")
                .eq("fileHash", file_hash)
                .limit(1)
                .execute()
            )
            if res and hasattr(res, "data") and res.data and len(res.data) > 0:
                record = res.data[0]
                return {
                    "candidateName": record.get("candidateName") or "Unknown",
                    "email": record.get("email"),
                    "jobRole": record.get("jobRole") or "Role not identified",
                    "skills": record.get("skills") or [],
                    "storageObjectPath": record.get("storageObjectPath"),
                    "source": "supabase_cache",
                }
        except Exception as exc:
            logger.warning("[ExtractionService] Cache lookup error: %s", exc)

        return None

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves current status and result of a job."""
        return self.jobs.get(job_id)

    async def submit_job(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str = "application/pdf",
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Enqueues an extraction job:
        - If already cached in Supabase: returns (200, cached_payload).
        - If queue is saturated: returns (429, error_payload).
        - Otherwise: creates job, enqueues it, and returns (202, job_metadata).
        """
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # 1. Check Supabase cache for duplicate/reused files
        cached_record = self.check_supabase_cache(file_hash)
        if cached_record:
            job_id = f"cache_{file_hash[:12]}"
            cached_data = dict(cached_record)
            cached_data["extractionSource"] = "cache"
            self.jobs[job_id] = {
                "id": job_id,
                "status": "completed",
                "file_hash": file_hash,
                "filename": filename,
                "cached": True,
                "extractionSource": "cache",
                "result": cached_data,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            logger.info("[ExtractionService] Job %s (%s) served from cache (source='cache').", job_id, filename)
            return 200, {
                "job_id": job_id,
                "status": "completed",
                "cached": True,
                "extractionSource": "cache",
                "result": cached_data,
            }

        # 2. Check Queue Capacity
        if self.queue.qsize() >= self.max_queue_size:
            logger.warning("[ExtractionService] Extraction queue at capacity (%d).", self.max_queue_size)
            return 429, {
                "error": "Extraction queue is at capacity. Please retry in a few moments.",
                "queue_size": self.queue.qsize(),
                "max_capacity": self.max_queue_size,
            }

        # 3. Create Pending Job & Enqueue
        job_id = str(uuid.uuid4())
        job_record = {
            "id": job_id,
            "status": "pending",
            "file_hash": file_hash,
            "filename": filename,
            "content_type": content_type,
            "cached": False,
            "result": None,
            "error": None,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        self.jobs[job_id] = job_record

        payload = (job_id, file_bytes, filename, content_type)
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            return 429, {"error": "Extraction queue full."}

        return 202, {
            "job_id": job_id,
            "status": "pending",
            "message": "Resume submitted for processing.",
        }

    def _infer_llm(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Synchronous model inference executed via asyncio.to_thread
        to keep the FastAPI async event loop unblocked.
        Uses canonical ChatML prompt identical to training and evaluation.
        """
        if not self.llm:
            return None

        prompt = build_chatml_prompt(raw_text, max_chars=12000)
        try:
            output = self.llm(
                prompt,
                max_tokens=1000,
                temperature=0.0,
                stop=["<|im_end|>", "\n\n\n"],
            )
            out_str = output["choices"][0]["text"].strip()
            # Strip markdown if present
            if "```" in out_str:
                out_str = re.sub(r"^```(?:json)?\s*", "", out_str, flags=re.I)
                out_str = re.sub(r"\s*```.*$", "", out_str, flags=re.DOTALL).strip()
            return json.loads(out_str)
        except Exception as exc:
            logger.warning("[ExtractionService] LLM inference failed: %s", exc)
            return None

    async def _worker_loop(self) -> None:
        """Single background worker loop processing jobs sequentially."""
        logger.info("[ExtractionService] Worker loop running...")
        while self.running:
            try:
                job_id, file_bytes, filename, content_type = await self.queue.get()
            except asyncio.CancelledError:
                break

            if job_id not in self.jobs:
                self.queue.task_done()
                continue

            job = self.jobs[job_id]
            job["status"] = "processing"
            job["updated_at"] = datetime.utcnow().isoformat()
            logger.info("[ExtractionService] Processing job %s (%s)...", job_id, filename)

            temp_file_path = None
            try:
                # 1. Parse text from document bytes via temporary file
                suffix = Path(filename).suffix if filename else ".pdf"
                if not suffix:
                    suffix = ".pdf"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(file_bytes)
                    temp_file_path = tmp.name

                success, raw_text, cleaned_text, file_hash, err = self.parser.parse_file(temp_file_path)
                if not success or not (raw_text or cleaned_text):
                    job["status"] = "failed"
                    job["error"] = err or "Failed to extract text from document."
                    job["updated_at"] = datetime.utcnow().isoformat()
                    continue

                text_to_extract = raw_text or cleaned_text

                # 2. Run model inference in thread (keeps event loop responsive)
                raw_extracted = await asyncio.to_thread(self._infer_llm, text_to_extract)

                # 3. Deterministic Post-Processing
                if raw_extracted:
                    processed = post_process_resume(raw_extracted)
                    extraction_source = "slm"
                else:
                    # Safe deterministic fallback if model is not loaded or failed
                    if self._fallback_extractor is None:
                        from app.services.structured_extractor import StructuredResumeExtractor
                        self._fallback_extractor = StructuredResumeExtractor()
                    fallback_data = self._fallback_extractor.get_safe_fallback_structure(text_to_extract)
                    processed = post_process_resume(fallback_data)
                    extraction_source = "fallback"

                dumped_result = processed.model_dump(by_alias=True)
                dumped_result["extractionSource"] = extraction_source

                job["status"] = "completed"
                job["result"] = dumped_result
                job["extractionSource"] = extraction_source
                job["updated_at"] = datetime.utcnow().isoformat()
                logger.info(
                    "[ExtractionService] Job %s (%s) completed successfully via source='%s'. Candidate: '%s', Role: '%s'",
                    job_id,
                    filename,
                    extraction_source,
                    dumped_result.get("candidateName"),
                    dumped_result.get("jobTitle"),
                )

            except Exception as exc:
                logger.error("[ExtractionService] Job %s encountered error: %s", job_id, exc, exc_info=True)
                job["status"] = "failed"
                job["error"] = str(exc)
                job["updated_at"] = datetime.utcnow().isoformat()
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        pass
                self.queue.task_done()
