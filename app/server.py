import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Query
from app.services.embedding_service import EmbeddingUnavailableError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Ensure ATS module importable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.adapters.screening_adapter import ATSScreeningAdapter
from app.services.resume_parser import ResumeParserService
from app.storage.supabase_client import is_supabase_configured, get_missing_supabase_env_vars
from app.storage.resume_storage import store_resume, ResumePersistenceError, StorageError
from app.storage.candidate_repository import (
    list_candidates,
    search_candidates,
    get_candidate,
    get_resume_download_url,
    delete_candidate,
    CandidateNotFoundError,
    RepositoryUnavailableError,
    PartialDeletionError,
    DatabaseError,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
)

logger = logging.getLogger("ats_server")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Hybrid Semantic ATS - Reference UI Server", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reference client build path
REF_CLIENT_DIST = Path(r"C:\Users\LENOVO\Downloads\Resume Screening and Shortlisting agent\Resume Screening and Shortlisting agent\e2e-chatbot-app-next\client\dist")

adapter = ATSScreeningAdapter()
parser = ResumeParserService()

@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": "Hybrid Semantic ATS Engine",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "service": "Hybrid Semantic ATS"}

@app.get("/api/session")
async def get_session(request: Request):
    """
    Session endpoint required by SessionContext.tsx.
    Detects Databricks auth headers when running on Databricks Apps,
    or returns an authenticated local reviewer session for browser access.
    """
    user_email = (
        request.headers.get("X-Forwarded-Email")
        or request.headers.get("X-Forwarded-User")
        or request.headers.get("X-Forwarded-Preferred-Username")
        or "reviewer@ats-workspace.local"
    )
    user_name = (
        request.headers.get("X-Forwarded-Preferred-Username")
        or request.headers.get("X-Forwarded-User")
        or "ATS Reviewer"
    )
    return {
        "user": {
            "email": user_email,
            "name": user_name,
            "preferredUsername": user_name,
        }
    }

@app.get("/api/config")
async def get_config():
    """
    Configuration endpoint required by AppConfigContext.tsx.
    """
    return {
        "features": {
            "chatHistory": False,
            "feedback": False,
        },
        "greeting": "Hybrid Semantic ATS Candidate Screening",
    }

def process_resume_upload(
    filename: str,
    content: bytes,
    content_type: str = "application/pdf"
) -> Dict[str, Any] | JSONResponse:
    """
    Core resume ingestion logic:
    1. Parses resume text from PDF, DOCX, or TXT.
    2. If Supabase is configured, extracts structured candidate details and persists
       both original file and metadata with SHA-256 duplicate detection.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in (".pdf", ".docx", ".txt"):
        return JSONResponse(
            status_code=400,
            content={"error": "Please upload a PDF, DOCX, or TXT resume."}
        )

    # Save to temp location
    temp_dir = ROOT_DIR / "data" / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / filename

    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        ok, raw_text, cleaned_text, file_hash, err = parser.parse_file(temp_path)
        if not ok or not (cleaned_text or raw_text):
            return JSONResponse(
                status_code=400,
                content={"error": err or "The resume was uploaded, but no readable text could be extracted."}
            )

        text_to_return = cleaned_text or raw_text
        response_payload = {
            "success": True,
            "pathname": filename,
            "contentType": content_type,
            "resumeText": text_to_return
        }

        # Persistent storage integration
        if is_supabase_configured():
            try:
                parsed_cand = adapter.nlp_extractor.extract_structured(
                    candidate_id="temp_upload",
                    source_file=filename,
                    file_hash=file_hash,
                    raw_text=raw_text or cleaned_text,
                    cleaned_text=cleaned_text or raw_text
                )

                candidate_name = parsed_cand.name or "Not available"
                candidate_role = parsed_cand.current_role or (
                    parsed_cand.job_titles[0] if parsed_cand.job_titles else "Role not identified"
                )

                storage_res = store_resume(
                    file_bytes=content,
                    filename=filename,
                    file_hash=file_hash,
                    candidate_name=candidate_name,
                    email=parsed_cand.email,
                    resume_text=text_to_return,
                    skills=parsed_cand.skills,
                    job_role=candidate_role,
                    content_type=content_type or "application/pdf"
                )

                response_payload.update({
                    "persistenceStatus": storage_res.get("persistenceStatus", "persisted"),
                    "candidateId": storage_res.get("candidateId"),
                    "isDuplicate": storage_res.get("isDuplicate", False),
                    "storageObjectPath": storage_res.get("storageObjectPath"),
                    "candidateName": storage_res.get("candidateName"),
                    "jobRole": storage_res.get("jobRole"),
                })
            except Exception as store_err:
                logger.error(
                    f"[Storage] Failed to persist resume '{filename}' (hash: {file_hash[:12]}...): {store_err}",
                    exc_info=True,
                )
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "error": f"Failed to persist resume in Supabase: {str(store_err)}",
                        "persistenceStatus": "failed",
                        "pathname": filename,
                    }
                )
        else:
            missing_vars = get_missing_supabase_env_vars()
            logger.warning(
                "[Storage] Supabase persistent storage is not configured (missing: %s). Operating in temporary in-memory mode.",
                ", ".join(missing_vars) if missing_vars else "unknown",
            )
            response_payload.update({
                "persistenceStatus": "temporary_in_memory",
                "candidateId": None,
                "isDuplicate": False,
                "storageObjectPath": None,
                "persistenceNotice": f"Resume processed in temporary in-memory mode. Missing configuration: {', '.join(missing_vars)}.",
            })

        return response_payload
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/api/files/upload")
async def upload_resume_file(file: UploadFile = File(...)):
    """
    Implements the exact /api/files/upload endpoint expected by NewChatPage.tsx.
    Extracts text from PDF, DOCX, and TXT using our layout-aware ATS ResumeParserService.
    If Supabase is configured, persists the resume file in Supabase Storage and candidate
    metadata in Supabase Postgres with SHA-256 duplicate detection.
    """
    filename = file.filename or "uploaded_resume.pdf"
    content = await file.read()
    return process_resume_upload(filename, content, file.content_type or "application/pdf")


# =============================================================================
# Batch 2: Candidate Repository API
# =============================================================================

def _supabase_required():
    """Return 503 JSON if Supabase is not configured."""
    if not is_supabase_configured():
        missing = get_missing_supabase_env_vars()
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Candidate repository is unavailable: Supabase not configured.",
                "missingConfig": missing,
            },
        )


def _require_delete_auth(request: Request) -> None:
    """
    Guard for destructive (DELETE) endpoints.

    Checks the `Authorization: Bearer <token>` header against the
    `ATS_ADMIN_TOKEN` environment variable on the backend.

    Security model:
      - ATS_ADMIN_TOKEN must be set as a secret env var on the HF Space backend.
      - The Render Node.js proxy should forward the token in an
        `Authorization: Bearer` header when calling DELETE /api/candidates/{id}.
      - If ATS_ADMIN_TOKEN is not set, the endpoint is disabled (503) to prevent
        unauthenticated destructive access in any deployment.
      - CORS and the Supabase service-role key are NOT used as user auth.
      - This token protects the Python backend's direct URL; the Node.js
        frontend's own authMiddleware provides a separate layer on the proxy side.

    Frontend configuration required (Render env or Node proxy):
      Forward the header:  Authorization: Bearer <ATS_ADMIN_TOKEN value>
      Env var name on HF Space: ATS_ADMIN_TOKEN
    """
    configured_token = os.environ.get("ATS_ADMIN_TOKEN", "").strip()
    if not configured_token:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Destructive operations are disabled: ATS_ADMIN_TOKEN is not configured on this deployment.",
                "action": "Set ATS_ADMIN_TOKEN as a secret environment variable on the backend to enable candidate deletion.",
            },
        )

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": "Authorization header with Bearer token is required for this operation."},
        )

    provided_token = auth_header[len("Bearer "):]
    # Constant-time comparison to prevent timing attacks
    import hmac
    if not hmac.compare_digest(provided_token, configured_token):
        raise HTTPException(
            status_code=403,
            detail={"error": "Invalid admin token."},
        )


@app.get("/api/candidates")
async def list_candidates_endpoint(
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
):
    """
    List stored candidates with pagination, sorted by newest first.
    Does not return full resume text or internal credentials.
    """
    _supabase_required()
    try:
        result = list_candidates(limit=limit, offset=offset)
        return result
    except RepositoryUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except DatabaseError as exc:
        logger.error("[API] list_candidates error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve candidates.")


@app.get("/api/candidates/search")
async def search_candidates_endpoint(
    q: str = Query(default="", description="Search query (name, email, skills, job role)"),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
):
    """
    Case-insensitive search across candidateName, email, jobRole, and skills.
    Returns paginated results. Empty query returns all candidates (same as list).
    """
    _supabase_required()
    try:
        result = search_candidates(query=q, limit=limit, offset=offset)
        return result
    except RepositoryUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except DatabaseError as exc:
        logger.error("[API] search_candidates error (q=%r): %s", q, exc)
        raise HTTPException(status_code=500, detail="Search failed.")


@app.get("/api/candidates/{candidate_id}")
async def get_candidate_endpoint(
    candidate_id: str,
    include_resume_text: bool = Query(default=False),
):
    """
    Retrieve a single candidate's metadata by UUID.
    Set include_resume_text=true to also return the full extracted resume text.
    """
    _supabase_required()
    try:
        candidate = get_candidate(candidate_id, include_resume_text=include_resume_text)
        return candidate
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except CandidateNotFoundError:
        raise HTTPException(status_code=404, detail=f"Candidate '{candidate_id}' not found.")
    except RepositoryUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except DatabaseError as exc:
        logger.error("[API] get_candidate(%s) error: %s", candidate_id, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve candidate.")


@app.get("/api/candidates/{candidate_id}/download")
async def get_candidate_download_url_endpoint(candidate_id: str):
    """
    Generate a short-lived signed URL (5 min TTL) for the candidate's resume file.
    The service-role key is never exposed; permanent public links are not issued.
    """
    _supabase_required()
    try:
        url = get_resume_download_url(candidate_id)
        return {"signedUrl": url, "expiresInSeconds": 300}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except CandidateNotFoundError:
        raise HTTPException(status_code=404, detail=f"Candidate '{candidate_id}' not found.")
    except RepositoryUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("[API] get_candidate_download_url(%s) error: %s", candidate_id, exc)
        raise HTTPException(status_code=500, detail="Failed to generate download URL.")


@app.delete("/api/candidates/{candidate_id}")
async def delete_candidate_endpoint(candidate_id: str, request: Request):
    """
    Delete a candidate's storage file and metadata record.

    Authorization: requires `Authorization: Bearer <ATS_ADMIN_TOKEN>` header.
    ATS_ADMIN_TOKEN must be set as a secret env var on the HF Space backend.

    Deletion order:
      1. Storage object is deleted first.
         - If Storage deletion fails, returns 409 with the error; the DB row is
           untouched and the operation is safe to retry.
         - If the Storage object is already absent, deletion proceeds normally.
      2. DB row is deleted after Storage succeeds.
         - If DB deletion fails after Storage was removed, returns 207 (partial
           failure) with enough detail to identify and clean up the orphaned row.
    """
    _require_delete_auth(request)
    _supabase_required()
    try:
        result = delete_candidate(candidate_id)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except CandidateNotFoundError:
        raise HTTPException(status_code=404, detail=f"Candidate '{candidate_id}' not found.")
    except RepositoryUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except PartialDeletionError as exc:
        # Storage was deleted but DB row could not be removed.
        # Return 207 Multi-Status so the caller knows the state precisely.
        logger.error("[API] Partial deletion for candidate %s: %s", candidate_id, exc)
        return JSONResponse(
            status_code=207,
            content={
                "deleted": False,
                "partialFailure": True,
                "candidateId": candidate_id,
                "storageDeleted": True,
                "storageAlreadyAbsent": False,
                "dbRowDeleted": False,
                "storageObjectPath": exc.storage_object_path,
                "error": (
                    "Storage file was deleted but the database row could not be removed. "
                    "Retry the DELETE request; the storage step will be skipped automatically."
                ),
            },
        )
    except StorageError as exc:
        # Storage deletion failed — DB row is intact, operation is safe to retry.
        logger.error("[API] Storage deletion failed for candidate %s: %s", candidate_id, exc)
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Could not delete the resume file from storage. The candidate record is intact and the operation is safe to retry.",
                "detail": str(exc),
            },
        )
    except DatabaseError as exc:
        logger.error("[API] delete_candidate(%s) DB error: %s", candidate_id, exc)
        raise HTTPException(status_code=500, detail="Failed to delete candidate record.")


@app.post("/invocations")
async def invocations_endpoint(request: Request):
    """
    Implements MLflow's ResponsesAgent / OpenAI Responses API /invocations endpoint.
    Consumes requests from NewChatPage.tsx (callAgent) and executes the deterministic ATS.
    """
    body = await request.json()
    messages = body.get("input", [])

    if not messages:
        return JSONResponse(
            status_code=400,
            content={"error": "Empty input messages in request."}
        )

    # Get last user prompt
    user_prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                user_prompt = content
                break
            elif isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                user_prompt = " ".join(parts).strip()
                break

    if not user_prompt:
        user_prompt = str(messages[-1])

    # Check if this is an interview question generation request
    if "interview questions" in user_prompt.lower():
        parsed_input = adapter.parse_screening_prompt(user_prompt)
        result_dict = adapter.generate_interview_questions(
            job_title=parsed_input["job_title"],
            job_description=parsed_input["job_description"],
            resume_text=parsed_input["resume_text"]
        )
        final_text = json.dumps(result_dict)
    else:
        # Standard candidate screening request
        parsed_input = adapter.parse_screening_prompt(user_prompt)
        try:
            result_dict = adapter.evaluate_candidate(
                job_title=parsed_input["job_title"],
                job_description=parsed_input["job_description"],
                resume_text=parsed_input["resume_text"]
            )
        except EmbeddingUnavailableError as e:
            # explicit failure instead of a fabricated semantic score (set ALLOW_DEGRADED_SEMANTIC=true to opt in)
            raise HTTPException(status_code=503, detail=str(e))
        final_text = json.dumps(result_dict)

    # Return standard MLflow ResponsesAgent response format
    return {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": final_text
                    }
                ]
            }
        ],
        "output_text": final_text,
        "text": final_text
    }

# Mount static files from reference client build if available
try:
    client_dist_exists = REF_CLIENT_DIST.exists()
except Exception:
    client_dist_exists = False

if client_dist_exists:
    app.mount("/assets", StaticFiles(directory=str(REF_CLIENT_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        target = REF_CLIENT_DIST / full_path
        no_cache_headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        if target.is_file():
            return FileResponse(target, headers=no_cache_headers)
        index_file = REF_CLIENT_DIST / "index.html"
        if index_file.exists():
            return FileResponse(index_file, headers=no_cache_headers)
        raise HTTPException(status_code=404, detail="File not found")

def main():
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app.server:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
