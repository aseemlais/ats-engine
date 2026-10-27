import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
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
from app.storage.resume_storage import store_resume, ResumePersistenceError

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
