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

@app.post("/api/files/upload")
async def upload_resume_file(file: UploadFile = File(...)):
    """
    Implements the exact /api/files/upload endpoint expected by NewChatPage.tsx.
    Extracts text from PDF, DOCX, and TXT using our layout-aware ATS ResumeParserService.
    """
    filename = file.filename or "uploaded_resume.pdf"
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

    content = await file.read()
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
        return {
            "success": True,
            "pathname": filename,
            "contentType": file.content_type,
            "resumeText": text_to_return
        }
    finally:
        if temp_path.exists():
            temp_path.unlink()

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
if REF_CLIENT_DIST.exists():
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
