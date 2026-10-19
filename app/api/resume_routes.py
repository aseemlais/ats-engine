import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Query

from app.config import config
from app.database.storage import MetadataStorage
from app.workers.ingestion_pipeline import IngestionPipeline
from app.models.resume import ParsedResume

router = APIRouter(prefix="/resumes", tags=["Resumes"])
storage = MetadataStorage()
pipeline = IngestionPipeline()

@router.get("", response_model=List[ParsedResume])
def list_resumes(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    """Lists indexed candidate resumes with pagination."""
    all_resumes = storage.get_all_resumes()
    return all_resumes[offset:offset + limit]

@router.get("/count")
def get_resume_count():
    """Returns total count of indexed resumes in storage and vector DB."""
    return {
        "storage_count": storage.count_resumes(),
        "vector_count": pipeline.qdrant_db.count_resumes()
    }

@router.get("/{candidate_id}", response_model=ParsedResume)
def get_resume_by_id(candidate_id: str):
    """Retrieves detailed structured information for a candidate."""
    resume = storage.get_resume(candidate_id)
    if not resume:
        raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found")
    return resume

@router.post("/upload")
def upload_single_resume(file: UploadFile = File(...)):
    """Uploads and ingests a single resume PDF/DOCX file."""
    if not file.filename.lower().endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported.")

    dest = config.resumes_dir / file.filename
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cid = f"CV_{storage.count_resumes() + 1:05d}"
    ok, parsed, err = pipeline.ingest_single_file(dest, cid)
    if not ok:
        raise HTTPException(status_code=422, detail=f"Failed to parse resume: {err}")

    storage.save_resume(parsed)
    return {"status": "success", "candidate_id": parsed.candidate_id, "name": parsed.name, "skills": parsed.skills}

@router.post("/bulk-upload")
def bulk_upload_resumes(files: List[UploadFile] = File(...)):
    """Uploads multiple resume files simultaneously."""
    uploaded_files = []
    for file in files:
        if file.filename.lower().endswith((".pdf", ".docx", ".txt")):
            dest = config.resumes_dir / file.filename
            with open(dest, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append(file.filename)

    # Trigger directory ingestion
    result = pipeline.ingest_directory(config.resumes_dir)
    return {"status": "success", "uploaded_count": len(uploaded_files), "ingestion_result": result}
