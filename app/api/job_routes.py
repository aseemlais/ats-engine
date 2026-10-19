from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.models.job import JobDescription
from app.services.jd_parser import JDParserService
from app.database.storage import MetadataStorage

router = APIRouter(prefix="/jobs", tags=["Jobs"])
jd_parser = JDParserService()
storage = MetadataStorage()

class JobCreateRequest(BaseModel):
    job_id: Optional[str] = None
    title: Optional[str] = None
    job_description: str

@router.post("", response_model=JobDescription)
def create_job(request: JobCreateRequest):
    """Parses raw Job Description text into structured requirements and persists it."""
    jid = request.job_id or f"JOB_{int(storage.get_connection().total_changes if hasattr(storage, 'get_connection') else 1):03d}"
    parsed_jd = jd_parser.parse_job_description(request.job_description, job_id=jid)
    if request.title:
        parsed_jd.title = request.title
    storage.save_job(parsed_jd)
    return parsed_jd

@router.get("/{job_id}", response_model=JobDescription)
def get_job(job_id: str):
    """Retrieves parsed Job Description by ID."""
    job = storage.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job
