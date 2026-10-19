from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class EducationEntry(BaseModel):
    degree: str
    tier: str = "none"
    institution: Optional[str] = None
    field_of_study: Optional[str] = None
    year: Optional[str] = None

class ExperienceEntry(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    duration_years: float = 0.0
    dates: Optional[str] = None
    description: Optional[str] = None

class ParsedResume(BaseModel):
    candidate_id: str
    source_file: str
    file_hash: str
    text_hash: str
    name: Optional[str] = "Unknown"
    email: Optional[str] = None
    phone: Optional[str] = None
    raw_text: str
    cleaned_text: str
    skills: List[str] = Field(default_factory=list)
    experience_years: float = 0.0
    work_experience_years: float = 0.0
    internship_years: float = 0.0
    job_titles: List[str] = Field(default_factory=list)
    education: List[EducationEntry] = Field(default_factory=list)
    degrees: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    section_texts: Dict[str, str] = Field(default_factory=lambda: {
        "skills": "",
        "experience": "",
        "projects": "",
        "education": ""
    })
