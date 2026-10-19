from typing import List, Optional
from pydantic import BaseModel, Field

class JobDescription(BaseModel):
    job_id: str = "JD_001"
    title: str = "Software Engineer"
    raw_text: str = ""
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    minimum_experience_years: float = 0.0
    education_requirements: List[str] = Field(default_factory=list)
    mandatory_skills: List[str] = Field(default_factory=list)
    # Tiers: required (= mandatory) / preferred / supporting (trainable, concept-level or responsibilities-derived)
    supporting_skills: List[str] = Field(default_factory=list)
    nonmandatory_skills: List[str] = Field(default_factory=list)   # explicitly stated as NOT mandatory
    maximum_experience_years: Optional[float] = None
    # None = derive from minimum_experience_years (> 0 means applicable). False = JD says freshers/no experience needed.
    experience_applicable: Optional[bool] = None
    # Groups of interchangeable required skills, e.g. "(Airflow, Redshift, or similar)" -> [["airflow","redshift"]]
    skill_alternatives: List[List[str]] = Field(default_factory=list)
    responsibilities: str = ""
    location: Optional[str] = None

    @property
    def has_experience_requirement(self) -> bool:
        """True only when the JD establishes a minimum experience requirement (never by default)."""
        if self.experience_applicable is not None:
            return bool(self.experience_applicable)
        return self.minimum_experience_years > 0
