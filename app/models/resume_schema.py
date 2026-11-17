from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


class DegreeTier(str, Enum):
    DOCTORATE = "doctorate"
    MASTER = "master"
    BACHELOR = "bachelor"
    ASSOCIATE = "associate"
    DIPLOMA = "diploma"
    NONE = "none"


class EducationItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    degree: str
    tier: DegreeTier = DegreeTier.NONE
    institution: Optional[str] = None
    field_of_study: Optional[str] = None
    year: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def coerce_from_string(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"degree": data}
        return data


class WorkExperienceItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    title: str
    company: Optional[str] = None
    dates: Optional[str] = None
    duration_years: float = 0.0
    is_current: bool = False
    is_internship: bool = False
    description: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def coerce_from_string(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"title": data}
        return data


class RawResumeExtraction(BaseModel):
    """
    Schema for raw semantic fields extracted directly by the LLM.
    Accepts raw un-normalized values (e.g., messy date strings, un-truncated
    parenthesized titles, un-validated emails/phones).
    Supports both camelCase and snake_case keys.
    """
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    candidate_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    job_title: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    work_experience: List[WorkExperienceItem] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)


class ProcessedResume(BaseModel):
    """
    Final standardized and validated resume schema produced by deterministic
    post-processing. All math, intervals, degree tiers, and validations are resolved.
    """
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    candidate_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    job_title: Optional[str] = None
    total_experience_years: float = 0.0
    work_experience_years: float = 0.0
    internship_years: float = 0.0
    skills: List[str] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    degrees: List[str] = Field(default_factory=list)
    work_experience: List[WorkExperienceItem] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
