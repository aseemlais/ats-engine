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

    degree: str = "Degree"
    tier: DegreeTier = DegreeTier.NONE
    institution: Optional[str] = None
    field_of_study: Optional[str] = None
    year: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def coerce_from_string(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"degree": data}
        if isinstance(data, dict):
            d = dict(data)
            if "degree" not in d:
                d["degree"] = d.get("degree_name") or d.get("course") or d.get("institution") or "Degree"
            return d
        return data


class WorkExperienceItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    title: str = "Role not specified"
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
        if isinstance(data, dict):
            d = dict(data)
            if "title" not in d and ("role" in d or "position" in d):
                d["title"] = d.get("role") or d.get("position")
            elif "title" not in d:
                d["title"] = d.get("company") or "Role not specified"
            return d
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

    @model_validator(mode="before")
    @classmethod
    def normalize_containers(cls, data: Any) -> Any:
        """
        Defensive normalization: handles cases where the LLM returns dictionary
        maps instead of arrays for education, work_experience, or projects.
        """
        if not isinstance(data, dict):
            return data

        d = dict(data)

        # 1. Normalize work_experience (could be dict, list, or under 'experience')
        work_key = next((k for k in ("work_experience", "workExperience", "experience") if k in d), None)
        if work_key:
            val = d[work_key]
            if isinstance(val, dict):
                items = []
                for comp_or_title, details in val.items():
                    if isinstance(details, dict):
                        item = dict(details)
                        if "company" not in item and "title" in item:
                            item["company"] = str(comp_or_title)
                        elif "title" not in item:
                            item["title"] = item.get("role") or item.get("position") or str(comp_or_title)
                            item.setdefault("company", str(comp_or_title))
                        items.append(item)
                    elif isinstance(details, str):
                        items.append({"title": details, "company": str(comp_or_title)})
                d["work_experience"] = items
            elif isinstance(val, list):
                items = []
                for it in val:
                    if isinstance(it, dict):
                        item = dict(it)
                        if "title" not in item and ("role" in item or "position" in item):
                            item["title"] = item.get("role") or item.get("position")
                        elif "title" not in item:
                            item["title"] = item.get("company") or "Role not specified"
                        items.append(item)
                    elif isinstance(it, str):
                        items.append({"title": it})
                    else:
                        items.append(it)
                d["work_experience"] = items

        # 2. Normalize education (could be dict, list, or under 'academics')
        edu_key = next((k for k in ("education", "academics") if k in d), None)
        if edu_key:
            val = d[edu_key]
            if isinstance(val, dict):
                items = []
                for inst_or_deg, details in val.items():
                    if isinstance(details, dict):
                        item = dict(details)
                        if "institution" not in item and "degree" in item:
                            item["institution"] = str(inst_or_deg)
                        elif "degree" not in item:
                            item["degree"] = item.get("degree_name") or item.get("course") or str(inst_or_deg)
                            item.setdefault("institution", str(inst_or_deg))
                        items.append(item)
                    elif isinstance(details, str):
                        items.append({"degree": details, "institution": str(inst_or_deg)})
                d["education"] = items
            elif isinstance(val, list):
                items = []
                for it in val:
                    if isinstance(it, dict):
                        item = dict(it)
                        if "degree" not in item and "institution" in item:
                            item["degree"] = item.get("degree_name") or item.get("course") or item.get("institution")
                        elif "degree" not in item:
                            item["degree"] = "Degree"
                        items.append(item)
                    elif isinstance(it, str):
                        items.append({"degree": it})
                    else:
                        items.append(it)
                d["education"] = items

        # 3. Normalize projects (must be List[str])
        proj_key = next((k for k in ("projects", "project") if k in d), None)
        if proj_key:
            val = d[proj_key]
            if isinstance(val, dict):
                items = []
                for proj_name, details in val.items():
                    if isinstance(details, dict):
                        desc = details.get("description") or details.get("summary") or ""
                        items.append(f"{proj_name}: {desc}".strip(": "))
                    elif isinstance(details, str):
                        items.append(f"{proj_name}: {details}")
                    else:
                        items.append(str(proj_name))
                d["projects"] = items
            elif isinstance(val, list):
                items = []
                for it in val:
                    if isinstance(it, dict):
                        name = it.get("name") or it.get("title") or "Project"
                        desc = it.get("description") or it.get("summary") or ""
                        items.append(f"{name}: {desc}".strip(": "))
                    else:
                        items.append(str(it))
                d["projects"] = items

        # 4. Normalize skills & certifications (must be lists)
        for list_field in ("skills", "certifications"):
            val = d.get(list_field)
            if val is not None:
                if isinstance(val, dict):
                    d[list_field] = [str(k) for k in val.keys()]
                elif isinstance(val, str):
                    d[list_field] = [s.strip() for s in val.split(",") if s.strip()]

        return d


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
