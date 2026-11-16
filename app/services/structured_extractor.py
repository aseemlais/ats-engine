import os
import re
import json
import logging
from typing import Dict, Any, Optional, List
import ftfy

from app.services.nlp_extractor import NLPExtractorService

logger = logging.getLogger(__name__)

EMAIL_VALIDATION_REGEX = re.compile(
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
)

# Strict JSON Schema prompt for databricks-gpt-oss-120b
DATABRICKS_EXTRACTION_PROMPT = """You are an expert AI resume extraction system leveraging databricks-gpt-oss-120b.
Extract structured candidate details from the provided resume text into a strict JSON object.

RESUME TEXT:
{resume_text}

JSON SCHEMA:
{{
  "candidateName": "<string or null>",
  "email": "<valid email string or null>",
  "phone": "<string or null>",
  "jobTitle": "<string or null>",
  "skills": ["<skill1>", "<skill2>"],
  "experienceYears": <number or null>,
  "education": ["<education entry>"]
}}

STRICT EXTRACTION RULES:
1. EMAIL EXTRACTION (STRICT):
   - Extract only an active, syntactically valid email address (e.g., "candidate@domain.com").
   - You are STRICTLY FORBIDDEN from outputting placeholder em-dashes ('—', '–', '-'), blank strings (""), "Not available", "None", or corrupted strings for the email field.
   - If the email is missing, masked, corrupted, or ambiguous, you MUST return null.

2. CANDIDATE NAME & JOB TITLE (CLEANING):
   - Extract the candidate's actual full name and primary or current job title / role.
   - Clean up any residual encoding artifacts, mojibake, binary-to-text glitches, or empty quotes (such as '' or "").
   - If candidate name or job title is not found or cannot be verified from the resume, return null.

3. SKILLS & EXPERIENCE:
   - Extract demonstrated technical, domain, and soft skills as a list of strings.
   - Extract total years of experience as a number or null.

4. OUTPUT FORMAT:
   - Return ONLY the raw JSON object.
   - Do NOT wrap your output in markdown code blocks like ```json ... ``` or ``` ... ```.
   - Do NOT add any introductory, conversational, or concluding text.
"""


def clean_extracted_text(val: Optional[str]) -> Optional[str]:
    """Cleans up text removing any residual encoding artifacts, mojibake, and empty quotes."""
    if val is None:
        return None
    val = ftfy.fix_text(str(val)).strip()
    # Strip empty quotes, double quotes, single quotes
    val = re.sub(r"^['\"]+|['\"]+$", "", val).strip()
    # Remove residual zero-width or replacement character artifacts
    val = re.sub(r"[\u200b\u200e\u200f\ufeff\ufffd]", "", val).strip()
    if not val or val.lower() in ("none", "null", "not available", "unknown", "—", "–", "-", "n/a"):
        return None
    return val


def clean_extracted_email(email_val: Optional[str]) -> Optional[str]:
    """Specifically validates email extraction, strictly forbidding em-dashes or corrupted strings."""
    if email_val is None:
        return None
    cleaned = ftfy.fix_text(str(email_val)).strip()
    # Strictly forbid placeholder em-dashes ('—', '–', '-'), blank or placeholder strings
    if cleaned in ("—", "–", "-", "null", "none", "None", "Not available", "N/A", ""):
        return None
    # Strip any accidental wrapping punctuation/brackets
    cleaned = re.sub(r"^[\s<(\['\"]+|[\s>)\]'\"]+$", "", cleaned).strip()
    # Strict regex check
    if EMAIL_VALIDATION_REGEX.match(cleaned):
        return cleaned
    # Check if a valid email substring is present
    m = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", cleaned)
    if m:
        return m.group(0)
    return None


def parse_llm_json_response(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Robust response parsing:
    - Strips markdown code blocks (e.g., ```json ... ``` or ``` ... ```)
    - Strips leading/trailing text and decodes JSON
    - Returns None if decoding fails so safe fallback structure can take over
    """
    if not raw_text or not isinstance(raw_text, str):
        return None

    cleaned = raw_text.strip()

    # Strip markdown code blocks like ```json ... ``` or ``` ... ```
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```.*$", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()

    data = None
    try:
        data = json.loads(cleaned)
    except Exception:
        # Fallback: search for first valid JSON object {...}
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except Exception:
                data = None

    if not isinstance(data, dict):
        return None

    return {
        "candidateName": clean_extracted_text(data.get("candidateName")),
        "email": clean_extracted_email(data.get("email")),
        "phone": clean_extracted_text(data.get("phone")),
        "jobTitle": clean_extracted_text(data.get("jobTitle") or data.get("currentRole")),
        "skills": [clean_extracted_text(s) for s in data.get("skills", []) if clean_extracted_text(s)] if isinstance(data.get("skills"), list) else [],
        "experienceYears": data.get("experienceYears") if isinstance(data.get("experienceYears"), (int, float)) else None,
        "education": [clean_extracted_text(e) for e in data.get("education", []) if clean_extracted_text(e)] if isinstance(data.get("education"), list) else [],
    }


class StructuredResumeExtractor:
    """
    Structured Resume Extractor leveraging databricks-gpt-oss-120b via Model Serving
    with strict JSON schema enforcement, ftfy normalization, and deterministic safe fallback.
    """

    def __init__(self, nlp_extractor: Optional[NLPExtractorService] = None):
        self.nlp_extractor = nlp_extractor or NLPExtractorService()
        self.model_name = os.getenv("DATABRICKS_SERVING_ENDPOINT", "databricks-gpt-oss-120b")
        self.databricks_host = os.getenv("DATABRICKS_HOST", "").rstrip("/")
        self.databricks_token = os.getenv("DATABRICKS_TOKEN", "")

    def get_safe_fallback_structure(self, raw_text: str, cleaned_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Safe fallback structure if LLM extraction or JSON decoding encounters an issue.
        Employs deterministic NLP extraction with ftfy normalization and strict email/role rules.
        """
        raw_fixed = ftfy.fix_text(raw_text or cleaned_text or "")
        clean_fixed = ftfy.fix_text(cleaned_text or raw_text or "")

        parsed = self.nlp_extractor.extract_structured(
            candidate_id="extract_fallback",
            source_file="resume_upload",
            file_hash="hash_fallback",
            raw_text=raw_fixed,
            cleaned_text=clean_fixed
        )

        name = clean_extracted_text(parsed.name)
        email = clean_extracted_email(parsed.email)
        phone = clean_extracted_text(parsed.phone)
        role = clean_extracted_text(parsed.current_role or (parsed.job_titles[0] if parsed.job_titles else None))

        return {
            "candidateName": name or "Unknown",
            "email": email,  # Strictly None if missing or em-dash
            "phone": phone,
            "jobTitle": role or "Role not identified",
            "skills": parsed.skills or [],
            "experienceYears": parsed.experience_years if parsed.experience_years > 0 else None,
            "education": [e.degree for e in parsed.education if e.degree] if parsed.education else [],
            "source": "deterministic_fallback"
        }

    def call_databricks_model(self, resume_text: str) -> Optional[str]:
        """Calls databricks-gpt-oss-120b via Databricks Model Serving endpoint if credentials exist."""
        if not self.databricks_host or not self.databricks_token:
            return None

        import requests
        endpoint_url = f"{self.databricks_host}/serving-endpoints/{self.model_name}/invocations"
        headers = {
            "Authorization": f"Bearer {self.databricks_token}",
            "Content-Type": "application/json"
        }

        prompt = DATABRICKS_EXTRACTION_PROMPT.format(resume_text=resume_text[:12000])
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 800,
            "temperature": 0.0
        }

        try:
            resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=15)
            if resp.ok:
                resp_json = resp.json()
                if "choices" in resp_json and len(resp_json["choices"]) > 0:
                    return resp_json["choices"][0].get("message", {}).get("content", "")
                if "predictions" in resp_json:
                    preds = resp_json["predictions"]
                    return preds[0] if isinstance(preds, list) and preds else str(preds)
                return resp.text
            else:
                logger.warning("[StructuredExtractor] Databricks returned status %s: %s", resp.status_code, resp.text)
                return None
        except Exception as exc:
            logger.warning("[StructuredExtractor] Failed calling Databricks endpoint: %s", exc)
            return None

    def extract(self, raw_text: str, cleaned_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Extracts structured candidate data from resume text using databricks-gpt-oss-120b
        with a strict JSON schema prompt and robust response parsing, falling back safely.
        """
        raw_fixed = ftfy.fix_text(raw_text or "")
        clean_fixed = ftfy.fix_text(cleaned_text or raw_fixed)

        # 1. Attempt Databricks model call if configured
        llm_output = self.call_databricks_model(raw_fixed)
        if llm_output:
            parsed = parse_llm_json_response(llm_output)
            if parsed and (parsed.get("candidateName") or parsed.get("email") or parsed.get("skills")):
                return {
                    "candidateName": parsed.get("candidateName") or "Unknown",
                    "email": parsed.get("email"),  # strictly None if em-dash or missing
                    "phone": parsed.get("phone"),
                    "jobTitle": parsed.get("jobTitle") or "Role not identified",
                    "skills": parsed.get("skills") or [],
                    "experienceYears": parsed.get("experienceYears"),
                    "education": parsed.get("education") or [],
                    "source": "databricks-gpt-oss-120b"
                }

        # 2. Safe fallback structure
        return self.get_safe_fallback_structure(raw_fixed, clean_fixed)
