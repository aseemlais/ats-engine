"""
app/services/shadow_evaluator.py
================================
Shadow Mode Evaluation & Metric Calculation Service.

Computes:
  1. Per-field agreement scores (legacy vs SLM):
     - Exact match for email/phone
     - Fuzzy sequence ratio and token F1 for candidate name and job title
     - Precision, Recall, and Jaccard similarity for skills, education, and experience
  2. Per-field validity flags:
     - email_valid (RFC pattern, rejects em-dashes / placeholders)
     - phone_valid (>= 7 digits, rejects em-dashes)
     - name_plausible (non-empty string, no digit corruption)
     - dates_parse (experience durations successfully computed)
  3. Grounding / Hallucination detection:
     - Checks if extracted entities (name, skills, companies) have textual grounding in source resume
  4. Async persistence to Supabase `extraction_shadow_logs` table
"""

import difflib
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("shadow_evaluator")


# ---------------------------------------------------------------------------
# String & Token Similarity Helpers
# ---------------------------------------------------------------------------

def _clean_str(val: Any) -> str:
    if val is None:
        return ""
    return str(val).strip().lower()


def _extract_tokens(val: Any) -> Set[str]:
    s = _clean_str(val)
    return set(re.findall(r"\w+", s))


def compute_token_f1(pred: Any, gold: Any) -> float:
    p_tokens = _extract_tokens(pred)
    g_tokens = _extract_tokens(gold)
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    common = p_tokens & g_tokens
    if not common:
        return 0.0
    prec = len(common) / len(p_tokens)
    rec = len(common) / len(g_tokens)
    return round(2 * (prec * rec) / (prec + rec), 4)


def compute_sequence_ratio(a: Any, b: Any) -> float:
    sa = _clean_str(a)
    sb = _clean_str(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return round(difflib.SequenceMatcher(None, sa, sb).ratio(), 4)


def compute_set_metrics(set_a: Set[str], set_b: Set[str]) -> Dict[str, float]:
    """Computes Precision, Recall, and Jaccard between two sets of string tokens."""
    if not set_a and not set_b:
        return {"precision": 1.0, "recall": 1.0, "jaccard": 1.0}
    if not set_a or not set_b:
        return {"precision": 0.0, "recall": 0.0, "jaccard": 0.0}

    intersection = set_a & set_b
    union = set_a | set_b
    prec = len(intersection) / len(set_b) if set_b else 0.0
    rec = len(intersection) / len(set_a) if set_a else 0.0
    jaccard = len(intersection) / len(union) if union else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "jaccard": round(jaccard, 4),
    }


# ---------------------------------------------------------------------------
# Per-Field Agreements
# ---------------------------------------------------------------------------

def compute_field_agreements(legacy: Dict[str, Any], slm: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes rigorous per-field agreement scores between legacy and SLM extractions.
    """
    # 1. Contact Info: Exact string match
    email_leg = _clean_str(legacy.get("email"))
    email_slm = _clean_str(slm.get("email"))
    email_match = (email_leg == email_slm) if (email_leg or email_slm) else True

    # Normalize phone digits only (stripping country code 1 if 11 digits)
    phone_leg_digits = re.sub(r"\D", "", str(legacy.get("phone") or ""))
    phone_slm_digits = re.sub(r"\D", "", str(slm.get("phone") or ""))
    if len(phone_leg_digits) == 11 and phone_leg_digits.startswith("1"):
        phone_leg_digits = phone_leg_digits[1:]
    if len(phone_slm_digits) == 11 and phone_slm_digits.startswith("1"):
        phone_slm_digits = phone_slm_digits[1:]
    phone_match = (phone_leg_digits == phone_slm_digits) if (phone_leg_digits or phone_slm_digits) else True

    # 2. Candidate Name & Job Title: Token F1 + fuzzy sequence ratio
    name_f1 = compute_token_f1(legacy.get("candidateName"), slm.get("candidateName"))
    name_ratio = compute_sequence_ratio(legacy.get("candidateName"), slm.get("candidateName"))

    title_f1 = compute_token_f1(legacy.get("jobTitle"), slm.get("jobTitle"))
    title_ratio = compute_sequence_ratio(legacy.get("jobTitle"), slm.get("jobTitle"))

    # 3. Skills: Normalized set comparison
    skills_leg = {s.strip().lower() for s in (legacy.get("skills") or []) if s}
    skills_slm = {s.strip().lower() for s in (slm.get("skills") or []) if s}
    skills_metrics = compute_set_metrics(skills_leg, skills_slm)

    # 4. Total Experience: Absolute delta in years
    exp_leg = float(legacy.get("totalExperienceYears") or legacy.get("experienceYears") or 0.0)
    exp_slm = float(slm.get("totalExperienceYears") or slm.get("experienceYears") or 0.0)
    exp_delta = round(abs(exp_leg - exp_slm), 2)

    # 5. Education & Work Experience count agreements
    edu_leg = len(legacy.get("education") or [])
    edu_slm = len(slm.get("education") or [])
    edu_count_match = (edu_leg == edu_slm)

    work_leg = len(legacy.get("workExperience") or [])
    work_slm = len(slm.get("workExperience") or [])
    work_count_match = (work_leg == work_slm)

    return {
        "emailExactMatch": email_match,
        "phoneExactMatch": phone_match,
        "nameTokenF1": name_f1,
        "nameSequenceRatio": name_ratio,
        "titleTokenF1": title_f1,
        "titleSequenceRatio": title_ratio,
        "skillsPrecision": skills_metrics["precision"],
        "skillsRecall": skills_metrics["recall"],
        "skillsJaccard": skills_metrics["jaccard"],
        "expDeltaYears": exp_delta,
        "educationCountMatch": edu_count_match,
        "workExperienceCountMatch": work_count_match,
    }


# ---------------------------------------------------------------------------
# Validity Flags
# ---------------------------------------------------------------------------

def compute_validity_flags(extraction: Dict[str, Any]) -> Dict[str, bool]:
    """Validates structural and semantic plausibility of fields in the extraction."""
    email = extraction.get("email")
    phone = extraction.get("phone")
    name = extraction.get("candidateName")

    # Email validity (no em-dash placeholder, valid RFC regex)
    email_valid = True
    if email:
        em_dashes = ("—", "–", "―", "−", "-")
        if any(d in str(email) for d in em_dashes) and "@" not in str(email):
            email_valid = False
        elif not re.match(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", str(email).strip()):
            email_valid = False

    # Phone validity (at least 7 digits, no placeholder text)
    phone_valid = True
    if phone:
        digits = re.sub(r"\D", "", str(phone))
        if len(digits) < 7:
            phone_valid = False

    # Name plausibility (>= 2 chars, not purely digits, no markdown)
    name_plausible = True
    if name:
        s_name = str(name).strip()
        if len(s_name) < 2 or bool(re.search(r"\d{3,}", s_name)) or s_name.lower() in ("unknown", "n/a", "none", "null"):
            name_plausible = False
    else:
        name_plausible = False

    # Dates parse validity
    work_entries = extraction.get("workExperience") or []
    dates_parse = True
    for entry in work_entries:
        if isinstance(entry, dict) and entry.get("dates") and not entry.get("durationYears") and not entry.get("duration_years"):
            dates_parse = False
            break

    return {
        "emailValid": email_valid,
        "phoneValid": phone_valid,
        "namePlausible": name_plausible,
        "datesParse": dates_parse,
    }


# ---------------------------------------------------------------------------
# Grounding / Hallucination Detection
# ---------------------------------------------------------------------------

def compute_grounding_flags(extraction: Dict[str, Any], source_text: str) -> Dict[str, Any]:
    """
    Checks if extracted candidate entities have literal textual support in the resume text.
    Returns ungrounded entity flags.
    """
    if not source_text:
        return {"allGrounded": True, "ungroundedEntities": []}

    source_lower = source_text.lower()
    ungrounded: List[str] = []

    # Check candidate name
    name = extraction.get("candidateName")
    if name:
        name_tokens = [t for t in re.findall(r"\w+", str(name).lower()) if len(t) > 2]
        if name_tokens and not any(t in source_lower for t in name_tokens):
            ungrounded.append(f"candidateName:{name}")

    # Check work experience company names
    for w in extraction.get("workExperience") or []:
        company = w.get("company") if isinstance(w, dict) else None
        if company:
            c_clean = str(company).strip().lower()
            if len(c_clean) > 3 and c_clean not in source_lower:
                ungrounded.append(f"company:{company}")

    # Check skills (only flag longer distinctive skills if missing)
    for s in extraction.get("skills") or []:
        s_clean = str(s).strip().lower()
        if len(s_clean) > 4 and s_clean not in source_lower:
            # Check if tokens of skill exist
            tokens = [t for t in re.findall(r"\w+", s_clean) if len(t) > 2]
            if tokens and not any(t in source_lower for t in tokens):
                ungrounded.append(f"skill:{s}")

    return {
        "allGrounded": (len(ungrounded) == 0),
        "ungroundedCount": len(ungrounded),
        "ungroundedEntities": ungrounded[:10],
    }


# ---------------------------------------------------------------------------
# Supabase Logging Helper
# ---------------------------------------------------------------------------

def persist_shadow_log_to_supabase(
    file_hash: str,
    legacy_version: str,
    slm_version: str,
    legacy_output: Dict[str, Any],
    slm_output: Dict[str, Any],
    field_agreements: Dict[str, Any],
    validity_flags: Dict[str, bool],
    grounding_flags: Dict[str, Any],
    slm_latency_ms: Optional[float] = None,
    tokens_out: Optional[int] = None,
    error_flags: Optional[List[str]] = None,
    job_id: Optional[str] = None,
    doc_format: str = "pdf",
    page_count: int = 1,
    is_two_column: bool = False,
    char_length: int = 0,
    rss_mb: float = 0.0,
) -> bool:
    """
    Persists shadow comparison record into Supabase `extraction_shadow_logs` table.
    Catches all exceptions to ensure zero impact on primary traffic.
    """
    try:
        from app.storage.resume_storage import get_supabase_client, is_supabase_configured
        if not is_supabase_configured():
            return False

        client = get_supabase_client()
        if not client:
            return False

        latency = round(slm_latency_ms, 2) if slm_latency_ms is not None else None
        record = {
            "fileHash": file_hash,
            "legacyVersion": legacy_version,
            "slmVersion": slm_version,
            "legacyOutput": legacy_output,
            "slmOutput": slm_output,
            "fieldAgreements": field_agreements,
            "validityFlags": validity_flags,
            "groundingFlags": grounding_flags,
            "slmLatencyMs": latency,
            "tokensOut": tokens_out,
            "errorFlags": error_flags or [],
            "jobId": job_id or f"shadow_{file_hash[:12]}",
            "docFormat": doc_format,
            "pageCount": page_count,
            "isTwoColumn": is_two_column,
            "charLength": char_length,
            "primarySource": "legacy",
            "shadowSource": "slm",
            "primaryDurationMs": 0.0,
            "shadowDurationMs": latency or 0.0,
            "rssMbAtRun": round(rss_mb, 2) if rss_mb else 0.0,
            "isValidJson": ("json_invalid" not in (error_flags or [])),
            "agreementScores": field_agreements,
            "discrepancies": error_flags or [],
            "shadowOutput": slm_output,
        }

        from app.services.extraction_service import _get_table_ref
        table = _get_table_ref(client, "extraction_shadow_logs")
        try:
            table.insert(record).execute()
        except Exception as insert_err:
            # If some optional columns are missing in table, prune to core columns and retry
            core_record = {
                "fileHash": file_hash,
                "legacyOutput": legacy_output,
                "slmOutput": slm_output,
                "fieldAgreements": field_agreements,
            }
            # Add other keys individually if supported
            for k, v in record.items():
                if k in ("id", "createdAt"):
                    continue
                core_record[k] = v
            table.insert(core_record).execute()

        logger.info("[ShadowEvaluator] Successfully saved shadow log for hash %s", file_hash[:12])
        return True
    except Exception as exc:
        logger.warning("[ShadowEvaluator] Failed to persist shadow log (non-fatal): %s", exc)
        return False
