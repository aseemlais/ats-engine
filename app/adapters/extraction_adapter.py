import logging
import os
import threading
from typing import Any, Dict, List, Optional, Set

from app.models.resume_schema import ProcessedResume, RawResumeExtraction
from app.services.post_processor import post_process_resume
from app.services.structured_extractor import StructuredResumeExtractor

logger = logging.getLogger("extraction_adapter")


class ExtractionAdapter:
    """
    Feature-flagged extraction adapter with automatic fallback and shadow mode.
    Controlled by environment variables:
      - USE_NEW_EXTRACTOR (default: false): Activates SLM as primary extractor.
      - EXTRACTION_SHADOW_MODE / SHADOW_MODE (default: false): Runs the alternate
        extractor asynchronously in the background and logs field-level differences
        without impacting latency or stability.
    """

    def __init__(
        self,
        extraction_service: Optional[Any] = None,
        legacy_extractor: Optional[StructuredResumeExtractor] = None,
    ):
        self.extraction_service = extraction_service
        self.legacy_extractor = legacy_extractor or StructuredResumeExtractor()
        self.latest_shadow_diff: Optional[Dict[str, Any]] = None

    @property
    def is_new_extractor_enabled(self) -> bool:
        """Checks if the new fine-tuned extractor is activated via environment variable."""
        val = os.getenv("USE_NEW_EXTRACTOR", "false").strip().lower()
        return val in ("true", "1", "yes", "on", "enabled")

    @property
    def is_shadow_mode_enabled(self) -> bool:
        """Checks if shadow mode is enabled for background comparison monitoring."""
        val = (
            os.getenv("EXTRACTION_SHADOW_MODE")
            or os.getenv("SHADOW_MODE", "false")
        ).strip().lower()
        return val in ("true", "1", "yes", "on", "enabled")

    def extract(self, raw_text: str, cleaned_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Unified extraction method matching the production extractor interface.
        Returns extracted dictionary with 'extractionSource' set to 'slm', 'fallback', or 'legacy'.
        If shadow mode is active, triggers background execution of the alternate extractor.
        """
        text_to_extract = raw_text or cleaned_text or ""
        primary_result: Dict[str, Any]
        primary_mode: str

        import hashlib
        file_hash = hashlib.sha256(text_to_extract.encode("utf-8")).hexdigest()

        should_slm = False
        if self.extraction_service and hasattr(self.extraction_service, "should_route_to_slm"):
            should_slm = self.extraction_service.should_route_to_slm(file_hash)
        else:
            should_slm = self.is_new_extractor_enabled

        if should_slm and self.extraction_service:
            try:
                logger.info("[ExtractionAdapter] Routing through new fine-tuned SLM extractor...")
                infer_out = self.extraction_service._infer_llm(text_to_extract)
                raw_data = None
                if infer_out:
                    if isinstance(infer_out, tuple):
                        d, reason = infer_out
                        if reason != "length" and isinstance(d, dict):
                            raw_data = d
                    elif isinstance(infer_out, dict):
                        raw_data = infer_out

                if raw_data and isinstance(raw_data, dict):
                    processed = post_process_resume(raw_data)
                    primary_result = processed.model_dump(by_alias=True)
                    primary_result["extractionSource"] = "slm"
                    primary_result["source"] = "qwen2.5-1.5b-instruct"
                    primary_mode = "slm"
                    logger.info(
                        "[ExtractionAdapter] Extraction completed via 'slm'. (skills: %d, exp: %.1f yrs)",
                        len(primary_result.get("skills") or []),
                        float(primary_result.get("totalExperienceYears") or 0.0),
                    )

                    if self.is_shadow_mode_enabled:
                        self._trigger_shadow_execution(text_to_extract, primary_result, primary_mode="slm")

                    return primary_result

                logger.warning("[ExtractionAdapter] SLM inference returned empty or invalid output. Falling back to legacy.")
            except Exception as exc:
                logger.warning("[ExtractionAdapter] New extractor failed with error: %s. Falling back to legacy.", exc)

            # Fallback path when new extractor was enabled but failed
            primary_result = self.legacy_extractor.extract(raw_text=text_to_extract, cleaned_text=cleaned_text)
            primary_result["extractionSource"] = "fallback"
            logger.info(
                "[ExtractionAdapter] Extraction completed via 'fallback'. (skills: %d, exp: %.1f yrs)",
                len(primary_result.get("skills") or []),
                float(primary_result.get("totalExperienceYears") or 0.0),
            )
            return primary_result

        # Legacy primary path (USE_NEW_EXTRACTOR=false)
        primary_result = self.legacy_extractor.extract(raw_text=text_to_extract, cleaned_text=cleaned_text)
        primary_result["extractionSource"] = "legacy"
        logger.info(
            "[ExtractionAdapter] Extraction completed via 'legacy'. (skills: %d, exp: %.1f yrs)",
            len(primary_result.get("skills") or []),
            float(primary_result.get("totalExperienceYears") or 0.0),
        )

        if self.is_shadow_mode_enabled:
            self._trigger_shadow_execution(text_to_extract, primary_result, primary_mode="legacy")

        return primary_result

    def _trigger_shadow_execution(
        self,
        text_to_extract: str,
        primary_result: Dict[str, Any],
        primary_mode: str,
    ) -> None:
        """Enqueues shadow job into the single prioritized ExtractionService queue, or runs via thread fallback."""
        import hashlib
        file_hash = hashlib.sha256(text_to_extract.encode("utf-8")).hexdigest()

        if self.extraction_service and hasattr(self.extraction_service, "enqueue_shadow_job"):
            enqueued = self.extraction_service.enqueue_shadow_job(
                file_hash=file_hash,
                text_to_extract=text_to_extract,
                primary_output=primary_result,
                primary_source=primary_mode,
            )
            if enqueued:
                return

        # Fallback daemon thread if service queue is unavailable
        def _shadow_worker():
            try:
                if primary_mode == "legacy" and self.extraction_service:
                    infer_out = self.extraction_service._infer_llm(text_to_extract)
                    raw_data = None
                    if infer_out:
                        if isinstance(infer_out, tuple):
                            d, reason = infer_out
                            if reason != "length" and isinstance(d, dict):
                                raw_data = d
                        elif isinstance(infer_out, dict):
                            raw_data = infer_out

                    if raw_data and isinstance(raw_data, dict):
                        processed = post_process_resume(raw_data)
                        shadow_result = processed.model_dump(by_alias=True)
                        shadow_result["extractionSource"] = "slm"
                        self._compare_and_log_diff(primary_result, shadow_result, primary_name="legacy", shadow_name="slm")
                elif primary_mode == "slm":
                    shadow_result = self.legacy_extractor.extract(raw_text=text_to_extract)
                    shadow_result["extractionSource"] = "legacy"
                    self._compare_and_log_diff(primary_result, shadow_result, primary_name="slm", shadow_name="legacy")
            except Exception as exc:
                logger.warning("[ShadowMode] Background shadow execution encountered error: %s", exc)

        thread = threading.Thread(target=_shadow_worker, daemon=True, name="ats_shadow_worker")
        thread.start()

    def _compare_and_log_diff(
        self,
        primary_res: Dict[str, Any],
        shadow_res: Dict[str, Any],
        primary_name: str,
        shadow_name: str,
    ) -> Dict[str, Any]:
        """Computes and logs field-level differences between primary and shadow extractions."""
        # Contact fields
        name_p, name_s = primary_res.get("candidateName"), shadow_res.get("candidateName")
        email_p, email_s = primary_res.get("email"), shadow_res.get("email")
        phone_p, phone_s = primary_res.get("phone"), shadow_res.get("phone")
        title_p, title_s = primary_res.get("jobTitle"), shadow_res.get("jobTitle")

        # Experience diff
        exp_p = float(primary_res.get("totalExperienceYears") or primary_res.get("experienceYears") or 0.0)
        exp_s = float(shadow_res.get("totalExperienceYears") or shadow_res.get("experienceYears") or 0.0)
        exp_delta = round(abs(exp_p - exp_s), 2)

        # Skills comparison (Jaccard similarity and symmetric differences)
        skills_p: Set[str] = {s.strip().lower() for s in (primary_res.get("skills") or []) if s}
        skills_s: Set[str] = {s.strip().lower() for s in (shadow_res.get("skills") or []) if s}

        intersection = skills_p & skills_s
        union = skills_p | skills_s
        jaccard = round(len(intersection) / len(union), 2) if union else 1.0
        added_in_shadow = list(skills_s - skills_p)
        missing_in_shadow = list(skills_p - skills_s)

        diff = {
            "primary": primary_name,
            "shadow": shadow_name,
            "nameMatch": (name_p == name_s),
            "primaryName": name_p,
            "shadowName": name_s,
            "emailMatch": (email_p == email_s),
            "primaryEmail": email_p,
            "shadowEmail": email_s,
            "phoneMatch": (phone_p == phone_s),
            "primaryPhone": phone_p,
            "shadowPhone": phone_s,
            "titleMatch": (title_p == title_s),
            "primaryTitle": title_p,
            "shadowTitle": title_s,
            "expDeltaYears": exp_delta,
            "skillsJaccard": jaccard,
            "addedInShadow": added_in_shadow[:5],
            "missingInShadow": missing_in_shadow[:5],
        }

        self.latest_shadow_diff = diff

        logger.info(
            "[ShadowMode] Diff [%s vs %s] | Name match: %s | Title match: %s | Email match: %s | Phone match: %s | ExpDelta: %.1f yrs | Skills Jaccard: %.2f (+%d, -%d)",
            primary_name,
            shadow_name,
            name_p == name_s,
            title_p == title_s,
            email_p == email_s,
            phone_p == phone_s,
            exp_delta,
            jaccard,
            len(added_in_shadow),
            len(missing_in_shadow),
        )
        return diff
