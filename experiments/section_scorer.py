"""
Isolated Section-Based Scorer for Local ATS Scoring Experiment.
Calculates section-level BM25 lexical relevance and Sentence Transformer semantic relevance,
aggregates into existing ATS core dimensions without global whole-resume dilution,
and computes an optional, low-weight, deduplicated bonus for evidence in additional sections.
"""
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
import numpy as np

from app.config import config
from app.models.job import JobDescription
from app.models.resume import ParsedResume
from app.services.bm25_query import build_bm25_query
from app.services.bm25_search import BM25SearchService
from app.services.certification_matcher import CertificationMatcherService
from app.services.education_matcher import EducationMatcherService
from app.services.embedding_service import EmbeddingService
from app.services.experience_matcher import ExperienceMatcherService
from app.services.nlp_extractor import NLPExtractorService
from app.services.score_composer import certification_status, compose_score
from app.services.semantic_search import SemanticSearchService
from app.services.skill_matcher import SkillMatcherService
from app.utils.skill_normalizer import extract_skills_from_text
from experiments.experiment_config import (
    CertificationScoringConfig,
    ExperienceScoringConfig,
    ProjectsScoringConfig,
    SectionDetectionConfig,
    SectionScoringConfig,
    SkillsScoringConfig,
)
from experiments.education_eligibility import (
    EducationEligibilityResult,
    EducationEligibilityService,
    EducationEligibilityStatus,
)
from experiments.experience_scorer import (
    ExperienceEvaluationResult,
    ExperienceRequirementStatus,
    ExperienceScorerService,
    ExperienceType,
)
from experiments.skills_projects_scorer import (
    CertificationEvaluationResult,
    ProjectsEvaluationResult,
    SkillsEvaluationResult,
    SkillsProjectsScorerService,
)
from experiments.spacy_section_detector import DetectedSection, DocumentExtractionResult, SpacySectionDetector


@dataclass
class SectionScore:
    """Detailed score and diagnostic container for an individual resume section."""
    raw_heading: str                     # Original section heading from resume
    category: str                        # Normalized category (skills, experience, projects, education, unknown, etc.)
    text_evidence: str                   # The text content used for matching
    char_length: int                     # Character length of evidence
    is_core: bool                        # Whether category is in CORE_CATEGORIES
    
    # Lexical BM25 (independent, not multiplied by cross-section weights)
    bm25_normalized: float               # theoretical_bound normalized score in [0.0, 1.0]
    bm25_raw: float                      # raw BM25 score
    matched_terms: List[str] = field(default_factory=list)  # query terms matched in this section
    query_text: str = ""                 # JD query text used
    
    # Semantic (independent, not multiplied by cross-section weights)
    semantic_raw_cosine: float = 0.0     # raw cosine similarity [-1.0, 1.0]
    semantic_calibrated: float = 0.0     # calibrated to [0.0, 1.0] via floor/ceil
    
    # Uncertainty & Diagnostics
    uncertainty_notes: List[str] = field(default_factory=list)


@dataclass
class SectionScoreResult:
    """Detailed breakdown of experimental section-based scoring."""
    candidate_id: str
    name: str
    core_score: float                  # Composed ATS core score (0-100%)
    additional_bonus: float            # Bonus points from additional sections (0-beta)
    final_score: float                 # min(100.0, core_score + additional_bonus)
    
    # Per-section independent scores (Batch 1: strictly independent, zero cross-section weights)
    detailed_section_scores: List[SectionScore] = field(default_factory=list)
    section_scores: Dict[str, SectionScore] = field(default_factory=dict) # category -> SectionScore
    unknown_sections: List[SectionScore] = field(default_factory=list)   # Preserved unknown/custom sections
    
    # Section-level BM25 breakdown (independent per section category)
    section_bm25_scores: Dict[str, float] = field(default_factory=dict)
    section_bm25_raw: Dict[str, float] = field(default_factory=dict)
    
    # Section-level Semantic breakdown (independent per section category)
    section_semantic_scores: Dict[str, float] = field(default_factory=dict)
    section_semantic_raw: Dict[str, float] = field(default_factory=dict)
    
    # Downstream bridge diagnostics (provisional unweighted mean of present core sections until aggregation redesign batch)
    bm25_core_aggregate: float = 0.0
    semantic_core_aggregate: float = 0.0
    
    # Core dimension scores (0-1 scale)
    core_dimension_scores: Dict[str, float] = field(default_factory=dict)
    
    # Additional section bonus breakdown
    additional_relevance: float = 0.0
    additional_bonus_breakdown: Dict[str, Any] = field(default_factory=dict)
    
    # Batch 2: Isolated Education Eligibility Filter (separate from relevance score)
    education_eligibility: Optional[EducationEligibilityResult] = None
    
    # Batch 3: Experience & Internship Evaluation (Duration x Type x Relevance)
    experience_evaluation: Optional[ExperienceEvaluationResult] = None
    fresher_experience_bonus: float = 0.0
    
    # Batch 4: Skills, Projects, and Certifications Evaluation
    skills_evaluation: Optional[SkillsEvaluationResult] = None
    projects_evaluation: Optional[ProjectsEvaluationResult] = None
    certifications_evaluation: Optional[CertificationEvaluationResult] = None
    
    # Extraction and classification metadata
    detected_sections_summary: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class SectionScorer:
    """
    Experimental section-level scorer.
    Evaluates lexical & semantic relevance strictly within classified resume sections.
    """

    def __init__(
        self,
        scoring_config: Optional[SectionScoringConfig] = None,
        detection_config: Optional[SectionDetectionConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
        bm25_service: Optional[BM25SearchService] = None,
        skill_service: Optional[SkillMatcherService] = None,
        experience_service: Optional[ExperienceMatcherService] = None,
        education_service: Optional[EducationMatcherService] = None,
        certification_service: Optional[CertificationMatcherService] = None,
        nlp_extractor: Optional[NLPExtractorService] = None,
        exp_config: Optional[ExperienceScoringConfig] = None,
        skills_config: Optional[SkillsScoringConfig] = None,
        projects_config: Optional[ProjectsScoringConfig] = None,
        cert_config: Optional[CertificationScoringConfig] = None,
    ):
        self.scoring_config = scoring_config or SectionScoringConfig()
        self.detection_config = detection_config or SectionDetectionConfig()
        self.exp_config = exp_config or ExperienceScoringConfig()
        self.skills_config = skills_config or SkillsScoringConfig()
        self.projects_config = projects_config or ProjectsScoringConfig()
        self.cert_config = cert_config or CertificationScoringConfig()
        self.detector = SpacySectionDetector(self.detection_config)

        # Service dependencies (reusing read-only existing services)
        self.embedding_service = embedding_service or EmbeddingService()
        self.semantic_service = SemanticSearchService(embedding_service=self.embedding_service)
        self.bm25_service = bm25_service or BM25SearchService()
        self.skill_service = skill_service or SkillMatcherService()
        self.experience_service = experience_service or ExperienceMatcherService()
        self.education_service = education_service or EducationMatcherService()
        self.certification_service = (
            certification_service
            or CertificationMatcherService(self.embedding_service)
        )
        self.nlp_extractor = nlp_extractor or NLPExtractorService()

        # Batch 2: Isolated education eligibility service (hard filter, not relevance scoring)
        self.education_eligibility_service = EducationEligibilityService()

        # Batch 3: Experience & Internship Scoring Service (Duration x Type x Duties Relevance)
        self.experience_scorer_service = ExperienceScorerService(
            config=self.exp_config,
            detection_config=self.detection_config,
        )

        # Batch 4: Skills and Projects Scoring Service (Provenance, Deduplication, Project Relevance)
        self.skills_projects_scorer = SkillsProjectsScorerService(
            skills_config=self.skills_config,
            projects_config=self.projects_config,
            cert_config=self.cert_config,
        )

    def build_section_queries(self, jd: JobDescription) -> Dict[str, str]:
        """
        Builds section-specific query strings from the Job Description.
        - skills: parsed JD required + preferred + supporting skills
        - experience: technical responsibilities
        - projects: skills + responsibilities
        - education: education requirement text / degree terms
        """
        skills_list: List[str] = []
        for s in list(jd.required_skills) + list(jd.preferred_skills) + list(jd.supporting_skills):
            if s not in skills_list:
                skills_list.append(s)
        skills_query = " ".join(skills_list).strip()

        resp_query = jd.responsibilities.strip() if jd.responsibilities else ""
        proj_query = f"{skills_query}\n{resp_query}".strip()
        edu_terms = " ".join(jd.education_requirements)
        edu_query = f"{edu_terms} degree in related field {jd.title}".strip() if edu_terms else f"degree in related field {jd.title}"

        # Fallback to full technical query if any is empty
        full_technical = build_bm25_query(jd).text
        return {
            "skills": skills_query or full_technical,
            "experience": resp_query or full_technical,
            "projects": proj_query or full_technical,
            "education": edu_query or full_technical,
            "full_technical": full_technical,
        }

    def score_section_bm25(
        self,
        section_text: str,
        query_text: str,
        candidate_id: str,
        section_name: str,
    ) -> Tuple[float, float, List[str]]:
        """
        Calculates BM25 lexical relevance for a single section using theoretical_bound normalization.
        Returns: (normalized_score [0,1], raw_score, matched_terms)
        """
        if not section_text or not section_text.strip() or not query_text.strip():
            return 0.0, 0.0, []

        doc_id = f"{candidate_id}#{section_name}"
        result = self.bm25_service.score_documents(
            query_text=query_text,
            candidate_ids=[doc_id],
            texts=[section_text],
            method="theoretical_bound",
        )
        norm_score = float(result.normalized.get(doc_id, 0.0))
        raw_score = float(result.raw.get(doc_id, 0.0))
        matched = result.matched_terms.get(doc_id, [])
        return norm_score, raw_score, matched

    def _provisional_downstream_aggregation(
        self,
        section_scores: Dict[str, SectionScore],
        section_texts: Dict[str, str],
        core_categories: Sequence[str],
    ) -> Tuple[float, float, int]:
        """
        Isolated provisional downstream bridge.
        Per Batch 1 requirements:
        - Cross-section weights (0.40, 0.30, 0.20, 0.10) are REMOVED.
        - Calculates an unweighted average of present core sections strictly to feed
          the existing downstream compose_score pipeline until the aggregation redesign batch.
        """
        present_bm25 = [
            section_scores[s].bm25_normalized
            for s in core_categories
            if section_texts.get(s, "").strip()
        ]
        bm25_agg = (sum(present_bm25) / len(present_bm25)) if present_bm25 else 0.0

        present_sem_raw = [
            section_scores[s].semantic_raw_cosine
            for s in core_categories
            if section_texts.get(s, "").strip()
        ]
        if present_sem_raw:
            mean_raw_cos = sum(present_sem_raw) / len(present_sem_raw)
            floor = self.scoring_config.SEMANTIC_COSINE_FLOOR
            ceil = self.scoring_config.SEMANTIC_COSINE_CEIL
            sem_agg = min(max((mean_raw_cos - floor) / (ceil - floor), 0.0), 1.0)
        else:
            sem_agg = 0.0

        return bm25_agg, sem_agg, len(present_sem_raw)

    def score_section_semantic(
        self,
        section_text: str,
        target_vec: Optional[np.ndarray],
    ) -> Tuple[float, float]:
        """
        Calculates Sentence Transformer semantic similarity for a single section.
        Calibrates raw cosine similarity with production floor/ceil bounds.
        Returns: (calibrated_score [0,1], raw_cosine [0,1])
        """
        if not section_text or not section_text.strip() or target_vec is None:
            return 0.0, 0.0

        sec_vec = self.embedding_service.encode_text(section_text)
        raw_cosine = self.semantic_service.compute_cosine_similarity(sec_vec, target_vec)

        # Apply production heuristic calibration
        floor = self.scoring_config.SEMANTIC_COSINE_FLOOR
        ceil = self.scoring_config.SEMANTIC_COSINE_CEIL
        calibrated = min(max((raw_cosine - floor) / (ceil - floor), 0.0), 1.0)
        return float(calibrated), float(raw_cosine)

    def compute_additional_bonus(
        self,
        detected_sections: List[DetectedSection],
        jd: JobDescription,
        jd_full_vector: Optional[np.ndarray],
        matched_core_skills: Set[str],
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Calculates the Additional-Section Relevance Bonus.
        Formula: AdditionalBonus = beta * AdditionalRelevance
        
        Rules:
        - Evaluates only non-core sections (achievements, publications, leadership, additional, unknown).
        - If no non-core sections or no evidence, bonus = 0.0.
        - Deduplication: skills already credited in core sections do not grant bonus.
        - Unrelated wordy sections receive 0.0 because semantic similarity falls below floor.
        - AdditionalRelevance must exceed threshold to trigger any bonus.
        - When config.ENABLE_ADDITIONAL_BONUS is False or BONUS_BETA == 0, bonus = 0.0.
        """
        if not self.scoring_config.ENABLE_ADDITIONAL_BONUS or self.scoring_config.BONUS_BETA <= 0.0:
            return 0.0, 0.0, {"reason": "additional_bonus_disabled"}

        non_core_sections = [
            s for s in detected_sections
            if not s.is_core and s.text.strip() and s.category in self.detection_config.ADDITIONAL_CATEGORIES and s.category != "education"
        ]

        if not non_core_sections:
            return 0.0, 0.0, {
                "reason": "no_non_core_sections",
                "non_core_sections_count": 0,
            }

        # 1. Lexical Relevance with Deduplication
        combined_text = "\n".join(s.text for s in non_core_sections)
        additional_skills = set(extract_skills_from_text(combined_text, expand=True))
        
        # Deduplicate against core skills
        new_additional_skills = additional_skills - matched_core_skills
        
        # Evaluate relevance of each new additional skill using BM25 and semantic similarity
        # Build JD skill set for reference
        jd_skills_all = set(
            s.lower() for s in (
                list(jd.required_skills)
                + list(jd.preferred_skills)
                + list(jd.supporting_skills)
            )
        )
        # Prepare containers
        relevant_new_skills_set: Set[str] = set()
        # Remaining JD skills after core matches (used for lexical proportion later)
        remaining_jd_skills = jd_skills_all - matched_core_skills

        # Use a fresh BM25 service instance to avoid contaminating the global index.
        local_bm25 = BM25SearchService()
        for raw_skill in new_additional_skills:
            # Canonicalize using existing skill service
            canonical = self.skills_projects_scorer.canonicalize_skill(raw_skill)
            if canonical in matched_core_skills:
                continue
            # BM25 relevance: treat skill term as query against combined additional text
            bm25_result = local_bm25.score_documents(
                query_text=canonical,
                candidate_ids=[f"add_skill_{canonical}"],
                texts=[combined_text],
                method="theoretical_bound",
            )
            bm25_norm = float(bm25_result.normalized.get(f"add_skill_{canonical}", 0.0))
            # Semantic relevance: embed skill term and compare with JD full vector
            semantic_norm = 0.0
            if jd_full_vector is not None:
                skill_vec = self.embedding_service.encode_text(canonical)
                raw_cos = self.semantic_service.compute_cosine_similarity(skill_vec, jd_full_vector)
                floor = self.scoring_config.SEMANTIC_COSINE_FLOOR
                ceil = self.scoring_config.SEMANTIC_COSINE_CEIL
                semantic_norm = min(max((raw_cos - floor) / (ceil - floor), 0.0), 1.0)
            # Determine if skill passes relevance threshold (either signal sufficient)
            if (
                bm25_norm >= self.scoring_config.ADDITIONAL_RELEVANCE_THRESHOLD
                or semantic_norm >= self.scoring_config.ADDITIONAL_RELEVANCE_THRESHOLD
            ):
                relevant_new_skills_set.add(canonical)

        # Lexical relevance now based on count of relevant new skills vs remaining JD skills
        lexical_relevance = 0.0
        if remaining_jd_skills:
            lexical_relevance = min(
                len(relevant_new_skills_set) / max(len(remaining_jd_skills), 1), 1.0
            )
        elif relevant_new_skills_set:
            lexical_relevance = 0.5

        # Ensure we have a list for breakdown
        relevant_new_skills = sorted(list(relevant_new_skills_set))

        # 2. Semantic Relevance of Additional Sections
        semantic_relevance = 0.0
        raw_cosine = 0.0
        if jd_full_vector is not None and combined_text.strip():
            calibrated, raw_cosine = self.score_section_semantic(combined_text, jd_full_vector)
            semantic_relevance = calibrated

        # 3. Combine Relevance
        # Blend: default 60% semantic + 40% lexical
        combined_relevance = (
            self.scoring_config.ADDITIONAL_SEMANTIC_WEIGHT * semantic_relevance
            + self.scoring_config.ADDITIONAL_LEXICAL_WEIGHT * lexical_relevance
        )

        # 4. Threshold Filter
        if combined_relevance < self.scoring_config.ADDITIONAL_RELEVANCE_THRESHOLD:
            additional_relevance = 0.0
            bonus = 0.0
            qualified = False
        else:
            additional_relevance = min(max(combined_relevance, 0.0), 1.0)
            bonus = self.scoring_config.BONUS_BETA * additional_relevance
            qualified = True

        breakdown = {
            "non_core_sections_evaluated": [
                {"category": s.category, "heading": s.raw_heading, "char_len": len(s.text)}
                for s in non_core_sections
            ],
            "additional_skills_found": sorted(list(additional_skills)),
            "new_additional_skills": sorted(list(new_additional_skills)),
            "relevant_new_skills": sorted(list(relevant_new_skills)),
            "deduplicated_core_skills_count": len(additional_skills.intersection(matched_core_skills)),
            "raw_cosine": round(raw_cosine, 4),
            "semantic_relevance": round(semantic_relevance, 4),
            "lexical_relevance": round(lexical_relevance, 4),
            "combined_relevance": round(combined_relevance, 4),
            "exceeded_threshold": qualified,
            "bonus_points": round(bonus, 3),
        }

        return round(additional_relevance, 4), round(bonus, 3), breakdown

    def score_extraction_result(
        self,
        extraction: DocumentExtractionResult,
        candidate_id: str,
        jd: JobDescription,
        parsed_resume: Optional[ParsedResume] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> SectionScoreResult:
        """
        Executes experimental section-based scoring on a DocumentExtractionResult.
        """
        notes = list(extraction.extraction_notes)
        if not extraction.success:
            notes.append(f"Extraction failed: {extraction.error_message}")
            return SectionScoreResult(
                candidate_id=candidate_id,
                name="Unknown",
                core_score=0.0,
                additional_bonus=0.0,
                final_score=0.0,
                notes=notes,
            )

        # Prepare JD queries and embeddings
        jd_queries = self.build_section_queries(jd)
        jd_vectors = self.embedding_service.compute_jd_vectors(jd)
        jd_full_vec = jd_vectors.get("full")

        # Synthesize or adopt ParsedResume for core rule-based matchers
        if parsed_resume is None:
            parsed_resume = self.nlp_extractor.extract_structured(
                candidate_id=candidate_id,
                source_file=extraction.file_path,
                file_hash="exp_hash",
                raw_text=extraction.raw_text,
                cleaned_text=extraction.cleaned_text,
            )

        # Update parsed_resume.section_texts with spaCy classified section text
        sec_map = extraction.section_map
        section_texts = {
            "skills": sec_map.get("skills", ""),
            "experience": sec_map.get("experience", ""),
            "projects": sec_map.get("projects", ""),
            "education": sec_map.get("education", ""),
            "certifications": sec_map.get("certifications", ""),
        }
        # If skills section was not explicitly detected, check if projects/summary contains skill text
        if not section_texts["skills"]:
            if "projects" in sec_map:
                section_texts["skills"] = sec_map["projects"]
                notes.append("Skills heading missing: using projects text for skills evaluation.")
            elif "summary" in sec_map:
                section_texts["skills"] = sec_map["summary"]
                notes.append("Skills heading missing: using summary text for skills evaluation.")

        parsed_resume.section_texts = section_texts

        # Isolate core skills to core sections so non-core evidence is evaluated cleanly
        # Batch 2: Education is EXCLUDED from core skills extraction and relevance scoring.
        # Education text is retained only for eligibility reporting.
        core_sections_text = "\n".join(
            s.text for s in extraction.sections
            if s.category in ("summary", "skills", "experience", "projects", "certifications")
        )
        core_skills_detected = set(
            s.lower() for s in extract_skills_from_text(core_sections_text, expand=True)
        )
        parsed_resume.skills = list(core_skills_detected)

        # 1. Independent Per-Section Scoring (Batch 1)
        # Strict Rule: No cross-section weights (0.40, 0.30, 0.20, 0.10) are used in calculating any section score.
        # Every section retains its own independent lexical BM25 relevance and semantic cosine similarity.
        detailed_section_scores: List[SectionScore] = []
        section_scores: Dict[str, SectionScore] = {}
        unknown_sections: List[SectionScore] = []
        section_bm25_norm: Dict[str, float] = {}
        section_bm25_raw: Dict[str, float] = {}
        section_sem_norm: Dict[str, float] = {}
        section_sem_raw: Dict[str, float] = {}

        # Batch 2: core_categories for SCORING excludes education.
        # Education is evaluated independently as a hard eligibility filter only.
        # Education section scores are still computed for diagnostics but excluded from aggregation.
        core_categories = ("skills", "experience", "projects")
        
        # All sections to compute BM25/semantic on (includes education for diagnostics)
        all_scored_categories = ("skills", "experience", "projects", "education")

        for sec_name in all_scored_categories:
            text = section_texts.get(sec_name, "")
            query = jd_queries.get(sec_name, jd_queries["full_technical"])
            raw_heading = sec_name.capitalize()
            for s in extraction.sections:
                if s.category == sec_name and s.raw_heading:
                    raw_heading = s.raw_heading
                    break

            # Batch 2: education is scored for diagnostics but NOT for relevance aggregation
            is_core_for_scoring = sec_name in core_categories

            if text and text.strip():
                norm_bm25, raw_bm25, matched = self.score_section_bm25(text, query, candidate_id, sec_name)
                target_vec = jd_vectors.get(sec_name, jd_full_vec)
                sem_cal, sem_cos = self.score_section_semantic(text, target_vec)
                uncertainty: List[str] = []
                if len(text.strip()) < 30:
                    uncertainty.append(f"Short text evidence ({len(text.strip())} chars) in section '{sec_name}'.")
                if not is_core_for_scoring:
                    uncertainty.append(f"Section '{sec_name}' scored for diagnostics only; excluded from relevance aggregation (Batch 2).")
            else:
                # Do NOT fabricate scores for absent or empty sections
                norm_bm25, raw_bm25, matched = 0.0, 0.0, []
                sem_cal, sem_cos = 0.0, 0.0
                uncertainty = [f"Section '{sec_name}' absent or empty: no relevance score fabricated."]

            sec_score = SectionScore(
                raw_heading=raw_heading,
                category=sec_name,
                text_evidence=text,
                char_length=len(text),
                is_core=is_core_for_scoring,
                bm25_normalized=norm_bm25,
                bm25_raw=raw_bm25,
                matched_terms=matched,
                query_text=query,
                semantic_raw_cosine=sem_cos,
                semantic_calibrated=sem_cal,
                uncertainty_notes=uncertainty,
            )
            detailed_section_scores.append(sec_score)
            section_scores[sec_name] = sec_score
            section_bm25_norm[sec_name] = norm_bm25
            section_bm25_raw[sec_name] = raw_bm25
            section_sem_norm[sec_name] = sem_cal
            section_sem_raw[sec_name] = sem_cos

        # Score non-core and unknown sections independently
        for s in extraction.sections:
            if not s.is_core and s.category not in core_categories:
                text = s.text.strip()
                query = jd_queries["full_technical"]
                if text:
                    norm_bm25, raw_bm25, matched = self.score_section_bm25(
                        text, query, candidate_id, f"{s.category}_{s.char_start}"
                    )
                    sem_cal, sem_cos = self.score_section_semantic(text, jd_full_vec)
                    uncertainty = []
                    if s.category == "unknown":
                        uncertainty.append(f"Unrecognized heading '{s.raw_heading}': content preserved as unknown section.")
                else:
                    norm_bm25, raw_bm25, matched = 0.0, 0.0, []
                    sem_cal, sem_cos = 0.0, 0.0
                    uncertainty = [f"Section '{s.raw_heading}' empty: no relevance score fabricated."]

                non_core_score = SectionScore(
                    raw_heading=s.raw_heading,
                    category=s.category,
                    text_evidence=text,
                    char_length=len(text),
                    is_core=False,
                    bm25_normalized=norm_bm25,
                    bm25_raw=raw_bm25,
                    matched_terms=matched,
                    query_text=query,
                    semantic_raw_cosine=sem_cos,
                    semantic_calibrated=sem_cal,
                    uncertainty_notes=uncertainty,
                )
                detailed_section_scores.append(non_core_score)
                if s.category == "unknown":
                    unknown_sections.append(non_core_score)

        # 2. PROVISIONAL DOWNSTREAM AGGREGATION BRIDGE (Batch 1 Isolation)
        # Note: Section weights (0.40, 0.30, 0.20, 0.10) are REMOVED.
        # This isolated helper computes an unweighted mean of present core sections strictly to feed
        # the existing downstream compose_score pipeline until the aggregation redesign batch.
        bm25_aggregate, semantic_aggregate, sem_active_count = self._provisional_downstream_aggregation(
            section_scores=section_scores,
            section_texts=section_texts,
            core_categories=core_categories,
        )
        notes.append("Batch 1: Section BM25 and semantic scores calculated independently without cross-section weights.")
        notes.append("Batch 2: Education excluded from relevance aggregation; evaluated as eligibility filter only.")
        notes.append("Batch 3: Experience evaluated using Duration x Type x Duties Relevance.")
        notes.append("Batch 4: Skills evaluated with token-safe alias matching, evidence provenance, and required-priority weighting.")
        notes.append("Batch 4: Projects scored on descriptions/contributions with duplicate-project suppression.")
        notes.append("Batch 4: Certifications evaluated conditionally on JD without absent-penalty.")

        # 3. Batch 4: Skills, Projects, and Certifications Evaluation
        skills_eval = self.skills_projects_scorer.evaluate_candidate_skills(
            extraction=extraction,
            jd=jd,
        )
        projects_eval = self.skills_projects_scorer.evaluate_candidate_projects(
            extraction=extraction,
            jd=jd,
            jd_query=jd_queries["projects"],
            jd_vector=jd_vectors.get("projects", jd_full_vec),
            bm25_scorer_fn=lambda t, q: self.score_section_bm25(t, q, candidate_id, "projects"),
            semantic_scorer_fn=lambda t, v: self.score_section_semantic(t, v),
        )

        # Batch 3: Experience & Internship Scoring (Duration x Type x Duties Relevance)
        exp_eval = self.experience_scorer_service.evaluate_candidate_experience(
            extraction=extraction,
            jd=jd,
            jd_query=jd_queries["experience"],
            jd_vector=jd_vectors.get("experience", jd_full_vec),
            bm25_scorer_fn=lambda t, q: self.score_section_bm25(t, q, candidate_id, "experience"),
            semantic_scorer_fn=lambda t, v: self.score_section_semantic(t, v),
        )

        cert_score_raw = self.certification_service.evaluate_certifications(parsed_resume, jd)
        cert_level = self.certification_service.requirement_level(jd)
        cert_eval = self.skills_projects_scorer.evaluate_certifications(
            extraction=extraction,
            jd=jd,
            existing_cert_score=cert_score_raw,
            req_level=cert_level,
        )

        # Determine skill score
        if skills_eval.is_applicable:
            skill_score = skills_eval.overall_skill_score
        else:
            skill_res = self.skill_service.evaluate_detailed(parsed_resume, jd)
            skill_score = skill_res.get("score", 0.0)

        # Certifications score
        cert_score = cert_eval.score

        # Matched core skills for additional bonus deduplication (Batch 4)
        matched_from_eval = set(s.lower() for s in (skills_eval.matched_required + skills_eval.matched_preferred))
        core_section_text = "\n".join(
            v for k, v in section_texts.items() if k != "education"
        )
        core_skills_detected = set(
            s.lower() for s in extract_skills_from_text(core_section_text, expand=True)
        )
        matched_core_skills = matched_from_eval.union(core_skills_detected)

        # 4. Compose Core Score using standard ATS weights
        w = config.scoring_weights(weights)
        legacy = config.na_policy == "legacy"

        # Determine whether experience is mandatory based on Batch 3 evaluation
        exp_is_mandatory = exp_eval.status != ExperienceRequirementStatus.NOT_REQUIRED and exp_eval.required_years > 0
        exp_score = exp_eval.final_experience_score if exp_is_mandatory else None

        scores = {
            "semantic": semantic_aggregate,
            "bm25": bm25_aggregate,
            "skill": skill_score,
            "experience": exp_score,
            "certification": cert_score,
        }
        status = {
            "semantic": "applied" if sem_active_count > 0 else "unavailable",
            "bm25": "applied" if bool([s for s in core_categories if section_texts.get(s, "").strip()]) else "not_applicable",
            "skill": "applied" if (skills_eval.is_applicable or self.skill_service.is_applicable(jd) or legacy) else "not_applicable",
            "experience": "applied" if exp_is_mandatory else "not_applicable",
        }
        # Certification status composition
        core_pre = compose_score(scores, {**status, "certification": "not_applicable"}, w)
        status["certification"] = certification_status(cert_level, cert_score, core_pre["final"], legacy)
        composed = compose_score(scores, status, w)

        cert_bonus = (
            config.cert_bonus_none * cert_score * 100.0
            if (cert_level == "none" and not legacy)
            else 0.0
        )
        core_score_percentage = min(max(composed["final"] * 100.0 + cert_bonus, 0.0), 100.0)

        # 5. Additional-Section Relevance Bonus
        add_rel, add_bonus, bonus_breakdown = self.compute_additional_bonus(
            detected_sections=extraction.sections,
            jd=jd,
            jd_full_vector=jd_full_vec,
            matched_core_skills=matched_core_skills,
        )

        # 6. Batch 2: Education Eligibility Assessment (hard filter, separate from score)
        education_eligibility = self.education_eligibility_service.evaluate_eligibility(jd, extraction)

        # Batch 3: Fresher experience bonus: only when NOT_REQUIRED, capped, and not compensating for failed education or missing mandatory skills
        fresher_exp_bonus = 0.0
        if not exp_is_mandatory and exp_eval.fresher_bonus > 0.0:
            edu_ok = education_eligibility.status != EducationEligibilityStatus.INELIGIBLE
            hard_skills_met = True
            if jd.mandatory_skills:
                have_skills = set(parsed_resume.skills)
                hard_skills_met = all(s.lower() in have_skills for s in jd.mandatory_skills)
            if edu_ok and hard_skills_met:
                fresher_exp_bonus = exp_eval.fresher_bonus

        # 7. Experimental Final Score (capped at 100.0)
        experimental_final_score = min(100.0, core_score_percentage + add_bonus + fresher_exp_bonus)

        # Summary of detected sections
        sec_summary = [
            {
                "category": s.category,
                "raw_heading": s.raw_heading,
                "is_core": s.is_core,
                "length_chars": len(s.text),
                "sentence_count": len(s.sentences),
            }
            for s in extraction.sections
        ]

        return SectionScoreResult(
            candidate_id=candidate_id,
            name=parsed_resume.name or "Unknown",
            core_score=round(core_score_percentage, 2),
            additional_bonus=round(add_bonus, 2),
            fresher_experience_bonus=round(fresher_exp_bonus, 2),
            final_score=round(experimental_final_score, 2),
            detailed_section_scores=detailed_section_scores,
            section_scores=section_scores,
            unknown_sections=unknown_sections,
            section_bm25_scores={k: round(v, 4) for k, v in section_bm25_norm.items()},
            section_bm25_raw={k: round(v, 4) for k, v in section_bm25_raw.items()},
            bm25_core_aggregate=round(bm25_aggregate, 4),
            section_semantic_scores={k: round(v, 4) for k, v in section_sem_norm.items()},
            section_semantic_raw={k: round(v, 4) for k, v in section_sem_raw.items()},
            semantic_core_aggregate=round(semantic_aggregate, 4),
            core_dimension_scores={k: round(v, 4) if v is not None else 0.0 for k, v in scores.items()},
            additional_relevance=round(add_rel, 4),
            additional_bonus_breakdown=bonus_breakdown,
            education_eligibility=education_eligibility,
            experience_evaluation=exp_eval,
            skills_evaluation=skills_eval,
            projects_evaluation=projects_eval,
            certifications_evaluation=cert_eval,
            detected_sections_summary=sec_summary,
            notes=notes,
        )

    def score_file(
        self,
        file_path: str,
        jd: JobDescription,
        candidate_id: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> SectionScoreResult:
        """Parses a resume file, segments into sections, and calculates experimental scores."""
        cid = candidate_id or f"cand_{abs(hash(file_path)) % 100000:05d}"
        extraction = self.detector.process_file(file_path)
        return self.score_extraction_result(extraction, cid, jd, weights=weights)

    def score_text(
        self,
        text: str,
        jd: JobDescription,
        candidate_id: str = "cand_custom",
        weights: Optional[Dict[str, float]] = None,
    ) -> SectionScoreResult:
        """Segments raw resume text and calculates experimental scores."""
        extraction = self.detector.detect_sections(text, file_path="inline_text", file_type="txt")
        return self.score_extraction_result(extraction, candidate_id, jd, weights=weights)
