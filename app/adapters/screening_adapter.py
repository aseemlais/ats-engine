import re
import json
import logging
from typing import Dict, Any, List

from app.models.resume import ParsedResume
from app.models.job import JobDescription
from app.services.nlp_extractor import NLPExtractorService
from app.services.jd_parser import JDParserService
from app.services.hybrid_ranker import HybridRankerService
from app.services.embedding_service import EmbeddingService, EmbeddingUnavailableError
from app.services.hard_filter import HardFilterService
from app.services.shortlist_policy import recommend_single
from app.config import config

logger = logging.getLogger(__name__)

class ATSScreeningAdapter:
    """
    Adapter bridging the conversational ResponsesAgent interface and NewChatPage.tsx
    to the deterministic Hybrid Semantic ATS pipeline without changing any existing ATS algorithms.
    """
    def __init__(
        self,
        nlp_extractor: NLPExtractorService | None = None,
        jd_parser: JDParserService | None = None,
        hybrid_ranker: HybridRankerService | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        self.nlp_extractor = nlp_extractor or NLPExtractorService()
        self.jd_parser = jd_parser or JDParserService()
        self.hybrid_ranker = hybrid_ranker or HybridRankerService()
        self.embedding_service = embedding_service or EmbeddingService()

    def parse_screening_prompt(self, prompt: str) -> Dict[str, str]:
        """
        Extracts JOB TITLE, JOB DESCRIPTION, and CANDIDATE RESUME from prompt sent by NewChatPage.tsx.
        """
        job_title = ""
        job_description = ""
        resume_text = ""

        # Match JOB TITLE
        m_title = re.search(r"JOB TITLE:\s*\n*(.*?)(?=\n+(?:JOB DESCRIPTION|JD|CANDIDATE RESUME|RESUME)|\Z)", prompt, re.DOTALL | re.I)
        if m_title:
            job_title = m_title.group(1).strip()

        # Match JOB DESCRIPTION
        m_jd = re.search(r"(?:JOB DESCRIPTION AND REQUIREMENTS|JOB REQUIREMENTS|JOB DESCRIPTION|JD):\s*\n*(.*?)(?=\n+(?:CANDIDATE RESUME|RESUME)|\Z)", prompt, re.DOTALL | re.I)
        if m_jd:
            job_description = m_jd.group(1).strip()

        # Match CANDIDATE RESUME
        m_resume = re.search(r"(?:CANDIDATE RESUME|RESUME):\s*\n*(.*?)(?=\n+Your task:|\n+Create questions|\Z)", prompt, re.DOTALL | re.I)
        if m_resume:
            resume_text = m_resume.group(1).strip()

        return {
            "job_title": job_title,
            "job_description": job_description or prompt,
            "resume_text": resume_text
        }

    def evaluate_candidate(self, job_title: str, job_description: str, resume_text: str) -> Dict[str, Any]:
        """
        Runs the candidate through the complete deterministic ATS pipeline:
        1. NLP extraction & section segmentation
        2. JD parsing (required, preferred skills, experience threshold)
        3. Hard filtering
        4. BM25 lexical match
        5. Section-level semantic embedding similarity
        6. Skill overlap & alias normalization
        7. Experience & role matching
        8. Certification scoring
        9. Composite score calculation
        """
        # Step 1: Parse candidate resume deterministically
        candidate_id = "CAND_SCREEN"
        parsed_resume = self.nlp_extractor.extract_structured(
            candidate_id=candidate_id,
            source_file="uploaded_resume",
            file_hash="hash_tmp",
            raw_text=resume_text,
            cleaned_text=resume_text
        )

        # Step 2: Parse Job Description
        full_jd_text = f"Job Title: {job_title}\n\n{job_description}" if job_title else job_description
        parsed_jd = self.jd_parser.parse_job_description(full_jd_text, job_id="JD_SCREEN")

        # Step 3: Compute candidate vectors
        try:
            resume_vectors = self.embedding_service.compute_resume_vectors(parsed_resume)
            vec_map = {candidate_id: resume_vectors}
        except EmbeddingUnavailableError:
            if not config.allow_degraded_semantic:
                raise                      # fail loudly: no silent semantic score
            vec_map = {}                   # explicit degraded mode: ranker flags the result

        # Step 4: Run through HybridRankerService
        # Keep BM25 sync on this candidate

        # In interactive 1-by-1 screening, evaluate all scoring dimensions (no hard rejection drops)
        prev_hf = self.hybrid_ranker.hard_filter_service.enabled
        try:
            self.hybrid_ranker.hard_filter_service.enabled = False
            search_response = self.hybrid_ranker.rank_candidates(
                resumes=[parsed_resume],
                resume_vectors_map=vec_map,
                jd=parsed_jd,
                top_k=1
            )
        finally:
            self.hybrid_ranker.hard_filter_service.enabled = prev_hf

        if search_response.results:
            breakdown = search_response.results[0]
            decision_score = breakdown.raw_ats_score if breakdown.raw_ats_score is not None else breakdown.final_score  # unrounded: all decisions
            overall_score = round(decision_score, 1)         # display value
            matched_skills = breakdown.matched_required_skills + breakdown.matched_preferred_skills
            skill_gaps = breakdown.missing_required_skills
        else:
            # Candidate was filtered out by hard pre-filters (e.g. strict mandatory requirements)
            decision_score = overall_score = 0.0
            matched_skills = []
            skill_gaps = parsed_jd.mandatory_skills or parsed_jd.required_skills

        # Eligibility (mandatory requirements) is evaluated separately from the score; the ranker ran with
        # the filter off above so that every dimension is scored.
        _hf = HardFilterService(enabled=True, strict=config.strict_hard_filter)
        eligible, elig_reason = _hf.passes_filter(parsed_resume, parsed_jd)
        elig_report = _hf.assess(parsed_resume, parsed_jd)
        # Fit / recommendation from the configurable thresholds, using the UNROUNDED score
        # (e.g. 59.96 must not become 60.0 -> Shortlisted).
        fit, recommendation = recommend_single(decision_score, eligible)

        # Format experience duration string
        work_exp = parsed_resume.work_experience_years
        intern_exp = parsed_resume.internship_years
        if work_exp > 0 and intern_exp > 0:
            exp_str = f"{work_exp:.1f} yrs full-time, {intern_exp:.1f} yrs internship"
        elif work_exp > 0:
            exp_str = f"{work_exp:.1f} years"
        elif intern_exp > 0:
            exp_str = f"{intern_exp:.1f} years internship"
        else:
            exp_str = f"{parsed_resume.experience_years:.1f} years"

        # Education string
        edu_list = [f"{e.degree} ({e.tier.title()})" for e in parsed_resume.education]
        edu_str = ", ".join(edu_list) if edu_list else "Not specified"

        # Build strengths & concerns deterministically
        strengths = []
        if matched_skills:
            strengths.append(f"Demonstrates strong alignment in key skills: {', '.join(matched_skills[:5])}")
        if work_exp >= parsed_jd.minimum_experience_years and parsed_jd.minimum_experience_years > 0:
            strengths.append(f"Meets or exceeds minimum required experience ({work_exp:.1f} yrs vs {parsed_jd.minimum_experience_years:.1f} yrs required)")
        if parsed_resume.certifications:
            strengths.append(f"Holds relevant verified certifications: {', '.join(parsed_resume.certifications[:3])}")
        if not strengths:
            strengths.append("Foundational technical background present in resume")

        concerns = []
        if skill_gaps:
            concerns.append(f"Missing key required skills from JD: {', '.join(skill_gaps[:4])}")
        if work_exp < parsed_jd.minimum_experience_years and parsed_jd.minimum_experience_years > 0:
            concerns.append(f"Experience below required threshold ({work_exp:.1f} yrs vs {parsed_jd.minimum_experience_years:.1f} yrs required)")
        if not concerns:
            concerns.append("No critical qualifications or experience gaps identified")

        # Recommendation Reason
        if recommendation == "Shortlisted":
            reason = f"Candidate achieved a strong hybrid ATS score of {overall_score:.1f}% matching {len(matched_skills)} required skills with verified background."
        elif recommendation == "Review":
            reason = f"Candidate ATS score is {overall_score:.1f}% (eligible). Demonstrates foundational alignment ({len(matched_skills)} matched skills) and is recommended for hiring team review."
        elif not eligible:
            reason = f"Candidate ATS score is {overall_score:.1f}%. Not eligible: {elig_reason}."
        else:
            reason = f"Candidate ATS score is {overall_score:.1f}%, below the review threshold of {config.review_threshold:.0f}%. Mandatory-requirement check: passed."

        # Summary
        summary = (
            f"Candidate: {parsed_resume.name}. Total effective experience: {parsed_resume.experience_years:.1f} yrs. "
            f"Education: {edu_str}. Matched {len(matched_skills)} skills against target job description. "
            f"Hybrid retrieval composite score: {overall_score:.1f}%."
        )

        # Relevant experience snippet
        relevant_exp = [l.strip("- ") for l in parsed_resume.section_texts.get("experience", "").splitlines() if len(l.strip()) > 10][:4]
        if not relevant_exp:
            relevant_exp = [l.strip("- ") for l in parsed_resume.section_texts.get("internship", "").splitlines() if len(l.strip()) > 10][:4]

        # Structure payload strictly adhering to ScreeningResult expected by NewChatPage.tsx
        return {
            "candidateName": parsed_resume.name or "Not available",
            "email": parsed_resume.email or "Not available",
            "phone": parsed_resume.phone or "Not available",
            "currentRole": parsed_resume.job_titles[0] if parsed_resume.job_titles else (job_title or "Not available"),
            "experience": exp_str,
            "skills": parsed_resume.skills,
            "educationDetails": edu_str,
            "atsScore": round(overall_score),
            "overallScore": round(overall_score, 1),
            "fit": fit,
            "recommendation": recommendation,
            "recommendationReason": reason,
            "summary": summary,
            "matchedSkills": matched_skills,
            "relevantExperience": relevant_exp,
            "education": edu_list,
            "skillGaps": skill_gaps,
            "strengths": strengths,
            "concerns": concerns,
            # Added, backward-compatible fields
            "rawAtsScore": round(decision_score, 2),
            "eligibility": "Eligible" if eligible else "Not Eligible",
            "eligibilityReason": elig_reason,
            "mandatoryRequirementsMet": elig_report.get("mandatory_requirements_met", eligible),
            "missingMandatorySkills": elig_report.get("missing_mandatory", []),
            "acceptedAlternatives": breakdown.accepted_alternatives if search_response.results else [],
            "degraded": breakdown.degraded if search_response.results else False,
            "thresholds": {"shortlist": config.shortlist_threshold, "review": config.review_threshold},
            "scoreBreakdown": (
                {
                    "semantic": breakdown.semantic_score, "bm25": breakdown.bm25_score,
                    "skill": breakdown.skill_score, "experience": breakdown.experience_score,
                    "certification": breakdown.certification_score,
                    "certificationEffective": breakdown.certification_effective_score,
                    "certificationRequirement": breakdown.certification_level,
                    "weights": search_response.weights_used,
                    "components": breakdown.components,
                    "experienceApplicable": breakdown.experience_applicable,
                    "scoringMode": breakdown.scoring_mode,
                    "degraded": breakdown.degraded,
                } if search_response.results else None
            ),
        }

    def generate_interview_questions(self, job_title: str, job_description: str, resume_text: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Generates deterministic, resume-tailored interview questions based on matched skills and candidate experience.
        """
        parsed_resume = self.nlp_extractor.extract_structured(
            candidate_id="CAND_INT",
            source_file="uploaded_resume",
            file_hash="hash_tmp",
            raw_text=resume_text,
            cleaned_text=resume_text
        )
        skills = parsed_resume.skills[:5] if parsed_resume.skills else ["Python", "Problem Solving"]
        
        tech_q = []
        for s in skills[:3]:
            tech_q.append({
                "question": f"Can you detail a production project where you utilized {s.title()} and how you optimized its implementation?",
                "expectedAnswer": f"Candidate should articulate system design, trade-offs, and practical hands-on experience using {s.title()}.",
                "evaluationCriteria": f"Assess depth of expertise in {s.title()}, architecture decisions, and code quality considerations."
            })

        behavioral_q = [
            {
                "question": "Describe a situation where a technical project fell behind schedule or faced unexpected requirements. How did you resolve it?",
                "expectedAnswer": "Structured response explaining the conflict, stakeholder communication, prioritization, and positive resolution.",
                "evaluationCriteria": "Evaluates problem-solving agility, ownership, communication, and emotional intelligence."
            },
            {
                "question": "Give an example of how you collaborated with cross-functional team members to deliver a feature.",
                "expectedAnswer": "Clear articulation of team alignment, division of labor, and feedback integration.",
                "evaluationCriteria": "Assess teamwork, empathy, and ability to bridge technical and business objectives."
            }
        ]

        domain_q = [
            {
                "question": f"How does your prior experience directly prepare you for the challenges of the {job_title or 'target'} role?",
                "expectedAnswer": "Concrete examples mapping past project achievements to the core requirements of this position.",
                "evaluationCriteria": "Evaluates candidate domain familiarity, role readiness, and proactive preparation."
            }
        ]

        return {
            "technical": tech_q,
            "behavioral": behavioral_q,
            "domain": domain_q
        }
