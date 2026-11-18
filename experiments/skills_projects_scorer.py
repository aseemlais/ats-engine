"""
Skills and Projects Scoring Service for Local ATS Scoring Experiment (Batch 4).
Evaluates candidate skills and project relevance with:
- Token-safe matching and configurable aliases (e.g. scikit-learn/sklearn, postgresql/postgres).
- Cross-section evidence extraction (Skills, Experience, Projects, Certifications).
- Strict exclusion of Education section from core skill evidence.
- Full evidence provenance tracking and strict deduplication across sections.
- Distinction between explicitly required vs preferred skills (preferred cannot outweigh missing required).
- Individual project relevance scoring based on descriptions/contributions (not titles alone).
- Duplicate project description detection (repeating the same project yields no extra score).
- Strict segregation of projects from employment duration.
- Conditional certification evaluation requiring direct credential evidence.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from app.models.job import JobDescription
from experiments.experiment_config import (
    CertificationScoringConfig,
    ProjectsScoringConfig,
    SkillsScoringConfig,
)
from experiments.spacy_section_detector import DetectedSection, DocumentExtractionResult


@dataclass
class SkillEvidence:
    """Provenance tracking for a single skill mention in the resume."""
    skill_name: str                     # Canonical skill name
    matched_alias: str                  # Exact alias string matched
    source_section: str                 # Section category (e.g., 'skills', 'experience', 'projects', 'certifications')
    source_heading: str                 # Original section heading
    excerpt: str                        # Concise surrounding context excerpt (~100 chars)
    char_start: int                     # Offset in section
    char_end: int
    has_active_context: bool            # True if accompanied by action verbs or usage details (not bare mention)


@dataclass
class SkillEvaluationItem:
    """Evaluation result for an individual skill requirement."""
    canonical_name: str
    requirement_type: str               # 'REQUIRED', 'PREFERRED', 'SUPPORTING'
    is_matched: bool
    matched_alias: Optional[str] = None
    evidence_items: List[SkillEvidence] = field(default_factory=list)
    has_proficiency_evidence: bool = False
    uncertainty_notes: List[str] = field(default_factory=list)


@dataclass
class SkillsEvaluationResult:
    """Overall evaluation of candidate skills against JD requirements."""
    is_applicable: bool
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    supporting_skills: List[str] = field(default_factory=list)
    matched_required: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    matched_preferred: List[str] = field(default_factory=list)
    missing_preferred: List[str] = field(default_factory=list)
    required_coverage: float = 0.0      # matched_required / total_required
    preferred_coverage: float = 0.0     # matched_preferred / total_preferred
    overall_skill_score: float = 0.0    # Composite score in [0.0, 1.0]
    evidence_provenance: Dict[str, List[SkillEvidence]] = field(default_factory=dict)
    skills_by_section: Dict[str, List[str]] = field(default_factory=dict)
    evaluations: Dict[str, SkillEvaluationItem] = field(default_factory=dict)
    deduplication_warnings: List[str] = field(default_factory=list)
    uncertainty_notes: List[str] = field(default_factory=list)


@dataclass
class ProjectRecord:
    """Structured representation of an individual project."""
    title: str
    description: str
    responsibilities_text: str
    technologies: List[str] = field(default_factory=list)
    char_length: int = 0
    bm25_normalized: float = 0.0
    bm25_raw: float = 0.0
    semantic_cosine: float = 0.0
    semantic_calibrated: float = 0.0
    relevance_score: float = 0.0
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    uncertainty_notes: List[str] = field(default_factory=list)


@dataclass
class ProjectsEvaluationResult:
    """Overall evaluation of candidate projects."""
    is_applicable: bool
    projects: List[ProjectRecord] = field(default_factory=list)
    unique_projects_count: int = 0
    duplicate_projects_count: int = 0
    overall_project_score: float = 0.0  # Composite score in [0.0, 1.0]
    top_project_relevances: List[float] = field(default_factory=list)
    aggregation_explanation: str = ""
    uncertainty_notes: List[str] = field(default_factory=list)


@dataclass
class CertificationEvaluationResult:
    """Evaluation of candidate certifications against JD."""
    requirement_level: str              # 'none', 'preferred', 'mandatory'
    is_applicable: bool
    required_certifications: List[str] = field(default_factory=list)
    matched_certifications: List[str] = field(default_factory=list)
    missing_certifications: List[str] = field(default_factory=list)
    score: float = 0.0                  # Score in [0.0, 1.0]
    has_direct_evidence: bool = False
    evidence_excerpts: List[str] = field(default_factory=list)
    uncertainty_notes: List[str] = field(default_factory=list)


# Active proficiency keywords indicating hands-on usage rather than bare mention
ACTIVE_CONTEXT_PATTERNS = re.compile(
    r"\b(implemented|built|developed|designed|engineered|architected|created|deployed|"
    r"optimized|automated|maintained|configured|managed|programmed|coded|trained|fine-tuned|"
    r"analyzed|evaluated|leveraged|utilized|integrated|spearheaded|led|authored)\b",
    re.IGNORECASE,
)


def build_token_safe_regex(term: str) -> re.Pattern:
    """
    Constructs a token-safe regex pattern for a skill term.
    Avoids false positives from partial substrings (e.g. 'c' matching 'CSS', 'r' matching 'React').
    Properly escapes special characters like 'c++', 'c#', 'node.js', '.net'.
    """
    term_clean = term.strip().lower()
    
    # Specific single-letter or punctuation tokens
    if term_clean == "c":
        # Must not be preceded or followed by alphanumeric or +, #, etc.
        return re.compile(r"(?<![a-zA-Z0-9#+])c(?![a-zA-Z0-9#+])", re.IGNORECASE)
    elif term_clean == "r":
        return re.compile(r"(?<![a-zA-Z0-9])r(?![a-zA-Z0-9])", re.IGNORECASE)
    elif term_clean == "c++":
        return re.compile(r"(?<![a-zA-Z0-9])c\+\+(?![a-zA-Z0-9])", re.IGNORECASE)
    elif term_clean == "c#":
        return re.compile(r"(?<![a-zA-Z0-9])c\#(?![a-zA-Z0-9])", re.IGNORECASE)
    elif term_clean in ("node.js", "nodejs"):
        return re.compile(r"(?<![a-zA-Z0-9])node(?:\.js|js)(?![a-zA-Z0-9])", re.IGNORECASE)
    elif term_clean in ("next.js", "nextjs"):
        return re.compile(r"(?<![a-zA-Z0-9])next(?:\.js|js)(?![a-zA-Z0-9])", re.IGNORECASE)
    elif term_clean in ("vue.js", "vuejs"):
        return re.compile(r"(?<![a-zA-Z0-9])vue(?:\.js|js)?(?![a-zA-Z0-9])", re.IGNORECASE)
    elif term_clean in ("react.js", "reactjs", "react"):
        return re.compile(r"(?<![a-zA-Z0-9])react(?:\.js|js)?(?![a-zA-Z0-9])", re.IGNORECASE)
    elif term_clean == ".net":
        return re.compile(r"(?<![a-zA-Z0-9])\.net(?![a-zA-Z0-9])", re.IGNORECASE)
    elif term_clean in ("go", "golang"):
        return re.compile(r"\b(?:golang|go)\b", re.IGNORECASE)
    else:
        # Standard word boundary regex with whitespace flexibility
        escaped = re.escape(term_clean)
        # Allow spaces in term to match multiple whitespace chars
        escaped = escaped.replace(r"\ ", r"\s+")
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


class SkillsProjectsScorerService:
    """
    Isolated scoring service for candidate Skills and Projects (Batch 4).
    """

    def __init__(
        self,
        skills_config: Optional[SkillsScoringConfig] = None,
        projects_config: Optional[ProjectsScoringConfig] = None,
        cert_config: Optional[CertificationScoringConfig] = None,
    ):
        self.skills_config = skills_config or SkillsScoringConfig()
        self.projects_config = projects_config or ProjectsScoringConfig()
        self.cert_config = cert_config or CertificationScoringConfig()

        # Build reverse alias lookup: variant -> canonical
        self.alias_to_canonical: Dict[str, str] = {}
        for canonical, aliases in self.skills_config.SKILL_ALIASES.items():
            for alias in aliases:
                self.alias_to_canonical[alias.lower()] = canonical.lower()
            self.alias_to_canonical[canonical.lower()] = canonical.lower()

    # -----------------------------------------------------------------------
    # Skill Normalization & Canonicalization
    # -----------------------------------------------------------------------
    def canonicalize_skill(self, skill_name: str) -> str:
        """Returns the canonical skill name if recognized in alias dictionary, otherwise lowercased term."""
        clean = skill_name.strip().lower()
        return self.alias_to_canonical.get(clean, clean)

    def get_skill_aliases(self, canonical_name: str) -> List[str]:
        """Returns all recognized search variants/aliases for a skill."""
        c = canonical_name.strip().lower()
        if c in self.skills_config.SKILL_ALIASES:
            return self.skills_config.SKILL_ALIASES[c]
        # Check reverse lookup
        actual_canonical = self.alias_to_canonical.get(c, c)
        if actual_canonical in self.skills_config.SKILL_ALIASES:
            return self.skills_config.SKILL_ALIASES[actual_canonical]
        return [c]

    # -----------------------------------------------------------------------
    # Skills Scoring Implementation
    # -----------------------------------------------------------------------
    def extract_evidence_for_skill(
        self,
        canonical_skill: str,
        extraction: DocumentExtractionResult,
    ) -> Tuple[bool, Optional[str], List[SkillEvidence], bool]:
        """
        Searches all non-excluded sections for occurrences of the skill or its aliases.
        Strictly excludes sections in EXCLUDED_SECTIONS (e.g. Education).
        Returns: (is_matched, matched_alias, evidence_list, has_active_proficiency)
        """
        aliases = self.get_skill_aliases(canonical_skill)
        evidence_list: List[SkillEvidence] = []
        matched_alias: Optional[str] = None
        has_active_prof = False

        for sec in extraction.sections:
            # STRICT POLICY: Education text is NEVER used as core skill evidence
            if sec.category in self.skills_config.EXCLUDED_SECTIONS:
                continue

            sec_text = sec.text
            if not sec_text or not sec_text.strip():
                continue

            for alias in aliases:
                pattern = build_token_safe_regex(alias)
                for match in pattern.finditer(sec_text):
                    start = match.start()
                    end = match.end()
                    
                    # Extract concise surrounding context (~50 chars before and after)
                    ctx_start = max(0, start - 50)
                    ctx_end = min(len(sec_text), end + 50)
                    excerpt = sec_text[ctx_start:ctx_end].replace("\n", " ").strip()
                    if ctx_start > 0:
                        excerpt = "..." + excerpt
                    if ctx_end < len(sec_text):
                        excerpt = excerpt + "..."

                    # Check for active proficiency signals
                    has_action = bool(ACTIVE_CONTEXT_PATTERNS.search(sec_text[max(0, start - 40):min(len(sec_text), end + 40)]))
                    if has_action or sec.category in ("projects", "experience"):
                        has_active_prof = True

                    evidence_list.append(
                        SkillEvidence(
                            skill_name=canonical_skill,
                            matched_alias=alias,
                            source_section=sec.category,
                            source_heading=sec.raw_heading,
                            excerpt=excerpt,
                            char_start=start,
                            char_end=end,
                            has_active_context=has_action,
                        )
                    )
                    if matched_alias is None:
                        matched_alias = alias

        is_matched = len(evidence_list) > 0
        return is_matched, matched_alias, evidence_list, has_active_prof

    def evaluate_candidate_skills(
        self,
        extraction: DocumentExtractionResult,
        jd: JobDescription,
    ) -> SkillsEvaluationResult:
        """
        Evaluates candidate skills against JD requirements.
        - Required skills vs preferred skills weighted transparently.
        - Deduplicated across all sections (evidence provenance tracked).
        - Education strictly excluded.
        - Missing required skills explicitly exposed and penalize score.
        """
        # 1. Parse JD skill tiers
        req_skills: List[str] = []
        for s in list(jd.mandatory_skills) + list(jd.required_skills):
            c = self.canonicalize_skill(s)
            if c not in req_skills:
                req_skills.append(c)

        pref_skills: List[str] = []
        for s in jd.preferred_skills:
            c = self.canonicalize_skill(s)
            if c not in req_skills and c not in pref_skills:
                pref_skills.append(c)

        supp_skills: List[str] = []
        for s in jd.supporting_skills:
            c = self.canonicalize_skill(s)
            if c not in req_skills and c not in pref_skills and c not in supp_skills:
                supp_skills.append(c)

        is_applicable = bool(req_skills or pref_skills or supp_skills)
        if not is_applicable:
            return SkillsEvaluationResult(
                is_applicable=False,
                overall_skill_score=0.0,
                uncertainty_notes=["No skills specified or extracted in JD; score not manufactured."],
            )

        # 2. Evaluate each skill requirement
        evaluations: Dict[str, SkillEvaluationItem] = {}
        provenance: Dict[str, List[SkillEvidence]] = {}
        skills_by_section: Dict[str, List[str]] = {}
        dedup_warnings: List[str] = []

        matched_req: List[str] = []
        missing_req: List[str] = []
        matched_pref: List[str] = []
        missing_pref: List[str] = []

        # Process Required Skills
        for skill in req_skills:
            matched, alias, ev_list, active_prof = self.extract_evidence_for_skill(skill, extraction)
            item = SkillEvaluationItem(
                canonical_name=skill,
                requirement_type="REQUIRED",
                is_matched=matched,
                matched_alias=alias,
                evidence_items=ev_list,
                has_proficiency_evidence=active_prof,
            )
            evaluations[skill] = item
            if matched:
                matched_req.append(skill)
                provenance[skill] = ev_list
                # Track provenance sections and detect duplication
                sections_found = set(e.source_section for e in ev_list)
                if len(sections_found) > 1:
                    dedup_warnings.append(
                        f"Skill '{skill}' found in multiple sections ({', '.join(sections_found)}); credited exactly once."
                    )
                for e in ev_list:
                    skills_by_section.setdefault(e.source_section, [])
                    if skill not in skills_by_section[e.source_section]:
                        skills_by_section[e.source_section].append(skill)
            else:
                missing_req.append(skill)
                item.uncertainty_notes.append(f"Required skill '{skill}' missing from resume evidence.")

        # Process Preferred Skills
        for skill in pref_skills:
            matched, alias, ev_list, active_prof = self.extract_evidence_for_skill(skill, extraction)
            item = SkillEvaluationItem(
                canonical_name=skill,
                requirement_type="PREFERRED",
                is_matched=matched,
                matched_alias=alias,
                evidence_items=ev_list,
                has_proficiency_evidence=active_prof,
            )
            evaluations[skill] = item
            if matched:
                matched_pref.append(skill)
                provenance[skill] = ev_list
                sections_found = set(e.source_section for e in ev_list)
                if len(sections_found) > 1:
                    dedup_warnings.append(
                        f"Preferred skill '{skill}' found in multiple sections ({', '.join(sections_found)}); credited exactly once."
                    )
                for e in ev_list:
                    skills_by_section.setdefault(e.source_section, [])
                    if skill not in skills_by_section[e.source_section]:
                        skills_by_section[e.source_section].append(skill)
            else:
                missing_pref.append(skill)

        # 3. Calculate Coverages
        req_cov = len(matched_req) / len(req_skills) if req_skills else 1.0
        pref_cov = len(matched_pref) / len(pref_skills) if pref_skills else 0.0

        # 4. Aggregation Formula
        # Required skills are heavily weighted. Preferred skills cannot compensate for missing required skills.
        w_req = self.skills_config.REQUIRED_TIER_WEIGHT
        w_pref = self.skills_config.PREFERRED_TIER_WEIGHT

        if req_skills and pref_skills:
            total_w = w_req + w_pref
            composite_score = (w_req * req_cov + w_pref * pref_cov) / total_w
        elif req_skills:
            composite_score = req_cov
        elif pref_skills:
            # Preferred only: capped at 0.5 without required base
            composite_score = min(0.5, pref_cov * 0.5)
        else:
            composite_score = 0.0

        final_score = min(self.skills_config.MAX_SKILL_SCORE, max(0.0, composite_score))

        uncertainty: List[str] = []
        if missing_req:
            uncertainty.append(f"Candidate missing {len(missing_req)} required skill(s): {', '.join(missing_req)}.")

        return SkillsEvaluationResult(
            is_applicable=is_applicable,
            required_skills=req_skills,
            preferred_skills=pref_skills,
            supporting_skills=supp_skills,
            matched_required=matched_req,
            missing_required=missing_req,
            matched_preferred=matched_pref,
            missing_preferred=missing_pref,
            required_coverage=round(req_cov, 4),
            preferred_coverage=round(pref_cov, 4),
            overall_skill_score=round(final_score, 4),
            evidence_provenance=provenance,
            skills_by_section=skills_by_section,
            evaluations=evaluations,
            deduplication_warnings=dedup_warnings,
            uncertainty_notes=uncertainty,
        )

    # -----------------------------------------------------------------------
    # Projects Scoring Implementation
    # -----------------------------------------------------------------------
    def extract_project_records(
        self,
        extraction: DocumentExtractionResult,
    ) -> List[ProjectRecord]:
        """
        Extracts individual project records from projects sections.
        Uses paragraph and bullet structure to detect project boundaries.
        Academic and personal projects are strictly segregated from employment.
        """
        project_sections = [
            s for s in extraction.sections
            if s.category == "projects" or "project" in s.raw_heading.lower()
        ]
        if not project_sections:
            return []

        records: List[ProjectRecord] = []

        for sec in project_sections:
            raw_text = sec.text.strip()
            if not raw_text:
                continue

            # Split into project blocks based on double newlines or bold title patterns
            blocks = [b.strip() for b in re.split(r"\n\s*\n", raw_text) if b.strip()]
            
            # If no double-newline separation, inspect line-by-line
            if len(blocks) <= 1:
                lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
                current_title = ""
                current_lines: List[str] = []
                for line in lines:
                    # Check if line looks like a project title (short, no leading bullet, not just a tool name)
                    is_title_candidate = (
                        not line.startswith(("-", "*", "•", "–"))
                        and len(line.split()) <= 10
                        and not line.endswith((".", ";"))
                    )
                    if is_title_candidate and current_lines:
                        # Flush previous
                        desc = "\n".join(current_lines)
                        records.append(
                            ProjectRecord(
                                title=current_title or "Untitled Project",
                                description=desc,
                                responsibilities_text=desc,
                                char_length=len(desc),
                            )
                        )
                        current_title = line
                        current_lines = []
                    elif is_title_candidate and not current_lines:
                        current_title = line
                    else:
                        current_lines.append(line)
                if current_lines:
                    desc = "\n".join(current_lines)
                    records.append(
                        ProjectRecord(
                            title=current_title or sec.raw_heading or "Project",
                            description=desc,
                            responsibilities_text=desc,
                            char_length=len(desc),
                        )
                    )
            else:
                for block in blocks:
                    lines = [l.strip() for l in block.splitlines() if l.strip()]
                    title = lines[0] if lines else "Project"
                    # Remove leading bullets from title if any
                    title = re.sub(r"^[-*•–]\s*", "", title)
                    desc = "\n".join(lines[1:]) if len(lines) > 1 else lines[0]
                    records.append(
                        ProjectRecord(
                            title=title,
                            description=desc,
                            responsibilities_text=desc,
                            char_length=len(desc),
                        )
                    )

        return records

    def detect_duplicate_projects(self, projects: List[ProjectRecord]) -> None:
        """
        Detects duplicate or repeated project descriptions using token overlap (Jaccard similarity).
        Flags duplicates so they earn 0.0 additional contribution.
        """
        for i in range(len(projects)):
            p1 = projects[i]
            tokens1 = set(re.findall(r"\b\w{3,}\b", p1.description.lower()))
            if not tokens1:
                continue

            for j in range(i + 1, len(projects)):
                p2 = projects[j]
                if p2.is_duplicate:
                    continue
                tokens2 = set(re.findall(r"\b\w{3,}\b", p2.description.lower()))
                if not tokens2:
                    continue

                intersection = len(tokens1 & tokens2)
                union = len(tokens1 | tokens2)
                sim = intersection / union if union > 0 else 0.0

                if sim >= self.projects_config.DUPLICATE_SIMILARITY_THRESHOLD:
                    p2.is_duplicate = True
                    p2.duplicate_of = p1.title
                    p2.uncertainty_notes.append(
                        f"Repeated project description: {sim:.0%} token similarity with '{p1.title}'. Ignored for scoring."
                    )

    def evaluate_candidate_projects(
        self,
        extraction: DocumentExtractionResult,
        jd: JobDescription,
        jd_query: str,
        jd_vector: Optional[Any],
        bm25_scorer_fn: Callable[[str, str], Tuple[float, float, List[str]]],
        semantic_scorer_fn: Callable[[str, Any], Tuple[float, float]],
    ) -> ProjectsEvaluationResult:
        """
        Evaluates candidate projects based on descriptions and contributions:
        - Scores each project independently using lexical BM25 and semantic similarity.
        - Detects and neutralizes repeated/duplicate project descriptions.
        - Aggregates unique projects using diminishing return weights.
        - Segregated from employment duration.
        """
        raw_projects = self.extract_project_records(extraction)
        if not raw_projects:
            return ProjectsEvaluationResult(
                is_applicable=False,
                overall_project_score=0.0,
                uncertainty_notes=["No projects section or project records detected in resume."],
            )

        # Detect duplicate projects
        self.detect_duplicate_projects(raw_projects)

        # Score each project independently on descriptions/contributions
        w_sem = self.projects_config.PROJECT_SEMANTIC_WEIGHT
        w_lex = self.projects_config.PROJECT_LEXICAL_WEIGHT

        scored_projects: List[ProjectRecord] = []
        for p in raw_projects:
            # Score strictly on responsibilities and description text (not title alone)
            eval_text = p.description.strip()
            if not eval_text or len(eval_text) < 15:
                p.uncertainty_notes.append("Project description too brief for reliable relevance evaluation.")
                p.relevance_score = 0.0
                scored_projects.append(p)
                continue

            bm25_norm, bm25_raw, matched = bm25_scorer_fn(eval_text, jd_query)
            sem_cal, sem_cos = semantic_scorer_fn(eval_text, jd_vector)

            p.bm25_normalized = round(bm25_norm, 4)
            p.bm25_raw = round(bm25_raw, 4)
            p.semantic_cosine = round(sem_cos, 4)
            p.semantic_calibrated = round(sem_cal, 4)

            # Combined relevance
            rel = (w_sem * sem_cal) + (w_lex * bm25_norm)
            p.relevance_score = round(min(1.0, max(0.0, rel)), 4)
            scored_projects.append(p)

        # Filter out duplicates for score aggregation
        unique_projects = [p for p in scored_projects if not p.is_duplicate]
        duplicate_projects = [p for p in scored_projects if p.is_duplicate]

        # Rank unique projects by relevance descending
        unique_projects.sort(key=lambda x: x.relevance_score, reverse=True)

        # Aggregation with diminishing returns
        weights = self.projects_config.TOP_PROJECT_WEIGHTS
        top_relevances = [p.relevance_score for p in unique_projects[:len(weights)]]

        if unique_projects:
            active_weights = weights[:len(unique_projects)]
            total_w = sum(active_weights)
            agg_score = sum(w * rel for w, rel in zip(active_weights, top_relevances)) / total_w if total_w > 0 else 0.0
            explanation = (
                f"Aggregated top {len(active_weights)} unique project(s) with diminishing weights "
                f"({', '.join(f'{w:.0%}' for w in active_weights)}): score = {agg_score:.4f}"
            )
        else:
            agg_score = 0.0
            explanation = "No non-duplicate projects with evaluable descriptions."

        final_proj_score = min(self.projects_config.MAX_PROJECT_SCORE, round(agg_score, 4))

        return ProjectsEvaluationResult(
            is_applicable=True,
            projects=scored_projects,
            unique_projects_count=len(unique_projects),
            duplicate_projects_count=len(duplicate_projects),
            overall_project_score=final_proj_score,
            top_project_relevances=top_relevances,
            aggregation_explanation=explanation,
        )

    # -----------------------------------------------------------------------
    # Certifications Scoring Implementation
    # -----------------------------------------------------------------------
    def evaluate_certifications(
        self,
        extraction: DocumentExtractionResult,
        jd: JobDescription,
        existing_cert_score: float = 0.0,
        req_level: str = "none",
    ) -> CertificationEvaluationResult:
        """
        Evaluates certifications conditionally on JD:
        - If JD does not require/value certifications ('none'): score is neutral (no penalty).
        - Direct credential evidence required; bare technology mention is NOT proof of certification.
        - Deduplicated from core skill evidence.
        """
        is_applicable = req_level in ("mandatory", "preferred")
        
        # If JD has no certification requirement, return neutral result without penalty
        if not is_applicable:
            return CertificationEvaluationResult(
                requirement_level="none",
                is_applicable=False,
                score=0.0,
                uncertainty_notes=["JD does not require or value certifications; no penalty applied."],
            )

        cert_sections = [
            s for s in extraction.sections
            if s.category == "certifications" or "certif" in s.raw_heading.lower()
        ]
        
        evidence_excerpts: List[str] = []
        has_direct_evidence = False

        # Check for credential keywords in certification sections or full text
        cred_keywords = self.cert_config.CREDENTIAL_KEYWORDS
        pattern = re.compile(rf"\b({'|'.join(cred_keywords)})\b", re.IGNORECASE)

        for sec in cert_sections:
            if pattern.search(sec.text):
                has_direct_evidence = True
                evidence_excerpts.append(sec.text.strip()[:200])

        score = existing_cert_score if has_direct_evidence else (existing_cert_score * 0.5)

        return CertificationEvaluationResult(
            requirement_level=req_level,
            is_applicable=is_applicable,
            score=round(score, 4),
            has_direct_evidence=has_direct_evidence,
            evidence_excerpts=evidence_excerpts,
            uncertainty_notes=[] if has_direct_evidence else ["No direct credential/license evidence found for certification."],
        )
