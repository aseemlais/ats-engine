"""
Configuration for Local spaCy Section-Based ATS Scoring Experiment.
This file is completely isolated and does not alter production ATS settings.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set

@dataclass
class SectionDetectionConfig:
    """Configurable heading patterns and mappings for resume section classification."""
    
    # Mapping of canonical section category to list of common header aliases/variations (case-insensitive)
    HEADING_MAP: Dict[str, List[str]] = field(default_factory=lambda: {
        "summary": [
            "summary", "professional summary", "career summary", "executive summary",
            "profile", "professional profile", "career profile", "personal profile",
            "about me", "about", "bio", "overview", "career objective", "objective",
            "statement", "summary of qualifications"
        ],
        "skills": [
            "skills", "technical skills", "core competencies", "key skills",
            "technologies", "tools", "tools & technologies", "competencies",
            "technical proficiencies", "technical expertise", "skills & tools",
            "areas of expertise", "programming languages", "tech stack",
            "hard skills", "soft skills", "skill set", "technical arsenal", "arsenal"
        ],
        "experience": [
            "experience", "work experience", "professional experience",
            "employment history", "work history", "career history",
            "professional background", "relevant experience", "industry experience",
            "internships", "internship", "internship experience", "practical experience",
            "where i have worked", "where i worked", "where i've worked"
        ],
        "education": [
            "education", "academic background", "academic qualifications",
            "educational background", "qualifications", "academics",
            "academic history", "education & training", "degrees", "degrees & credentials"
        ],
        "projects": [
            "projects", "academic projects", "technical projects", "key projects",
            "personal projects", "portfolio projects", "selected projects",
            "project work", "capstone projects", "notable projects"
        ],
        "certifications": [
            "certifications", "certificates", "licenses & certifications",
            "licenses", "professional certifications", "credentials",
            "accreditations", "training & certifications"
        ],
        "achievements": [
            "achievements", "honors & awards", "awards", "honors",
            "key achievements", "accomplishments", "recognition", "accolades"
        ],
        "publications": [
            "publications", "research", "research publications", "papers",
            "patents", "presentations", "conference proceedings"
        ],
        "leadership": [
            "leadership", "leadership experience", "extracurricular",
            "extracurricular activities", "volunteering", "volunteer experience",
            "community service", "activities", "affiliations"
        ],
        "additional": [
            "additional information", "miscellaneous", "other details",
            "languages", "known languages", "personal details", "interests", "hobbies"
        ]
    })
    
    # Core categories that map into the existing ATS dimensions
    CORE_CATEGORIES: Set[str] = field(default_factory=lambda: {
        "skills", "experience", "education", "projects", "certifications"
    })
    
    # Categories considered 'additional' that can contribute to the additional relevance bonus
    ADDITIONAL_CATEGORIES: Set[str] = field(default_factory=lambda: {
        "achievements", "publications", "leadership", "additional", "unknown"
    })
    
    # Max words on a line to be considered a possible section heading
    MAX_HEADING_WORDS: int = 7
    # Minimum characters in heading
    MIN_HEADING_CHARS: int = 3
    # Use spaCy sentencizer and POS tagger for sentence splitting & heading detection
    SPACY_MODEL: str = "en_core_web_sm"

@dataclass
class SectionScoringConfig:
    """Configurable weights and thresholds for section-level scoring and bonus."""
    
    # DEPRECATED (Batch 1): Cross-section percentage weights (e.g. Skills 40%, Experience 30%,
    # Projects 20%, Education 10%) are strictly prohibited from aggregating section-level BM25
    # and semantic scores. Every section maintains its own independent lexical and semantic scores.
    # Kept as empty dictionaries for backwards compatibility.
    SECTION_BM25_WEIGHTS: Dict[str, float] = field(default_factory=dict)
    SECTION_SEMANTIC_WEIGHTS: Dict[str, float] = field(default_factory=dict)
    
    # Additional-section relevance bonus parameters:
    # Experimental formula: ExperimentalFinalScore = min(100.0, CoreScore + (beta * AdditionalRelevance))
    # beta represents the maximum bonus in score points (default: 3.0, beta=0.0 disables bonus)
    ENABLE_ADDITIONAL_BONUS: bool = True
    BONUS_BETA: float = 3.0
    
    # Minimum semantic/lexical similarity required for additional evidence to be considered relevant
    # Prevents keyword stuffing or irrelevant text from triggering the bonus
    ADDITIONAL_RELEVANCE_THRESHOLD: float = 0.20
    
    # Semantic calibration floor and ceiling (matching production settings)
    SEMANTIC_COSINE_FLOOR: float = 0.15
    SEMANTIC_COSINE_CEIL: float = 0.70
    
    # Weight blend for additional relevance: 60% semantic similarity + 40% contextual lexical overlap
    ADDITIONAL_SEMANTIC_WEIGHT: float = 0.60
    ADDITIONAL_LEXICAL_WEIGHT: float = 0.40


@dataclass
class ExperienceScoringConfig:
    """Configurable weights, multipliers, and thresholds for experience scoring (Batch 3)."""
    # Type multipliers
    MULTIPLIER_PROFESSIONAL: float = 1.0
    MULTIPLIER_INTERNSHIP: float = 0.4
    MULTIPLIER_OTHER: float = 0.7
    MULTIPLIER_UNKNOWN: float = 0.5  # Conservative treatment when type is uncertain
    
    # Duties relevance weight blend: 60% semantic + 40% lexical
    DUTIES_SEMANTIC_WEIGHT: float = 0.60
    DUTIES_LEXICAL_WEIGHT: float = 0.40
    
    # Max duration ratio cap
    MAX_DURATION_RATIO: float = 1.0
    
    # Max experience score cap
    MAX_EXPERIENCE_SCORE: float = 1.0
    
    # Fresher experience bonus when JD experience is NOT_REQUIRED
    ENABLE_FRESHER_EXPERIENCE_BONUS: bool = True
    FRESHER_EXPERIENCE_BONUS_CAP: float = 2.0  # Max points added to experimental final score
    
    # Reference evaluation year and month for current/present roles
    REFERENCE_YEAR: int = 2026
    REFERENCE_MONTH: int = 10


@dataclass
class SkillsScoringConfig:
    """Configurable weights, tier ratios, and alias dictionary for skills scoring (Batch 4)."""
    # Tier importance weights
    REQUIRED_TIER_WEIGHT: float = 0.80
    PREFERRED_TIER_WEIGHT: float = 0.20
    SUPPORTING_TIER_WEIGHT: float = 0.10
    
    # Maximum skill score cap
    MAX_SKILL_SCORE: float = 1.0
    
    # Policy: Preferred skills cannot compensate for missing required skills
    ENFORCE_REQUIRED_PRIORITY: bool = True
    
    # Excluded sections for core skill evidence (Education is strictly excluded)
    EXCLUDED_SECTIONS: Set[str] = field(default_factory=lambda: {"education"})
    
    # Canonical skill -> list of aliases / variants (token-safe matching applied)
    SKILL_ALIASES: Dict[str, List[str]] = field(default_factory=lambda: {
        "scikit-learn": ["scikit-learn", "sklearn"],
        "postgresql": ["postgresql", "postgres", "psql"],
        "natural language processing": ["natural language processing", "nlp", "text processing"],
        "machine learning": ["machine learning", "ml", "statistical learning"],
        "deep learning": ["deep learning", "dl", "neural networks"],
        "docker": ["docker", "containerization", "containers"],
        "kubernetes": ["kubernetes", "k8s"],
        "react": ["react", "react.js", "reactjs"],
        "node.js": ["node.js", "nodejs"],
        "c++": ["c++", "cpp"],
        "c#": ["c#", "csharp", "c sharp"],
        "golang": ["golang", "go language"],
        "c": ["c language", "ansi c"],
        "r": ["r language", "r programming"],
        "aws": ["aws", "amazon web services"],
        "gcp": ["gcp", "google cloud", "google cloud platform"],
        "azure": ["azure", "microsoft azure"],
        "power bi": ["power bi", "powerbi"],
        "tableau": ["tableau"],
        "snowflake": ["snowflake"],
        "spark": ["spark", "apache spark", "pyspark"],
        "tensorflow": ["tensorflow", "tf", "keras"],
        "pytorch": ["pytorch", "torch"],
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "git": ["git", "github", "gitlab"],
        "sql": ["sql", "structured query language"],
        "python": ["python", "python3"],
        "java": ["java", "core java"],
        "linux": ["linux", "unix", "ubuntu"],
        "ci/cd": ["ci/cd", "continuous integration", "continuous deployment"],
    })


@dataclass
class ProjectsScoringConfig:
    """Configurable weights, aggregation decay, and duplication thresholds for projects scoring (Batch 4)."""
    # Relevance blend for each individual project: 60% semantic + 40% BM25
    PROJECT_SEMANTIC_WEIGHT: float = 0.60
    PROJECT_LEXICAL_WEIGHT: float = 0.40
    
    # Cap for single project and overall project score
    MAX_PROJECT_SCORE: float = 1.0
    
    # Jaccard token overlap threshold above which two projects are treated as duplicate descriptions
    DUPLICATE_SIMILARITY_THRESHOLD: float = 0.75
    
    # Diminishing return weights for top unique projects (best project 50%, 2nd 30%, 3rd 20%)
    TOP_PROJECT_WEIGHTS: List[float] = field(default_factory=lambda: [0.50, 0.30, 0.20])
    
    # Maximum distinct projects evaluated
    MAX_PROJECTS_EVALUATED: int = 5


@dataclass
class CertificationScoringConfig:
    """Configurable settings for certification evaluation (Batch 4)."""
    # Penalize absent certifications only when JD explicitly requires them
    PENALIZE_ABSENT_WHEN_NOT_REQUIRED: bool = False
    
    # Explicit credential keywords required to prove certification earned
    CREDENTIAL_KEYWORDS: List[str] = field(default_factory=lambda: [
        "certified", "certification", "credential", "license", "passed exam", "cert"
    ])

