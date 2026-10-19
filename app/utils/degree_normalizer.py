import re
from typing import Dict, List, Tuple

DEGREE_LEVELS: Dict[str, int] = {
    "doctorate": 4,
    "master": 3,
    "bachelor": 2,
    "associate": 1,
    "diploma": 1,
    "none": 0
}

DEGREE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(ph\.?d|doctorate|doctor of philosophy)\b", re.I), "doctorate"),
    (re.compile(r"\b(master(?:'?s)?|m\.?s(?:c)?|m\.?tech|m\.?e|mca|mba|m\.?b\.?a|m\.?phil|m\.?eng)\b", re.I), "master"),
    (re.compile(r"\b(bachelor(?:'?s)?|b\.?tech|b\.?e|b\.?s(?:c)?|bca|bba|b\.?com|b\.?eng)\b", re.I), "bachelor"),
    (re.compile(r"\b(associate(?:'?s)?|a\.?s|a\.?a|diploma)\b", re.I), "associate")
]

def normalize_degree(degree_str: str) -> str:
    """Classifies a degree string into a canonical tier: doctorate, master, bachelor, associate, or none."""
    if not degree_str:
        return "none"
    for pattern, tier in DEGREE_PATTERNS:
        if pattern.search(degree_str):
            return tier
    return "none"

def degree_rank(degree_str: str) -> int:
    """Returns numerical hierarchy rank of a degree tier."""
    tier = normalize_degree(degree_str)
    return DEGREE_LEVELS.get(tier, 0)

def is_degree_satisfied(candidate_degrees: List[str], required_degree: str) -> bool:
    """Checks if any candidate degree meets or exceeds the required degree tier."""
    req_rank = degree_rank(required_degree)
    if req_rank == 0:
        return True
    return any(degree_rank(deg) >= req_rank for deg in candidate_degrees)
