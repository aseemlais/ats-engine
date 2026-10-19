from typing import Tuple
from app.models.resume import ParsedResume
from app.models.job import JobDescription
from app.utils.degree_normalizer import degree_rank, is_degree_satisfied

class EducationMatcherService:
    def evaluate_education(self, resume: ParsedResume, jd: JobDescription) -> Tuple[float, str]:
        """
        Evaluates education requirement satisfaction based on canonical hierarchy.
        Returns: (education_score, highest_tier)
        """
        req_degrees = jd.education_requirements
        if not req_degrees:
            return 1.0, "any"

        highest_cand_tier = "none"
        highest_cand_rank = 0
        for deg in resume.degrees:
            rank = degree_rank(deg)
            if rank > highest_cand_rank:
                highest_cand_rank = rank
                highest_cand_tier = deg

        # Check against each required degree requirement
        req_rank = min(degree_rank(d) for d in req_degrees)  # any listed degree is accepted
        if highest_cand_rank >= req_rank:
            return 1.0, highest_cand_tier
        elif highest_cand_rank > 0:
            # Partial degree match
            return max(0.5, highest_cand_rank / req_rank), highest_cand_tier
        else:
            return 0.3, "none"
