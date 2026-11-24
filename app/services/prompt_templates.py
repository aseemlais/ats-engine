"""
prompt_templates.py
===================
Canonical single source of truth for ATS resume extraction prompts.
Ensures 100% strict alignment across:
  1. Synthetic Dataset Generation (build_dataset.py)
  2. Fine-Tuning (train_unsloth.py via ChatML)
  3. Offline Golden Benchmark Evaluation (evaluate_extractor.py)
  4. Real-time Inference & Serving (extraction_service.py)
"""

from typing import Any, Optional

EXTRACTION_SYSTEM_PROMPT = (
    "You are a professional ATS resume information extraction system.\n"
    "Extract structured candidate details from the provided resume text into strict JSON conforming to this schema:\n"
    "{\n"
    '  "candidateName": "Candidate Full Name",\n'
    '  "email": "email@example.com",\n'
    '  "phone": "+1234567890",\n'
    '  "location": "City, Country",\n'
    '  "jobTitle": "Clean Professional Job Title Only",\n'
    '  "skills": ["Skill 1", "Skill 2"],\n'
    '  "education": [{"degree": "B.Tech/BS/MS", "institution": "Institution Name", "fieldOfStudy": "Field", "year": "2022"}],\n'
    '  "workExperience": [{"title": "Job Title", "company": "Company", "dates": "2020-2023", "durationYears": 3.0, "isCurrent": false, "description": "Key accomplishments"}],\n'
    '  "certifications": ["Certification 1"],\n'
    '  "projects": ["Project Name: Description"]\n'
    "}\n"
    "IMPORTANT RULES:\n"
    "- 'candidateName': Extract the person's actual human name (e.g. 'Shashi Bawa'). NEVER extract job titles, roles, or banners (e.g. 'Java Backend Developer' is a job title, NOT a name).\n"
    "- 'jobTitle': Extract ONLY the professional title (e.g. 'Java Backend Developer' or 'Generative AI Intern'). DO NOT include company names, pipe characters ('|'), locations, or modalities like '(On-site Intern)'.\n"
    "- When a header line contains both a job title and a candidate name (e.g. 'Java Backend Developer    Shashi Bawa'): candidateName MUST be 'Shashi Bawa' and jobTitle MUST be 'Java Backend Developer'.\n"
    "- 'education', 'workExperience', and 'projects' MUST be arrays/lists of objects, NEVER dictionary mappings.\n"
    "- Return ONLY raw minified JSON. Do not include markdown codeblocks (no ```json), explanations, or preambles."
)

# ---------------------------------------------------------------------------
# Shared Token & Character Budgets (Zero Drift across Serving, Train, & Eval)
# ---------------------------------------------------------------------------
# Context window: 4096 tokens (fits in llama.cpp on free-tier 2-vCPU / 16GB RAM)
# Output reserve: 2048 tokens (for rich JSON representation of all fields without truncation)
# System prompt + template overhead: ~150 tokens
# Input budget: 4096 - 2048 - 150 = 1898 tokens (~6,800 chars)
DEFAULT_N_CTX: int = 4096
DEFAULT_MAX_OUTPUT_TOKENS: int = 2048
SYSTEM_PROMPT_TOKEN_RESERVE: int = 150
MAX_INPUT_TOKEN_BUDGET: int = DEFAULT_N_CTX - DEFAULT_MAX_OUTPUT_TOKENS - SYSTEM_PROMPT_TOKEN_RESERVE  # 1898
APPROX_CHARS_PER_TOKEN: float = 3.6
MAX_INPUT_CHARS_BUDGET: int = int(MAX_INPUT_TOKEN_BUDGET * APPROX_CHARS_PER_TOKEN)  # ~6,830 chars


def truncate_resume_text(
    resume_text: str,
    max_tokens: int = MAX_INPUT_TOKEN_BUDGET,
    tokenizer: Optional[Any] = None,
) -> str:
    """
    Truncates raw resume text to fit within the shared input token budget.
    Uses tokenizer if provided, otherwise falls back to character ratio.
    """
    if not resume_text:
        return ""
    if tokenizer is not None:
        tokens = tokenizer.encode(resume_text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            return tokenizer.decode(tokens)
        return resume_text
    # Character budget fallback
    char_budget = int(max_tokens * APPROX_CHARS_PER_TOKEN)
    if len(resume_text) > char_budget:
        return resume_text[:char_budget]
    return resume_text


def build_chatml_prompt(resume_text: str, max_chars: Optional[int] = MAX_INPUT_CHARS_BUDGET) -> str:
    """
    Constructs the exact ChatML prompt used by Qwen2.5-Instruct:
    <|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n
    <|im_start|>user\nRESUME TEXT:\n{text}<|im_end|>\n
    <|im_start|>assistant\n
    """
    text = resume_text[:max_chars] if max_chars is not None else resume_text
    return (
        f"<|im_start|>system\n{EXTRACTION_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nRESUME TEXT:\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
