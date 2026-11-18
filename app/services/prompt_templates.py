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
    "You are a professional ATS resume information extraction system. "
    "Extract structured candidate details from the provided resume text into strict JSON conforming to the schema. "
    "Return ONLY valid minified JSON without markdown code blocks, preamble, or conversational text."
)

# ---------------------------------------------------------------------------
# Shared Token & Character Budgets (Zero Drift across Serving, Train, & Eval)
# ---------------------------------------------------------------------------
# Context window: 4096 tokens (fits in llama.cpp on free-tier 2-vCPU / 16GB RAM)
# Output reserve: 1000 tokens (for JSON representation of all fields)
# System prompt + template overhead: ~96 tokens
# Input budget: 4096 - 1000 - 96 = 3000 tokens
DEFAULT_N_CTX: int = 4096
DEFAULT_MAX_OUTPUT_TOKENS: int = 1000
SYSTEM_PROMPT_TOKEN_RESERVE: int = 96
MAX_INPUT_TOKEN_BUDGET: int = DEFAULT_N_CTX - DEFAULT_MAX_OUTPUT_TOKENS - SYSTEM_PROMPT_TOKEN_RESERVE  # 3000
APPROX_CHARS_PER_TOKEN: float = 3.6
MAX_INPUT_CHARS_BUDGET: int = int(MAX_INPUT_TOKEN_BUDGET * APPROX_CHARS_PER_TOKEN)  # 10,800 chars


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
