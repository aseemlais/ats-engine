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

from typing import Optional

EXTRACTION_SYSTEM_PROMPT = (
    "You are a professional ATS resume information extraction system. "
    "Extract structured candidate details from the provided resume text into strict JSON conforming to the schema. "
    "Return ONLY valid minified JSON without markdown code blocks, preamble, or conversational text."
)


def build_chatml_prompt(resume_text: str, max_chars: Optional[int] = None) -> str:
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
