import random
import re
from typing import List

BULLET_GLYPHS = ["•", "▪", "●", "‣", "⁃", "◦", "*", "–", "—", ">"]

HEADER_FOOTER_TEMPLATES = [
    "Page {page} of {total_pages}",
    "CONFIDENTIAL - {name}",
    "Curriculum Vitae | {name}",
    "Generated via TalentPortal | Page {page}",
    "References available upon request",
    "Candidate Dossier - Strict Confidentiality",
]

MOJIBAKE_CHARS = ["\ufffd", "\u200b", "\u200e", "\u00a0", "â€™", "â€œ", "â€\x9d", "â€“"]


def apply_midword_line_wraps(text: str, prob: float = 0.08) -> str:
    """
    Simulates PDF layout-induced mid-word line wraps with hyphens,
    e.g. 'Software' -> 'Soft-\nware', 'Developer' -> 'Devel-\noper'.
    """
    words = text.split(" ")
    augmented_words = []
    for word in words:
        if len(word) >= 7 and word.isalpha() and random.random() < prob:
            split_idx = random.randint(3, len(word) - 3)
            hyphenated = word[:split_idx] + "-\n" + word[split_idx:]
            augmented_words.append(hyphenated)
        else:
            augmented_words.append(word)
    return " ".join(augmented_words)


def apply_bullet_glyph_variants(text: str) -> str:
    """
    Replaces standard bullets with diverse unicode glyph variants found in messy PDFs.
    """
    lines = text.splitlines()
    augmented_lines = []
    current_bullet = random.choice(BULLET_GLYPHS)
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("-", "*", "•", "–")):
            new_bullet = current_bullet if random.random() < 0.85 else random.choice(BULLET_GLYPHS)
            augmented_lines.append(re.sub(r"^[-*•–]\s*", f"{new_bullet} ", line))
        else:
            augmented_lines.append(line)
    return "\n".join(augmented_lines)


def inject_header_footer(text: str, candidate_name: str) -> str:
    """
    Injects realistic headers/footers and page markers into multi-page text flows.
    """
    lines = text.splitlines()
    if len(lines) < 15:
        return text

    split_point = len(lines) // 2
    header_footer = random.choice(HEADER_FOOTER_TEMPLATES).format(
        page=random.choice([1, 2]),
        total_pages=2,
        name=candidate_name or "Candidate",
    )

    page_break = f"\n\n--- {header_footer} ---\n\n"
    return "\n".join(lines[:split_point]) + page_break + "\n".join(lines[split_point:])


def inject_unicode_mojibake(text: str, prob: float = 0.04) -> str:
    """
    Injects common PDF font decoding artifacts and mojibake characters.
    """
    chars = list(text)
    for i in range(len(chars)):
        if chars[i] == " " and random.random() < prob:
            chars[i] = random.choice(MOJIBAKE_CHARS)
    return "".join(chars)


def augment_resume_text(text: str, candidate_name: str = "", p_augment: float = 0.85) -> str:
    """
    Full adversarial augmentation pipeline applying realistic layout noise.
    """
    if random.random() > p_augment:
        return text

    augmented = text

    # 1. Random bullet glyph variants
    if random.random() < 0.7:
        augmented = apply_bullet_glyph_variants(augmented)

    # 2. Mid-word line wraps
    if random.random() < 0.5:
        augmented = apply_midword_line_wraps(augmented, prob=0.04)

    # 3. Header/footer page numbering
    if random.random() < 0.6:
        augmented = inject_header_footer(augmented, candidate_name)

    # 4. Mojibake/replacement artifacts
    if random.random() < 0.4:
        augmented = inject_unicode_mojibake(augmented, prob=0.02)

    return augmented
