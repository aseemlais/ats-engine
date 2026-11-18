"""
spaCy-Powered Layout-Aware Section Detector for Resume Preprocessing.
Extracts text from PDF/DOCX/TXT files, preprocesses with spaCy, and classifies
sections using a configurable, alias-driven heading recognition engine.
"""
import re
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import spacy
from spacy.language import Language

from experiments.experiment_config import SectionDetectionConfig

@dataclass
class DetectedSection:
    """Represents a classified resume section."""
    category: str              # Normalized category (e.g. 'skills', 'experience', 'projects', 'unknown')
    raw_heading: str           # The actual heading text in the document
    text: str                  # The section body text
    char_start: int            # Start character offset in full text
    char_end: int              # End character offset in full text
    line_start: int            # Starting line number
    sentences: List[str] = field(default_factory=list) # Extracted sentences via spaCy
    is_core: bool = False      # Whether it belongs to the 5 core ATS dimensions


@dataclass
class DocumentExtractionResult:
    """Container for document extraction and section classification output."""
    file_path: str
    file_type: str
    raw_text: str
    cleaned_text: str
    sections: List[DetectedSection] = field(default_factory=list)
    section_map: Dict[str, str] = field(default_factory=dict) # category -> concatenated text
    success: bool = True
    error_message: Optional[str] = None
    extraction_notes: List[str] = field(default_factory=list)


class SpacySectionDetector:
    """Preprocesses resumes and segments them into structured sections using spaCy and heading taxonomy."""
    
    def __init__(self, config: Optional[SectionDetectionConfig] = None):
        self.config = config or SectionDetectionConfig()
        self.nlp = self._load_spacy()
        self._compiled_alias_map = self._build_alias_index()

    def _load_spacy(self) -> Language:
        """Loads spaCy model with sentencizer and tokenizer, with fallback to blank English model."""
        try:
            nlp = spacy.load(self.config.SPACY_MODEL, disable=["ner"])
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp
        except Exception:
            # Fallback to lightweight blank model with sentencizer
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            return nlp

    def _build_alias_index(self) -> Dict[str, str]:
        """Creates a normalized lookup dictionary from alias phrases to canonical categories."""
        index: Dict[str, str] = {}
        for canonical, aliases in self.config.HEADING_MAP.items():
            for alias in aliases:
                norm_alias = self._normalize_heading_candidate(alias)
                index[norm_alias] = canonical
        return index

    @staticmethod
    def _normalize_heading_candidate(text: str) -> str:
        """Normalizes a candidate line by stripping punctuation, numbers, bullets, and excessive spaces."""
        t = text.strip().lower()
        # Remove markdown headers (###, etc.), list bullets, and leading numbers like "1.", "I."
        t = re.sub(r"^[\s#\*\-\•\–\—\d\.\:\)\(]+", "", t).strip()
        # Remove trailing colons, dashes, pipes, underlines
        t = re.sub(r"[\:\-\–\—\|\_\s]+$", "", t).strip()
        # Collapse multi-spaces
        t = re.sub(r"\s+", " ", t)
        return t

    def extract_file(self, file_path: str | Path) -> DocumentExtractionResult:
        """Extracts text from PDF, DOCX, or TXT file and detects sections."""
        path = Path(file_path)
        if not path.exists():
            return DocumentExtractionResult(
                file_path=str(path),
                file_type="unknown",
                raw_text="",
                cleaned_text="",
                success=False,
                error_message=f"File not found: {path}"
            )
        
        suffix = path.suffix.lower()
        raw_text = ""
        notes = []

        try:
            if suffix == ".pdf":
                raw_text, notes = self._extract_pdf(path)
            elif suffix == ".docx":
                raw_text, notes = self._extract_docx(path)
            elif suffix in (".txt", ".text", ".md"):
                raw_text = path.read_text(encoding="utf-8", errors="replace")
            else:
                return DocumentExtractionResult(
                    file_path=str(path),
                    file_type=suffix,
                    raw_text="",
                    cleaned_text="",
                    success=False,
                    error_message=f"Unsupported file format: {suffix}. Supported: .pdf, .docx, .txt"
                )
        except Exception as e:
            return DocumentExtractionResult(
                file_path=str(path),
                file_type=suffix,
                raw_text="",
                cleaned_text="",
                success=False,
                error_message=f"Extraction failure: {type(e).__name__}: {str(e)}"
            )

        if not raw_text or not raw_text.strip():
            return DocumentExtractionResult(
                file_path=str(path),
                file_type=suffix,
                raw_text="",
                cleaned_text="",
                success=False,
                error_message="Document text extraction resulted in empty content. Document may be an image-only scan without OCR.",
                extraction_notes=notes
            )

        return self.detect_sections(raw_text, file_path=str(path), file_type=suffix, extraction_notes=notes)

    # Alias for convenience
    process_file = extract_file

    def _extract_pdf(self, path: Path) -> Tuple[str, List[str]]:
        """Extracts text from PDF using PyMuPDF (fitz) with layout awareness."""
        notes = []
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz  # type: ignore

        doc = fitz.open(path)
        pages_text = []
        is_scanned_likely = True

        for pno, page in enumerate(doc):
            text = page.get_text("text") or ""
            if text.strip():
                is_scanned_likely = False
            pages_text.append(text)

        doc.close()
        full_text = "\n".join(pages_text)
        
        if is_scanned_likely:
            notes.append("Warning: PDF contains 0 extractable text characters across all pages. Likely a scanned image.")
            
        return full_text, notes

    def _extract_docx(self, path: Path) -> Tuple[str, List[str]]:
        """Extracts text from DOCX using python-docx."""
        notes = []
        import docx
        doc = docx.Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)
        return "\n".join(paragraphs), notes

    def detect_sections(
        self,
        text: str,
        file_path: str = "in_memory",
        file_type: str = "txt",
        extraction_notes: Optional[List[str]] = None
    ) -> DocumentExtractionResult:
        """Preprocesses text using spaCy and partitions lines into classified sections."""
        lines = text.splitlines()
        notes = list(extraction_notes or [])
        
        # 1. Identify section boundary line candidates
        section_boundaries: List[Tuple[int, str, str]] = [] # (line_idx, raw_heading, category)
        
        for idx, line in enumerate(lines):
            raw_line = line.strip()
            if not raw_line:
                continue

            # Check if line qualifies as a section header
            category, heading = self._classify_line_as_heading(raw_line)
            if category is not None:
                section_boundaries.append((idx, heading, category))

        # 2. Extract section text chunks and compute offsets
        sections: List[DetectedSection] = []
        char_cursor = 0
        
        if not section_boundaries:
            # No explicit headings detected: treat entire document as 'summary' / general profile
            notes.append("No explicit section headings detected. Preserving entire document content under 'summary'.")
            spacy_doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in spacy_doc.sents if sent.text.strip()]
            sec = DetectedSection(
                category="summary",
                raw_heading="Document Content",
                text=text.strip(),
                char_start=0,
                char_end=len(text),
                line_start=0,
                sentences=sentences,
                is_core=False
            )
            sections.append(sec)
        else:
            # If there is content before the first heading, capture it as summary / profile header
            first_idx, first_head, _ = section_boundaries[0]
            if first_idx > 0:
                pre_text = "\n".join(lines[:first_idx]).strip()
                if pre_text:
                    spacy_doc = self.nlp(pre_text)
                    sentences = [sent.text.strip() for sent in spacy_doc.sents if sent.text.strip()]
                    sec = DetectedSection(
                        category="summary",
                        raw_heading="Header & Profile Summary",
                        text=pre_text,
                        char_start=0,
                        char_end=len(pre_text),
                        line_start=0,
                        sentences=sentences,
                        is_core=False
                    )
                    sections.append(sec)
                    char_cursor = len(pre_text)

            # Process identified sections
            for b_idx in range(len(section_boundaries)):
                start_line_idx, raw_heading, category = section_boundaries[b_idx]
                end_line_idx = (
                    section_boundaries[b_idx + 1][0]
                    if b_idx + 1 < len(section_boundaries)
                    else len(lines)
                )
                
                # Exclude the heading line itself from the section body
                body_lines = lines[start_line_idx + 1 : end_line_idx]
                body_text = "\n".join(body_lines).strip()
                
                # Process sentences with spaCy
                sentences = []
                if body_text:
                    spacy_doc = self.nlp(body_text)
                    sentences = [sent.text.strip() for sent in spacy_doc.sents if sent.text.strip()]

                sec_start = char_cursor
                sec_end = sec_start + len(body_text)
                char_cursor = sec_end + 1

                is_core = category in self.config.CORE_CATEGORIES

                sec = DetectedSection(
                    category=category,
                    raw_heading=raw_heading,
                    text=body_text,
                    char_start=sec_start,
                    char_end=sec_end,
                    line_start=start_line_idx,
                    sentences=sentences,
                    is_core=is_core
                )
                sections.append(sec)

        # 3. Build concatenated section map for multi-section combinations
        section_map: Dict[str, str] = {}
        for s in sections:
            if s.text:
                if s.category in section_map:
                    section_map[s.category] += "\n" + s.text
                else:
                    section_map[s.category] = s.text

        cleaned_text = "\n\n".join(f"[{s.raw_heading.upper()}]\n{s.text}" for s in sections if s.text)

        return DocumentExtractionResult(
            file_path=file_path,
            file_type=file_type,
            raw_text=text,
            cleaned_text=cleaned_text,
            sections=sections,
            section_map=section_map,
            success=True,
            extraction_notes=notes
        )

    def _classify_line_as_heading(self, line: str) -> Tuple[Optional[str], str]:
        """Determines if a given line is a section heading and returns (canonical_category, raw_heading)."""
        words = line.split()
        if len(words) > self.config.MAX_HEADING_WORDS:
            return None, line
        if len(line) < self.config.MIN_HEADING_CHARS:
            return None, line

        # Reject common contact info lines (phone numbers, emails, addresses, links)
        if "@" in line or "http" in line or "www." in line:
            return None, line
        if re.search(r"\+?\d{1,4}[-.\s]??\d{3}[-.\s]??\d{4}", line):
            return None, line

        norm = self._normalize_heading_candidate(line)
        if not norm:
            return None, line

        # 1. Exact match against normalized alias map
        if norm in self._compiled_alias_map:
            return self._compiled_alias_map[norm], line

        # 2. Check prefix/suffix matches against common headings
        for alias_norm, category in self._compiled_alias_map.items():
            # e.g., "Relevant Skills", "Key Technical Skills"
            if norm == alias_norm or norm.startswith(alias_norm + " ") or norm.endswith(" " + alias_norm):
                return category, line

        # 3. Detect custom or unknown section headings (e.g. "Research and Patents", "Community Work")
        # Strong heading signals: line is ALL CAPS or Title Case, short (<= 4 words), ends with colon, or in bold style
        is_uppercase = line.isupper() and len(words) <= 4 and len(line) >= 4
        has_trailing_colon = line.strip().endswith(":") and len(words) <= 4
        
        # Avoid treating normal sentences as headings
        ends_with_period = line.strip().endswith(".")
        if ends_with_period:
            return None, line

        if is_uppercase or has_trailing_colon:
            # Classified as an unknown/custom section to preserve its content
            return "unknown", line

        return None, line
