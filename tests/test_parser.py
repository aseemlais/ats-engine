import pytest
from pathlib import Path
import tempfile
from tests._helpers import require_module

from app.services.resume_parser import ResumeParserService

def test_parse_txt_resume():
    parser = ResumeParserService()
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Jane Doe\nPython Developer with 5 years experience\nSkills: Python, SQL, Docker")
        f_path = f.name

    try:
        ok, raw, cleaned, f_hash, err = parser.parse_file(f_path)
        assert ok is True
        assert "Jane Doe" in raw
        assert "python" in cleaned.lower()
        assert f_hash is not None
        assert err is None
    finally:
        Path(f_path).unlink(missing_ok=True)

def test_parse_pdf_resume():
    fitz = require_module("fitz")   # needs real PyMuPDF; skipped when absent
    parser = ResumeParserService()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "John Smith\nSenior Machine Learning Engineer\nSkills: PyTorch, NLP, AWS")
    doc.save(pdf_path)
    doc.close()

    try:
        ok, raw, cleaned, f_hash, err = parser.parse_file(pdf_path)
        assert ok is True
        assert "John Smith" in raw
        assert "Machine Learning" in cleaned
        assert f_hash is not None
    finally:
        Path(pdf_path).unlink(missing_ok=True)

def test_corrupted_file_handling():
    parser = ResumeParserService()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"NOT_A_VALID_PDF_CORRUPT")
        bad_path = f.name

    try:
        ok, raw, cleaned, f_hash, err = parser.parse_file(bad_path)
        assert ok is False
        assert "Invalid PDF header signature" in err
    finally:
        Path(bad_path).unlink(missing_ok=True)
