import os
import tempfile
from typing import Any, Dict, Tuple
import fitz  # PyMuPDF
from docx import Document

from app.services.resume_parser import ResumeParserService


def render_single_column_pdf(profile: Dict[str, Any], filepath: str) -> None:
    """
    Renders a standard single-column resume into a PDF document using PyMuPDF.
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # Standard A4

    lines = []
    # Header
    lines.append(profile.get("candidateName") or "Candidate")
    lines.append(profile.get("jobTitle") or "Professional")

    contacts = []
    if profile.get("email"):
        contacts.append(profile["email"])
    if profile.get("phone"):
        contacts.append(profile["phone"])
    if profile.get("location"):
        contacts.append(profile["location"])
    if contacts:
        lines.append(" | ".join(contacts))
    lines.append("")

    # Skills
    if profile.get("skills"):
        lines.append("TECHNICAL SKILLS")
        lines.append(", ".join(profile["skills"]))
        lines.append("")

    # Work Experience
    if profile.get("workExperience"):
        lines.append("PROFESSIONAL EXPERIENCE")
        for exp in profile["workExperience"]:
            comp = f" – {exp['company']}" if exp.get("company") else ""
            dates = f"   |   {exp['dates']}" if exp.get("dates") else ""
            lines.append(f"{exp['title']}{comp}{dates}")
            if exp.get("description"):
                lines.append(f"• {exp['description']}")
            lines.append("")

    # Education
    if profile.get("education"):
        lines.append("EDUCATION")
        for edu in profile["education"]:
            inst = f" – {edu['institution']}" if edu.get("institution") else ""
            yr = f" ({edu['year']})" if edu.get("year") else ""
            lines.append(f"{edu['degree']}{inst}{yr}")
        lines.append("")

    # Projects
    if profile.get("projects"):
        lines.append("KEY PROJECTS")
        for proj in profile["projects"]:
            lines.append(f"• {proj}")
        lines.append("")

    # Certifications
    if profile.get("certifications"):
        lines.append("CERTIFICATIONS")
        for cert in profile["certifications"]:
            lines.append(f"• {cert}")

    text_content = "\n".join(lines)
    rect = fitz.Rect(40, 40, 555, 800)
    page.insert_textbox(rect, text_content, fontsize=10, fontname="helv")

    doc.save(filepath)
    doc.close()


def render_two_column_pdf(profile: Dict[str, Any], filepath: str) -> None:
    """
    Renders a two-column resume:
    Left column (width: 170): Contact info, Skills, Education, Certifications.
    Right column (width: 320): Name, Title, Work Experience, Projects.
    Simulates real two-column PDF block ordering quirks.
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # 1. Right Column (Main: Name, Title, Experience, Projects)
    main_lines = []
    main_lines.append(profile.get("candidateName") or "Candidate")
    main_lines.append(profile.get("jobTitle") or "Professional")
    main_lines.append("")

    if profile.get("workExperience"):
        main_lines.append("WORK EXPERIENCE")
        for exp in profile["workExperience"]:
            comp = f" at {exp['company']}" if exp.get("company") else ""
            dates = f" ({exp['dates']})" if exp.get("dates") else ""
            main_lines.append(f"{exp['title']}{comp}{dates}")
            if exp.get("description"):
                main_lines.append(f"• {exp['description']}")
            main_lines.append("")

    if profile.get("projects"):
        main_lines.append("SELECTED PROJECTS")
        for proj in profile["projects"]:
            main_lines.append(f"• {proj}")

    main_text = "\n".join(main_lines)
    main_rect = fitz.Rect(210, 40, 555, 800)
    page.insert_textbox(main_rect, main_text, fontsize=10, fontname="helv")

    # 2. Left Column (Sidebar: Contact, Skills, Education)
    side_lines = []
    side_lines.append("CONTACT")
    if profile.get("email"):
        side_lines.append(profile["email"])
    if profile.get("phone"):
        side_lines.append(profile["phone"])
    if profile.get("location"):
        side_lines.append(profile["location"])
    side_lines.append("")

    if profile.get("skills"):
        side_lines.append("CORE SKILLS")
        for s in profile["skills"][:8]:
            side_lines.append(f"• {s}")
        side_lines.append("")

    if profile.get("education"):
        side_lines.append("ACADEMICS")
        for edu in profile["education"]:
            side_lines.append(edu["degree"])
            if edu.get("institution"):
                side_lines.append(edu["institution"])
            if edu.get("year"):
                side_lines.append(edu["year"])
        side_lines.append("")

    side_text = "\n".join(side_lines)
    side_rect = fitz.Rect(40, 40, 195, 800)
    page.insert_textbox(side_rect, side_text, fontsize=9, fontname="helv")

    doc.save(filepath)
    doc.close()


def render_docx(profile: Dict[str, Any], filepath: str) -> None:
    """
    Renders a clean DOCX resume document using python-docx.
    """
    doc = Document()

    # Name & Title
    doc.add_heading(profile.get("candidateName") or "Candidate", 0)
    doc.add_paragraph(profile.get("jobTitle") or "Professional")

    # Contact line
    contacts = []
    if profile.get("email"):
        contacts.append(profile["email"])
    if profile.get("phone"):
        contacts.append(profile["phone"])
    if profile.get("location"):
        contacts.append(profile["location"])
    if contacts:
        doc.add_paragraph(" | ".join(contacts))

    # Experience
    if profile.get("workExperience"):
        doc.add_heading("Work Experience", level=1)
        for exp in profile["workExperience"]:
            p = doc.add_paragraph()
            comp = f" – {exp['company']}" if exp.get("company") else ""
            dates = f"   ({exp['dates']})" if exp.get("dates") else ""
            p.add_run(f"{exp['title']}{comp}{dates}").bold = True
            if exp.get("description"):
                doc.add_paragraph(exp["description"], style="List Bullet")

    # Education
    if profile.get("education"):
        doc.add_heading("Education", level=1)
        for edu in profile["education"]:
            inst = f" – {edu['institution']}" if edu.get("institution") else ""
            yr = f" ({edu['year']})" if edu.get("year") else ""
            doc.add_paragraph(f"{edu['degree']}{inst}{yr}")

    # Skills
    if profile.get("skills"):
        doc.add_heading("Technical Skills", level=1)
        doc.add_paragraph(", ".join(profile["skills"]))

    # Projects
    if profile.get("projects"):
        doc.add_heading("Projects", level=1)
        for proj in profile["projects"]:
            doc.add_paragraph(proj, style="List Bullet")

    doc.save(filepath)


def render_and_extract_text(profile: Dict[str, Any], template: str = "single_column_pdf") -> Tuple[str, str]:
    """
    Renders profile to a file using the specified template, extracts text using
    the actual production ResumeParserService to capture authentic artifacts,
    and returns (extracted_raw_text, format_type).
    """
    parser = ResumeParserService()

    with tempfile.TemporaryDirectory() as tmpdir:
        if template == "two_column_pdf":
            filepath = os.path.join(tmpdir, "resume.pdf")
            render_two_column_pdf(profile, filepath)
            extracted_text = parser.extract_text_from_pdf(filepath)
            fmt = "pdf_two_column"
        elif template == "docx":
            filepath = os.path.join(tmpdir, "resume.docx")
            render_docx(profile, filepath)
            extracted_text = parser.extract_text_from_docx(filepath)
            fmt = "docx"
        else:  # default single_column_pdf
            filepath = os.path.join(tmpdir, "resume.pdf")
            render_single_column_pdf(profile, filepath)
            extracted_text = parser.extract_text_from_pdf(filepath)
            fmt = "pdf_single_column"

    return extracted_text, fmt
