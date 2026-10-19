import argparse
import random
from pathlib import Path
import fitz  # PyMuPDF
from docx import Document

# Sample candidate profiles
CANDIDATE_POOLS = [
    {
        "category": "NLP_AI_ENGINEER",
        "roles": ["Senior NLP Engineer", "AI Research Engineer", "Machine Learning Specialist"],
        "skills": ["python", "pytorch", "transformers", "nlp", "huggingface", "fastapi", "docker", "machine learning", "deep learning", "bert", "sql"],
        "exp_range": (3, 8),
        "degree": "Master of Science in Computer Science",
        "projects": [
            "Developed transformer-based semantic retrieval system reducing candidate screening latency by 45%",
            "Fine-tuned BERT models for domain-specific entity extraction and classification with 94% F1-score"
        ]
    },
    {
        "category": "DATA_SCIENTIST",
        "roles": ["Data Scientist", "Lead Data Scientist", "Senior Data Analyst"],
        "skills": ["python", "sql", "machine learning", "pandas", "numpy", "scikit-learn", "xgboost", "tableau", "aws"],
        "exp_range": (2, 6),
        "degree": "Bachelor of Technology in Information Technology",
        "projects": [
            "Engineered end-to-end churn prediction pipeline using XGBoost and SHAP explainability",
            "Built automated SQL ETL pipelines in AWS processing 10M records daily"
        ]
    },
    {
        "category": "FULLSTACK_PYTHON",
        "roles": ["Full Stack Engineer", "Python Backend Developer", "Software Engineer"],
        "skills": ["python", "fastapi", "django", "postgresql", "react", "javascript", "typescript", "docker", "git", "rest api"],
        "exp_range": (3, 7),
        "degree": "Bachelor of Engineering in Computer Science",
        "projects": [
            "Designed microservices architecture using FastAPI, PostgreSQL, and Docker",
            "Created responsive frontend web applications in React and TypeScript"
        ]
    },
    {
        "category": "DEVOPS_ENGINEER",
        "roles": ["DevOps Engineer", "Cloud Infrastructure Engineer", "Site Reliability Engineer"],
        "skills": ["aws", "docker", "kubernetes", "terraform", "ci/cd", "linux", "bash", "git", "ansible"],
        "exp_range": (2, 5),
        "degree": "Bachelor of Science in Computer Science",
        "projects": [
            "Automated multi-region AWS cloud infrastructure using Terraform and Ansible",
            "Deployed production Kubernetes clusters with zero-downtime CI/CD workflows"
        ]
    },
    {
        "category": "NON_TECH_FINANCE",
        "roles": ["Financial Analyst", "Senior Accountant", "Auditor"],
        "skills": ["financial analysis", "budgeting", "excel", "quickbooks", "sap", "accounting"],
        "exp_range": (2, 9),
        "degree": "Master of Business Administration (MBA)",
        "projects": [
            "Managed corporate quarterly budgeting and financial audit reports",
            "Conducted risk assessment models and regulatory tax compliance analysis"
        ]
    },
    {
        "category": "NON_TECH_HR",
        "roles": ["HR Manager", "Talent Acquisition Specialist", "HR Business Partner"],
        "skills": ["recruitment", "talent acquisition", "communication", "leadership", "employee relations", "excel"],
        "exp_range": (3, 8),
        "degree": "Bachelor of Business Administration (BBA)",
        "projects": [
            "Led full-cycle recruitment campaigns across technical and operational departments",
            "Implemented company-wide performance management systems"
        ]
    }
]

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Robin", "Casey", "Drew", "Priya", "Rahul", "Aarav", "Elena", "Liam", "Sophia", "Wei", "Jin", "Fatima", "Tariq"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Sharma", "Verma", "Patel", "Chen", "Wang", "Kim", "Alvarez", "Martinez", "Kumar", "Singh"]

def generate_resume_text(idx: int, profile: dict) -> tuple[str, str, str]:
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    cid = f"CV_{idx:05d}"
    email = f"{first.lower()}.{last.lower()}{idx}@example.com"
    phone = f"+1 (555) {random.randint(100, 999)}-{random.randint(1000, 9999)}"
    
    role = random.choice(profile["roles"])
    exp_years = random.randint(profile["exp_range"][0], profile["exp_range"][1])
    start_year = 2026 - exp_years
    
    skills_subset = random.sample(profile["skills"], k=random.randint(max(3, len(profile["skills"]) - 3), len(profile["skills"])))
    skills_line = ", ".join(skills_subset)

    text = f"""{name}
{email} | {phone} | Location: Remote / San Francisco, CA

PROFESSIONAL SUMMARY
Results-driven {role} with {exp_years} years of experience specializing in {profile['category'].replace('_', ' ').lower()}. Proven track record of delivering high-impact projects.

TECHNICAL SKILLS
{skills_line}

WORK EXPERIENCE
{role} - Tech Innovations Corp
{start_year} to Present ({exp_years} years of experience)
- {profile['projects'][0]}
- {profile['projects'][1]}
- Collaborated across cross-functional teams to deliver production-grade scalable systems.

EDUCATION
{profile['degree']} - State University of Technology
Graduated: {start_year - 1}

PROJECTS
- Scalable Data Infrastructure: Implemented high-throughput pipelines handling real-time data streams.
- Enterprise Optimization: Optimized computational workflows reducing operational cost by 30%.

CERTIFICATIONS
- Certified Professional Specialist (2023)
"""
    return cid, name, text

def save_as_pdf(text: str, dest_path: Path):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842) # A4
    rect = fitz.Rect(50, 50, 545, 792)
    page.insert_textbox(rect, text, fontsize=10, fontname="helv")
    doc.save(str(dest_path))
    doc.close()

def save_as_docx(text: str, dest_path: Path):
    doc = Document()
    for line in text.splitlines():
        if line.strip():
            doc.add_paragraph(line)
    doc.save(str(dest_path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10, help="Number of test resumes to generate")
    parser.add_argument("--output-dir", type=str, default="data/resumes", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.count} sample resumes into {out_dir}...")
    for i in range(1, args.count + 1):
        profile = random.choice(CANDIDATE_POOLS)
        cid, name, text = generate_resume_text(i, profile)

        # Distribute formats: 70% PDF, 20% DOCX, 10% TXT
        rand_fmt = random.random()
        if rand_fmt < 0.70:
            file_path = out_dir / f"{cid}_{name.replace(' ', '_')}.pdf"
            save_as_pdf(text, file_path)
        elif rand_fmt < 0.90:
            file_path = out_dir / f"{cid}_{name.replace(' ', '_')}.docx"
            save_as_docx(text, file_path)
        else:
            file_path = out_dir / f"{cid}_{name.replace(' ', '_')}.txt"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

    print(f"Successfully created {args.count} resumes in {out_dir}!")

if __name__ == "__main__":
    main()
