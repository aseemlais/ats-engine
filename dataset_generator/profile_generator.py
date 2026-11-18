import random
from typing import Any, Dict, List, Optional
from faker import Faker

fake = Faker()

# Edge case pools for realistic ATS resumes
PARENTHESIZED_TITLES = [
    "Full-Stack Developer (TypeScript / Node.js / React)",
    "Senior Cloud Architect (AWS / Azure) - Enterprise Systems",
    "Machine Learning Engineer (NLP / PyTorch / LLMs)",
    "Backend Software Engineer (Go / Kubernetes / gRPC)",
    "DevOps & Platform Engineer (Docker / Terraform / CI/CD)",
    "Mobile Application Developer (iOS / Swift / SwiftUI)",
    "Frontend Engineer (React / Next.js / Tailwind CSS)",
    "Data Engineer (Python / Spark / Snowflake)",
    "Embedded Systems Engineer (C / C++ / RTOS)",
    "Security Analyst (SIEM / SOC / Threat Intelligence)",
    "Full-Stack Engineer (Python / Django / React)",
    "Site Reliability Engineer (Linux / Prometheus / Grafana)",
]

STANDARD_TITLES = [
    "Software Engineer",
    "Senior Software Engineer",
    "Lead Developer",
    "Data Scientist",
    "Product Manager",
    "QA Automation Engineer",
    "Systems Administrator",
    "Database Administrator",
    "Technical Solutions Architect",
]

INTERNSHIP_TITLES = [
    "Software Engineering Intern",
    "Data Science Intern",
    "Frontend Developer Intern",
    "Machine Learning Research Intern",
    "Summer Technology Analyst (Intern)",
    "Graduate Trainee Engineer",
    "Undergraduate Research Apprentice",
]

COMPANIES = [
    "Tata Consultancy Services (TCS)",
    "Infosys Limited",
    "Google LLC",
    "Amazon Web Services",
    "Microsoft Corporation",
    "Trackinsight Technologies",
    "Revo Karma Hub",
    "Stripe Payments",
    "Datadog Inc.",
    "Snowflake Inc.",
    "Shopify",
    "Deloitte Digital",
    "Accenture Solutions",
    "Capgemini",
]

SKILL_SETS = {
    "frontend": ["React", "TypeScript", "JavaScript", "Next.js", "Redux", "Tailwind CSS", "HTML5", "CSS3", "Webpack", "Vite", "Vue.js", "Angular"],
    "backend": ["Node.js", "Python", "Go", "Java", "Express.js", "FastAPI", "Django", "Spring Boot", "GraphQL", "REST APIs", "gRPC", "PostgreSQL", "MongoDB", "MySQL", "Redis"],
    "cloud_devops": ["AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "CI/CD", "Terraform", "GitHub Actions", "Linux", "Git", "Jenkins", "Ansible"],
    "data_ai": ["Python", "PyTorch", "TensorFlow", "Pandas", "NumPy", "Scikit-Learn", "Spark", "SQL", "Snowflake", "Databricks", "Hugging Face"],
}

DEGREES = [
    ("Doctor of Philosophy in Computer Science", "doctorate"),
    ("Ph.D. in Artificial Intelligence", "doctorate"),
    ("Master of Computer Applications (MCA)", "master"),
    ("Master of Science in Software Engineering", "master"),
    ("M.Tech in Computer Science and Engineering", "master"),
    ("Master of Business Administration (MBA)", "master"),
    ("Bachelor of Science in Information Technology", "bachelor"),
    ("Bachelor of Technology (B.Tech) in Computer Science", "bachelor"),
    ("Bachelor of Engineering (B.E.) in Electronics", "bachelor"),
    ("Bachelor of Computer Applications (BCA)", "bachelor"),
    ("Associate of Applied Science in Information Technology", "associate"),
    ("Diploma in Web Applications Development", "diploma"),
    ("Higher National Diploma in Computing", "diploma"),
]

UNIVERSITIES = [
    "CHARUSAT University, India",
    "Sardar Patel University, India",
    "University of Toronto, Canada",
    "University of Waterloo, Canada",
    "National University of Singapore",
    "University of California, Berkeley",
    "Carnegie Mellon University",
    "University of Washington",
    "Indian Institute of Technology (IIT) Bombay",
    "Delhi Technological University",
    "San Jose State University",
]

CERTIFICATIONS = [
    "AWS Certified Solutions Architect – Associate",
    "Certified Kubernetes Administrator (CKA)",
    "Microsoft Certified: Azure Developer Associate",
    "Google Professional Cloud Architect",
    "HashiCorp Certified: Terraform Associate",
    "Oracle Certified Professional: Java SE 11 Developer",
]

PROJECTS = [
    "High-throughput Financial Cheque Fraud Detection Platform with Azure & microservices",
    "Real-time Distributed Log Processing Engine using Apache Kafka & Go",
    "E-Commerce Microservices Storefront built with Next.js, FastAPI, and Stripe",
    "Automated Resume Screening Engine with NLP & Vector Search",
    "Edge IoT Device Telemetry System built with C++ and MQTT",
]


def generate_candidate_profile(profile_type: str = "standard") -> Dict[str, Any]:
    """
    Generates a rich, ground-truth candidate profile covering key edge cases:
    - standard: 2-4 professional jobs
    - parenthesized_title: complex role with tech stacks in parens
    - fresher: 0 experience, recent graduate, coursework & projects
    - overlapping: multiple simultaneous roles
    - career_gap: multi-year gap between roles
    - missing_contact: null email or phone, or em-dash simulation
    """
    first_name = fake.first_name()
    last_name = fake.last_name()
    full_name = f"{first_name} {last_name}"

    # Determine contact info
    if profile_type == "missing_contact" and random.random() < 0.6:
        email = None
    else:
        email = f"{first_name.lower()}.{last_name.lower()}@{fake.free_email_domain()}"

    if profile_type == "missing_contact" and random.random() < 0.5:
        phone = None
    else:
        phone = fake.phone_number()

    location = f"{fake.city()}, {fake.country()}" if random.random() < 0.85 else "Remote"

    # Select skills
    skill_category = random.choice(["frontend", "backend", "cloud_devops", "data_ai"])
    core_skills = random.sample(SKILL_SETS[skill_category], k=random.randint(4, 7))
    other_skills = random.sample(SKILL_SETS[random.choice(list(SKILL_SETS.keys()))], k=random.randint(2, 4))
    skills = list(dict.fromkeys(core_skills + other_skills))

    # Education
    edu_count = random.choice([1, 2])
    education_entries = []
    if edu_count == 2:
        # Bachelor + Master
        deg_b, tier_b = random.choice([d for d in DEGREES if d[1] == "bachelor"])
        deg_m, tier_m = random.choice([d for d in DEGREES if d[1] in ("master", "doctorate")])
        education_entries.append({
            "degree": deg_m,
            "tier": tier_m,
            "institution": random.choice(UNIVERSITIES),
            "year": str(random.randint(2018, 2023)),
        })
        education_entries.append({
            "degree": deg_b,
            "tier": tier_b,
            "institution": random.choice(UNIVERSITIES),
            "year": str(random.randint(2014, 2017)),
        })
    else:
        deg, tier = random.choice(DEGREES)
        education_entries.append({
            "degree": deg,
            "tier": tier,
            "institution": random.choice(UNIVERSITIES),
            "year": str(random.randint(2016, 2024)),
        })

    # Work Experience
    work_experience = []
    job_title = None

    if profile_type == "fresher":
        job_title = "Aspiring " + random.choice(["Software Engineer", "Frontend Developer", "Data Scientist"])
        work_experience = []
    elif profile_type == "parenthesized_title":
        main_title = random.choice(PARENTHESIZED_TITLES)
        job_title = main_title
        work_experience.append({
            "title": main_title,
            "company": random.choice(COMPANIES),
            "dates": "Jan 2022 - Present",
            "isCurrent": True,
            "isInternship": False,
            "description": f"Designed and delivered mission-critical applications using modern tools and agile practices.",
        })
        work_experience.append({
            "title": random.choice(STANDARD_TITLES),
            "company": random.choice(COMPANIES),
            "dates": "Jun 2019 - Dec 2021",
            "isCurrent": False,
            "isInternship": False,
            "description": f"Collaborated with product teams to build modular REST APIs and backend services.",
        })
    elif profile_type == "overlapping":
        main_title = random.choice(PARENTHESIZED_TITLES)
        job_title = main_title
        # Overlapping roles: Job 1 (2021-Present), Job 2 (2022-2024 part-time/consultant)
        work_experience.append({
            "title": main_title,
            "company": random.choice(COMPANIES),
            "dates": "Mar 2021 - Present",
            "isCurrent": True,
            "isInternship": False,
            "description": "Led core architecture and cross-functional feature development.",
        })
        work_experience.append({
            "title": "Technical Consultant (" + random.choice(["Cloud / DevOps", "React / Node.js", "AI Solutions"]) + ")",
            "company": random.choice(COMPANIES),
            "dates": "Jan 2022 - Dec 2023",
            "isCurrent": False,
            "isInternship": False,
            "description": "Provided architectural advisory and performance optimization.",
        })
        work_experience.append({
            "title": random.choice(INTERNSHIP_TITLES),
            "company": random.choice(COMPANIES),
            "dates": "Jun 2020 - Dec 2020",
            "isCurrent": False,
            "isInternship": True,
            "description": "Contributed to internal tools and automation scripts.",
        })
    elif profile_type == "career_gap":
        job_title = random.choice(STANDARD_TITLES)
        work_experience.append({
            "title": job_title,
            "company": random.choice(COMPANIES),
            "dates": "Jan 2023 - Present",
            "isCurrent": True,
            "isInternship": False,
            "description": "Full-stack feature engineering and cloud deployment.",
        })
        # 2-year career gap: 2021 to 2023
        work_experience.append({
            "title": random.choice(STANDARD_TITLES),
            "company": random.choice(COMPANIES),
            "dates": "Feb 2018 - Nov 2020",
            "isCurrent": False,
            "isInternship": False,
            "description": "Maintained legacy services and planned microservices migration.",
        })
    else:  # standard
        title_cand = random.choice(PARENTHESIZED_TITLES if random.random() < 0.6 else STANDARD_TITLES)
        job_title = title_cand
        work_experience.append({
            "title": title_cand,
            "company": random.choice(COMPANIES),
            "dates": "Dec 2021 - Jun 2025",
            "isCurrent": False,
            "isInternship": False,
            "description": "Built resilient full-stack services and transactional database schemas.",
        })
        work_experience.append({
            "title": random.choice(STANDARD_TITLES),
            "company": random.choice(COMPANIES),
            "dates": "Aug 2019 - Nov 2021",
            "isCurrent": False,
            "isInternship": False,
            "description": "Developed single-page web applications and automated CI/CD pipelines.",
        })
        if random.random() < 0.4:
            work_experience.append({
                "title": random.choice(INTERNSHIP_TITLES),
                "company": random.choice(COMPANIES),
                "dates": "Jan 2019 - Jul 2019",
                "isCurrent": False,
                "isInternship": True,
                "description": "Internship supporting QA automation and documentation.",
            })

    certifications = random.sample(CERTIFICATIONS, k=random.randint(1, 2)) if random.random() < 0.7 else []
    projects = random.sample(PROJECTS, k=random.randint(1, 2)) if random.random() < 0.8 else []

    return {
        "candidateName": full_name,
        "email": email,
        "phone": phone,
        "location": location,
        "jobTitle": job_title,
        "skills": skills,
        "education": education_entries,
        "workExperience": work_experience,
        "certifications": certifications,
        "projects": projects,
        "_meta": {
            "profile_type": profile_type,
        }
    }
