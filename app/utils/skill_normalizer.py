import re
from typing import Dict, Set, List

# Canonical skill -> list of aliases / variants
SKILL_TAXONOMY: Dict[str, List[str]] = {
    # Programming Languages
    "python": ["python", "py", "python3"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "csharp", "c sharp"],
    "c": ["c language", "ansi c"],
    "java": ["java", "core java", "j2ee"],
    "javascript": ["javascript", "js", "ecmascript"],
    "typescript": ["typescript", "ts"],
    "golang": ["golang", "go language"],
    "rust": ["rust", "rustlang"],
    "ruby": ["ruby", "ruby on rails", "rails"],
    "php": ["php"],
    "scala": ["scala"],
    "r": ["r language", "r programming"],
    "kotlin": ["kotlin"],
    "swift": ["swift"],
    "sql": ["sql", "structured query language"],
    "bash": ["bash", "shell scripting", "shell script", "zsh"],

    # AI, ML, Data Science
    "machine learning": ["machine learning", "ml", "statistical learning"],
    "deep learning": ["deep learning", "dl", "neural networks", "deep neural networks"],
    "nlp": ["nlp", "natural language processing", "text processing"],
    "computer vision": ["computer vision", "cv", "image processing", "opencv"],
    "large language models": ["large language models", "llm", "llms", "generative ai", "genai"],
    "transformers": ["transformers", "huggingface", "hugging face", "bert", "gpt"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "tf", "keras"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scipy": ["scipy"],
    "xgboost": ["xgboost", "xgb"],
    "lightgbm": ["lightgbm", "lgb"],
    "catboost": ["catboost"],
    "data science": ["data science", "data scientist"],
    "data analysis": ["data analysis", "data analytics", "data analyst", "analyzing data", "analysing data", "exploratory data analysis", "eda"],
    "data engineering": ["data engineering", "etl", "elt", "etl/elt", "data pipelines", "data pipeline", "data ingestion"],
    "cloud computing": ["cloud computing", "cloud concepts", "cloud basics", "cloud platforms", "cloud technologies", "cloud services", "cloud fundamentals"],
    "databases": ["databases", "dbms", "rdbms", "database management", "database management system", "database management systems", "database technologies", "relational databases"],
    "databricks": ["databricks", "azure databricks", "databricks sql"],
    "delta lake": ["delta lake", "delta tables"],
    "medallion architecture": ["medallion architecture", "bronze-silver-gold", "bronze silver gold"],
    "reinforcement learning": ["reinforcement learning", "rl"],

    # Databases & Big Data
    "postgresql": ["postgresql", "postgres", "psql"],
    "mysql": ["mysql"],
    "sqlite": ["sqlite", "sqlite3"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "cassandra": ["cassandra"],
    "elasticsearch": ["elasticsearch", "elastic search", "opensearch"],
    "spark": ["spark", "apache spark", "pyspark"],
    "hadoop": ["hadoop", "mapreduce"],
    "kafka": ["kafka", "apache kafka"],
    "snowflake": ["snowflake"],
    "bigquery": ["bigquery", "google bigquery"],
    "airflow": ["airflow", "apache airflow"],
    "redshift": ["redshift", "amazon redshift"],
    "data warehousing": ["data warehousing", "data warehouse", "data warehouses"],
    "a/b testing": ["a/b testing", "a/b test", "a/b tests", "ab testing", "split testing"],

    # Cloud & DevOps
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "docker": ["docker", "containerization", "containers"],
    "kubernetes": ["kubernetes", "k8s"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],
    "ci/cd": ["ci/cd", "continuous integration", "continuous deployment", "jenkins", "github actions", "gitlab ci"],
    "linux": ["linux", "ubuntu", "centos", "redhat", "unix"],
    "git": ["git", "github", "gitlab", "bitbucket", "version control"],

    # Web & Backend
    "fastapi": ["fastapi"],
    "flask": ["flask"],
    "django": ["django"],
    "react": ["react", "react.js", "reactjs"],
    "next.js": ["next.js", "nextjs"],
    "vue": ["vue", "vue.js", "vuejs"],
    "node.js": ["node.js", "nodejs", "node"],
    "express": ["express", "express.js"],
    "spring boot": ["spring boot", "spring framework", "spring"],
    "rest api": ["rest api", "restful api", "restful"],
    "graphql": ["graphql"],
    "microservices": ["microservices", "microservice architecture"],
    "html": ["html", "html5"],
    "css": ["css", "css3", "sass", "tailwind", "bootstrap"],

    # Business, Analytics & Tools
    "project management": ["project management", "pmp"],
    "agile": ["agile", "scrum", "kanban"],
    "jira": ["jira", "confluence"],
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "excel": ["excel", "microsoft excel", "advanced excel"]
}

# Skills that are interchangeable "kinds" of the same thing. Used ONLY to read "A, B or C" lists as one
# either/or group when all members share a category (e.g. AWS, Azure or GCP). Not an equivalence rule.
SKILL_CATEGORY: Dict[str, str] = {
    **{k: "sql_db" for k in ("sql", "mysql", "postgresql", "sqlite")},
    **{k: "cloud" for k in ("aws", "azure", "gcp")},
    **{k: "nosql_db" for k in ("mongodb", "redis", "cassandra")},
    **{k: "bi_tool" for k in ("tableau", "power bi")},
    **{k: "dl_framework" for k in ("pytorch", "tensorflow")},
    **{k: "frontend_framework" for k in ("react", "vue")},
    **{k: "orchestrator" for k in ("airflow",)},
}

# Carefully defined implications for RESUME skills: holding the specific skill is evidence for the general
# one (MySQL is a SQL database; AWS/Azure/GCP are cloud platforms). The reverse never holds.
IMPLIES: Dict[str, List[str]] = {
    "mysql": ["sql", "databases"], "postgresql": ["sql", "databases"], "sqlite": ["sql", "databases"],
    "sql": ["databases"],
    "aws": ["cloud computing"], "azure": ["cloud computing"], "gcp": ["cloud computing"],
}

# Inverted index: alias -> canonical skill name
ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canonical, aliases in SKILL_TAXONOMY.items():
    ALIAS_TO_CANONICAL[canonical.lower()] = canonical.lower()
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias.lower()] = canonical.lower()

# Compiled regex patterns for skill extraction
def build_skill_patterns():
    patterns = []
    # Sort aliases by length descending so multi-word aliases match before single-word substrings
    all_terms = sorted(ALIAS_TO_CANONICAL.keys(), key=len, reverse=True)
    for term in all_terms:
        # Handle terms with special regex chars like c++, c#, .net, next.js
        escaped = re.escape(term)
        # Use word boundaries or custom boundaries for special terms
        if term in ("c++", "c#", ".net", "c"):
            pattern = re.compile(rf"(?:^|[\s,;/(]){escaped}(?:[\s,;/)]|$)", re.IGNORECASE)
        else:
            pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
        patterns.append((pattern, term, ALIAS_TO_CANONICAL[term]))
    return patterns

COMPILED_SKILL_PATTERNS = build_skill_patterns()

def normalize_skill(skill_str: str) -> str:
    """Normalizes an individual skill token or alias to its canonical form."""
    cleaned = skill_str.strip().lower()
    return ALIAS_TO_CANONICAL.get(cleaned, cleaned)

def expand_implied(skills) -> List[str]:
    """Adds skills implied by more specific ones (see IMPLIES). Never adds the reverse direction."""
    out = set(skills)
    for sk in list(out):
        out.update(IMPLIES.get(sk, []))
    return sorted(out)


def extract_skills_from_text(text: str, expand: bool = False) -> List[str]:
    """Deterministically extracts and normalizes all canonical skills found in text.
    expand=True also adds implied general skills (use for RESUMES, not for job descriptions)."""
    if not text:
        return []

    found_skills: Set[str] = set()
    for pattern, term, canonical in COMPILED_SKILL_PATTERNS:
        if pattern.search(text):
            found_skills.add(canonical)

    return expand_implied(found_skills) if expand else sorted(found_skills)


def extract_skill_mentions(text: str) -> List[tuple]:
    """Ordered, non-overlapping skill mentions as (start, end, canonical); longer aliases win overlaps."""
    cands = []
    for pattern, term, canonical in COMPILED_SKILL_PATTERNS:
        for m in pattern.finditer(text or ""):
            cands.append((m.start(), m.end(), canonical))
    cands.sort(key=lambda c: (c[0], -(c[1] - c[0])))
    picked, last_end = [], -1
    for st, en, canon in cands:
        if st >= last_end:
            picked.append((st, en, canon)); last_end = en
    return picked
