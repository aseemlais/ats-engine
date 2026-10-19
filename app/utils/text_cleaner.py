import re
import unicodedata

# Bump whenever tokenize_for_bm25 changes: a persisted index built with another version is stale and must be rebuilt.
TOKENIZER_VERSION = 2   # v2: compound skills (ci/cd, pl/sql, tcp/ip, power bi), context-gated R/C, spelling aliases

# Protected technical tokens regex to protect during tokenization / cleaning
PROTECTED_TERMS = {
    "c++", "c#", ".net", "node.js", "vue.js", "next.js", "react.js",
    "scikit-learn", "ci/cd", "tcp/ip", "pl/sql", "t-sql", "objective-c"
}

def clean_text(text: str) -> str:
    """Cleans resume text while preserving technical terminology and section headers."""
    if not text:
        return ""
    
    # Normalize unicode (decompose ligatures like fi -> f i, normalize accents)
    text = unicodedata.normalize("NFKD", text)
    
    # Replace non-breaking spaces and exotic whitespace
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    
    # Replace bullet points and weird dashes with uniform dash or space
    text = re.sub(r"[•●▪■◆★\u2022\u2023\u25e6\u2043\u2219]", "\n- ", text)
    text = re.sub(r"[\u2013\u2014\u2015\u2212]", "-", text)
    
    # Replace smart quotes
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    
    # Replace excessive horizontal whitespace but keep newline structure for section boundaries
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
            
    return "\n".join(lines)


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "shall", "should", "may", "might", "must", "can",
    "could", "that", "this", "these", "those", "it", "its", "as", "if"
}

# One token per match. Compounds are matched FIRST so "CI/CD" is one token "ci/cd" (not "ci" + "cd") and
# "PL/SQL" is one token "pl/sql" - it must NOT also yield "sql": PL/SQL is a related but distinct technology.
_TOKEN_RE = re.compile(
    r"(?P<compound>\bci\s*[/-]\s*cd\b|\bcicd\b|\bpl\s*/\s*sql\b|\btcp\s*/\s*ip\b|\bpower\s+bi\b|"
    r"\bscikit[\s-]learn\b|\bc\s+sharp\b)"
    r"|(?P<tok>[a-z0-9]+(?:\.[a-z0-9]+)+|[a-z0-9]+\+\+|[a-z0-9]+\#|\.net|[a-z0-9]+(?:-[a-z0-9]+)+|[a-z0-9]+)",
    re.I)

# Spelling variants of the SAME technology only. Deliberately absent (distinct technologies, never merged):
# java/javascript, spark/pyspark, sql/mysql/postgresql/pl-sql/t-sql, c/c++/c#, go/golang-vs-"go" the verb.
_CANONICAL = {
    "cicd": "ci/cd", "k8s": "kubernetes", "postgres": "postgresql", "cpp": "c++", "csharp": "c#", "c sharp": "c#",
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn", "powerbi": "powerbi", "power bi": "powerbi",
    "js": "javascript", "nodejs": "node.js", "reactjs": "react.js", "vuejs": "vue.js", "nextjs": "next.js",
}

# R and C are real languages but, as bare letters, mostly noise ("Plan C", "Vitamin C", "Option B/C", "R&D").
# A single-letter token is kept only when an ADJACENT word is a programming context word.
_TECH_CONTEXT = {
    "python", "java", "javascript", "sql", "matlab", "sas", "stata", "spss", "scala", "julia", "perl", "golang",
    "c++", "c#", "ansi", "programming", "language", "languages", "rstudio", "studio", "shiny", "tidyverse", "ggplot",
    "ggplot2", "dplyr", "cran", "pandas", "numpy", "bash", "vba", "fortran", "rust", "kotlin", "swift", "php", "ruby",
}
_WORD = re.compile(r"[A-Za-z0-9+#.]+")


def _neighbour_is_tech(text: str, start: int, end: int):
    """(previous_is_tech, next_is_tech) for the words adjacent to text[start:end], punctuation ignored."""
    prev = _WORD.findall(text[max(0, start - 40):start])
    nxt = _WORD.findall(text[end:end + 40])
    return (bool(prev) and prev[-1].lower() in _TECH_CONTEXT, bool(nxt) and nxt[0].lower() in _TECH_CONTEXT)


def tokenize_for_bm25(text: str, query: bool = False) -> list[str]:
    """Tokenizes text preserving technical terms (c++, c#, .net, node.js, ci/cd, pl/sql, t-sql).

    query=True is for the structured skill query (case is lost there): a single R/C is kept if ONE neighbour is a
    programming word. Documents (query=False) keep an upper-case R/C with one such neighbour, or a lower-case one
    with two, so "Plan C" and "R&D" never create a token. A text that is just "R" or "C" is kept."""
    if not text:
        return []
    sole = text.strip().lower() in ("r", "c")
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        raw = m.group(0)
        low = re.sub(r"\s+", " ", raw.lower()) if m.lastgroup == "compound" else raw.lower()
        if m.lastgroup == "compound":
            low = re.sub(r"\s*([/-])\s*", r"/", low) if low.startswith(("ci", "pl", "tcp")) else low
            if low.startswith("ci") and low != "cicd":
                low = "ci/cd"
        if len(low) == 1:
            if low not in ("r", "c"):
                continue
            before, after = _neighbour_is_tech(text, m.start(), m.end())
            keep = sole or (before or after if (query or raw.isupper()) else before and after)
            if not keep:
                continue
        low = _CANONICAL.get(low, low)
        if low in _STOPWORDS:
            continue
        out.append(low)
    return out
