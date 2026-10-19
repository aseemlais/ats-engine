import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app.models.resume import EducationEntry, ExperienceEntry, ParsedResume
from app.utils.hash_utils import compute_text_hash
from app.utils.skill_normalizer import extract_skills_from_text
from app.utils.degree_normalizer import normalize_degree

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

# Regex for common resume section headers
SECTION_PATTERNS = {
    "skills": re.compile(r"^(?:technical\s+skills|skills|core\s+competencies|technologies|tools\s+&\s+technologies)\b", re.I),
    "internship": re.compile(r"^(?:internships?|internship\s+experience|intern\s+experience|project\s+&\s+internship|projects\s+&\s+internships)\b", re.I),
    "experience": re.compile(r"^(?:work\s+experience|professional\s+experience|experience|employment\s+history|career\s+history)\b", re.I),
    "education": re.compile(r"^(?:education|academic\s+background|education\s+&\s+credentials|qualifications|academic\s+details|academics)\b", re.I),
    "projects": re.compile(r"^(?:projects|key\s+projects|academic\s+projects|technical\s+projects)\b", re.I),
    "certifications": re.compile(r"^(?:certifications|certificates|licenses\s+&\s+certifications|credentials|achievements\s+&\s+certifications)\b", re.I),
}

# Date patterns for experience calculation
YEAR_RANGE_RE = re.compile(
    r"\b((?:19|20)\d{2})\s*(?:-|–|—|to)\s*((?:19|20)\d{2}|present|current|now)\b",
    re.I
)
MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}
DATE_WITH_MONTH_RE = re.compile(
    r"\b(?:(\d{1,2})\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+((?:19|20)\d{2})\s*(?:-|–|—|to)\s*(?:(?:(\d{1,2})\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+((?:19|20)\d{2})|(present|current|now))\b",
    re.I
)
DATE_SLASH_RE = re.compile(
    r"\b(0?[1-9]|1[0-2])[\/\.-]((?:19|20)\d{2})\s*(?:-|–|—|to)\s*(?:(0?[1-9]|1[0-2])[\/\.-]((?:19|20)\d{2})|(present|current|now))\b",
    re.I
)
# DD.MM.YYYY to DD.MM.YYYY (e.g. 06.06.2025 to 05.07.2025)
DATE_DOT_RE = re.compile(
    r"\b(\d{1,2})[\.\/\-](\d{1,2})[\.\/\-]((?:19|20)\d{2})\s*(?:-|–|—|to)\s*(\d{1,2})[\.\/\-](\d{1,2})[\.\/\-]((?:19|20)\d{2})\b",
    re.I
)
# Single month+year without an end range (e.g. 'June 2025') -> counts as 1 month
SINGLE_MONTH_YEAR_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+((?:19|20)\d{2})\b",
    re.I
)

EXPLICIT_EXP_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp|in\s+software|in\s+data|in\s+development)\b",
    re.I
)

# Months duration pattern including hyphenated forms (e.g. '7-Month Internship', '6 months internship', '3 months')
MONTHS_DURATION_RE = re.compile(
    r"\b(\d+)\s*(?:-|–|—)?\s*(?:months?|mos?)\b",
    re.I
)

# Weeks duration pattern (e.g. '4 weeks', '6 weeks')
WEEKS_DURATION_RE = re.compile(
    r"\b(\d+)\s*(?:weeks?|wks?)\b",
    re.I
)

# Common degrees regex for extraction
DEGREE_LINE_RE = re.compile(
    r"\b(Ph\.?D|Doctorate|Master(?:'?s)?|M\.?S(?:c)?|M\.?Tech|MBA|Bachelor(?:'?s)?|B\.?Tech|B\.?E|B\.?S(?:c)?|BCA|BBA|Diploma|Associate)\b[^\n,]*",
    re.I
)

class NLPExtractorService:
    def __init__(self):
        self.current_year = datetime.now().year

    def extract_name(self, text: str) -> str:
        """Heuristic name extraction from the header of the resume."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines[:8]:
            # Filter out lines that look like emails, URLs, phones, or section headings
            if "@" in line or "http" in line or len(line) > 50 or re.search(r"\+?\d{10}", line):
                continue
            if any(p.match(line) for p in SECTION_PATTERNS.values()):
                continue
            if re.search(r"\b(objective|summary|resume|curriculum|vitae|contact)\b", line, re.I):
                continue
            # Must contain letters, mostly alphabetic
            words = line.split()
            if 1 <= len(words) <= 4 and all(re.match(r"^[a-zA-Z\.]+$", w) for w in words):
                return line.title()
        return "Unknown"

    def extract_contact(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        email_match = EMAIL_RE.search(text)
        email = email_match.group(0) if email_match else None

        phone_match = PHONE_RE.search(text)
        phone = phone_match.group(0) if phone_match else None

        return email, phone

    def segment_sections(self, text: str) -> Dict[str, str]:
        """Segments raw/cleaned resume text into functional sections."""
        lines = text.splitlines()
        sections: Dict[str, List[str]] = {
            "skills": [],
            "experience": [],
            "internship": [],
            "education": [],
            "projects": [],
            "certifications": [],
            "general": []
        }

        current_sec = "general"
        for line in lines:
            trimmed = line.strip()
            if not trimmed:
                continue

            # Check if trimmed line is a header (short line matching header pattern)
            matched_sec = None
            if len(trimmed.split()) <= 6:
                for sec_name, pattern in SECTION_PATTERNS.items():
                    if pattern.match(trimmed):
                        matched_sec = sec_name
                        break

            if matched_sec:
                current_sec = matched_sec
            else:
                sections[current_sec].append(trimmed)

        return {k: "\n".join(v) for k, v in sections.items()}

    def _parse_duration_from_text(self, text: str) -> float:
        """Parses duration in years strictly from given text slice."""
        if not text.strip():
            return 0.0

        # Check explicit months (e.g., '6 months', '7-month', '3 months')
        month_matches = MONTHS_DURATION_RE.findall(text)
        total_months = 0
        if month_matches:
            for m in month_matches:
                try:
                    val = int(m)
                    if 1 <= val <= 36:
                        total_months += val
                except Exception:
                    pass

        # Check explicit years mention (e.g. '2 years of experience')
        explicit_matches = EXPLICIT_EXP_RE.findall(text)
        if explicit_matches:
            try:
                values = [float(v) for v in explicit_matches]
                max_exp = max(values)
                if 0.5 <= max_exp <= 40.0:
                    return max_exp
            except Exception:
                pass

        # Check explicit weeks (e.g. '4 weeks', '6 weeks')
        week_matches = WEEKS_DURATION_RE.findall(text)
        if week_matches:
            for w in week_matches:
                try:
                    total_months += round(int(w) / 4.33, 1)
                except Exception:
                    pass

        # Check DD.MM.YYYY ranges
        for m in DATE_DOT_RE.finditer(text):
            try:
                d1, m1, y1, d2, m2, y2 = m.groups()
                dur = (int(y2) - int(y1)) * 12 + (int(m2) - int(m1))
                total_months += max(dur, 1)
            except Exception:
                pass

        # Check explicit month ranges (e.g. '20 May 2025 - 12 June 2025' or 'May 2023 - Present')
        for m in DATE_WITH_MONTH_RE.finditer(text):
            try:
                _, start_m_str, start_y_str, _, end_m_str, end_y_str, present_str = m.groups()
                start_m = MONTH_MAP.get(start_m_str.lower()[:3], 1)
                start_y = int(start_y_str)
                if present_str:
                    end_m = datetime.now().month
                    end_y = self.current_year
                else:
                    end_m = MONTH_MAP.get(end_m_str.lower()[:3], 1) if end_m_str else 1
                    end_y = int(end_y_str)

                duration_m = (end_y - start_y) * 12 + (end_m - start_m)
                if duration_m > 0:
                    total_months += duration_m
                elif duration_m == 0:
                    total_months += 1
            except Exception:
                pass

        # Check single month+year mentions
        if total_months == 0:
            for m in SINGLE_MONTH_YEAR_RE.finditer(text):
                total_months += 1

        # Parse date intervals (e.g., 2021 - 2023)
        intervals: List[Tuple[int, int]] = []
        for match in YEAR_RANGE_RE.finditer(text):
            start_str, end_str = match.groups()
            try:
                start_year = int(start_str)
                if end_str.lower() in ("present", "current", "now"):
                    end_year = self.current_year
                else:
                    end_year = int(end_str)

                if 1980 <= start_year <= self.current_year and end_year >= start_year:
                    intervals.append((start_year, min(end_year, self.current_year)))
            except Exception:
                continue

        if intervals:
            intervals.sort(key=lambda x: x[0])
            merged = [intervals[0]]
            for start, end in intervals[1:]:
                last_start, last_end = merged[-1]
                if start <= last_end:
                    merged[-1] = (last_start, max(last_end, end))
                else:
                    merged.append((start, end))

            years_from_intervals = sum(end - start for start, end in merged)
            if years_from_intervals > 0:
                return float(years_from_intervals)

        if total_months > 0:
            return round(total_months / 12.0, 2)

        return 0.0

    def extract_experience_years(self, text: str) -> float:
        """Backwards compatible helper returning effective experience years."""
        sections = self.segment_sections(text)
        _, _, effective = self.extract_experience_breakdown(text, sections)
        return effective

    def _parse_entry_duration_months(self, block_text: str) -> int:
        """Calculates total months from date ranges in a single experience entry block."""
        months = 0
        
        # 1. Check DD.MM.YYYY ranges (e.g., 06.06.2025 to 05.07.2025)
        for m in DATE_DOT_RE.finditer(block_text):
            try:
                d1, m1, y1, d2, m2, y2 = m.groups()
                dur = (int(y2) - int(y1)) * 12 + (int(m2) - int(m1))
                months += max(dur, 1)
            except Exception:
                pass

        # 2. Check MM/YYYY slash ranges (e.g., 05/2026 - 08/2026)
        for m in DATE_SLASH_RE.finditer(block_text):
            try:
                sm, sy, em, ey, pres = m.groups()
                start_m, start_y = int(sm), int(sy)
                if pres:
                    end_m, end_y = datetime.now().month, self.current_year
                else:
                    end_m, end_y = int(em), int(ey)
                dur = (end_y - start_y) * 12 + (end_m - start_m)
                months += max(dur, 1)
            except Exception:
                pass

        # 3. Check Month Name ranges (e.g., Sept 2024 - Oct 2024, May 2023 - Present)
        for m in DATE_WITH_MONTH_RE.finditer(block_text):
            try:
                _, sm_str, sy_str, _, em_str, ey_str, pres = m.groups()
                start_m = MONTH_MAP.get(sm_str.lower()[:3], 1)
                start_y = int(sy_str)
                if pres:
                    end_m, end_y = datetime.now().month, self.current_year
                else:
                    end_m = MONTH_MAP.get(em_str.lower()[:3], 1) if em_str else 1
                    end_y = int(ey_str)
                dur = (end_y - start_y) * 12 + (end_m - start_m)
                months += max(dur, 1)
            except Exception:
                pass

        # 4. Check explicit duration mentions (e.g. '7-Month Internship', '6 months', '3 mos')
        if months == 0:
            for w in MONTHS_DURATION_RE.findall(block_text):
                try:
                    months += int(w)
                except Exception:
                    pass

        # 5. Check Single Month + Year mentions (e.g. 'June 2025' -> 1 month)
        if months == 0:
            single_matches = SINGLE_MONTH_YEAR_RE.findall(block_text)
            if single_matches:
                months += len(single_matches)

        # 6. Check year-only ranges if months not found (e.g., 2022 - 2024)
        if months == 0:
            for m in YEAR_RANGE_RE.finditer(block_text):
                try:
                    sy_str, ey_str = m.groups()
                    start_y = int(sy_str)
                    end_y = self.current_year if ey_str.lower() in ("present", "current", "now") else int(ey_str)
                    if 1980 <= start_y <= self.current_year and end_y >= start_y:
                        months += (end_y - start_y) * 12
                except Exception:
                    pass

        return months

    def _find_date_matches(self, text: str):
        """Finds date intervals in text while avoiding double-counting single months inside ranges."""
        primary_matches = (
            list(DATE_DOT_RE.finditer(text)) +
            list(DATE_SLASH_RE.finditer(text)) +
            list(DATE_WITH_MONTH_RE.finditer(text)) +
            list(YEAR_RANGE_RE.finditer(text)) +
            list(MONTHS_DURATION_RE.finditer(text))
        )
        all_matches = list(primary_matches)
        for sm in SINGLE_MONTH_YEAR_RE.finditer(text):
            # Only include single month if it does not overlap with any existing range match
            if not any(pm.start() <= sm.start() and sm.end() <= pm.end() for pm in primary_matches):
                all_matches.append(sm)
        all_matches.sort(key=lambda m: m.start())
        return all_matches

    _INTERN_RE = re.compile(r"\b(intern|internship|trainee)\b", re.I)
    _ROLE_RE = re.compile(r"\b(experience|developer|engineer|analyst|specialist|consultant|manager|worked\s+as|working\s+at)\b", re.I)
    _PRESENT = ("present", "current", "now")

    def _match_interval(self, m):
        """(start_month_index, end_month_index) for range-type date matches, else None (durations only)."""
        rx = m.re
        now_idx = self.current_year * 12 + datetime.now().month
        try:
            if rx is DATE_WITH_MONTH_RE:
                _, sm, sy, _, em, ey, pres = m.groups()
                st = int(sy) * 12 + MONTH_MAP.get(sm.lower()[:3], 1)
                en = now_idx if pres else int(ey) * 12 + (MONTH_MAP.get(em.lower()[:3], 1) if em else 1)
            elif rx is DATE_SLASH_RE:
                sm, sy, em, ey, pres = m.groups()
                st = int(sy) * 12 + int(sm)
                en = now_idx if pres else int(ey) * 12 + int(em)
            elif rx is YEAR_RANGE_RE:
                sy, ey = m.groups()
                if not (1980 <= int(sy) <= self.current_year):
                    return None
                st = int(sy) * 12
                en = self.current_year * 12 if ey.lower() in self._PRESENT else int(ey) * 12
                if en < st:
                    return None
            elif rx is DATE_DOT_RE:
                _, m1, y1, _, m2, y2 = m.groups()
                st, en = int(y1) * 12 + int(m1), int(y2) * 12 + int(m2)
            else:
                return None
        except Exception:
            return None
        return (st, max(en, st + 1))  # same 1-month floor as _parse_entry_duration_months

    @staticmethod
    def _union_months(intervals) -> int:
        """Total months covered by the union of (start, end) intervals: concurrent jobs are not counted twice."""
        total, cur = 0, None
        for st, en in sorted(intervals):
            if cur and st <= cur[1]:
                cur = (cur[0], max(cur[1], en))
            else:
                if cur:
                    total += cur[1] - cur[0]
                cur = (st, en)
        if cur:
            total += cur[1] - cur[0]
        return total

    def _entry_header(self, text: str, dm, prev_end: int) -> str:
        """Header text of the entry that OWNS this date range: the date's own line, up to two preceding
        non-bullet lines (after the previous date match), and - only when the date line carries no title
        words - the next non-bullet, date-free line. Bullets/lines of neighbouring entries are never used
        (the old code classified an entry using the text up to the NEXT date, i.e. the next entry's title)."""
        line_start = text.rfind("\n", 0, dm.start()) + 1
        line_end = text.find("\n", dm.end())
        line_end = len(text) if line_end == -1 else line_end
        date_line = text[line_start:line_end]
        bullets = ("-", "\u2022", "*", "\u2013")
        before = [l.strip() for l in text[max(prev_end, 0):line_start].splitlines()
                  if l.strip() and not l.strip().startswith(bullets)][-2:]
        header = " ".join(before + [date_line])
        remainder = (date_line[:dm.start() - line_start] + " " + date_line[dm.end() - line_start:])
        if len(re.findall(r"[A-Za-z]{3,}", remainder)) < 1:
            for l in text[line_end:].splitlines():
                l = l.strip()
                if not l:
                    continue
                if not l.startswith(bullets) and not self._find_date_matches(l):
                    header += " " + l
                break
        return header

    def _accumulate_entries(self, text: str, date_matches, require_role_words: bool):
        """Returns (work_months, intern_months). Range dates are unioned per class (no double counting of
        concurrent jobs); pure durations ('6 months') are additive."""
        work_iv, intern_iv, work_add, intern_add = [], [], 0, 0
        prev_end = 0
        for idx, dm in enumerate(date_matches):
            start_boundary = date_matches[idx - 1].end() if idx > 0 else 0
            end_boundary = date_matches[idx + 1].start() if idx + 1 < len(date_matches) else len(text)
            header = self._entry_header(text, dm, prev_end)
            prev_end = dm.end()
            is_intern = bool(self._INTERN_RE.search(header))
            if require_role_words and not is_intern and not self._ROLE_RE.search(header):
                continue
            iv = self._match_interval(dm)
            if iv is not None:
                (intern_iv if is_intern else work_iv).append(iv)
                continue
            months = self._parse_entry_duration_months(text[dm.start():dm.end()])
            if months == 0:
                months = self._parse_entry_duration_months(text[start_boundary:end_boundary])
            if is_intern:
                intern_add += months
            else:
                work_add += months
        return self._union_months(work_iv) + work_add, self._union_months(intern_iv) + intern_add

    def extract_experience_breakdown(self, full_text: str, sections: Dict[str, str]) -> Tuple[float, float, float]:
        """
        Extracts duration strictly from professional/work experience and internships,
        strictly ignoring education, projects, and certification sections.
        Returns: (work_experience_years, internship_years, total_effective_years)
        """
        exp_text = sections.get("experience", "").strip()
        intern_text = sections.get("internship", "").strip()

        total_work_months = 0
        total_intern_months = 0
        undated_intern = False  # internship listed without any duration (unverified experience)

        # Case 1: Pure internship section found
        if intern_text:
            date_matches = self._find_date_matches(intern_text)
            if date_matches:
                date_matches.sort(key=lambda m: m.start())
                for dm in date_matches:
                    start_idx = max(0, dm.start() - 60)
                    end_idx = min(len(intern_text), dm.end() + 60)
                    block = intern_text[start_idx:end_idx]
                    total_intern_months += self._parse_entry_duration_months(block)
            else:
                total_intern_months += self._parse_entry_duration_months(intern_text)

            # If still 0 months detected but candidate clearly listed internships without explicit dates,
            # mark with -0.01 as a flag for semantic/lexical experience relevance scoring
            if total_intern_months == 0 and len(intern_text.split()) >= 4:
                undated_intern = True

        # Case 2: Experience section found
        if exp_text:
            date_matches = self._find_date_matches(exp_text)
            date_matches.sort(key=lambda m: m.start())

            if date_matches:
                w_m, i_m = self._accumulate_entries(exp_text, date_matches, require_role_words=False)
                total_work_months += w_m
                total_intern_months += i_m
            else:
                entry_months = self._parse_entry_duration_months(exp_text)
                if re.search(r"\b(intern|internship|trainee)\b", exp_text, re.I):
                    if entry_months > 0:
                        total_intern_months += entry_months
                    elif len(exp_text.split()) >= 4:
                        undated_intern = True
                else:
                    total_work_months += entry_months

        # Case 3: If no experience or internship headers exist in sections, check full text
        # BUT strictly ignore education, projects, skills, and certifications sections
        if total_work_months == 0 and total_intern_months == 0 and not undated_intern:
            edu_text = sections.get("education", "")
            proj_text = sections.get("projects", "")
            cert_text = sections.get("certifications", "")
            skills_text = sections.get("skills", "")

            ignored_content = set(
                edu_text.splitlines() + 
                proj_text.splitlines() + 
                cert_text.splitlines() + 
                skills_text.splitlines()
            )

            candidate_lines = []
            for line in full_text.splitlines():
                if line.strip() in ignored_content:
                    continue
                if DEGREE_LINE_RE.search(line) or re.search(r"\b(college|university|school|cgpa|gpa|percentage|degree)\b", line, re.I):
                    continue
                candidate_lines.append(line)

            candidate_text = "\n".join(candidate_lines)
            date_matches = (
                list(DATE_DOT_RE.finditer(candidate_text)) +
                list(DATE_SLASH_RE.finditer(candidate_text)) + 
                list(DATE_WITH_MONTH_RE.finditer(candidate_text)) + 
                list(YEAR_RANGE_RE.finditer(candidate_text)) +
                list(MONTHS_DURATION_RE.finditer(candidate_text))
            )
            date_matches.sort(key=lambda m: m.start())

            w_m, i_m = self._accumulate_entries(candidate_text, date_matches, require_role_words=True)
            total_work_months += w_m
            total_intern_months += i_m

        work_years = round(total_work_months / 12.0, 2)
        if undated_intern and total_intern_months == 0:
            intern_years = -0.01  # Special marker: internship exists without explicit duration
            effective_total = work_years
        else:
            intern_years = round(total_intern_months / 12.0, 2)
            effective_total = round(work_years + (intern_years * 0.5), 2)

        return work_years, intern_years, effective_total

    def extract_education(self, text: str, education_sec: str) -> List[EducationEntry]:
        """Extracts degrees, institutions, and canonical tiers."""
        entries: List[EducationEntry] = []
        source_text = education_sec if education_sec.strip() else text

        for line in source_text.splitlines():
            line_str = line.strip()
            match = DEGREE_LINE_RE.search(line_str)
            if match:
                raw_deg = match.group(0).strip()
                tier = normalize_degree(raw_deg)
                
                inst = None
                inst_match = re.search(r"\b([A-Za-z\s]+(?:University|College|Institute|Academy|School))\b", line_str, re.I)
                if inst_match:
                    inst = inst_match.group(1).strip()

                entries.append(EducationEntry(
                    degree=raw_deg,
                    tier=tier,
                    institution=inst
                ))

        if not entries:
            for pattern_str in ["Ph.D", "PhD", "Master", "M.Tech", "MSc", "MS", "B.Tech", "BE", "Bachelor", "BSc", "BS"]:
                if re.search(rf"\b{pattern_str}\b", text, re.I):
                    tier = normalize_degree(pattern_str)
                    entries.append(EducationEntry(degree=pattern_str, tier=tier))
                    break

        return entries

    def extract_structured(self, candidate_id: str, source_file: str, file_hash: str, raw_text: str, cleaned_text: str) -> ParsedResume:
        """Orchestrates structured deterministic extraction without LLM calls."""
        name = self.extract_name(cleaned_text)
        email, phone = self.extract_contact(raw_text)
        sections = self.segment_sections(cleaned_text)
        
        # Extract skills across the full text and weighted in skills section
        skills = extract_skills_from_text(cleaned_text, expand=True)
        work_exp_years, intern_years, effective_exp_years = self.extract_experience_breakdown(cleaned_text, sections)
        education_entries = self.extract_education(cleaned_text, sections.get("education", ""))
        degrees = [e.tier for e in education_entries if e.tier != "none"]

        # Extract project snippets
        project_lines = [l.strip("- ") for l in sections.get("projects", "").splitlines() if len(l.strip()) > 10][:5]

        # Extract certification lines (clean issuer prefixes if needed)
        cert_raw = sections.get("certifications", "")
        cert_lines = [l.strip("- ") for l in cert_raw.splitlines() if len(l.strip()) > 3][:10]

        text_hash = compute_text_hash(cleaned_text)

        return ParsedResume(
            candidate_id=candidate_id,
            source_file=source_file,
            file_hash=file_hash,
            text_hash=text_hash,
            name=name,
            email=email,
            phone=phone,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            skills=skills,
            experience_years=effective_exp_years,
            work_experience_years=work_exp_years,
            internship_years=intern_years,
            education=education_entries,
            degrees=degrees,
            projects=project_lines,
            certifications=cert_lines,
            section_texts={
                "skills": sections.get("skills", ""),
                "experience": sections.get("experience", ""),
                "internship": sections.get("internship", ""),
                "projects": sections.get("projects", ""),
                "education": sections.get("education", ""),
                "certifications": cert_raw
            }
        )
