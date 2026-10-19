import sqlite3
import json
from app.services.nlp_extractor import NLPExtractorService
from scripts.export_structured_summary import export_summary

def sync_all():
    conn = sqlite3.connect("data/ats_metadata.db")
    c = conn.cursor()
    extractor = NLPExtractorService()

    rows = c.execute("SELECT candidate_id, parsed_json FROM resumes").fetchall()
    print(f"Syncing experience breakdown for {len(rows)} resumes...")
    for cid, p_json in rows:
        parsed = json.loads(p_json)
        cleaned = parsed.get("cleaned_text", "")
        sections = extractor.segment_sections(cleaned)
        work, intern, effective = extractor.extract_experience_breakdown(cleaned, sections)

        parsed["experience_years"] = effective
        parsed["work_experience_years"] = work
        parsed["internship_years"] = intern

        c.execute("UPDATE resumes SET experience_years = ?, parsed_json = ? WHERE candidate_id = ?",
                  (effective, json.dumps(parsed), cid))

    conn.commit()

    # Query Akalya and Praveen specifically
    for pattern in ["%Akalya%", "%Praveen_N%"]:
        row = c.execute("SELECT candidate_id, name, experience_years, parsed_json, source_file FROM resumes WHERE source_file LIKE ?", (pattern,)).fetchone()
        if row:
            p = json.loads(row[3])
            print("--------------------------------------------------")
            print(f"File: {row[4]}")
            print(f"Candidate: {row[1]} ({row[0]})")
            print(f"  - Full-Time Work Experience: {p.get('work_experience_years', 0.0)} yrs")
            print(f"  - Internship Experience:    {p.get('internship_years', 0.0)} yrs")
            print(f"  - Effective Experience:     {row[2]} yrs")

    conn.close()
    print("--------------------------------------------------")
    export_summary()

if __name__ == "__main__":
    sync_all()
