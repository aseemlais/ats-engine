import sqlite3
import json
from pathlib import Path

def export_summary():
    db_path = Path("data/ats_metadata.db")
    if not db_path.exists():
        print("Database not found.")
        return

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    rows = c.execute(
        "SELECT candidate_id, name, source_file, email, phone, experience_years, skills_json, degrees_json, parsed_json FROM resumes ORDER BY candidate_id"
    ).fetchall()
    conn.close()

    output_path = Path("extracted_resumes_summary.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 📄 Candidate Resumes Structured Repository\n\n")
        f.write(f"**Total Verified Real Resumes in Database:** {len(rows)}  \n")
        f.write(f"**Database Source:** `data/ats_metadata.db`  \n\n")
        f.write("---\n\n")

        for idx, (cid, name, src, email, phone, exp_years, skills_json, degrees_json, p_json) in enumerate(rows, start=1):
            parsed = json.loads(p_json)
            skills = json.loads(skills_json) if skills_json else []
            degrees = json.loads(degrees_json) if degrees_json else []
            work_exp = parsed.get("work_experience_years", 0.0)
            intern_exp = parsed.get("internship_years", 0.0)
            education_list = parsed.get("education", [])
            projects = parsed.get("projects", [])
            certs = parsed.get("certifications", [])

            f.write(f"## {idx}. Candidate `{cid}`: {name or 'Unknown'}\n\n")
            f.write(f"- **Source File:** `{src}`\n")
            f.write(f"- **Email:** {email or 'N/A'}\n")
            f.write(f"- **Phone:** {phone or 'N/A'}\n")
            f.write(f"- **Experience:**\n")
            f.write(f"  - Full-Time Work Experience: **{work_exp} years**\n")
            f.write(f"  - Internship Experience: **{intern_exp} years**\n")
            f.write(f"  - Effective Experience: **{exp_years} years**\n")

            f.write(f"- **Education:**\n")
            if education_list:
                for edu in education_list:
                    deg = edu.get("degree", "N/A")
                    inst = edu.get("institution", "N/A")
                    tier = edu.get("tier", "N/A")
                    f.write(f"  - {deg} ({tier.title()}) — {inst or 'Institution Not Specified'}\n")
            else:
                deg_str = ", ".join(degrees) if degrees else "Not Specified"
                f.write(f"  - {deg_str}\n")

            f.write(f"- **Skills Extracted ({len(skills)}):**\n")
            if skills:
                skills_formatted = ", ".join([f"`{s}`" for s in skills])
                f.write(f"  {skills_formatted}\n")
            else:
                f.write("  *None detected*\n")

            if projects:
                f.write(f"- **Key Projects ({len(projects)}):**\n")
                for p in projects:
                    clean_p = p.replace("\n", " ").strip()
                    f.write(f"  - {clean_p}\n")

            if certs:
                f.write(f"- **Certifications ({len(certs)}):**\n")
                for cert in certs:
                    clean_c = cert.replace("\n", " ").strip()
                    f.write(f"  - {clean_c}\n")

            f.write("\n---\n\n")

    print(f"Export complete: {output_path} generated successfully with {len(rows)} candidates.")

if __name__ == "__main__":
    export_summary()
