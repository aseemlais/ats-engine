import sqlite3
import json
from pathlib import Path
from app.services.nlp_extractor import NLPExtractorService
from app.services.resume_parser import ResumeParserService
from app.workers.ingestion_pipeline import IngestionPipeline
from scripts.export_structured_summary import export_summary

def run_resync():
    # 1. Purge synthetic txt files
    conn = sqlite3.connect("data/ats_metadata.db")
    c = conn.cursor()
    c.execute("DELETE FROM resumes WHERE source_file LIKE '%.txt'")
    conn.commit()
    print("Purged any .txt records from database.")
    
    # 2. Re-ingest real resumes directory from disk
    pipeline = IngestionPipeline()
    real_folder = Path(r"C:\Users\LENOVO\OneDrive\Documents\Resumes & JD\Resume 2\Resume")
    if real_folder.exists():
        print(f"Re-ingesting all real resumes from {real_folder} with fresh parser & extractor...")
        # Reset checkpoint to parse all files fresh with new block sorting and date extraction
        res = pipeline.ingest_directory(real_folder, reset_checkpoint=True)
        print("Ingestion results:", res)
    else:
        print(f"Resume directory not found: {real_folder}")

    # 3. Print verification for candidates
    c = conn.cursor()
    test_names = ["%Anantharam%", "%Dhaksha%", "%Dharanya%", "%Adeeswaran%", "%Fidhaa%", "%Akalya%", "%Praveen_N%"]
    for pat in test_names:
        row = c.execute("SELECT candidate_id, name, experience_years, parsed_json, source_file FROM resumes WHERE source_file LIKE ?", (pat,)).fetchone()
        if row:
            p = json.loads(row[3])
            print("--------------------------------------------------")
            print(f"Candidate: {row[1]} ({row[0]}) | File: {row[4]}")
            print(f"  - Work Experience: {p.get('work_experience_years', 0.0)} yrs")
            print(f"  - Internship Experience: {p.get('internship_years', 0.0)} yrs")
            print(f"  - Effective Experience: {row[2]} yrs")
            print(f"  - Certifications count: {len(p.get('certifications', []))}")
        else:
            print(f"Candidate pattern not found in DB: {pat}")
    
    conn.close()

if __name__ == "__main__":
    run_resync()
