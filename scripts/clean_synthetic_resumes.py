import sqlite3
import json
from pathlib import Path
from app.database.storage import MetadataStorage
from app.services.bm25_search import BM25SearchService

def clean_database():
    db_file = Path("data/ats_metadata.db")
    if db_file.exists():
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute("DELETE FROM resumes WHERE source_file LIKE 'CV_%'")
        conn.commit()
        remaining = c.execute("SELECT COUNT(*) FROM resumes").fetchone()[0]
        conn.close()
        print(f"Purged synthetic records. Remaining real resumes in SQLite: {remaining}")

    # Clean checkpoint
    chk_file = Path("data/ingestion_checkpoint.json")
    if chk_file.exists():
        with open(chk_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        cleaned_chk = [item for item in data if not item.startswith("CV_")]
        with open(chk_file, "w", encoding="utf-8") as f:
            json.dump(cleaned_chk, f, indent=2)
        print(f"Cleaned ingestion checkpoint. Retained {len(cleaned_chk)} real resume entries.")

    # Delete synthetic files from data/resumes
    resumes_dir = Path("data/resumes")
    if resumes_dir.exists():
        deleted = 0
        for f in resumes_dir.glob("CV_*"):
            try:
                f.unlink()
                deleted += 1
            except Exception:
                pass
        print(f"Removed {deleted} synthetic resume files from data/resumes.")

    # Rebuild BM25 index on only real resumes
    storage = MetadataStorage()
    real_resumes = storage.get_all_resumes()
    bm25 = BM25SearchService()
    c_ids = [r.candidate_id for r in real_resumes]
    texts = [r.cleaned_text for r in real_resumes]
    bm25.build_index(c_ids, texts)
    print(f"BM25 index rebuilt successfully on {len(real_resumes)} remaining real resumes.")

if __name__ == "__main__":
    clean_database()
