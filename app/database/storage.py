import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any

from app.config import config
from app.models.resume import ParsedResume
from app.models.job import JobDescription

class MetadataStorage:
    """Manages SQLite storage for structured resume metadata, jobs, and search logs."""
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(config.data_dir / "ats_metadata.db")
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS resumes (
                candidate_id TEXT PRIMARY KEY,
                source_file TEXT,
                file_hash TEXT,
                text_hash TEXT,
                name TEXT,
                email TEXT,
                phone TEXT,
                experience_years REAL,
                skills_json TEXT,
                degrees_json TEXT,
                parsed_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                title TEXT,
                raw_text TEXT,
                parsed_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_hash ON resumes (file_hash);
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_text_hash ON resumes (text_hash);
        """)
        conn.commit()
        conn.close()

    def is_duplicate(self, file_hash: str, text_hash: str) -> bool:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT candidate_id FROM resumes WHERE file_hash = ? OR text_hash = ? LIMIT 1", (file_hash, text_hash))
        row = c.fetchone()
        conn.close()
        return row is not None

    def save_resume(self, resume: ParsedResume):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO resumes (
                candidate_id, source_file, file_hash, text_hash, name, email, phone,
                experience_years, skills_json, degrees_json, parsed_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            resume.candidate_id,
            resume.source_file,
            resume.file_hash,
            resume.text_hash,
            resume.name,
            resume.email,
            resume.phone,
            resume.experience_years,
            json.dumps(resume.skills),
            json.dumps(resume.degrees),
            resume.model_dump_json()
        ))
        conn.commit()
        conn.close()

    def save_resumes_batch(self, resumes: List[ParsedResume]):
        conn = self._get_connection()
        c = conn.cursor()
        data = [
            (
                r.candidate_id,
                r.source_file,
                r.file_hash,
                r.text_hash,
                r.name,
                r.email,
                r.phone,
                r.experience_years,
                json.dumps(r.skills),
                json.dumps(r.degrees),
                r.model_dump_json()
            )
            for r in resumes
        ]
        c.executemany("""
            INSERT OR REPLACE INTO resumes (
                candidate_id, source_file, file_hash, text_hash, name, email, phone,
                experience_years, skills_json, degrees_json, parsed_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        conn.commit()
        conn.close()

    def get_resume(self, candidate_id: str) -> Optional[ParsedResume]:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT parsed_json FROM resumes WHERE candidate_id = ?", (candidate_id,))
        row = c.fetchone()
        conn.close()
        if row:
            return ParsedResume.model_validate_json(row[0])
        return None

    def get_all_resumes(self) -> List[ParsedResume]:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT parsed_json FROM resumes")
        rows = c.fetchall()
        conn.close()
        return [ParsedResume.model_validate_json(r[0]) for r in rows]

    def count_resumes(self) -> int:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM resumes")
        count = c.fetchone()[0]
        conn.close()
        return count

    def save_job(self, job: JobDescription):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO jobs (job_id, title, raw_text, parsed_json)
            VALUES (?, ?, ?, ?)
        """, (job.job_id, job.title, job.raw_text, job.model_dump_json()))
        conn.commit()
        conn.close()

    def get_job(self, job_id: str) -> Optional[JobDescription]:
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT parsed_json FROM jobs WHERE job_id = ?", (job_id,))
        row = c.fetchone()
        conn.close()
        if row:
            return JobDescription.model_validate_json(row[0])
        return None
