"""Rebuild the persisted BM25 reference corpus from the resumes stored in data/ats_metadata.db (no embedding model needed).

Why: the previous screening code overwrote data/bm25_index.pkl with a ONE-document corpus ('CAND_SCREEN'). BM25
IDF/length statistics from such an index are meaningless. Run this once (and after any bulk change of the resume
store) to restore a proper reference corpus. It refuses to write an empty corpus and never touches the resume DB.

Usage (project root):  python scripts/rebuild_bm25_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.database.storage import MetadataStorage
from app.services.bm25_search import BM25SearchService


def main() -> int:
    resumes = MetadataStorage().get_all_resumes()
    if not resumes:
        print("No resumes in the metadata DB - nothing written (existing index left untouched).")
        return 1
    svc = BM25SearchService()
    before = len(svc.candidate_ids)
    svc.build_index([r.candidate_id for r in resumes], [r.cleaned_text for r in resumes], persist=True)
    print(f"BM25 reference corpus rebuilt: {before} -> {len(resumes)} documents ({svc.index_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
