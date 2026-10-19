import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from qdrant_client.models import PointStruct

from app.config import config
from app.models.resume import ParsedResume
from app.services.resume_parser import ResumeParserService
from app.services.nlp_extractor import NLPExtractorService
from app.services.embedding_service import EmbeddingService
from app.database.storage import MetadataStorage
from app.database.qdrant import QdrantVectorDB
from app.services.bm25_search import BM25SearchService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IngestionPipeline")

class IngestionPipeline:
    def __init__(
        self,
        parser_service: Optional[ResumeParserService] = None,
        nlp_service: Optional[NLPExtractorService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        storage: Optional[MetadataStorage] = None,
        qdrant_db: Optional[QdrantVectorDB] = None,
        bm25_service: Optional[BM25SearchService] = None
    ):
        self.parser_service = parser_service or ResumeParserService()
        self.nlp_service = nlp_service or NLPExtractorService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.storage = storage or MetadataStorage()
        self.qdrant_db = qdrant_db or QdrantVectorDB()
        self.bm25_service = bm25_service or BM25SearchService()
        self.checkpoint_file = config.checkpoint_file
        self.vectors_cache_file = config.data_dir / "vectors_cache.npy"
        self.vectors_index_file = config.data_dir / "vectors_index.json"

    def load_checkpoint(self) -> set:
        """Loads set of processed file paths/hashes from checkpoint."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    data = json.load(f)
                    return set(data.get("processed_files", []))
            except Exception as e:
                logger.warning(f"Failed to read checkpoint: {e}")
        return set()

    def save_checkpoint(self, processed_files: set):
        """Saves current progress to disk."""
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, "w") as f:
            json.dump({"processed_files": list(processed_files), "updated_at": time.time()}, f, indent=2)

    def ingest_single_file(self, file_path: Path, candidate_id: str) -> Tuple[bool, Optional[ParsedResume], str]:
        """Parses and extracts structured data from a single resume file with error handling."""
        t0 = time.time()
        success, raw_text, cleaned_text, file_hash, err = self.parser_service.parse_file(file_path)
        if not success:
            logger.error(f"[{candidate_id}] Failed parsing {file_path.name}: {err}")
            return False, None, err

        try:
            parsed = self.nlp_service.extract_structured(
                candidate_id=candidate_id,
                source_file=file_path.name,
                file_hash=file_hash or "",
                raw_text=raw_text,
                cleaned_text=cleaned_text
            )
            elapsed = time.time() - t0
            logger.info(f"[{candidate_id}] Parsed {file_path.name} in {elapsed:.2f}s ({len(parsed.skills)} skills, {parsed.experience_years:.1f} yrs exp)")
            return True, parsed, ""
        except Exception as e:
            logger.error(f"[{candidate_id}] NLP extraction failed: {e}")
            return False, None, str(e)

    def ingest_directory(
        self,
        directory: Path | str,
        batch_size: int = 128,
        reset_checkpoint: bool = False
    ) -> Dict[str, Any]:
        """
        Processes an entire directory of resumes with batching, duplicate detection,
        checkpointing, vector generation, and BM25 indexing.
        """
        dir_path = Path(directory)
        files = sorted([f for f in dir_path.glob("*") if f.suffix.lower() in (".pdf", ".docx", ".txt")])
        logger.info(f"Starting ingestion of {len(files)} resume files from {dir_path}")

        processed_files = set() if reset_checkpoint else self.load_checkpoint()
        new_resumes: List[ParsedResume] = []
        errors: List[Dict[str, str]] = []
        duplicates: int = 0

        # Pass 1: Parse and extract structured text
        for idx, file_path in enumerate(files, start=1):
            file_key = str(file_path.name)
            if file_key in processed_files:
                continue

            cid = f"CV_{idx:05d}"
            ok, parsed, err = self.ingest_single_file(file_path, cid)
            if not ok:
                errors.append({"file": file_path.name, "error": err})
                continue

            # Duplicate check
            if self.storage.is_duplicate(parsed.file_hash, parsed.text_hash):
                logger.info(f"Skipping duplicate resume: {file_path.name}")
                duplicates += 1
                processed_files.add(file_key)
                continue

            new_resumes.append(parsed)
            processed_files.add(file_key)

            # Periodic checkpointing
            if len(new_resumes) % config.ingestion_batch_size == 0:
                self.storage.save_resumes_batch(new_resumes[-config.ingestion_batch_size:])
                self.save_checkpoint(processed_files)
                logger.info(f"Checkpoint saved. Total processed so far: {len(processed_files)}")

        if new_resumes:
            self.storage.save_resumes_batch(new_resumes)
            self.save_checkpoint(processed_files)

        # Pass 2: Embeddings & Vector Indexing
        logger.info("Computing embeddings and indexing into Qdrant vector database...")
        all_resumes = self.storage.get_all_resumes()
        
        qdrant_points: List[PointStruct] = []
        t_embed_start = time.time()
        
        # Batch encode entire corpus
        vectors_dict = self.embedding_service.compute_corpus_vectors_batched(all_resumes)

        for idx, r in enumerate(all_resumes):
            vecs = vectors_dict.get(r.candidate_id)
            if vecs:
                qdrant_points.append(
                    PointStruct(
                        id=idx + 1,
                        vector=vecs["full"].tolist(),
                        payload={
                            "candidate_id": r.candidate_id,
                            "name": r.name,
                            "skills": r.skills,
                            "experience_years": r.experience_years,
                            "degrees": r.degrees,
                            "source_file": r.source_file
                        }
                    )
                )

        if qdrant_points:
            self.qdrant_db.upsert_points(qdrant_points, batch_size=256)
        embed_time = time.time() - t_embed_start
        logger.info(f"Vector embeddings generated and stored in {embed_time:.2f}s")

        # Pass 3: Index BM25
        logger.info("Building BM25 lexical index...")
        c_ids = [r.candidate_id for r in all_resumes]
        texts = [r.cleaned_text for r in all_resumes]
        self.bm25_service.build_index(c_ids, texts)
        logger.info(f"BM25 index built over {len(c_ids)} documents.")

        # Pass 4: Auto-update extracted_resumes_summary.md for VS Code viewing
        try:
            from scripts.export_structured_summary import export_summary
            export_summary()
            logger.info("Auto-updated extracted_resumes_summary.md with latest candidate repository.")
        except Exception as e:
            logger.warning(f"Could not auto-update markdown summary: {e}")

        return {
            "total_files": len(files),
            "newly_parsed": len(new_resumes),
            "total_indexed": len(all_resumes),
            "duplicates": duplicates,
            "errors": errors,
            "embedding_time": embed_time,
            "vectors_cache": vectors_dict
        }
