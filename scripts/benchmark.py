import time
import os
import sys
import tracemalloc
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import config
from app.database.storage import MetadataStorage
from app.services.resume_parser import ResumeParserService
from app.services.nlp_extractor import NLPExtractorService
from app.services.embedding_service import EmbeddingService
from app.services.bm25_search import BM25SearchService
from app.services.jd_parser import JDParserService
from app.services.hybrid_ranker import HybridRankerService

def run_benchmarks():
    print("=" * 70)
    print("         SCALABLE HYBRID SEMANTIC ATS - SYSTEM BENCHMARK")
    print("=" * 70)

    tracemalloc.start()

    storage = MetadataStorage()
    parser_service = ResumeParserService()
    nlp_service = NLPExtractorService()
    embedding_service = EmbeddingService()
    bm25_service = BM25SearchService()
    jd_parser = JDParserService()
    ranker = HybridRankerService()

    resumes = storage.get_all_resumes()
    n_resumes = len(resumes)
    print(f"Total Resumes Loaded in Database: {n_resumes}")
    if n_resumes == 0:
        print("Please ingest resumes first before benchmarking.")
        return

    # 1. PARSING & NLP EXTRACTION BENCHMARK
    test_slice = resumes[:min(50, n_resumes)]
    t0 = time.time()
    for r in test_slice:
        _ = nlp_service.extract_structured(r.candidate_id, r.source_file, r.file_hash, r.raw_text, r.cleaned_text)
    nlp_time = time.time() - t0
    nlp_per_sec = len(test_slice) / nlp_time if nlp_time > 0 else 0
    print(f"[1] NLP Extraction Speed:     {nlp_time:.3f}s for {len(test_slice)} resumes ({nlp_per_sec:.1f} resumes/sec)")

    # 2. EMBEDDING GENERATION BENCHMARK
    sample_texts = [r.cleaned_text[:500] for r in test_slice]
    t0 = time.time()
    _ = embedding_service.encode_batch(sample_texts)
    embed_time = time.time() - t0
    embed_per_sec = len(sample_texts) / embed_time if embed_time > 0 else 0
    print(f"[2] Embedding Generation:     {embed_time:.3f}s for {len(sample_texts)} resumes ({embed_per_sec:.1f} emb/sec)")

    # 3. BM25 INDEX & SEARCH BENCHMARK
    t0 = time.time()
    cids = [r.candidate_id for r in resumes]
    texts = [r.cleaned_text for r in resumes]
    bm25_service.build_index(cids, texts)
    bm25_index_time = time.time() - t0
    print(f"[3] BM25 Index Build:         {bm25_index_time:.3f}s for {n_resumes} resumes")

    # 4. SEARCH LATENCY OVER DIFFERENT POOL SIZES
    sample_jd = """Job Title: Senior NLP Engineer
Requirements: Python, PyTorch, Transformers, NLP, Machine Learning, 3+ years experience
Preferred Skills: FastAPI, Docker, Kubernetes, AWS"""
    parsed_jd = jd_parser.parse_job_description(sample_jd)

    vec_map = {}
    for r in resumes:
        vec_map[r.candidate_id] = embedding_service.compute_resume_vectors(r)

    print("\n--- SEARCH LATENCY BENCHMARK (End-to-End Hybrid Ranking) ---")
    sizes_to_test = [10, 25, 50, 75, 100, n_resumes]
    sizes_to_test = sorted(list(set([s for s in sizes_to_test if s <= n_resumes])))

    print(f"{'Pool Size':<12} | {'Hard Filtered':<14} | {'Latency (ms)':<14} | {'Throughput (cand/s)'}")
    print("-" * 62)

    for sz in sizes_to_test:
        sub_resumes = resumes[:sz]
        t0 = time.perf_counter()
        resp = ranker.rank_candidates(
            resumes=sub_resumes,
            resume_vectors_map=vec_map,
            jd=parsed_jd,
            top_k=min(50, sz)
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        throughput = sz / (latency_ms / 1000) if latency_ms > 0 else 0
        print(f"{sz:<12} | {resp.hard_filtered_count:<14} | {latency_ms:<14.2f} | {throughput:.0f}")

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("-" * 62)
    print(f"Memory Overhead Tracked:      Current: {current_mem / (1024*1024):.2f} MB | Peak: {peak_mem / (1024*1024):.2f} MB")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmarks()
