"""
BM25+ lexical scoring with documented score semantics.

Three DIFFERENT quantities are returned and never mixed:
  raw         BM25+ sum over unique query terms (unbounded above, depends on corpus statistics). Not a percentage.
  normalized  raw mapped to [0, 1] by `normalization`:
                theoretical_bound    raw / sum_t idf_t * (delta + k1 + 1) over unique, SEEN query terms. The per-term
                                     bound is the tf -> infinity limit, so raw <= bound always (no clipping needed).
                                     Reads as "fraction of the theoretical maximum lexical evidence"; a resume that
                                     mentions every term once scores roughly 0.5, never 1. Absolute, batch independent.
                reference_percentile EXPERIMENTAL. Mid-rank percentile of raw among the reference corpus scored with the
                                     same query. Relative to the reference population, not an absolute match level.
                legacy_ceiling       the previous method (kept for comparison/reproduction): raw / score of a document
                                     holding every query term at the query's own frequency. Not an upper bound
                                     (resumes can exceed it and are clipped to 1.0); unseen terms deflate it.
  percentile  always available as a diagnostic: reference percentile of raw.

Corpus handling: the persisted index is the REFERENCE corpus (IDF, average length). With a reliable reference
(>= BM25_MIN_REFERENCE_DOCS documents) only the reference feeds the statistics, so scores are exactly independent of the
scored batch; query terms absent from the reference are reported as unseen and excluded from raw score and bound. Scoring candidates never mutates
shared state (score_documents); only build_index(persist=True) - ingestion - rewrites the stored index.
Duplicate ids or identical documents are counted once in the statistics.
"""
import hashlib
import math
import os
import pickle
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from app.config import config
from app.utils.text_cleaner import TOKENIZER_VERSION, tokenize_for_bm25

METHODS = ("theoretical_bound", "reference_percentile", "legacy_ceiling")


@dataclass
class BM25Result:
    normalized: Dict[str, float]
    raw: Dict[str, float]
    percentile: Dict[str, float]
    method: str
    query_tokens: List[str]
    unique_terms: List[str]
    unseen_terms: List[str]            # terms found in no document (reference + scored)
    denominator: float                 # normalisation denominator (bound / ceiling); 0 for percentile
    reference_size: int
    reference_reliability: str         # "ok" | "low"
    matched_terms: Dict[str, List[str]] = field(default_factory=dict)
    unmatched_terms: Dict[str, List[str]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


class BM25SearchService:
    K1, B, DELTA = 1.5, 0.75, 1.0  # standard BM25+ parameters (Lv & Zhai, 2011)

    def __init__(self, index_path: Path | None = None):
        self.index_path = index_path or config.bm25_index_file
        self._lock = threading.RLock()
        self.candidate_ids: List[str] = []          # legacy attributes: the currently loaded index
        self.corpus_tokens: List[List[str]] = []
        self.reference_corpus: Dict[str, List[str]] = {}
        self._loaded_sig: Optional[Tuple[int, int]] = None
        self.tokenizer_version_loaded: Optional[int] = None
        self._load_index()
        self.reference_corpus = dict(zip(self.candidate_ids, self.corpus_tokens))
        self._ref_cache: Optional[Tuple[Counter, int, int]] = None

    # ------------------------------------------------------------------ persistence
    def _file_sig(self) -> Optional[Tuple[int, int]]:
        try:
            st = self.index_path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def _load_index(self):
        self._ref_cache = None
        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                ids, toks = data.get("candidate_ids", []), data.get("corpus_tokens", [])
                self.tokenizer_version_loaded = data.get("tokenizer_version", 1)
                self.candidate_ids, self.corpus_tokens = list(ids), list(toks)
            except Exception:
                self.candidate_ids, self.corpus_tokens = [], []
        self._loaded_sig = self._file_sig()

    def save_index(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"candidate_ids": self.candidate_ids, "corpus_tokens": self.corpus_tokens,
                         "tokenizer_version": TOKENIZER_VERSION}, f)
        os.replace(tmp, self.index_path)               # atomic: readers never see a half-written file
        self.tokenizer_version_loaded = TOKENIZER_VERSION
        self._loaded_sig = self._file_sig()

    def reload_index(self):
        """Explicit reload of the persisted reference corpus (e.g. after ingestion in another process)."""
        with self._lock:
            self._load_index()
            self.reference_corpus = dict(zip(self.candidate_ids, self.corpus_tokens))

    def is_stale(self) -> bool:
        """True if the file changed since it was loaded, or was tokenised by another tokenizer version."""
        return self._file_sig() != self._loaded_sig or (
            self.tokenizer_version_loaded is not None and self.tokenizer_version_loaded != TOKENIZER_VERSION)

    def refresh_if_stale(self) -> bool:
        with self._lock:
            if self._file_sig() != self._loaded_sig:
                self.reload_index()
                return True
        return False

    def build_index(self, candidate_ids: List[str], texts: List[str], persist: bool = True):
        """
        persist=True (ingestion): replaces the reference corpus and writes the index atomically.
        persist=False: legacy in-memory mode that MUTATES this service's loaded index; request code should use
        score_documents() instead, which never mutates shared state.
        Duplicate ids keep the LAST text; identical texts under different ids are kept (they are real documents)
        but counted once in corpus statistics.
        """
        with self._lock:
            dedup: Dict[str, List[str]] = {}
            for cid, t in zip(candidate_ids, texts):
                dedup[cid] = tokenize_for_bm25(t)
            self.candidate_ids, self.corpus_tokens = list(dedup), list(dedup.values())
            self._ref_cache = None
            if persist and self.corpus_tokens:
                self.reference_corpus = dict(dedup)
                self.save_index()

    # ------------------------------------------------------------------ statistics
    @staticmethod
    def _fp(tokens: Sequence[str]) -> str:
        return hashlib.md5(" ".join(tokens).encode()).hexdigest()

    def _reference_stats(self) -> Tuple[Counter, int, int]:
        if self._ref_cache is None:
            df: Counter = Counter()
            total = 0
            for toks in self.reference_corpus.values():
                total += len(toks)
                df.update(set(toks))
            self._ref_cache = (df, len(self.reference_corpus), total)
        return self._ref_cache

    def reference_info(self) -> Dict:
        n = len(self.reference_corpus)
        return {"documents": n, "reliability": "ok" if n >= config.bm25_min_reference_docs else "low",
                "min_documents_for_reliable": config.bm25_min_reference_docs,
                "tokenizer_version": self.tokenizer_version_loaded, "stale": self.is_stale()}

    def _tf_score(self, idf: float, f: int, dl: int, avgdl: float) -> float:
        norm = self.K1 * (1 - self.B + self.B * dl / max(avgdl, 1e-9))
        return idf * (self.DELTA + f * (self.K1 + 1) / (f + norm))

    # ------------------------------------------------------------------ scoring
    def score_documents(self, query_text: str, candidate_ids: Sequence[str], texts: Sequence[str],
                        method: Optional[str] = None, top_k: Optional[int] = None,
                        pretokenized: bool = False) -> BM25Result:
        """Scores the given documents against the query using the reference corpus statistics. Never mutates
        shared state, so concurrent requests cannot interfere."""
        method = (method or config.bm25_normalization)
        if method not in METHODS:
            raise ValueError(f"Unknown BM25 normalization '{method}'; choose one of {METHODS}")
        with self._lock:                                  # consistent snapshot of the reference
            ref = self.reference_corpus
            ref_df, n_ref, ref_len = self._reference_stats()
            ref_items = None
        docs: Dict[str, List[str]] = {}
        for cid, t in zip(candidate_ids, texts):
            docs[cid] = t.split() if pretokenized else tokenize_for_bm25(t)   # duplicate ids: last wins

        q_tokens = tokenize_for_bm25(query_text, query=True)
        uniq = list(dict.fromkeys(q_tokens))               # repeated query terms count once
        notes: List[str] = []
        ref_ok = n_ref >= config.bm25_min_reference_docs

        # Statistics. Reliable reference (and not the legacy method): reference ONLY, so a candidate's score cannot
        # depend on which other documents are scored with it. Unreliable/small reference (or legacy): reference +
        # scored documents that are not already in the reference (by id or identical content), counted once.
        use_scored_in_stats = (not ref_ok) or method == "legacy_ceiling"
        ref_fps = None
        extra_df: Counter = Counter()
        extra_n = extra_len = 0
        seen_fp = set()
        for cid, toks in docs.items():
            if not use_scored_in_stats or cid in ref:
                continue
            fp = self._fp(toks)
            if fp in seen_fp:
                continue
            if ref_fps is None:
                ref_fps = {self._fp(v) for v in ref.values()}
            if fp in ref_fps:
                continue
            seen_fp.add(fp)
            extra_n += 1; extra_len += len(toks); extra_df.update(set(toks))
        N = n_ref + extra_n
        avgdl = (ref_len + extra_len) / N if N else 1.0
        df = {t: ref_df.get(t, 0) + extra_df.get(t, 0) for t in uniq}
        idf = {t: math.log((N + 1) / d) for t, d in df.items() if d > 0}
        unseen = [t for t in uniq if df[t] == 0]
        max_idf = math.log(N + 1) if N else 0.0

        empty = BM25Result({}, {}, {}, method, q_tokens, uniq, unseen, 0.0, n_ref, "ok" if ref_ok else "low", notes=notes)
        if not docs:
            return empty
        if not uniq:
            notes.append("empty query")
            z = {cid: 0.0 for cid in docs}
            # normalized stays EMPTY for an empty query (existing contract: the ranker then marks BM25 not applicable)
            return BM25Result({}, dict(z), dict(z), method, q_tokens, uniq, unseen, 0.0, n_ref, empty.reference_reliability, notes=notes)

        def raw_score(tf: Counter, dl: int) -> float:
            return sum(self._tf_score(idf[t], tf[t], dl, avgdl) for t in uniq if t in idf and tf.get(t, 0) > 0)

        raw: Dict[str, float] = {}
        matched: Dict[str, List[str]] = {}
        for cid, toks in docs.items():
            tf = Counter(toks)
            raw[cid] = raw_score(tf, max(len(toks), 1))
            matched[cid] = [t for t in uniq if tf.get(t, 0) > 0]
        unmatched = {cid: [t for t in uniq if t not in set(m)] for cid, m in matched.items()}

        # Percentile of raw within the reference corpus (diagnostic; also the experimental normalisation).
        # A candidate that IS a reference document is never compared with itself (leave-one-out).
        import bisect
        ref_raw_by_id = {rid: raw_score(Counter(toks), max(len(toks), 1)) for rid, toks in ref.items()}
        ref_sorted = sorted(ref_raw_by_id.values())
        def pct(cid: str, v: float) -> float:
            lo, hi = bisect.bisect_left(ref_sorted, v - 1e-12), bisect.bisect_right(ref_sorted, v + 1e-12)
            n = len(ref_sorted)
            if cid in ref_raw_by_id:                       # remove the candidate's own reference entry
                own = ref_raw_by_id[cid]
                n -= 1
                if own < v - 1e-12:
                    lo -= 1; hi -= 1
                elif own <= v + 1e-12:
                    hi -= 1
            return 0.0 if n <= 0 else (lo + 0.5 * (hi - lo)) / n
        percentile = {cid: pct(cid, r) for cid, r in raw.items()}

        if method == "theoretical_bound":
            terms = [t for t in uniq if t in idf]
            if not ref_ok:
                # unreliable (small) reference: keep unseen terms in the denominator at maximum idf (conservative)
                bound = sum(idf[t] * (self.DELTA + self.K1 + 1) for t in terms) + len(unseen) * max_idf * (self.DELTA + self.K1 + 1)
                notes.append(f"reference corpus has {n_ref} < {config.bm25_min_reference_docs} documents: unseen terms kept in the denominator")
            else:
                bound = sum(idf[t] * (self.DELTA + self.K1 + 1) for t in terms)
            norm = {cid: (min(max(r / bound, 0.0), 1.0) if bound > 0 else 0.0) for cid, r in raw.items()}
            denominator = bound
        elif method == "reference_percentile":
            norm = dict(percentile); denominator = 0.0
            if not ref_ok:
                notes.append(f"reference corpus has {n_ref} < {config.bm25_min_reference_docs} documents: percentile unreliable")
        else:  # legacy_ceiling
            q_tf = Counter(q_tokens)
            avg_dl = round(avgdl)
            ceiling = sum(self._tf_score(idf.get(t, max_idf), f, avg_dl, avgdl) for t, f in q_tf.items())
            norm = {cid: (min(max(r / ceiling, 0.0), 1.0) if ceiling > 0 else 0.0) for cid, r in raw.items()}
            denominator = ceiling

        if top_k is not None and top_k < len(norm):
            keep = sorted(norm, key=lambda c: (-norm[c], c))[:max(top_k, 0)]
            norm = {c: norm[c] for c in keep}
        return BM25Result(norm, raw, percentile, method, q_tokens, uniq, unseen, denominator, n_ref,
                          "ok" if ref_ok else "low", matched, unmatched, notes)

    def search(self, query_text: str, top_k: Optional[int] = None, method: Optional[str] = None) -> Dict[str, float]:
        """Backward compatible: scores the CURRENTLY LOADED index; returns {candidate_id: normalized in [0, 1]}.
        top_k=None returns every document; a number returns only the best top_k by normalized score."""
        with self._lock:
            ids, toks = list(self.candidate_ids), list(self.corpus_tokens)
        if not ids:
            return {}
        # stored tokens are already final: they are NOT re-tokenised (context-gated tokens such as R/C would be lost)
        res = self.score_documents(query_text, ids, [" ".join(t) for t in toks], method=method, top_k=top_k,
                                 pretokenized=True)
        return res.normalized
