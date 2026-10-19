"""Compare BM25 normalisation methods on the SAME resumes and queries (reproducible, no model needed).

Usage (project root):  python scripts/bm25_normalization_compare.py [--out reports/BM25_NORMALIZATION_COMPARISON.md]
Reference corpus = the persisted data/bm25_index.pkl (read only). Nothing here is a labelled evaluation: it measures
mechanical properties (scale, sensitivity, stability), not ranking quality against human judgements.
"""
import argparse, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    import pydantic  # noqa: F401
except ImportError:      # offline sandbox only: labelled test shim, same as tests/run_offline.py
    sys.path.insert(0, str(ROOT / "tests" / "_offline_shims")); print("[note] pydantic missing: using offline shim")
from app.database.storage import MetadataStorage
from app.services.bm25_query import build_bm25_query
from app.services.bm25_search import BM25SearchService, METHODS
from app.services.jd_parser import JDParserService

FIX = ROOT / "tests" / "fixtures"
parse = lambda t: JDParserService().parse_job_description(t, job_id="CMP")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "reports" / "BM25_NORMALIZATION_COMPARISON.md"))
    out_path = Path(ap.parse_args().out)
    svc = BM25SearchService()
    jd_text = (FIX / "jd_databricks_trainee.txt").read_text(encoding="utf-8")
    fresher = (FIX / "resume_fresher_databricks.txt").read_text(encoding="utf-8")
    jd = parse(jd_text); q = build_bm25_query(jd)
    db = MetadataStorage().get_all_resumes()
    ref_n = len(svc.reference_corpus)
    L = [f"# BM25 normalisation comparison (generated)\n", f"Reference corpus: {ref_n} documents (reliability: {svc.reference_info()['reliability']}). "
         f"Query method: `{q.method}`, {len(q.tokens)} tokens ({len(set(q.tokens))} unique).\n"]

    def score(query, docs, method):
        ids = list(docs); return svc.score_documents(query, ids, [docs[i] for i in ids], method=method)

    # 1. strong fresher vs reference population
    docs = {"FRESHER": fresher, **{r.candidate_id: r.cleaned_text for r in db}}
    L += ["## 1. Strong fresher match vs the reference population", "",
          "| method | fresher raw | fresher normalised | median of other resumes | max of other resumes | gap (fresher - best other) |", "|---|---|---|---|---|---|"]
    for m in METHODS:
        r = score(q.text, docs, m); others = [v for k, v in r.normalized.items() if k != "FRESHER"]
        L.append(f"| {m} | {r.raw['FRESHER']:.2f} | {r.normalized['FRESHER']:.3f} | {statistics.median(others):.3f} | {max(others):.3f} | {r.normalized['FRESHER']-max(others):+.3f} |")
    L.append("")

    # 2. administrative text in the JD (query builder bypassed = worst case; builder = production)
    admin = "\nSalary 8-12 LPA, night shift rotational\nInterview process: 3 rounds, notice period 30 days\nLocation: Pune, hybrid, relocation assistance\n"
    q_admin_builder = build_bm25_query(parse(jd_text + admin))
    L += ["## 2. Administrative JD text", "", "| method | clean query | +admin via query builder (production) | +admin, builder bypassed (full text) |", "|---|---|---|---|"]
    for m in METHODS:
        a = score(q.text, {"F": fresher}, m).normalized["F"]
        b = score(q_admin_builder.text, {"F": fresher}, m).normalized["F"]
        c = score(jd_text + admin, {"F": fresher}, m).normalized["F"]
        L.append(f"| {m} | {a:.3f} | {b:.3f} | {c:.3f} |")
    L.append("")

    # 3. query length
    skills = list(dict.fromkeys(q.tokens))
    L += ["## 3. Query length (first k unique query terms; fresher resume)", "", "| k | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for k in (3, 6, 10, len(skills)):
        qq = " ".join(skills[:k]); L.append(f"| {min(k, len(skills))} | " + " | ".join(f"{score(qq, {'F': fresher}, m).normalized['F']:.3f}" for m in METHODS) + " |")
    L.append("")

    # 4. repeated terms: query repeated x5 and a keyword-stuffed resume
    L += ["## 4. Repeated terms", "", "| scenario | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    stuffed = fresher + "\n" + ("python sql databricks " * 30)
    for label, qq, d in (("baseline", q.text, fresher), ("query repeated x5", (q.text + "\n") * 5, fresher),
                         ("resume stuffed with python/sql/databricks x30", q.text, stuffed)):
        L.append(f"| {label} | " + " | ".join(f"{score(qq, {'F': d}, m).normalized['F']:.3f}" for m in METHODS) + " |")
    L.append("")

    # 5. unrelated resumes
    unrelated = {"ACCOUNTING": "accounts payable ledger audit tax invoices reconciliation month end close gaap excel",
                 "NURSING": "registered nurse patient care ward medication administration charting icu",
                 "GIBBERISH": "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor"}
    L += ["## 5. Unrelated documents", "", "| document | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for name, t in unrelated.items():
        L.append(f"| {name} | " + " | ".join(f"{score(q.text, {name: t}, m).normalized[name]:.3f}" for m in METHODS) + " |")
    out_path.parent.mkdir(parents=True, exist_ok=True); out_path.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L)); print(f"\n[written to {out_path}]")


if __name__ == "__main__":
    main()
