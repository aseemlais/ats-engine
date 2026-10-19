"""Evaluate BM25 normalisation methods against human labels (CSV from evaluation/labelled_pairs_TEMPLATE.csv).
Refuses to run without labelled data; produces no numbers otherwise.   Usage: python scripts/evaluate_bm25_labelled.py FILE.csv [--k 5] [--query-mode both|structured|filtered]
Every (query mode x normalisation method) pair is reported so the BM25 query strategies can be compared on the same labels."""
import csv, math, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    import pydantic  # noqa: F401
except ImportError:
    sys.path.insert(0, str(ROOT / "tests" / "_offline_shims"))
from app.services.bm25_query import QUERY_MODES, build_bm25_query
from app.services.bm25_search import BM25SearchService, METHODS
from app.services.jd_parser import JDParserService

MIN_PAIRS = 30


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]: j += 1
        for t in range(i, j + 1): r[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b); n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)); den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def ndcg(scores, labels, k):
    idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    dcg = sum((2 ** labels[i] - 1) / math.log2(p + 2) for p, i in enumerate(idx))
    ideal = sorted(labels, reverse=True)[:k]; idcg = sum((2 ** l - 1) / math.log2(p + 2) for p, l in enumerate(ideal))
    return dcg / idcg if idcg else float("nan")


def main():
    if len(sys.argv) < 2 or not Path(sys.argv[1]).exists():
        print("No labelled file supplied - nothing evaluated (no labelled dataset exists yet)."); return 2
    k = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 5
    rows = [r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r.get("human_label", "").strip() != ""]
    if not rows:
        print("The file has no labelled rows - nothing evaluated."); return 2
    qm = sys.argv[sys.argv.index("--query-mode") + 1] if "--query-mode" in sys.argv else "both"
    if qm != "both" and qm not in QUERY_MODES:
        print(f"--query-mode must be 'both' or one of {QUERY_MODES}"); return 2
    modes = QUERY_MODES if qm == "both" else (qm,)
    svc, parser, by_jd = BM25SearchService(), JDParserService(), defaultdict(list)
    for r in rows: by_jd[r["jd_file"]].append(r)
    res = {(qmode, m): {"scores": [], "labels": [], "ndcg": []} for qmode in modes for m in METHODS}
    for jd_file, items in by_jd.items():
        jd = parser.parse_job_description((ROOT / jd_file).read_text(encoding="utf-8"), job_id=jd_file)
        ids = [i["resume_file"] for i in items]; texts = [(ROOT / i["resume_file"]).read_text(encoding="utf-8", errors="ignore") for i in items]
        labels = [float(i["human_label"]) for i in items]
        for qmode in modes:
            q = build_bm25_query(jd, qmode)
            for m in METHODS:
                s = svc.score_documents(q.text, ids, texts, method=m).normalized; sc = [s.get(i, 0.0) for i in ids]
                res[(qmode, m)]["scores"] += sc; res[(qmode, m)]["labels"] += labels
                if len(items) >= 2: res[(qmode, m)]["ndcg"].append(ndcg(sc, labels, k))
    n = len(rows); print(f"{n} labelled pairs over {len(by_jd)} JDs")
    if n < MIN_PAIRS: print(f"WARNING: fewer than {MIN_PAIRS} pairs - too few to conclude anything.")
    for key, v in res.items():
        nd = [x for x in v["ndcg"] if x == x]
        print(f"{key[0]:10s} {key[1]:22s} spearman={spearman(v['scores'], v['labels']):.3f}  mean nDCG@{k}={sum(nd)/len(nd) if nd else float('nan'):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
