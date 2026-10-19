"""Standard-library test runner (no pytest needed). Usage (from project root):
    python tests/run_offline.py [-k substring] [--verbose]

If real `pydantic`/`pytest` import fine they are used; otherwise the clearly-labelled shims in
tests/_offline_shims are used and the run header says so. Supports: module-level @pytest.fixture functions
(with fixture-to-fixture dependencies), tmp_path, monkeypatch, @pytest.mark.parametrize, skip/skipif.
"""
import importlib, inspect, itertools, os, shutil, sys, tempfile, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SHIMS = Path(__file__).resolve().parent / "_offline_shims"
sys.path.insert(0, str(SHIMS))
used_shims = ["pydantic", "pytest"]
import pytest  # noqa: E402


class MonkeyPatch:
    def __init__(self): self._undo = []
    def setattr(self, target, name, value=None):
        if isinstance(target, str):
            mod_path, attr = target.rsplit(".", 1)
            obj, name, value = importlib.import_module(mod_path), attr, name
        else:
            obj = target
        old = getattr(obj, name, None); self._undo.append(("a", obj, name, old)); setattr(obj, name, value)
    def setenv(self, k, v):
        self._undo.append(("e", k, os.environ.get(k))); os.environ[k] = str(v)
    def delenv(self, k, raising=True):
        self._undo.append(("e", k, os.environ.get(k))); os.environ.pop(k, None)
    def undo(self):
        for rec in reversed(self._undo):
            if rec[0] == "a": setattr(rec[1], rec[2], rec[3])
            else:
                if rec[2] is None: os.environ.pop(rec[1], None)
                else: os.environ[rec[1]] = rec[2]


def build(fn_name, fixtures, cache, cleanup):
    if fn_name in cache: return cache[fn_name]
    if fn_name == "tmp_path":
        d = Path(tempfile.mkdtemp(prefix="ats_t_")); cleanup.append(lambda: shutil.rmtree(d, ignore_errors=True)); val = d
    elif fn_name == "monkeypatch":
        m = MonkeyPatch(); cleanup.append(m.undo); val = m
    else:
        f = fixtures[fn_name]
        args = {p: build(p, fixtures, cache, cleanup) for p in inspect.signature(f).parameters}
        val = f(**args)
        if inspect.isgenerator(val):
            gen = val; val = next(gen)
            cleanup.append(lambda g=gen: next(g, None))
    cache[fn_name] = val
    return val


def main():
    argv = sys.argv[1:]; kw = argv[argv.index("-k") + 1] if "-k" in argv else ""; verbose = "--verbose" in argv
    print(f"[offline runner] python {sys.version.split()[0]}; shims in use: {used_shims or 'none (real pydantic & pytest)'}")
    results = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}; failures = []
    t0 = time.time()
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        conf = ROOT / "tests" / "conftest.py"
        if conf.exists() and "tests.conftest" not in sys.modules:
            importlib.import_module("tests.conftest")
        try:
            mod = importlib.import_module(f"tests.{path.stem}")
        except Exception as e:
            results["errors"] += 1; failures.append((path.name, "IMPORT", traceback.format_exc(limit=3))); print(f"ERROR  {path.name}: import failed: {type(e).__name__}: {e}"); continue
        conf_mod = sys.modules.get("tests.conftest")
        conf_fixtures = {n: f for n, f in vars(conf_mod).items() if getattr(f, "_is_fixture", False)} if conf_mod else {}
        fixtures = {**conf_fixtures, **{n: f for n, f in vars(mod).items() if getattr(f, "_is_fixture", False)}}
        skip_exc = getattr(pytest, "_Skip", getattr(pytest, "skip", None) and getattr(pytest.skip, "Exception", ()))
        for name, fn in vars(mod).items():
            if not name.startswith("test_") or not callable(fn) or kw not in f"{path.stem}::{name}": continue
            sets = [dict()]
            for names, values in getattr(fn, "_params", []):
                nl = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)
                sets = [{**s, **dict(zip(nl, v if len(nl) > 1 else (v,)))} for s in sets for v in values]
            for i, params in enumerate(sets):
                label = f"{path.stem}::{name}" + (f"[{i}]" if len(sets) > 1 else "")
                if getattr(fn, "_skip", None):
                    results["skipped"] += 1; print(f"SKIP   {label} ({fn._skip})"); continue
                cache, cleanup = dict(params), []
                try:
                    args = {p: build(p, fixtures, cache, cleanup) for p in inspect.signature(fn).parameters}
                    fn(**args); results["passed"] += 1
                    if verbose: print(f"PASS   {label}")
                except skip_exc as e:
                    results["skipped"] += 1; print(f"SKIP   {label} ({e})")
                except BaseException:
                    results["failed"] += 1; failures.append((label, "FAIL", traceback.format_exc(limit=4))); print(f"FAIL   {label}")
                finally:
                    for c in reversed(cleanup):
                        try: c()
                        except Exception: pass
    print("\n" + "=" * 70)
    for label, kind, tb in failures: print(f"--- {kind}: {label}\n{tb}")
    print(f"RESULT: {results['passed']} passed, {results['failed']} failed, {results['skipped']} skipped, {results['errors']} import errors in {time.time()-t0:.1f}s")
    sys.exit(1 if results["failed"] or results["errors"] else 0)


if __name__ == "__main__":
    main()
