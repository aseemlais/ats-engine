"""OFFLINE TEST SHIM - NOT the real pytest. Used only by tests/run_offline.py when pytest is not installed.
Provides approx, raises, fixture, mark.parametrize/skip/skipif - the subset these tests use."""
import math

IS_OFFLINE_SHIM = True


class _Skip(Exception):
    pass


def skip(reason=""):
    raise _Skip(reason)


class approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected, self.rel, self.abs = expected, rel, abs

    def __eq__(self, other):
        if isinstance(self.expected, (list, tuple)):
            return len(other) == len(self.expected) and all(a == approx(b, self.rel, self.abs) for a, b in zip(other, self.expected))
        return math.isclose(other, self.expected, rel_tol=self.rel, abs_tol=self.abs)

    def __repr__(self):
        return f"approx({self.expected})"


class _Raises:
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"DID NOT RAISE {self.exc}")
        if not issubclass(et, self.exc):
            return False
        if self.match:
            import re
            assert re.search(self.match, str(ev)), f"{self.match!r} not in {ev!r}"
        return True


def raises(exc, match=None):
    return _Raises(exc, match)


def fixture(fn=None, **kw):
    def deco(f):
        f._is_fixture = True
        return f
    return deco(fn) if callable(fn) else deco


class _Mark:
    def parametrize(self, names, values, ids=None):
        def deco(f):
            f.__dict__.setdefault("_params", []).append((names, list(values)))
            return f
        return deco

    def skip(self, reason=""):
        def deco(f):
            f._skip = reason or "skipped"
            return f
        return deco

    def skipif(self, cond, reason=""):
        def deco(f):
            if cond:
                f._skip = reason or "skipif"
            return f
        return deco

    def __getattr__(self, name):          # unknown marks (e.g. integration) are no-ops
        return lambda *a, **k: (lambda f: f)


mark = _Mark()
