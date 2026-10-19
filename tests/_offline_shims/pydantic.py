"""OFFLINE TEST SHIM - NOT the real pydantic.

Used only by tests/run_offline.py when the real `pydantic` package cannot be imported (e.g. an
air-gapped sandbox). It implements just the surface this project uses: BaseModel with annotated defaults,
Field(default / default_factory / ge / le), model_dump(), model_dump_json(), model_validate_json(),
and no type coercion/validation beyond ge/le. Results obtained with this shim validate the project's
scoring logic, NOT pydantic's validation behaviour. Always run the real test-suite with real pydantic/pytest
(`python -m pytest`) before releasing.
"""
import copy
import json
from pathlib import Path

IS_OFFLINE_SHIM = True
_MISSING = object()


class _FieldInfo:
    def __init__(self, default=_MISSING, default_factory=None, ge=None, le=None, **_kw):
        self.default, self.default_factory, self.ge, self.le = default, default_factory, ge, le


def Field(default=_MISSING, default_factory=None, ge=None, le=None, **kw):
    return _FieldInfo(default, default_factory, ge, le, **kw)


class BaseModel:
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        fields = {}
        for base in reversed(cls.__mro__[1:]):
            fields.update(getattr(base, "__shim_fields__", {}))
        for name in getattr(cls, "__annotations__", {}):
            fields[name] = cls.__dict__.get(name, _MISSING)
        cls.__shim_fields__ = fields

    def __init__(self, **data):
        cls = type(self)
        unknown = set(data) - set(cls.__shim_fields__)
        for name, dflt in cls.__shim_fields__.items():
            if name in data:
                val = data[name]
            elif isinstance(dflt, _FieldInfo):
                if dflt.default_factory is not None:
                    val = dflt.default_factory()
                elif dflt.default is not _MISSING:
                    val = copy.deepcopy(dflt.default)
                else:
                    raise ValueError(f"field required: {name}")
            elif dflt is _MISSING:
                raise ValueError(f"field required: {name}")
            else:
                val = copy.deepcopy(dflt)
            if isinstance(dflt, _FieldInfo) and val is not None:
                if dflt.ge is not None and val < dflt.ge or dflt.le is not None and val > dflt.le:
                    raise ValueError(f"{name} out of bounds")
            object.__setattr__(self, name, val)
        _ = unknown  # extra fields ignored (pydantic default)

    def model_dump(self):
        def conv(v):
            if isinstance(v, BaseModel):
                return v.model_dump()
            if isinstance(v, (list, tuple)):
                return [conv(x) for x in v]
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            return v
        return {n: conv(getattr(self, n)) for n in type(self).__shim_fields__}

    def model_dump_json(self):
        return json.dumps(self.model_dump(), default=str)

    @classmethod
    def model_validate_json(cls, s):
        return cls(**json.loads(s))

    @classmethod
    def model_validate(cls, d):
        return cls(**d)

    def __repr__(self):
        return f"{type(self).__name__}({self.model_dump()})"
