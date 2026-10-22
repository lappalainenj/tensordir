"""`Spec`: frozen dataclasses whose canonical form determines record keys."""

import dataclasses
import importlib
import types
import typing
from collections.abc import Mapping
from enum import Enum
from pathlib import PurePath
from typing import Any, ClassVar, Self

import numpy as np

from ._canonical import Digest, canonical, sha256, untag

__all__ = ["Spec", "field", "FieldOptions"]

_META = "tensordir"


@dataclasses.dataclass(frozen=True)
class FieldOptions:
    """How a spec field enters the key.

    `hashed=False` keeps the field out of the key (devices, worker counts, batch sizes).
    `since=v` marks a field added in version `v` of the spec: it enters the key only when it
    differs from its default, so records built before the field existed keep their keys.
    """

    hashed: bool = True
    since: Any = None


def field(
    default: Any = dataclasses.MISSING,
    *,
    default_factory: Any = dataclasses.MISSING,
    hashed: bool = True,
    since: Any = None,
    **kwargs: Any,
) -> Any:
    """A spec field with tensordir options; other keyword arguments go to `dataclasses.field`."""
    has_default = default is not dataclasses.MISSING or default_factory is not dataclasses.MISSING
    if since is not None and not has_default:
        raise TypeError("a field with since= needs a default; records built before it existed used it")
    if since is not None and not hashed:
        raise TypeError("since= only applies to hashed fields")
    metadata = dict(kwargs.pop("metadata", None) or {})
    metadata[_META] = FieldOptions(hashed=hashed, since=since)
    return dataclasses.field(default=default, default_factory=default_factory, metadata=metadata, **kwargs)


def _options(f: dataclasses.Field) -> FieldOptions:
    return f.metadata.get(_META, FieldOptions())


def _default(f: dataclasses.Field) -> Any:
    if f.default is not dataclasses.MISSING:
        return f.default
    return f.default_factory()


class Spec:
    """Base class of specs. Subclasses become frozen, keyword-only dataclasses.

    ```python
    class FitSpec(tensordir.Spec):
        n_sweeps: int
        boundary: str = "censored"
        device: str = tensordir.field("cpu", hashed=False)
        prior_scale: float = tensordir.field(1.0, since=2)
    ```
    """

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        dataclasses.dataclass(frozen=True, kw_only=True)(cls)

    def __post_init__(self) -> None:
        hints = _hints(type(self))
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if hints.get(f.name) is float and isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
                object.__setattr__(self, f.name, float(value))
            nested = _nested_spec(hints.get(f.name))
            if nested is not None and isinstance(value, Mapping):
                object.__setattr__(self, f.name, nested(**value))
        canonical(self, full=True, _path=type(self).__qualname__)

    def __tensordir_canonical__(self, *, full: bool = False) -> dict:
        out = {}
        for f in dataclasses.fields(self):
            value = canonical(getattr(self, f.name), full=full, _path=f"{type(self).__qualname__}.{f.name}")
            opts = _options(f)
            if not full:
                if not opts.hashed:
                    continue
                if opts.since is not None and value == canonical(_default(f), full=False):
                    continue
            out[f.name] = value
        return dict(sorted(out.items()))

    def keys(self) -> list[str]:
        """The field names, so `Fit(**spec)` passes a spec's settings."""
        return [f.name for f in dataclasses.fields(self)]

    def __getitem__(self, name: str) -> Any:
        """The value of field `name` (nested specs stay specs)."""
        if name not in self.__dataclass_fields__:
            raise KeyError(name)
        return getattr(self, name)

    def to_dict(self) -> dict:
        """All fields in canonical JSON form (including fields excluded from the key)."""
        return canonical(self, full=True)

    def hashed_dict(self) -> dict:
        """The fields that enter the key, in canonical JSON form."""
        return canonical(self, full=False)

    def key(self) -> str:
        """sha256 of the hashed fields (the spec's own identity, without a record type)."""
        return sha256(self.hashed_dict())

    def replace(self, **changes: Any) -> Self:
        """A copy with `changes`; nested fields by dotted or `__` names.

        ```python
        spec.replace(seed=1, model__lr=0.01)        # same as
        spec.replace(**{"seed": 1, "model.lr": 0.01})
        ```
        """
        direct: dict[str, Any] = {}
        nested: dict[str, dict[str, Any]] = {}
        for name, value in changes.items():
            head, dot, rest = name.replace("__", ".").partition(".")
            if dot:
                nested.setdefault(head, {})[rest] = value
            else:
                direct[head] = value
        for head, sub in nested.items():
            base = direct.get(head, getattr(self, head))
            if not isinstance(base, Spec):
                raise TypeError(f"{type(self).__qualname__}.{head} is not a nested spec; cannot set {sorted(sub)}")
            direct[head] = base.replace(**sub)
        return dataclasses.replace(self, **direct)

    def flat(self) -> dict[str, Any]:
        """All fields by dotted name (`{"model.lr": 0.1, ...}`), in canonical JSON form."""
        from .collection import flatten

        return flatten(self.to_dict())

    def diff(self, other: Any, *, hashed: bool = False) -> Any:
        """`tensordir.diff(self, other)`: added, removed and changed fields by dotted name."""
        from ._diff import diff

        return diff(self, other, hashed=hashed)

    def _rows(self, prefix: str = "") -> list[tuple[str, Any, str]]:
        rows = []
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            opts = _options(f)
            if isinstance(value, Spec):
                rows.extend(value._rows(f"{prefix}{f.name}."))
                continue
            note = "not hashed" if not opts.hashed else (f"since {opts.since}" if opts.since is not None else "")
            rows.append((f"{prefix}{f.name}", value, note))
        return rows

    def _repr_html_(self) -> str:
        import html

        body = "".join(
            f"<tr><td><code>{html.escape(name)}</code></td><td>{html.escape(repr(value))}</td><td>{note}</td></tr>"
            for name, value, note in self._rows()
        )
        return (
            f"<p><b>{html.escape(type(self).__qualname__)}</b></p>"
            f"<table><tbody>{body}</tbody></table>"
        )

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Rebuild a spec from `to_dict()` output, reviving nested specs, paths, tuples and enums."""
        hints = _hints(cls)
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"{cls.__qualname__} has no fields {sorted(unknown)}")
        return cls(**{k: _revive(hints.get(k), untag(v)) for k, v in data.items()})

    def __reduce__(self):
        return _restore_spec, (f"{type(self).__module__}:{type(self).__qualname__}", self.to_dict())


def _hints(cls: type) -> dict:
    try:
        return typing.get_type_hints(cls, include_extras=True)
    except Exception:
        return {}


def _nested_spec(tp: Any) -> type[Spec] | None:
    """The spec class of a field annotated with one (also `Annotated` or `| None`), else None."""
    while True:
        origin = typing.get_origin(tp)
        if origin is typing.Annotated:
            tp = typing.get_args(tp)[0]
        elif origin in (typing.Union, types.UnionType):
            args = [a for a in typing.get_args(tp) if a is not type(None)]
            if len(args) != 1:
                return None
            tp = args[0]
        else:
            return tp if isinstance(tp, type) and issubclass(tp, Spec) else None


def _revive(tp: Any, value: Any) -> Any:
    if value is None or tp is None:
        return value
    origin = typing.get_origin(tp)
    if origin is typing.Annotated:
        return _revive(typing.get_args(tp)[0], value)
    if origin in (typing.Union, types.UnionType):
        for arg in typing.get_args(tp):
            if arg is type(None):
                continue
            try:
                return _revive(arg, value)
            except Exception:
                continue
        return value
    if isinstance(tp, type):
        revive = getattr(tp, "__tensordir_revive__", None)
        if revive is not None and isinstance(value, dict):
            return revive(value)
        if issubclass(tp, Spec) and isinstance(value, dict):
            return tp.from_dict(value)
        if issubclass(tp, PurePath) and isinstance(value, str):
            return tp(value)
        if issubclass(tp, Enum):
            return tp(value)
        if tp is float and isinstance(value, int):
            return float(value)
        if issubclass(tp, Digest):
            return value
    if origin is tuple and isinstance(value, list):
        args = typing.get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_revive(args[0], v) for v in value)
        if args and len(args) == len(value):
            return tuple(_revive(a, v) for a, v in zip(args, value, strict=True))
        return tuple(value)
    if origin in (frozenset, set) and isinstance(value, list):
        (arg,) = typing.get_args(tp) or (None,)
        return origin(_revive(arg, v) for v in value)
    if origin is list and isinstance(value, list):
        (arg,) = typing.get_args(tp) or (None,)
        return [_revive(arg, v) for v in value]
    if origin is dict and isinstance(value, dict):
        args = typing.get_args(tp)
        return {k: _revive(args[1] if args else None, v) for k, v in value.items()}
    return value


def _restore_spec(path: str, data: dict) -> Spec:
    module, _, qualname = path.partition(":")
    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj.from_dict(data)
