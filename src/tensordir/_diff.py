"""`diff`: what differs between two specs, records or spec dicts, by dotted field name."""

import dataclasses
import html
from collections.abc import Mapping
from typing import Any

from ._canonical import canonical
from ._where import display

__all__ = ["SpecDiff", "diff"]


@dataclasses.dataclass(frozen=True)
class SpecDiff:
    """Fields only in `b` (`added`), only in `a` (`removed`), and in both with different values
    (`changed`, as `(a, b)` pairs), keyed by dotted name. Values are shown as in tables. A diff is
    falsy when the two sides agree."""

    added: dict[str, Any]
    removed: dict[str, Any]
    changed: dict[str, tuple[Any, Any]]

    def __bool__(self) -> bool:
        return bool(self.added or self.removed or self.changed)

    def names(self) -> list[str]:
        """Every dotted name that differs, sorted."""
        return sorted({*self.added, *self.removed, *self.changed})

    def lines(self) -> list[str]:
        out = []
        for name in self.names():
            if name in self.removed:
                out.append(f"- {name}: {self.removed[name]!r}")
            elif name in self.added:
                out.append(f"+ {name}: {self.added[name]!r}")
            else:
                a, b = self.changed[name]
                out.append(f"~ {name}: {a!r} -> {b!r}")
        return out

    def __repr__(self) -> str:
        if not self:
            return "SpecDiff(no differences)"
        return "SpecDiff(\n" + "".join(f"  {line}\n" for line in self.lines()) + ")"

    def _repr_html_(self) -> str:
        if not self:
            return "<p>SpecDiff: no differences</p>"
        rows = []
        for name in self.names():
            if name in self.removed:
                a, b = repr(self.removed[name]), "—"
            elif name in self.added:
                a, b = "—", repr(self.added[name])
            else:
                a, b = (repr(v) for v in self.changed[name])
            rows.append(f"<tr><td><code>{html.escape(name)}</code></td><td>{html.escape(a)}</td><td>{html.escape(b)}</td></tr>")
        return "<table><thead><tr><th>field</th><th>a</th><th>b</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _as_dict(obj: Any, hashed: bool) -> dict:
    from .collection import spec_dict
    from .record import Record
    from .spec import Spec
    from .view import View

    if isinstance(obj, View):
        obj = obj.record
    if isinstance(obj, Spec):
        return obj.hashed_dict() if hashed else obj.to_dict()
    if isinstance(obj, Record):
        if not hashed:
            return spec_dict(type(obj), obj.meta["spec"])
        if type(obj) is Record:
            raise TypeError("diff(hashed=True) needs records of a typed Record subclass")
        return obj.spec.hashed_dict()
    if isinstance(obj, Mapping):
        return canonical(dict(obj), full=True)
    raise TypeError(f"diff compares specs, records, views or mappings, not {type(obj).__qualname__}")


def diff(a: Any, b: Any, *, hashed: bool = False) -> SpecDiff:
    """What changes from `a` to `b`: two specs, records (their stored specs), views or dicts.

    `hashed=True` compares only the fields that enter the key, i.e. explains why two keys differ.

    ```python
    >>> td.diff(Fit.Spec(n_columns=8), Fit.Spec(n_columns=16, seed=1))
    SpecDiff(
      ~ n_columns: 8 -> 16
      ~ seed: 0 -> 1
    )
    ```
    """
    from .collection import flatten

    left, right = flatten(_as_dict(a, hashed)), flatten(_as_dict(b, hashed))
    return SpecDiff(
        added={k: display(right[k]) for k in sorted(right.keys() - left.keys())},
        removed={k: display(left[k]) for k in sorted(left.keys() - right.keys())},
        changed={
            k: (display(left[k]), display(right[k])) for k in sorted(left.keys() & right.keys()) if left[k] != right[k]
        },
    )
