"""`Collection`: an ordered set of records with a table view and an xarray `map`.

`table()` generalises the `info()` helper of another project (MIT, Copyright (c) 2024 mackelab): one row
per record with status, timestamps, size and the flattened spec, columns ordered by nesting depth
and filtered by depth or regex. `map()` generalises flyvis's `generic_responses` (MIT, Copyright
(c) 2023 Janne K. Lappalainen et al.): one labeled result per record, concatenated along a new
dimension with spec fields as coordinates.
"""

import json
import os
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

from ._canonical import canonical
from ._root import get_root
from . import index

__all__ = ["Collection", "query", "flatten"]


def _leaf(value: Any) -> bool:
    return not isinstance(value, dict) or any(k.startswith("__") and k.endswith("__") for k in value)


def flatten(spec: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten a canonical spec dict to dotted names; tagged values (digests, records) stay leaves."""
    out = {}
    for k, v in spec.items():
        name = f"{prefix}{k}"
        if _leaf(v):
            out[name] = v
        else:
            out.update(flatten(v, f"{name}."))
    return out


def _display(value: Any) -> Any:
    if isinstance(value, dict):
        if "__float__" in value:
            return float(value["__float__"])
        if "__digest__" in value:
            return f"digest:{value['__digest__'][:16]}"
        if "__record__" in value:
            return f"{value['__record__']}:{value['key'][:16]}"
        return json.dumps(value)
    if isinstance(value, list):
        return json.dumps(value)
    return value


def _matches(spec: dict, filters: dict[str, Any]) -> bool:
    flat = flatten(spec)
    for name, wanted in filters.items():
        name = name.replace("__", ".")
        if name not in flat:
            return False
        value = flat[name]
        if callable(wanted):
            if not wanted(_display(value)):
                return False
        elif value != canonical(wanted, full=True):
            return False
    return True


def query(cls, *, root: str | os.PathLike | None = None, status: str | Sequence[str] | None = "done", **filters: Any) -> "Collection":
    base = get_root(root)
    wanted = {status} if isinstance(status, str) else (set(status) if status is not None else None)
    found = []
    for path, meta in index.records(base, cls.dirname()):
        if meta.get("type") != cls.type_id():
            continue
        record = cls._new(path, key=meta["key"], managed_root=base)
        if wanted is not None:
            current = meta["status"] if meta["status"] in ("done", "failed") else record.status
            if current not in wanted:
                continue
        if _matches(meta.get("spec", {}), filters):
            found.append(record)
    return Collection(found)


class Collection(Sequence):
    """Records in a fixed order. Build one with `Record.query(...)` or from any iterable."""

    def __init__(self, records: Iterable[Any] = ()):
        self._records = list(records)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._records)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Collection(self._records[index])
        return self._records[index]

    def __repr__(self) -> str:
        types = sorted({type(r).__qualname__ for r in self._records})
        return f"Collection({len(self)} records of {', '.join(types) or 'no type'})"

    def where(self, predicate: Callable[[Any], bool] | None = None, /, **filters: Any) -> "Collection":
        """Records for which `predicate(record)` holds and whose spec matches `filters`."""
        return Collection(
            r for r in self._records
            if (predicate is None or predicate(r)) and _matches(r.meta.get("spec", {}), filters)
        )

    def keys(self) -> list[str]:
        return [r.key for r in self._records]

    def table(self, *, depth: int | None = None, include: Sequence[str] = (), sizes: bool = True):
        """A DataFrame with one row per record.

        Columns: type, key (16 hex), status, created, finished, duration_s, size_mb (if `sizes`),
        n_fields, path, then `spec.<name>` columns ordered by nesting depth and name. `depth=k`
        keeps spec columns nested at most `k` levels deep; `include` regexes bring back others.
        """
        import pandas as pd

        rows = []
        for record in self._records:
            meta = record._disk_meta() or {}
            row = {
                "type": meta.get("type", type(record).type_id()),
                "key": (record.key or "")[:16],
                "status": record.status,
                "created": pd.Timestamp(meta["created"]) if meta.get("created") else pd.NaT,
                "finished": pd.Timestamp(meta["finished"]) if meta.get("finished") else pd.NaT,
                "duration_s": meta.get("duration_s"),
            }
            if sizes:
                row["size_mb"] = record.size_bytes() / 2**20
            row["n_fields"] = len(meta.get("fields", {}))
            row["path"] = str(record.path)
            row.update({f"spec.{k}": _display(v) for k, v in flatten(meta.get("spec", {})).items()})
            rows.append(row)
        frame = pd.DataFrame(rows)
        fixed = [c for c in frame.columns if not c.startswith("spec.")]
        spec_columns = sorted((c for c in frame.columns if c.startswith("spec.")), key=lambda c: (c.count("."), c))
        keep = [
            c for c in spec_columns
            if depth is None or c.count(".") <= depth or any(re.search(p, c) for p in include)
        ]
        return frame[fixed + keep].convert_dtypes()

    def map(
        self,
        func: Callable[[Any], Any],
        *,
        dim: str = "record",
        coords: Sequence[str] = (),
        memory: Any = None,
    ):
        """Apply `func(record)` (memoized with `memory`, a joblib.Memory, if given) and concatenate
        the xarray results along a new `dim` indexed by record key, with the flattened spec fields
        named in `coords` (`"model.lr"` or `"model__lr"`) as coordinates along `dim`."""
        import xarray as xr

        if not self._records:
            raise ValueError("cannot map over an empty collection")
        call = memory.cache(func) if memory is not None else func
        outputs = []
        for record in self._records:
            out = call(record)
            if not isinstance(out, (xr.DataArray, xr.Dataset)):
                raise TypeError(f"map needs xarray results, got {type(out).__qualname__} for {record!r}")
            outputs.append(out)
        result = xr.concat(outputs, dim=dim, data_vars="all", coords="minimal", compat="override", join="outer")
        result = result.assign_coords({dim: [r.key[:16] for r in self._records]})
        specs = [flatten(r.meta.get("spec", {})) for r in self._records]
        for name in coords:
            dotted = name.replace("__", ".")
            result = result.assign_coords({dotted: (dim, [_display(s.get(dotted)) for s in specs])})
        return result
