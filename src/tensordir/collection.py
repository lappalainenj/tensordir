"""`Collection`: an ordered set of records with a table view and an xarray `map`.

`table()` generalises the `info()` helper of another project (MIT, Copyright (c) 2024 mackelab): one row
per record with status, timestamps, size and the flattened spec, columns ordered by nesting depth
and filtered by depth or regex. `map()` generalises flyvis's `generic_responses` (MIT, Copyright
(c) 2023 Janne K. Lappalainen et al.): one labeled result per record, concatenated along a new
dimension with spec fields as coordinates.
"""

import html
import importlib
import importlib.util
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from ._root import get_root
from ._where import conditions as _conditions
from ._where import display as _display
from ._where import matches as _match
from . import index

__all__ = ["Collection", "query", "flatten", "spec_dict"]


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


def spec_dict(cls: type, stored: dict) -> dict:
    """A record's stored spec with defaults filled in for fields added later (`since=`).

    Records built before a field existed don't store it; reviving the stored spec through the
    class's current `Spec` restores the default, so queries and tables see it.
    """
    from .record import Record

    if not stored or cls is Record or not issubclass(cls, Record):
        return stored
    try:
        return cls.Spec.from_dict(stored).to_dict()
    except Exception:
        return stored


def require(module: str, extra: str, what: str) -> Any:
    """Import an optional dependency or say which extra installs it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(f"{what} needs {module}: pip install 'tensordir[{extra}]'") from exc


def _spec_of(record: Any) -> dict:
    meta = record._disk_meta() or {}
    return spec_dict(type(record), meta.get("spec", {}))


def _matches(spec: dict, conds: dict[str, Any]) -> bool:
    flat = flatten(spec)
    return all(name in flat and _match(_display(flat[name]), wanted) for name, wanted in conds.items())


def query(cls, *, root: str | os.PathLike | None = None, status: str | Sequence[str] | None = "done", **filters: Any) -> "Collection":
    base = get_root(root)
    wanted = {status} if isinstance(status, str) else (set(status) if status is not None else None)
    conds = _conditions(None, filters)
    found = []
    for path, meta in index.records(base, cls.dirname()):
        if meta.get("type") != cls.type_id():
            continue
        record = cls._new(path, key=meta["key"], managed_root=base)
        if wanted is not None:
            current = meta["status"] if meta["status"] in ("done", "failed") else record.status
            if current not in wanted:
                continue
        if not conds or _matches(spec_dict(cls, meta.get("spec", {})), conds):
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

    def status_counts(self) -> dict[str, int]:
        """How many records have each status, e.g. `{"done": 10, "failed": 2}`."""
        counts: Counter = Counter()
        for r in self._records:
            try:
                counts[r.status] += 1
            except Exception:
                counts["?"] += 1
        return dict(counts.most_common())

    def __repr__(self) -> str:
        types = sorted({type(r).__qualname__ for r in self._records})
        counts = ", ".join(f"{n} {s}" for s, n in self.status_counts().items())
        return f"Collection({len(self)} records of {', '.join(types) or 'no type'}{': ' + counts if counts else ''})"

    def _repr_html_(self) -> str:
        head = f"<p>{html.escape(repr(self))}</p>"
        if not self._records or importlib.util.find_spec("pandas") is None:
            return head
        frame = self.table(sizes=False, varying_only=len(self) > 1)
        keep = ["key", "status", "finished", "duration_s"] + [c for c in frame.columns if c.startswith("spec.")]
        if frame["type"].nunique() > 1:
            keep.insert(0, "type")
        return head + frame[keep]._repr_html_()

    def where(
        self, predicate: Callable[[Any], bool] | Mapping[str, Any] | None = None, /, **conditions: Any
    ) -> "Collection":
        """Records for which `predicate(record)` holds and whose flattened spec meets `conditions`.

        Conditions follow `tensordir.where`: `seed=[0, 1]`, `model__lr=">1e-3"`, a regex or a
        callable. Pass a mapping instead of a predicate for dotted names: `where({"model.lr": 0.1})`.
        """
        if isinstance(predicate, Mapping):
            conds, predicate = _conditions(predicate, conditions), None
        else:
            conds = _conditions(None, conditions)
        return Collection(
            r for r in self._records
            if (predicate is None or predicate(r)) and (not conds or _matches(_spec_of(r), conds))
        )

    def keys(self) -> list[str]:
        return [r.key for r in self._records]

    def delete(self, *, force: bool = False) -> int:
        """Delete every record; returns how many. Running builds stop it unless `force=True`.

        `Fit.query(status="failed").delete()` clears failures before a rerun.
        """
        from .errors import RecordBusy

        if not force:
            busy = [r for r in self._records if r.status == "running"]
            if busy:
                raise RecordBusy(busy[0])
        for r in self._records:
            r.delete(force=force)
        return len(self._records)

    def table(
        self,
        *,
        depth: int | None = None,
        include: Sequence[str] = (),
        sizes: bool = True,
        varying_only: bool = False,
    ):
        """A DataFrame with one row per record.

        Columns: type, key (16 hex), status, created, finished, duration_s, size_mb (if `sizes`),
        n_fields, path, then `spec.<name>` columns ordered by nesting depth and name. `depth=k`
        keeps spec columns nested at most `k` levels deep; `include` regexes bring back others.
        `varying_only=True` keeps only spec columns whose value differs between records. Filter
        rows with `tensordir.where(table, ...)` or `table.pipe(tensordir.where, ...)`.
        """
        pd = require("pandas", "tables", "Collection.table")

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
            row.update({f"spec.{k}": _display(v) for k, v in flatten(spec_dict(type(record), meta.get("spec", {}))).items()})
            rows.append(row)
        frame = pd.DataFrame(rows)
        fixed = [c for c in frame.columns if not c.startswith("spec.")]
        spec_columns = sorted((c for c in frame.columns if c.startswith("spec.")), key=lambda c: (c.count("."), c))
        keep = [
            c for c in spec_columns
            if (depth is None or c.count(".") <= depth or any(re.search(p, c) for p in include))
            and (not varying_only or frame[c].nunique(dropna=False) > 1)
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
        xr = require("xarray", "xarray", "Collection.map")

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
        specs = [flatten(_spec_of(r)) for r in self._records]
        for name in coords:
            dotted = name.replace("__", ".")
            result = result.assign_coords({dotted: (dim, [_display(s.get(dotted)) for s in specs])})
        return result
