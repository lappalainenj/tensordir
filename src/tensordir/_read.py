"""`Record.read`: rows of several array fields in one call, with merged, concurrent reads.

Ranges are held as int64 arrays of starts and stops, so a request of many thousands of ranges
costs a few numpy calls, not a Python loop per range.
"""

import asyncio
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["read_rows"]


def _parse(value: Any, n: int) -> tuple[str, np.ndarray, np.ndarray | None]:
    """`("ranges", starts, stops)` or `("gather", rows, None)` for one field's request."""
    if isinstance(value, slice):
        return ("ranges", *_slices([value], n))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if any(isinstance(v, slice) for v in value):
            if not all(isinstance(v, slice) for v in value):
                raise TypeError("mix slices with other range forms by converting them to slices")
            return ("ranges", *_slices(value, n))
        value = np.asarray(value) if len(value) else np.empty((0, 2), dtype=np.int64)
    if isinstance(value, np.ndarray):
        if value.dtype == bool:
            if value.shape != (n,):
                raise IndexError(f"boolean index of shape {value.shape} for {n} rows")
            return "gather", np.flatnonzero(value), None
        if value.dtype.kind in "iu" and value.ndim == 2 and value.shape[1] == 2:
            return ("ranges", *_pairs(value[:, 0], value[:, 1], n))
        if value.dtype.kind in "iu" and value.ndim == 1:
            return "gather", _indices(value, n), None
        raise TypeError(f"cannot read rows by a {value.dtype} array of shape {value.shape}")
    raise TypeError(f"cannot read rows by {type(value).__qualname__}; use a slice, ranges or an index array")


def _slices(slices: Sequence[slice], n: int) -> tuple[np.ndarray, np.ndarray]:
    starts, stops = [], []
    for s in slices:
        start, stop, step = s.indices(n)
        if step != 1:
            raise ValueError(f"slice {s} has step {step}; read ranges with step 1 or pass an index array")
        starts.append(start)
        stops.append(max(start, stop))
    return np.asarray(starts, dtype=np.int64), np.asarray(stops, dtype=np.int64)


def _pairs(starts: np.ndarray, stops: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """`(start, stop)` pairs with slice semantics: negatives count from the end, bounds clip."""
    starts, stops = starts.astype(np.int64), stops.astype(np.int64)
    starts = np.clip(np.where(starts < 0, starts + n, starts), 0, n)
    stops = np.clip(np.where(stops < 0, stops + n, stops), 0, n)
    return starts, np.maximum(starts, stops)


def _indices(index: np.ndarray, n: int) -> np.ndarray:
    index = index.astype(np.int64)
    index = np.where(index < 0, index + n, index)
    if index.size and (index.min() < 0 or index.max() >= n):
        raise IndexError(f"row index out of range for {n} rows")
    return index


def _merge(starts: np.ndarray, stops: np.ndarray, gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Sorted, disjoint spans covering the non-empty ranges, merging ranges less than `gap`
    rows apart."""
    keep = stops > starts
    starts, stops = starts[keep], stops[keep]
    if not starts.size:
        return starts, stops
    order = np.argsort(starts, kind="stable")
    starts, stops = starts[order], stops[order]
    reach = np.maximum.accumulate(stops)
    new = np.empty(starts.size, dtype=bool)
    new[0] = True
    new[1:] = starts[1:] > reach[:-1] + gap
    first = np.flatnonzero(new)
    return starts[first], np.maximum.reduceat(stops, first)


def _runs(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximal runs of consecutive values in sorted, unique `rows`, as starts and stops."""
    if not rows.size:
        return rows, rows
    breaks = np.flatnonzero(np.diff(rows) > 1)
    starts = np.concatenate([rows[:1], rows[breaks + 1]])
    stops = np.concatenate([rows[breaks], rows[-1:]]) + 1
    return starts, stops


def _gap(array: Any) -> int:
    chunks = getattr(array, "chunks", None)
    return int(chunks[0]) if chunks else 0


class _Plan:
    """What to read from one field and how to assemble the result."""

    def __init__(self, array: Any, value: Any):
        self.array = array
        shape = tuple(array.shape)
        if not shape:
            raise TypeError("cannot read rows of a 0-d array")
        self.n, self.trailing, self.dtype = shape[0], shape[1:], array.dtype
        self.kind, a, b = _parse(value, self.n)
        if self.kind == "ranges":
            self.starts, self.stops = a, b
        else:
            self.unique, self.inverse = np.unique(a, return_inverse=True)
            self.starts, self.stops = _runs(self.unique)
        self.span_starts, self.span_stops = _merge(self.starts, self.stops, _gap(array))
        self.spans = list(zip(self.span_starts.tolist(), self.span_stops.tolist()))

    def assemble(self, datas: list[Any]) -> np.ndarray:
        """The requested rows from the span reads `datas`, in request order."""
        empty = np.empty((0, *self.trailing), dtype=self.dtype)
        if not datas:
            return empty
        data = np.concatenate([np.asarray(d) for d in datas]) if len(datas) > 1 else np.asarray(datas[0])
        offsets = np.concatenate([[0], np.cumsum(self.span_stops - self.span_starts)[:-1]])

        def position(rows: np.ndarray) -> np.ndarray:  # row number -> index into `data`
            span = np.searchsorted(self.span_starts, rows, side="right") - 1
            return offsets[span] + rows - self.span_starts[span]

        if self.kind == "gather":
            return data[position(self.unique)[self.inverse.reshape(-1)]]
        lengths = self.stops - self.starts
        total = int(lengths.sum())
        if not total:
            return empty
        keep = lengths > 0
        firsts = position(self.starts[keep])
        lengths = lengths[keep]
        index = np.repeat(firsts - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths) + np.arange(total)
        return data[index]


ENGINES = ("zarr", "tensorstore")


def read_rows(record: Any, request: Mapping[str, Any], *, engine: str | None = None) -> dict[str, np.ndarray]:
    """The rows `request` asks of each field of `record`; see `Record.read`."""
    import zarr

    engine = engine or os.environ.get("TENSORDIR_READ_ENGINE") or "zarr"
    if engine not in ENGINES:
        raise ValueError(f"unknown read engine {engine!r}; use one of {ENGINES}")
    plans: dict[str, _Plan] = {}
    for name, value in request.items():
        array = record[name]
        if not hasattr(array, "shape") or not hasattr(array, "dtype"):
            raise TypeError(f"{name}: read() needs an array field, got {type(array).__qualname__}")
        plans[name] = _Plan(array, value)
    zarr_plans = [p for p in plans.values() if isinstance(p.array, zarr.Array) and p.spans]
    datas: dict[int, list] = {}
    if engine == "tensorstore" and zarr_plans:
        stores = {id(p): record._tensorstore(name) for name, p in plans.items() if p in zarr_plans}
        by_ts = [p for p in zarr_plans if stores[id(p)] is not None]
        datas.update(_read_tensorstore(by_ts, stores))
        zarr_plans = [p for p in zarr_plans if stores[id(p)] is None]
    if zarr_plans:
        from zarr.core.sync import sync

        async def fetch() -> list:
            return await asyncio.gather(
                *(p.array.async_array.getitem((slice(a, b),)) for p in zarr_plans for a, b in p.spans)
            )

        flat = iter(sync(fetch()))
        datas.update({id(p): [next(flat) for _ in p.spans] for p in zarr_plans})
    out = {}
    for name, plan in plans.items():
        data = datas.get(id(plan))
        if data is None:  # HDF5 and other array-likes: one read per merged span
            data = [plan.array[a:b] for a, b in plan.spans]
        out[name] = plan.assemble(data)
    return out


def _read_tensorstore(plans: list[_Plan], stores: dict[int, Any]) -> dict[int, list]:
    """Every span of `plans` read in one tensorstore batch."""
    if not plans:
        return {}
    import tensorstore as ts

    with ts.Batch() as batch:
        futures = {id(p): [stores[id(p)][a:b].read(batch=batch) for a, b in p.spans] for p in plans}
    return {k: [f.result() for f in fs] for k, fs in futures.items()}
