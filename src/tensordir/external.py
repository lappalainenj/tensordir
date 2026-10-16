"""Pinned references to external tables: Delta Lake snapshots and parquet directories.

A reference names one immutable snapshot of a table that lives outside tensordir (on GCS, S3 or
a local disk). It can be a spec field or a `get` input; its canonical form, and so the key of a
record depending on it, is its URI and version:

```python
class Edges(td.Record):
    class Spec(td.Spec):
        synapses: td.DeltaRef
        min_synapses: int = 1

    def build(self, spec):
        frame = spec.synapses.to_pandas(columns=["pre_pt_root_id", "post_pt_root_id"])
        ...

synapses = td.DeltaRef("gs://bucket/synapses_pni_2/post_pt_root_id", version=3)
Edges.get({"synapses": synapses})
```

Data is read lazily through pyarrow datasets, with column selection and filter pushdown. Needs
`pyarrow`; Delta tables also need `deltalake` (`pip install 'tensordir[delta]'`).
"""

import dataclasses
import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar

from ._where import AnyOf, Exactly, _comparisons
from .errors import TensordirError

__all__ = ["DeltaRef", "ParquetRef", "SnapshotChanged", "TableRef", "expression"]


class SnapshotChanged(TensordirError):
    """A parquet directory no longer matches the version (listing digest) a reference pinned.

    **Experimental:** may change or be removed; see Design, Experimental features.
    """


def _require(module: str, what: str) -> Any:
    from .collection import require

    return require(module, "delta", what)


def _uri(uri: str | os.PathLike) -> str:
    text = os.fspath(uri)
    return text if "://" in text else Path(text).expanduser().absolute().as_posix()


def expression(filter: Any) -> Any:
    """A pyarrow filter expression from an expression, None, or a mapping of column conditions.

    **Experimental:** may change or be removed; see Design, Experimental features.

    Mapping values follow the pushdown-able part of `tensordir.where`: a value (equality, `None`
    for null), a list/tuple/set or `any_of(...)` of values (any of), `exactly(value)` for a
    scalar, or comparisons such as `">=2, <8"`. Conditions AND.
    """
    if filter is None or not isinstance(filter, Mapping):
        return filter
    import pyarrow as pa
    import pyarrow.compute as pc

    out = None
    for name, condition in filter.items():
        column = pc.field(*name.split(".")) if "." in name else pc.field(name)
        if isinstance(condition, Exactly):
            if isinstance(condition.value, (list, tuple, set, frozenset, Mapping)):
                raise TypeError(f"{name}: exactly() of a sequence cannot be pushed down; filter the result")
            term = column.is_null() if condition.value is None else column == condition.value
        elif condition is None:
            term = column.is_null()
        elif isinstance(condition, (list, tuple, set, frozenset, AnyOf)):
            items = condition.conditions if isinstance(condition, AnyOf) else condition
            term = column.isin(pa.array(list(items)))
        elif isinstance(condition, str) and (parsed := _comparisons(condition)) is not None:
            term = None
            for op, target in parsed:
                compare = {">": column > target, ">=": column >= target, "<": column < target,
                           "<=": column <= target, "==": column == target, "!=": column != target}[op]
                term = compare if term is None else term & compare
        elif callable(condition) or hasattr(condition, "pattern"):
            raise TypeError(f"{name}: regex and callable conditions cannot be pushed down; filter the result")
        else:
            term = column == condition
        out = term if out is None else out & term
    return out


@dataclasses.dataclass(frozen=True)
class TableRef:
    """Base class of pinned external tables. Identity: `(kind, uri, version)`."""

    kind: ClassVar[str]

    uri: str
    version: Any = None

    def __tensordir_canonical__(self, *, full: bool = False) -> dict:
        return {f"__{self.kind}__": self.uri, "version": self.version}

    @classmethod
    def __tensordir_revive__(cls, value: Mapping) -> "TableRef":
        return cls(value[f"__{cls.kind}__"], version=value["version"])

    def dataset(self) -> Any:
        """The snapshot as a lazy `pyarrow.dataset.Dataset`."""
        raise NotImplementedError

    def scanner(self, columns: list[str] | None = None, filter: Any = None, **kwargs: Any) -> Any:
        """A `pyarrow.dataset.Scanner` reading only `columns` and rows matching `filter`."""
        return self.dataset().scanner(columns=columns, filter=expression(filter), **kwargs)

    def to_table(self, columns: list[str] | None = None, filter: Any = None) -> Any:
        """The selected columns and rows as a `pyarrow.Table`."""
        return self.scanner(columns, filter).to_table()

    def to_pandas(self, columns: list[str] | None = None, filter: Any = None) -> Any:
        return self.to_table(columns, filter).to_pandas()

    def to_batches(self, columns: list[str] | None = None, filter: Any = None, batch_size: int = 1 << 20) -> Iterator[Any]:
        """Stream `pyarrow.RecordBatch`es (empty batches skipped)."""
        for batch in self.scanner(columns, filter, batch_size=batch_size).to_batches():
            if batch.num_rows:
                yield batch

    def to_reader(self, columns: list[str] | None = None, filter: Any = None) -> Any:
        """A `pyarrow.RecordBatchReader`: streams into `record.write` without loading the table."""
        return self.scanner(columns, filter).to_reader()

    def to_polars(self) -> Any:
        """A lazy `polars.LazyFrame` over the snapshot (polars pushes down its own filters)."""
        import polars as pl

        return pl.scan_pyarrow_dataset(self.dataset())

    def duckdb(self, connection: Any = None) -> Any:
        """A duckdb relation over the snapshot; duckdb pushes down projections and filters.

        **Experimental:** may change or be removed; see Design, Experimental features.
        """
        import duckdb

        return (connection or duckdb).from_arrow(self.dataset())

    def materialize(self, record: Any, name: str, *, columns: list[str] | None = None, filter: Any = None) -> None:
        """Copy the selected columns and rows into `record` as the parquet field `name`.

        **Experimental:** may change or be removed; see Design, Experimental features.

        Streams batch by batch. Call it inside `build` (or on a workspace record); `record[name]`
        then opens a `pyarrow.dataset.Dataset`, `record.load(name)` a `pyarrow.Table`.
        """
        record.write(name, self.to_reader(columns, filter), codec="arrow")


@dataclasses.dataclass(frozen=True)
class DeltaRef(TableRef):
    """A Delta Lake table at one version. `version=None` pins the current version on creation.

    `storage_options` (credentials, endpoints) are used for reading only: they never enter the
    key or the stored spec.
    """

    kind: ClassVar[str] = "delta"

    storage_options: Mapping[str, str] | None = dataclasses.field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", _uri(self.uri))
        if self.version is None:
            object.__setattr__(self, "version", self.table().version())
        elif not isinstance(self.version, int) or isinstance(self.version, bool):
            raise TypeError(f"Delta versions are integers, got {self.version!r}")

    def table(self) -> Any:
        """The `deltalake.DeltaTable` at the pinned version (or the latest while pinning)."""
        deltalake = _require("deltalake", "DeltaRef")
        options = dict(self.storage_options) if self.storage_options else None
        return deltalake.DeltaTable(self.uri, version=self.version, storage_options=options)

    def dataset(self) -> Any:
        return self.table().to_pyarrow_dataset()

    def files(self) -> list[str]:
        """The URIs of the snapshot's live data files."""
        return list(self.table().file_uris())


@dataclasses.dataclass(frozen=True)
class ParquetRef(TableRef):
    """A directory of parquet files (hive-partitioned or flat), pinned by a digest of its listing.

    **Experimental:** may change or be removed; see Design, Experimental features.

    The version (the "digest") is a SHA-256 fingerprint of the directory's file listing: the
    sorted `(relative path, size in bytes)` pairs of its data files, written as JSON and hashed.
    Names starting with `.` or `_` are skipped, as pyarrow does. The file contents are not read.
    `version=None` computes it on creation.

    The version is part of the reference's canonical form, so it is stored in the spec of a record
    that uses the reference and enters that record's key: once files are added, removed or
    resized, a new `ParquetRef(path)` has a new version and gives new keys, while records built
    from the old listing keep their keys. Opening the data (`dataset()`, `to_table()`, ...)
    recomputes the digest and raises `SnapshotChanged` if it differs from the pinned version.
    A file rewritten with the same size and name goes unnoticed. `DeltaRef` pins a Delta table
    version instead, which Delta keeps readable until it is vacuumed.
    """

    kind: ClassVar[str] = "parquet"

    partitioning: str | None = dataclasses.field(default="hive", compare=False)
    filesystem: Any = dataclasses.field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", _uri(self.uri))
        if self.version is None:
            object.__setattr__(self, "version", self.listing_digest())

    def _fs(self) -> tuple[Any, str]:
        fs = _require("pyarrow.fs", "ParquetRef")
        if self.filesystem is not None:
            return self.filesystem, self.uri.split("://", 1)[-1]
        return fs.FileSystem.from_uri(self.uri)

    def files(self) -> list[tuple[str, int]]:
        """`(relative path, size)` of every data file, sorted."""
        fs = _require("pyarrow.fs", "ParquetRef")
        filesystem, base = self._fs()
        infos = filesystem.get_file_info(fs.FileSelector(base, recursive=True))
        out = []
        for info in infos:
            if info.type != fs.FileType.File:
                continue
            relative = info.path[len(base):].lstrip("/")
            if any(part.startswith((".", "_")) for part in relative.split("/")):
                continue
            out.append((relative, info.size))
        return sorted(out)

    def listing_digest(self) -> str:
        files = self.files()
        if not files:
            raise FileNotFoundError(f"no parquet files under {self.uri}")
        return hashlib.sha256(json.dumps(files, separators=(",", ":")).encode()).hexdigest()

    def dataset(self) -> Any:
        current = self.listing_digest()
        if current != self.version:
            raise SnapshotChanged(f"{self.uri} changed since it was pinned ({self.version[:16]} -> {current[:16]})")
        ds = _require("pyarrow.dataset", "ParquetRef")
        filesystem, base = self._fs()
        return ds.dataset(base, filesystem=filesystem, format="parquet", partitioning=self.partitioning)
