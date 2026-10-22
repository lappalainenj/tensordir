"""`Record`: a content-addressed directory, built once from a spec and reopened afterwards."""

import dataclasses
import datetime
import json
import os
import re
import shutil
import threading
import time
import traceback
import types
import typing
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

import numpy as np

from . import _provenance
from ._canonical import canonical, sha256, untag
from ._directory import Directory
from ._files import read_file, write_file
from ._lease import Lease, lease_state
from ._root import get_root
from .codecs import atomic_path, codec_for, get_codec, remove_tree
from .dims import dims_for
from .errors import BuildFailed, DimsError, KeyCollision, ReadOnlyError, RecordBusy
from ._signature import Interface, interface_of
from ._stages import Progress, check_required, progress_of, stages_of
from .spec import Spec as _Spec

__all__ = ["Record", "Status", "FORMAT"]

FORMAT = 1
META = "_meta.json"
INTERNAL = ".tensordir"
Status = Literal["absent", "running", "done", "failed", "stale"]

_REGISTRY: dict[str, type["Record"]] = {}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", name)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, NotADirectoryError):
        return None


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME.match(name) or name.endswith((".tmp", ".lock")):
        raise ValueError(f"invalid field name {name!r}: use letters, digits, '_', '.', '-' and no leading '_' or '.'")


def _unwrap(annotation: Any) -> Any:
    while True:
        origin = typing.get_origin(annotation)
        if origin is typing.Annotated:
            annotation = typing.get_args(annotation)[0]
        elif origin in (typing.Union, types.UnionType):
            args = [a for a in typing.get_args(annotation) if a is not type(None)]
            if len(args) != 1:
                return annotation
            annotation = args[0]
        else:
            return annotation


def _wants_numpy(annotation: Any) -> bool:
    base = _unwrap(annotation)
    return base is np.ndarray or getattr(base, "array_type", None) is np.ndarray


class _ClassOrInstance:
    """A method bound to the class when looked up on the class, to the instance otherwise."""

    def __init__(self, on_class: Any, on_instance: Any):
        self._on_class, self._on_instance = on_class, on_instance

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        if obj is None:
            return types.MethodType(self._on_class, owner)
        return types.MethodType(self._on_instance, obj)


class Record(Directory):
    """A directory `<root>/<RecordName>/<key[:16]>/` holding `_meta.json` and the record's data.

    `__init__` is the build. Its positional-only parameters are the inputs, its keyword-only
    parameters the settings (the generated `Fit.Spec`):

    ```python
    class Fit(tensordir.Record):
        centres: Float[np.ndarray, "n_columns n_dims"]

        def __init__(self, points: np.ndarray, /, *, n_columns: int, seed: int = 0):
            self.centres = fit(points, n_columns, seed)

    fit = Fit(points, n_columns=8)  # builds once, then reopens without running __init__
    ```

    The key is the sha256 of the record type id, `schema_version`, the hashed settings and the
    identity of the inputs by position (records by key, arrays by digest).
    """

    #: Type id, part of every key; defaults to the class's qualified name (without the module), so
    #: keys survive moving the class between modules. Set it to keep keys across a class rename,
    #: or when two modules define classes with the same name.
    __record_name__: ClassVar[str]
    #: Bump to invalidate every key of this record type after an incompatible recipe change.
    schema_version: ClassVar[int] = 0
    #: Build directly in the final directory (readers see partial data, e.g. training logs) instead
    #: of a temporary directory renamed into place. Contents from an interrupted attempt are kept.
    in_place: ClassVar[bool] = False
    #: Seconds after the last heartbeat at which a build lease is considered stale.
    lease_ttl: ClassVar[float] = 300.0
    #: Seconds between heartbeats; None means lease_ttl / 4.
    heartbeat_interval: ClassVar[float | None] = None
    #: Size-name → dimension-name overrides for this record's typed fields.
    dims: ClassVar[Mapping[str, str]] = {}
    #: Root of this record type (also set by `@tensordir.root(path)`), joined onto the base root
    #: (`use_root`, `set_root`, `$TENSORDIR_ROOT` or the working directory) on every access: a
    #: relative path extends it, an absolute one replaces it. A `root=` argument replaces it.
    root: ClassVar[str | os.PathLike | None] = None
    #: Workspace record: fields may be written, appended and deleted after the build is done (by
    #: any process; each edit holds the lease briefly). The key still names the spec and inputs,
    #: but no longer guarantees the contents. Without an `__init__`, `get` creates an empty record.
    mutable: ClassVar[bool] = False
    #: Codec replacements for values written without `codec=`, by the name of the codec that
    #: would be chosen: `{"ndarray": "hdf5"}` stores arrays, including `append`ed ones, as HDF5
    #: files instead of zarr (see `tensordir.codecs.Hdf5Codec` for the trade-offs).
    codecs: ClassVar[Mapping[str, str]] = {}

    _is_record: ClassVar[bool] = True

    #: The settings as a frozen dataclass, generated from the keyword-only `__init__` parameters.
    Spec: ClassVar[type[_Spec]]
    _interface: ClassVar[Interface]

    def __init__(self) -> None:
        """The build of a record without inputs or settings: nothing to write."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "Spec" in cls.__dict__:
            raise TypeError(
                f"{cls.__qualname__}: declare settings as keyword-only __init__ parameters; the Spec is generated"
            )
        cls._interface = interface_of(cls, cls.__init__)
        cls.Spec = cls._interface.spec
        for stages in stages_of(cls):
            for name in stages.stages:
                if hasattr(Record, name):
                    raise TypeError(f"{cls.__qualname__}: stage {name!r} shadows Record.{name}; rename the stage")
        if stages_of(cls) and "in_place" not in cls.__dict__:
            cls.in_place = True  # so a failed staged build resumes where it stopped
        type_id = cls.type_id()
        other = _REGISTRY.get(type_id)
        if (
            other is not None
            and other.__module__ != cls.__module__
            and "__record_name__" not in cls.__dict__
            and "__record_name__" not in other.__dict__
        ):
            raise TypeError(
                f"record type id {type_id!r} is used by {other.__module__}.{other.__qualname__} and "
                f"{cls.__module__}.{cls.__qualname__}; set __record_name__ on one of them"
            )
        _REGISTRY[type_id] = cls

    # ----- identity -------------------------------------------------------------------------

    @classmethod
    def type_id(cls) -> str:
        return cls.__dict__.get("__record_name__") or cls.__qualname__

    @classmethod
    def dirname(cls) -> str:
        return _safe(cls.__dict__.get("__record_name__") or cls.__qualname__)

    @classmethod
    def _typed_fields(cls) -> dict[str, Any]:
        cache = "_tensordir_fields"
        if cache not in cls.__dict__:
            try:
                hints = typing.get_type_hints(cls, include_extras=True)
            except Exception:
                hints = {}
                for klass in reversed(cls.__mro__):
                    hints.update(getattr(klass, "__annotations__", {}))
            fields = {
                k: v
                for k, v in hints.items()
                if not k.startswith("_") and typing.get_origin(v) is not ClassVar and v is not ClassVar
                and k not in Record.__dict__
            }
            setattr(cls, cache, fields)
        return cls.__dict__[cache]

    @classmethod
    def _construct(cls, *inputs: Any, **settings: Any) -> Self:
        """Calling a record type gets the record: `Fit(points, n_columns=8)` is `Fit.get(...)`."""
        return cls.get(*inputs, **settings)

    @classmethod
    def open(cls, path: str | os.PathLike = ".") -> Self:
        """The record at `path`: `Record.at(path)`."""
        return Record.at(path) if cls is Record else cls.at(path)

    @classmethod
    def compute_key(cls, *inputs: Any, **settings: Any) -> str:
        """The full sha256 key for these inputs and settings (no filesystem access)."""
        inputs, spec, _ = cls._interface.bind(inputs, settings)
        return cls._key_of(spec, cls._identify(inputs))

    @classmethod
    def _identify(cls, inputs: Sequence[Any]) -> list:
        names = cls._interface.inputs
        return [canonical(v, digest_arrays=True, _path=f"inputs.{names[i]}") for i, v in enumerate(inputs)]

    @classmethod
    def _key_of(cls, spec: _Spec, identity: list) -> str:
        return sha256(
            {"type": cls.type_id(), "schema_version": cls.schema_version, "spec": spec.hashed_dict(), "inputs": identity}
        )

    @classmethod
    def _new(cls, path: Path, *, key: str | None, managed_root: Path | None, spec: _Spec | None = None) -> Self:
        self = object.__new__(cls)
        path = Path(path)
        if managed_root is not None:
            lock = managed_root / INTERNAL / "locks" / path.parent.name / path.name
            tmp = managed_root / INTERNAL / "tmp"
        else:
            lock = path.with_name(f".{path.name}.lock")
            tmp = path.parent
        state = {
            "_path": path, "_dir": path, "_key": key, "_spec": spec, "_lock": lock, "_tmp": tmp,
            "_writable": False, "_meta": None, "_done_meta": None, "_done_mtime": None,
            "_parent": None, "_meta_cache": None, "_root": managed_root,
        }
        self.__dict__.update(state)
        return self

    @classmethod
    def resolve_root(cls, root: str | os.PathLike | None = None) -> Path:
        """Where records of this type live: `tensordir.get_root(cls, root=root)`, i.e. the base root
        joined with `root` if given, else with the class's `root`."""
        return get_root(cls, root=root)

    @classmethod
    def locate(cls, *inputs: Any, root: str | os.PathLike | None = None, **settings: Any) -> Self:
        """The record for these inputs and settings, whatever its status; nothing is built."""
        inputs, spec, transients = cls._interface.bind(inputs, settings)
        key = cls._key_of(spec, cls._identify(inputs))
        base = cls.resolve_root(root)
        record = cls._new(base / cls.dirname() / key[:16], key=key, managed_root=base, spec=spec)
        record.__dict__["_call"] = (inputs, transients)
        return record

    # ----- build or reopen -------------------------------------------------------------------

    @classmethod
    def get(
        cls,
        *inputs: Any,
        root: str | os.PathLike | None = None,
        block: bool = True,
        timeout: float | None = None,
        retry_failed: bool = False,
        **settings: Any,
    ) -> Self:
        """Return the finished record for these inputs and settings, running `__init__` to build
        it if needed; `Fit(...)` is `Fit.get(...)` without the options.

        If another process is building it, wait (up to `timeout` seconds; `block=False` raises
        `RecordBusy` instead). A previous failure raises `BuildFailed` unless `retry_failed`.
        """
        record = cls.locate(*inputs, root=root, **settings)
        return record._obtain(block=block, timeout=timeout, retry_failed=retry_failed)

    @classmethod
    def lookup(cls, *inputs: Any, root: str | os.PathLike | None = None, **settings: Any) -> Self | None:
        """The finished record for these inputs and settings, or None; nothing is built."""
        record = cls.locate(*inputs, root=root, **settings)
        return record if record.status == "done" else None

    @classmethod
    def at(
        cls,
        path: str | os.PathLike,
        *inputs: Any,
        block: bool = True,
        timeout: float | None = None,
        retry_failed: bool = False,
        **settings: Any,
    ) -> Self:
        """The record in a hand-named directory.

        `td.Record.at(path)` opens whatever record is there, as its registered class. On a record
        type, `Fit.at(path, *inputs, **settings)` builds that record at `path` if the directory
        holds none, and opens it otherwise; a different record there raises `KeyCollision`.
        """
        path = Path(path).expanduser().absolute()
        if cls is Record:
            meta = _read_json(path / META)
            if meta is None:
                raise FileNotFoundError(f"no record at {path}")
            klass = _REGISTRY.get(meta["type"], cls)
            return klass._new(path, key=meta["key"], managed_root=_managed_root(path, meta["key"]))
        inputs, spec, transients = cls._interface.bind(inputs, settings)
        key = cls._key_of(spec, cls._identify(inputs))
        # a managed record's path shares its lease with the keyed call, so neither builds twice
        record = cls._new(path, key=key, managed_root=_managed_root(path, key), spec=spec)
        record.__dict__["_call"] = (inputs, transients)
        return record._obtain(block=block, timeout=timeout, retry_failed=retry_failed)

    @classmethod
    def query(cls, *, root: str | os.PathLike | None = None, status: str | None = "done", **filters: Any):
        """A `Collection` of this type's records under `resolve_root(root)` whose flattened spec
        matches `filters`.

        Filter names use `__` for nesting (`model__lr=0.1`). Values follow `tensordir.where`:
        equality, any of a list (`seed=[0, 1]`), comparisons (`lr=">1e-3, <0.1"`), a compiled
        regex, or a callable. `status=None` includes records of any status.
        """
        from .collection import query

        return query(cls, root=root, status=status, **filters)

    def _info_of_type(
        cls,
        *,
        root: str | os.PathLike | None = None,
        status: str | None = None,
        depth: int | None = None,
        include: Sequence[str] = (),
        sizes: bool = True,
        varying_only: bool = False,
        **filters: Any,
    ):
        return cls.query(root=root, status=status, **filters).table(
            depth=depth, include=include, sizes=sizes, varying_only=varying_only
        )

    def _info_of_record(self, *, depth: int | None = None, include: Sequence[str] = (), sizes: bool = True):
        from .collection import Collection

        return Collection([self]).table(depth=depth, include=include, sizes=sizes)

    info = _ClassOrInstance(_info_of_type, _info_of_record)
    """A table of records (another project's `info(Type, depth, include)`), needs pandas.

    On the class, `Fit.info(depth=..., include=..., **filters)` is
    `Fit.query(root=root, status=status, **filters).table(depth=..., include=..., sizes=...,
    varying_only=...)` with `status=None` (all statuses) by default. On a record,
    `fit.info(depth=..., include=..., sizes=...)` is its one-row table.
    """
    del _info_of_type, _info_of_record

    def _obtain(self, *, block: bool, timeout: float | None, retry_failed: bool) -> Self:
        """Wait for the build, or take the lease and run `__init__` as the build."""
        cls = type(self)
        inputs, transients = self.__dict__.pop("_call")
        deadline = None if timeout is None else time.monotonic() + timeout
        delay = 0.01
        checked = False
        while True:
            meta = self._disk_meta()
            if meta and meta["status"] == "done":
                return self
            if meta and meta["status"] == "failed" and not retry_failed:
                raise BuildFailed(self, meta.get("error") or {})
            if not checked:
                self._check_upstream(inputs)
                checked = True
            lease = Lease.acquire(self._lock, ttl=cls.lease_ttl, heartbeat=cls.heartbeat_interval)
            if lease is not None:
                with lease:
                    meta = self._disk_meta()
                    if meta and meta["status"] == "done":
                        return self
                    if meta and meta["status"] == "failed" and not retry_failed:
                        raise BuildFailed(self, meta.get("error") or {})
                    _BUILDING[self._lock] = threading.get_ident()
                    try:
                        self._run_build(inputs, transients, previous=meta)
                    finally:
                        _BUILDING.pop(self._lock, None)
                return self
            if _BUILDING.get(self._lock) == threading.get_ident():
                raise RuntimeError(
                    f"{self!r} is being built by this thread, so this call would wait for itself; inside the "
                    f"build, use `self`, or open the directory with td.Record.at(path)"
                )
            if not block:
                raise RecordBusy(self)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"{self!r} is still being built by another process after {timeout} s")
            time.sleep(delay)
            delay = min(delay * 2, 2.0)

    def _check_upstream(self, inputs: tuple) -> None:
        """Raise before a build if a record among the settings or inputs is not done."""
        names = type(self)._interface.inputs
        for name, value in [("spec", self.spec), *zip(names, inputs)]:
            for record in _records_in(value):
                status = record.status
                if status != "done":
                    raise ValueError(
                        f"cannot build {self!r}: its {'settings hold' if name == 'spec' else f'input {name!r} is or holds'} "
                        f"{record!r}, which is {status}; build it first"
                    )

    def _run_build(self, inputs: tuple, transients: dict, previous: dict | None) -> None:
        cls = type(self)
        t0 = time.monotonic()
        if cls.in_place:
            workdir = self._path
            workdir.mkdir(parents=True, exist_ok=True)
            fields = dict(previous["fields"]) if previous else {}
        else:
            workdir = self._tmp / f".{self._path.parent.name}-{self._path.name}-{uuid.uuid4().hex[:8]}"
            workdir.mkdir(parents=True)
            fields = {}
        meta = {
            "tensordir": FORMAT,
            "type": cls.type_id(),
            "schema_version": cls.schema_version,
            "key": self._key,
            "spec_type": f"{type(self.spec).__module__}:{type(self.spec).__qualname__}",
            "spec": self.spec.to_dict(),
            "inputs": cls._identify(inputs),
            "upstream": {r.key: self._relative(r.path) for v in (self.spec, list(inputs)) for r in _records_in(v)},
            "status": "running",
            "fields": fields,
            "created": (previous or {}).get("created") or _now(),
            "started": _now(),
            "finished": None,
            "duration_s": None,
            "modified": None,
            "error": None,
            "provenance": _provenance.collect(cls),
        }
        if cls.in_place and previous and previous.get("stages"):
            meta["stages"] = previous["stages"]
        self.__dict__.update(_dir=workdir, _meta=meta, _writable=True, _stage_calls={})
        try:
            self._write_meta()
            if cls.__init__ is Record.__init__ and not cls.mutable:
                raise NotImplementedError(f"{cls.__qualname__} defines no __init__; only workspace records may leave it out")
            settings = {name: getattr(self.spec, name) for name in cls._interface.settings}
            cls.__init__(self, *inputs, **settings, **transients)
            check_required(self)
            self._check_sizes()
        except Exception as exc:
            self._writable = False
            error = {
                "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            meta.update(status="failed", error=error, finished=_now(), duration_s=time.monotonic() - t0)
            if cls.in_place:
                self._write_meta()
            else:
                shutil.rmtree(workdir, ignore_errors=True)
                failed = self._tmp / f".{self._path.parent.name}-{self._path.name}-{uuid.uuid4().hex[:8]}"
                failed.mkdir(parents=True)
                meta["fields"] = {}
                self.__dict__["_dir"] = failed
                self._write_meta()
                self._place(failed)
            self.__dict__.update(_dir=self._path, _meta=None)
            raise BuildFailed(self, error) from exc
        except BaseException:
            self._writable = False
            if not cls.in_place:
                shutil.rmtree(workdir, ignore_errors=True)
            self.__dict__.update(_dir=self._path, _meta=None)
            raise
        self._writable = False
        meta.update(status="done", finished=_now(), duration_s=time.monotonic() - t0)
        self._write_meta()
        if not cls.in_place:
            self._place(workdir)
        self.__dict__.update(_dir=self._path, _meta=None)

    def _relative(self, path: Path) -> str:
        """`path` relative to this record's root when it lies inside it, else absolute; stored
        relative paths keep a copied or moved root linked."""
        if self._root is not None and Path(path).is_relative_to(self._root):
            return Path(path).relative_to(self._root).as_posix()
        return str(path)

    def _place(self, src: Path) -> None:
        final = self._path
        final.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(src, final)
            return
        except OSError:
            pass
        existing = _read_json(final / META)
        if existing and existing.get("status") == "done":
            shutil.rmtree(src, ignore_errors=True)  # another builder finished first
            return
        remove_tree(final)
        os.rename(src, final)

    def _check_sizes(self) -> None:
        sizes: dict[str, tuple[int, str]] = {}
        for name, info in self._manifest().items():
            for dim, size in zip(info.get("dims") or (), info.get("shape") or (), strict=bool(info.get("dims"))):
                if dim in sizes and sizes[dim][0] != size:
                    other_size, other = sizes[dim]
                    raise DimsError(f"dimension {dim!r} has size {other_size} in {other!r} but {size} in {name!r}")
                sizes.setdefault(dim, (size, name))

    # ----- state ----------------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def key(self) -> str:
        if self._key is None:
            self.__dict__["_key"] = (self._disk_meta() or {}).get("key")
        return self._key

    @property
    def spec(self) -> Any:
        """The spec this handle was created with, else the one stored on disk."""
        if self._spec is None:
            stored = self.meta["spec"]
            if type(self) is Record:
                return stored
            self.__dict__["_spec"] = type(self).Spec.from_dict(stored)
        return self._spec

    def _disk_meta(self) -> dict | None:
        mutable = type(self).mutable
        mtime = None
        if mutable:  # edits rewrite _meta.json: trust the cache only while its mtime holds
            try:
                mtime = (self._path / META).stat().st_mtime_ns
            except (FileNotFoundError, NotADirectoryError):
                mtime = None
        if self._done_meta is not None and (not mutable or mtime == self._done_mtime):
            return self._done_meta
        meta = _read_json(self._path / META)
        if meta is None:
            return None
        if self._key is not None and meta.get("key") != self._key:
            raise KeyCollision(f"{self._path} holds key {meta.get('key')}, expected {self._key}")
        if meta.get("status") == "done":
            self.__dict__.update(_done_meta=meta, _done_mtime=mtime)
        return meta

    @property
    def meta(self) -> dict:
        """The contents of `_meta.json` (spec, status, fields, timings, provenance, error)."""
        if self._writable:
            return self._meta
        meta = self._disk_meta()
        if meta is None:
            raise FileNotFoundError(f"{self!r} does not exist (status {self.status})")
        return meta

    def progress(self) -> Progress:
        """Every declared stage with its state, calls, times and fields, read from disk, so it
        follows a build running in another process; empty for a record type without stages."""
        meta = self._meta if self._writable else (self._disk_meta() or {})
        return progress_of(self, meta, self.status)

    @property
    def status(self) -> Status:
        """`absent`, `running`, `done`, `failed`, or `stale` (a build whose lease expired). A failed
        record that is being rebuilt (`retry_failed=True`) is `running`."""
        meta = self._disk_meta()
        if meta is not None and meta["status"] == "done":
            return "done"
        state = lease_state(self._lock, type(self).lease_ttl)
        if meta is None:
            return {"absent": "absent", "fresh": "running", "stale": "stale"}[state]
        if meta["status"] == "failed":
            return "running" if state == "fresh" else "failed"
        return "running" if state == "fresh" else "stale"

    @property
    def error(self) -> dict | None:
        meta = self._disk_meta()
        return meta.get("error") if meta else None

    @property
    def inputs(self) -> tuple[Any, ...]:
        """The inputs this record was built from, in order: records reopened, arrays as their
        `Digest`, other values in canonical form.

        **Experimental:** may change or be removed; see Design, Experimental features.

        An input record is reopened from the path stored at build time, else from its type's
        current root; if it is in neither place, the handle points at the stored path and its
        status is `absent`.
        """
        meta = self.meta
        return tuple(_revive_tags(value, meta.get("upstream", {}), self._root) for value in meta["inputs"])

    def lineage(self) -> list["Record"]:
        """Every record this one was built from, as an input or a spec field, directly or through
        other records: each once, upstream records before the records built from them.

        **Experimental:** may change or be removed; see Design, Experimental features.
        """
        order: list[Record] = []
        seen: set[str] = set()

        def visit(record: Record) -> None:
            meta = record._disk_meta()
            if meta is None:
                return
            for upstream in _upstream_of(meta, record._root):
                if upstream.key not in seen:
                    seen.add(upstream.key)
                    visit(upstream)
                    order.append(upstream)

        visit(self)
        return order

    def _manifest(self) -> dict:
        if self._writable:
            return self._meta["fields"]
        meta = self._disk_meta()
        return meta["fields"] if meta else {}

    def _write_meta(self) -> None:
        with atomic_path(self._dir / META) as tmp:
            tmp.write_text(json.dumps(self._meta, indent=1))

    # ----- fields ---------------------------------------------------------------------------

    def keys(self) -> list[str]:
        return sorted(self._manifest())

    def __contains__(self, name: object) -> bool:
        return name in self._manifest()

    def __iter__(self):
        return iter(self.keys())

    def _read(self, name: str, lazy: bool) -> Any:
        info = self._manifest().get(name)
        if info is None:
            raise KeyError(f"{self!r} has no field {name!r}; fields: {self.keys()}")
        return get_codec(info["codec"]).read(self._dir, name, info, lazy=lazy)

    def __getitem__(self, name: str) -> Any:
        """A lazy handle: `zarr.Array` for arrays, an unloaded xarray object for xarray fields."""
        return self._read(name, lazy=True)

    def load(self, name: str) -> Any:
        """Field `name` fully in memory (`numpy.ndarray` for arrays); any other file in the
        record directory by its suffix, as `Directory.load` reads it."""
        if name in self._manifest() or not (self._dir / name).exists():
            return self._read(name, lazy=False)
        return read_file(self._dir / name)

    def save(self, name: str, value: Any) -> Path:
        """Write `value` as the file `name` by its suffix (not a field): inside the build, or on a
        workspace record."""
        if not self._writable and not type(self).mutable:
            raise ReadOnlyError(f"{self!r} is read-only: save files inside __init__, or set `mutable = True`")
        path = self._dir / name
        write_file(path, value)
        return path

    def move(self, id: str) -> Self:
        """Records live where their key puts them; copy a record to another root instead."""
        raise TypeError(f"{type(self).__qualname__} is content-addressed and cannot be moved")

    def archive(self) -> Self:
        """Records live where their key puts them; delete it or keep it instead."""
        raise TypeError(f"{type(self).__qualname__} is content-addressed and cannot be archived")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or "_path" not in self.__dict__:
            raise AttributeError(name)
        typed = type(self)._typed_fields()
        if name in self._manifest():
            return self._read(name, lazy=not (name in typed and _wants_numpy(typed[name])))
        if name in typed:
            raise AttributeError(f"{self!r}: field {name!r} has not been written")
        raise AttributeError(f"{type(self).__qualname__!r} has no attribute or field {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        """`record.name = value` writes field `name`, except for private names and class attributes."""
        if name.startswith("_") or (name not in type(self)._typed_fields() and hasattr(type(self), name)):
            object.__setattr__(self, name, value)
        else:
            self.write(name, value)

    def __delattr__(self, name: str) -> None:
        """`del record.name` deletes field `name` (in `build`, or on a workspace record)."""
        if name.startswith("_") or name in self.__dict__:
            object.__delattr__(self, name)
        elif name in self._manifest():
            self.delete_field(name)
        else:
            raise AttributeError(f"{self!r} has no field {name!r}")

    def __setitem__(self, name: str, value: Any) -> None:
        self.write(name, value)

    def __delitem__(self, name: str) -> None:
        self.delete_field(name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._manifest()))

    @contextmanager
    def _edit(self) -> Iterator[dict]:
        """Yield the field manifest for changes and persist it afterwards.

        Inside `build` this is the build's manifest. On a finished workspace record it is the
        manifest on disk, re-read under the record's lease, which is held until the edit ends.
        """
        if self._writable:
            yield self._meta["fields"]
            self._write_meta()
            return
        cls = type(self)
        if not cls.mutable:
            raise ReadOnlyError(
                f"{self!r} is read-only: write fields inside __init__, or set `mutable = True` on "
                f"{cls.__qualname__} to make it a workspace record"
            )
        status = self.status
        if status != "done":
            raise ReadOnlyError(f"{self!r} is {status}; a workspace record takes edits once its build is done")
        lease = self._edit_lease()
        try:
            meta = _read_json(self._path / META)
            if meta is None or meta.get("key") != self.key:
                raise ReadOnlyError(f"{self!r} disappeared or was replaced during the edit")
            self.__dict__.update(_dir=self._path, _meta=meta)
            yield meta["fields"]
            meta["modified"] = _now()
            self._write_meta()
        finally:
            self.__dict__.update(_meta=None, _done_meta=None)
            lease.release()

    @contextmanager
    def _stage_edit(self, stage: str) -> Iterator[dict]:
        """Run optional `stage` on a finished record: under the lease, writable, and limited to
        new fields and the fields this stage wrote before. Yields the stage progress."""
        lease = self._edit_lease()
        try:
            meta = _read_json(self._path / META)
            if meta is None or meta.get("key") != self.key:
                raise ReadOnlyError(f"{self!r} disappeared or was replaced")
            self.__dict__.update(_dir=self._path, _meta=meta, _writable=True, _stage_owner=stage)
            yield meta.setdefault("stages", {})
            meta["modified"] = _now()
            self._write_meta()
        finally:
            self.__dict__.update(_meta=None, _done_meta=None, _writable=False, _stage_owner=None)
            lease.release()

    def _check_owner(self, name: str) -> None:
        """Inside an optional stage on a finished record, refuse fields the stage did not add."""
        stage = self.__dict__.get("_stage_owner")
        if stage is None or name not in self._meta["fields"]:
            return
        if name not in self._meta.get("stages", {}).get(stage, {}).get("fields", ()):
            raise ReadOnlyError(f"{self!r}: optional stage {stage!r} may not rewrite field {name!r}")

    def _edit_lease(self) -> Lease:
        cls = type(self)
        deadline = time.monotonic() + cls.lease_ttl
        delay = 0.005
        while True:
            lease = Lease.acquire(self._lock, ttl=cls.lease_ttl, heartbeat=cls.heartbeat_interval)
            if lease is not None:
                return lease
            if time.monotonic() >= deadline:
                raise RecordBusy(self)
            time.sleep(delay)
            delay = min(delay * 2, 0.5)

    def write(self, name: str, value: Any, *, codec: str | None = None, dims: tuple[str, ...] | None = None) -> None:
        """Store `value` as field `name`, replacing any earlier value.

        Allowed inside `build`, and on finished workspace records (`mutable = True`); elsewhere it
        raises `ReadOnlyError`. The codec follows the value's type unless given; `dims` overrides
        the dimension names derived from the field annotation. An array written over a field
        created by `append` replaces its rows and stays appendable.
        """
        _validate_name(name)
        annotation = type(self)._typed_fields().get(name)
        if annotation is not None:
            _check_type(name, annotation, value)
            if dims is None:
                dims = dims_for(annotation, type(self).dims)
        chosen = get_codec(codec) if codec else codec_for(value)
        if codec is None and chosen.name in type(self).codecs:
            chosen = get_codec(type(self).codecs[chosen.name])
        with self._edit() as fields:
            self._check_owner(name)
            old = fields.get(name)
            if old is not None and old.get("append") and codec is None and chosen.name == old["codec"]:
                chosen.delete(self._dir, name, old)
                fields[name] = {"codec": chosen.name, **chosen.append(self._dir, name, value, dims=dims)}
                return
            if old is not None and old["codec"] != chosen.name:
                get_codec(old["codec"]).delete(self._dir, name, old)
            info = chosen.write(self._dir, name, value, dims=dims)
            fields[name] = {"codec": chosen.name, **info}

    def append(
        self,
        name: str,
        values: Any,
        *,
        dims: tuple[str, ...] | None = None,
        chunk: int = 1024,
        codec: str | None = None,
    ) -> None:
        """Append rows (axis 0) to array field `name`, creating it on first use. With `in_place`
        records, readers see appended rows while the build runs (training logs); workspace records
        accept appends after the build too.

        A new field uses `codec`, else the record's `codecs["ndarray"]`, else zarr (`ndarray`);
        later appends keep the field's codec. Rows must match the trailing shape and are cast to
        the stored dtype.
        """
        _validate_name(name)
        annotation = type(self)._typed_fields().get(name)
        if dims is None and annotation is not None:
            dims = dims_for(annotation, type(self).dims)
        with self._edit() as fields:
            self._check_owner(name)
            old = fields.get(name)
            if old is not None and not old.get("append"):
                raise ValueError(f"{name!r} was written with write(); append() needs a field created by append()")
            if old is not None and codec is not None and codec != old["codec"]:
                raise ValueError(f"{name!r} is appended with codec {old['codec']!r}, not {codec!r}")
            chosen = get_codec(old["codec"] if old is not None else codec or type(self).codecs.get("ndarray", "ndarray"))
            fields[name] = {"codec": chosen.name, **chosen.append(self._dir, name, values, dims=dims, chunk=chunk)}

    def delete_field(self, name: str) -> None:
        """Remove field `name` (inside `build`, or on a workspace record); also `del record.name`."""
        with self._edit() as fields:
            self._check_owner(name)
            info = fields.pop(name, None)
            if info is None:
                raise KeyError(f"{self!r} has no field {name!r}")
            get_codec(info["codec"]).delete(self._dir, name, info)

    def to_xarray(self, name: str) -> Any:
        """An array field as an `xarray.DataArray` with its stored dimension names (loaded)."""
        import xarray as xr

        info = self._manifest()[name]
        value = self.load(name)
        if isinstance(value, (xr.DataArray, xr.Dataset, xr.DataTree)):
            return value
        dims = info.get("dims") or [f"{name}_dim_{i}" for i in range(np.ndim(value))]
        return xr.DataArray(value, dims=dims, name=name)

    # ----- housekeeping ----------------------------------------------------------------------

    def size_bytes(self) -> int:
        total = 0
        for dirpath, _, files in os.walk(self._path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
        return total

    def delete(self, *, force: bool = False) -> None:
        """Remove the record directory. A running build is only removed with `force=True`."""
        if self.status == "running" and not force:
            raise RecordBusy(self)
        remove_tree(self._path)
        self.__dict__["_done_meta"] = None

    @property
    def memory(self):
        """A `joblib.Memory` cache stored in this record's `__cache__` directory."""
        from ._memory import memory

        return memory(self._path / "__cache__")

    # ----- identity protocol -----------------------------------------------------------------

    def __tensordir_canonical__(self, *, full: bool = False) -> dict:
        """`{"__record__": type, "key": key}`; a workspace record adds the time of its last edit
        (`modified`), so records built from it get new keys after it changes. The full form, stored
        in `_meta.json`, adds the path the record is reopened from."""
        out = {"__record__": type(self).type_id(), "key": self.key}
        if type(self).mutable:
            meta = self._disk_meta()
            done = meta is not None and meta.get("status") == "done"
            out["modified"] = (meta.get("modified") or meta.get("finished")) if done else None
        if full:
            out["path"] = str(self._path)
        return out

    @classmethod
    def __tensordir_revive__(cls, value: Mapping) -> "Record":
        """Reopen a record from its full canonical form (a record stored in a spec field)."""
        return _open_upstream(value, {}, cls)

    def __reduce__(self):
        return _reopen, (type(self).type_id(), self.key, str(self._path))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Record) and (type(self).type_id(), self.key, self._path) == (
            type(other).type_id(), other.key, other._path,
        )

    def __hash__(self) -> int:
        return hash((type(self).type_id(), self.key, self._path))

    def __repr__(self) -> str:
        key = (self.key or "?")[:16]
        try:
            status = self.status
        except Exception:
            status = "?"
        return f"<{type(self).__qualname__} {key} {status}>"

    def _field_rows(self) -> list[tuple[str, str, str]]:
        rows = []
        for name, info in sorted(self._manifest().items()):
            parts = []
            if info.get("dtype"):
                parts.append(np.dtype(info["dtype"]).name)
            if info.get("shape") is not None:
                parts.append(str(tuple(info["shape"])))
            if info.get("dims"):
                parts.append("[" + ", ".join(info["dims"]) + "]")
            if info.get("kind"):
                parts.append(info["kind"])
            if info.get("append"):
                parts.append("appendable")
            rows.append((name, info["codec"], " ".join(parts)))
        return rows

    def _tree(self) -> str:
        rows = self._field_rows()
        if (self._path / "__cache__").is_dir():
            rows.append(("__cache__/", "joblib", ""))
        width = max((len(name) for name, _, _ in rows), default=0)
        lines = [f"{self!r} {self._path}"]
        for i, (name, codec, details) in enumerate(rows):
            branch = "└──" if i == len(rows) - 1 else "├──"
            lines.append(f"{branch} {name:<{width}}  {codec:<11} {details}".rstrip())
        return "\n".join(lines)

    def tree(self) -> None:
        """Print the record's type, key, status, path and stored fields with their shapes and dims.

        IPython shows the same tree for a record that ends a cell; `repr` stays one line."""
        print(self._tree())

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        try:
            text = repr(self) if cycle else self._tree()
        except Exception:
            text = repr(self)
        printer.text(text)

    def _repr_html_(self) -> str:
        import html

        from ._where import display
        from .collection import flatten, spec_dict

        try:
            status = self.status
            meta = self._disk_meta()
        except Exception as exc:
            return f"<p>{html.escape(repr(self))}: {html.escape(str(exc))}</p>"
        out = [
            f"<p><b>{html.escape(type(self).__qualname__)}</b> <code>{(self.key or '?')[:16]}</code> "
            f"<i>{status}</i> <small>{html.escape(str(self._path))}</small></p>"
        ]
        if meta is None:
            return out[0]
        if meta.get("error"):
            error = meta["error"]
            out.append(f"<pre>{html.escape(error.get('type', ''))}: {html.escape(error.get('message', ''))}</pre>")
        spec = flatten(spec_dict(type(self), meta.get("spec", {})))
        if spec:
            out.append("<table><thead><tr><th>spec</th><th></th></tr></thead><tbody>")
            out.extend(
                f"<tr><td><code>{html.escape(k)}</code></td><td>{html.escape(repr(display(v)))}</td></tr>"
                for k, v in spec.items()
            )
            out.append("</tbody></table>")
        fields = self._field_rows()
        if fields:
            out.append("<table><thead><tr><th>field</th><th>codec</th><th></th></tr></thead><tbody>")
            out.extend(
                f"<tr><td><code>{html.escape(n)}</code></td><td>{c}</td><td>{html.escape(d)}</td></tr>"
                for n, c, d in fields
            )
            out.append("</tbody></table>")
        return "".join(out)


Record._interface = interface_of(Record, Record.__init__)
Record.Spec = Record._interface.spec


def _records_in(value: Any) -> Iterator[Record]:
    """The records in `value`, searching specs, mappings and sequences."""
    if isinstance(value, Record):
        yield value
    elif isinstance(value, _Spec):
        for f in dataclasses.fields(value):
            yield from _records_in(getattr(value, f.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _records_in(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _records_in(item)


def _tags_in(value: Any) -> Iterator[dict]:
    """The record tags (`{"__record__": ...}`) in a canonical value."""
    if isinstance(value, dict):
        if "__record__" in value:
            yield value
        else:
            for item in value.values():
                yield from _tags_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from _tags_in(item)


def _revive_tags(value: Any, paths: Mapping[str, str], base: Path | None = None) -> Any:
    if isinstance(value, dict):
        if "__record__" in value:
            return _open_upstream(value, paths, base=base)
        if set(value) <= {"__float__", "__digest__"}:
            return untag(value)
        return {k: _revive_tags(v, paths, base) for k, v in value.items()}
    if isinstance(value, list):
        return [_revive_tags(v, paths, base) for v in value]
    return value


def _upstream_of(meta: dict, base: Path | None = None) -> list[Record]:
    """The records named in a record's stored spec and inputs."""
    paths = meta.get("upstream", {})
    return [_open_upstream(tag, paths, base=base) for part in (meta.get("spec"), meta.get("inputs")) for tag in _tags_in(part)]


def _open_upstream(tag: Mapping, paths: Mapping[str, str], cls: type[Record] | None = None, base: Path | None = None) -> Record:
    """The record a tag names: at its stored path (relative paths joined onto `base`, the root of
    the record that names it), else under its type's current root."""
    type_id, key = tag["__record__"], tag["key"]
    klass = resolve_type(type_id)
    if cls is not None and not issubclass(klass, cls):
        if klass is not Record:
            raise TypeError(f"{type_id} {key[:16]} is a {klass.__qualname__}, not a {cls.__qualname__}")
        klass = cls
    stored = tag.get("path") or paths.get(key)
    candidates = []
    if stored:
        path = Path(stored)
        if not path.is_absolute() and base is not None:
            path = base / path
        candidates.append(path)
    if klass is not Record:
        candidates.append(klass.resolve_root() / klass.dirname() / key[:16])
    if not candidates:
        raise FileNotFoundError(f"no path is stored for {type_id} {key[:16]} and its type is not registered")
    for path in candidates:
        meta = _read_json(path / META)
        if meta is not None and meta.get("key") == key:
            return klass._new(path, key=key, managed_root=_managed_root(path, key))
    return klass._new(candidates[0], key=key, managed_root=_managed_root(candidates[0], key))


def _check_type(name: str, annotation: Any, value: Any) -> None:
    base = _unwrap(annotation)
    if hasattr(base, "dim_str") or isinstance(base, type):
        try:
            ok = isinstance(value, base)
        except TypeError:
            return
        if not ok:
            raise TypeError(f"field {name!r}: {type(value).__qualname__} {getattr(value, 'shape', '')} does not match {annotation}")


def resolve_type(type_id: str) -> type[Record]:
    """The record class registered under `type_id`; `Record` if no such class is defined."""
    return _REGISTRY.get(type_id, Record)


#: The builds running in this process, by lock path, with the thread running each.
_BUILDING: dict[Path, int] = {}

def _managed_root(path: Path, key: str | None) -> Path | None:
    """The root of a content-addressed record at `path`, or None for a hand-named directory."""
    if key and path.name == key[:16] and (path.parent.parent / INTERNAL).is_dir():
        return path.parent.parent
    return None


def _reopen(type_id: str, key: str, path: str) -> Record:
    path = Path(path)
    return resolve_type(type_id)._new(path, key=key, managed_root=_managed_root(path, key))
