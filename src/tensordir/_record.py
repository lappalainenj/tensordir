"""`Record`: a content-addressed directory, built once from a spec and reopened afterwards."""

import dataclasses
import datetime
import json
import os
import re
import shutil
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
from ._lease import Lease, lease_state
from ._root import get_root
from .codecs import atomic_path, codec_for, get_codec, remove_tree
from .dims import dims_for
from .errors import BuildFailed, DimsError, KeyCollision, ReadOnlyError, RecordBusy
from .spec import Spec as _Spec

__all__ = ["Record", "Status", "FORMAT"]

FORMAT = 1
META = "_record.json"
INTERNAL = ".tensordir"
Status = Literal["absent", "running", "done", "failed", "stale"]

_REGISTRY: dict[str, type["Record"]] = {}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


class _EmptySpec(_Spec):
    pass


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


class Record:
    """A directory `<root>/<RecordName>/<key[:16]>/` holding `_record.json` and the record's data.

    Subclass it, declare a nested `Spec`, typed fields, and a `build` recipe:

    ```python
    class Fit(tensordir.Record):
        __record_name__ = "Fit"

        class Spec(tensordir.Spec):
            n_sweeps: int
            seed: int = 0

        centres: Float[np.ndarray, "n_columns n_dims"]

        def build(self, spec, points):
            self.centres = fit(points, spec.n_sweeps, spec.seed)

    fit = Fit.get(Fit.Spec(n_sweeps=100), points=points)  # builds once, then reopens
    ```

    The key is the sha256 of the record type id, `schema_version`, the hashed spec fields and the
    identity of the inputs (records by key, arrays by digest).
    """

    #: Stable type id; defaults to "module:qualname". Set it so keys survive moves and renames.
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
    #: but no longer guarantees the contents. Without a `build`, `get` creates an empty record.
    mutable: ClassVar[bool] = False
    #: Codec replacements for values written without `codec=`, by the name of the codec that
    #: would be chosen: `{"ndarray": "hdf5"}` stores arrays, including `append`ed ones, as HDF5
    #: files instead of zarr (see `tensordir.codecs.Hdf5Codec` for the trade-offs).
    codecs: ClassVar[Mapping[str, str]] = {}

    Spec: ClassVar[type[_Spec]] = _EmptySpec

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not (isinstance(cls.Spec, type) and issubclass(cls.Spec, _Spec)):
            raise TypeError(f"{cls.__qualname__}.Spec must subclass tensordir.Spec")
        _REGISTRY[cls.type_id()] = cls

    # ----- identity -------------------------------------------------------------------------

    @classmethod
    def type_id(cls) -> str:
        return cls.__dict__.get("__record_name__") or f"{cls.__module__}:{cls.__qualname__}"

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
    def _coerce_spec(cls, spec: Any) -> _Spec:
        if spec is None:
            return cls.Spec()
        if isinstance(spec, cls.Spec):
            return spec
        if isinstance(spec, Mapping):
            return cls.Spec(**spec)
        raise TypeError(f"{cls.__qualname__} expects a {cls.Spec.__qualname__} or a mapping, got {type(spec).__qualname__}")

    @classmethod
    def compute_key(cls, spec: Any = None, /, **inputs: Any) -> str:
        """The full sha256 key for `spec` and `inputs` (no filesystem access)."""
        return cls._key_of(cls._coerce_spec(spec), cls._identify(inputs))

    @classmethod
    def _identify(cls, inputs: Mapping[str, Any]) -> dict:
        return {k: canonical(v, digest_arrays=True, _path=f"inputs.{k}") for k, v in sorted(inputs.items())}

    @classmethod
    def _key_of(cls, spec: _Spec, identity: dict) -> str:
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
        }
        self.__dict__.update(state)
        return self

    @classmethod
    def resolve_root(cls, root: str | os.PathLike | None = None) -> Path:
        """Where records of this type live: `tensordir.get_root(cls, root=root)`, i.e. the base root
        joined with `root` if given, else with the class's `root`."""
        return get_root(cls, root=root)

    @classmethod
    def locate(cls, spec: Any = None, /, *, root: str | os.PathLike | None = None, **inputs: Any) -> Self:
        """The record for `spec` and `inputs`, whatever its status; nothing is built."""
        spec = cls._coerce_spec(spec)
        key = cls._key_of(spec, cls._identify(inputs))
        base = cls.resolve_root(root)
        return cls._new(base / cls.dirname() / key[:16], key=key, managed_root=base, spec=spec)

    # ----- build or reopen -------------------------------------------------------------------

    @classmethod
    def get(
        cls,
        spec: Any = None,
        /,
        *,
        root: str | os.PathLike | None = None,
        block: bool = True,
        timeout: float | None = None,
        retry_failed: bool = False,
        **inputs: Any,
    ) -> Self:
        """Return the finished record for `spec` and `inputs`, building it if needed.

        If another process is building it, wait (up to `timeout` seconds; `block=False` raises
        `RecordBusy` instead). A previous failure raises `BuildFailed` unless `retry_failed`.
        """
        record = cls.locate(spec, root=root, **inputs)
        return record._obtain(inputs, block=block, timeout=timeout, retry_failed=retry_failed)

    @classmethod
    def lookup(cls, spec: Any = None, /, *, root: str | os.PathLike | None = None, **inputs: Any) -> Self | None:
        """The finished record for `spec` and `inputs`, or None."""
        record = cls.locate(spec, root=root, **inputs)
        return record if record.status == "done" else None

    @classmethod
    def at(
        cls,
        path: str | os.PathLike,
        spec: Any = None,
        /,
        *,
        block: bool = True,
        timeout: float | None = None,
        retry_failed: bool = False,
        **inputs: Any,
    ) -> Self:
        """A record in a hand-named directory.

        Without a spec, open the record at `path` (as its registered class when called on
        `Record`). With a spec, build it there if absent; a different record there is an error.
        """
        path = Path(path).expanduser().absolute()
        if spec is None and not inputs:
            meta = _read_json(path / META)
            if meta is None:
                raise FileNotFoundError(f"no record at {path}")
            klass = _REGISTRY.get(meta["type"], cls) if cls is Record else cls
            return klass._new(path, key=meta["key"], managed_root=_managed_root(path, meta["key"]))
        spec = cls._coerce_spec(spec)
        record = cls._new(path, key=cls._key_of(spec, cls._identify(inputs)), managed_root=None, spec=spec)
        return record._obtain(inputs, block=block, timeout=timeout, retry_failed=retry_failed)

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

    def build(self, spec: Any, **inputs: Any) -> None:
        """The recipe. Write fields with `self.name = value`, `self["name"] = value` or `append`.

        Workspace records (`mutable = True`) may leave it out: their fields come later."""
        if type(self).mutable:
            return
        raise NotImplementedError(f"{type(self).__qualname__} does not define build()")

    def _obtain(
        self,
        inputs: dict,
        *,
        block: bool,
        timeout: float | None,
        retry_failed: bool,
        recipe: Callable[[], None] | None = None,
    ) -> Self:
        """Wait for or run the build; `recipe` replaces `self.build(spec, **inputs)` if given."""
        cls = type(self)
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
                    self._run_build(inputs, previous=meta, recipe=recipe)
                return self
            if not block:
                raise RecordBusy(self)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"{self!r} is still being built by another process after {timeout} s")
            time.sleep(delay)
            delay = min(delay * 2, 2.0)

    def _check_upstream(self, inputs: dict) -> None:
        """Raise before a build if a record among the spec fields or inputs is not done."""
        for name, value in [("spec", self.spec), *sorted(inputs.items())]:
            for record in _records_in(value):
                status = record.status
                if status != "done":
                    raise ValueError(
                        f"cannot build {self!r}: its {'spec holds' if name == 'spec' else f'input {name!r} is or holds'} "
                        f"{record!r}, which is {status}; build it first with {type(record).__qualname__}.get(...)"
                    )

    def _run_build(self, inputs: dict, previous: dict | None, recipe: Callable[[], None] | None = None) -> None:
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
            "upstream": {r.key: str(r.path) for v in (self.spec, inputs) for r in _records_in(v)},
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
        self.__dict__.update(_dir=workdir, _meta=meta, _writable=True)
        try:
            self._write_meta()
            if recipe is None:
                self.build(self.spec, **inputs)
            else:
                recipe()
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
        if mutable:  # edits rewrite _record.json: trust the cache only while its mtime holds
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
        """The contents of `_record.json` (spec, status, fields, timings, provenance, error)."""
        if self._writable:
            return self._meta
        meta = self._disk_meta()
        if meta is None:
            raise FileNotFoundError(f"{self!r} does not exist (status {self.status})")
        return meta

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
    def inputs(self) -> dict[str, Any]:
        """The keyword inputs this record was built from, by name: records reopened, arrays as
        their `Digest`, other values in canonical form.

        **Experimental:** may change or be removed; see Design, Experimental features.

        An input record is reopened from the path stored at build time, else from its type's
        current root; if it is in neither place, the handle points at the stored path and its
        status is `absent`.
        """
        meta = self.meta
        return {name: _revive_tags(value, meta.get("upstream", {})) for name, value in meta["inputs"].items()}

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
            for upstream in _upstream_of(meta):
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
        """The field's value fully in memory (`numpy.ndarray` for arrays)."""
        return self._read(name, lazy=False)

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
                f"{self!r} is read-only: write fields inside build(), or set `mutable = True` on "
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
        in `_record.json`, adds the path the record is reopened from."""
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


def _revive_tags(value: Any, paths: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        if "__record__" in value:
            return _open_upstream(value, paths)
        if set(value) <= {"__float__", "__digest__"}:
            return untag(value)
        return {k: _revive_tags(v, paths) for k, v in value.items()}
    if isinstance(value, list):
        return [_revive_tags(v, paths) for v in value]
    return value


def _upstream_of(meta: dict) -> list[Record]:
    """The records named in a record's stored spec and inputs."""
    paths = meta.get("upstream", {})
    return [_open_upstream(tag, paths) for part in (meta.get("spec"), meta.get("inputs")) for tag in _tags_in(part)]


def _open_upstream(tag: Mapping, paths: Mapping[str, str], cls: type[Record] | None = None) -> Record:
    """The record a tag names: at its stored path, else under its type's current root."""
    type_id, key = tag["__record__"], tag["key"]
    klass = resolve_type(type_id)
    if cls is not None and not issubclass(klass, cls):
        if klass is not Record:
            raise TypeError(f"{type_id} {key[:16]} is a {klass.__qualname__}, not a {cls.__qualname__}")
        klass = cls
    stored = tag.get("path") or paths.get(key)
    candidates = [Path(stored)] if stored else []
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
    """The record class registered (or importable) under `type_id`; `Record` if unknown."""
    if type_id in _REGISTRY:
        return _REGISTRY[type_id]
    module, sep, qualname = type_id.partition(":")
    if sep:
        try:
            import importlib

            obj: Any = importlib.import_module(module)
            for part in qualname.split("."):
                obj = getattr(obj, part)
            return obj
        except (ImportError, AttributeError):
            pass
    return Record


def _managed_root(path: Path, key: str | None) -> Path | None:
    """The root of a content-addressed record at `path`, or None for a hand-named directory."""
    if key and path.name == key[:16] and (path.parent.parent / INTERNAL).is_dir():
        return path.parent.parent
    return None


def _reopen(type_id: str, key: str, path: str) -> Record:
    path = Path(path)
    return resolve_type(type_id)._new(path, key=key, managed_root=_managed_root(path, key))
