"""`Directory`: a folder addressed by name, with typed metadata, typed children and files.

```python
class Dev(td.Directory):
    class Track(td.Directory):
        title: str = ""

        class Study(td.Directory):
            class Version(td.Directory):
                names = "v{n}"
                title: str = ""
                status: Literal["active", "superseded", "private"] = "active"
                builds_on: list[td.NodeId] = []

dev = Dev("dev")
v = dev.track("toy-model").study("inverse").version("v4")
dev.track.study.version.keys()          # IDs of every version below dev
study.version.new(title="...")           # the next free vK
v.move("toy-model/ablations/v1")  # the old ID becomes an alias
v.archive()                              # moves under archive/, keeps its ID
```

- Opening a directory writes nothing. A typed directory writes `_meta.yaml` (its type, metadata
  fields and aliases, in a fixed order and without timestamps, so it can be tracked by git) when
  a field is set, when it is created with `new`, or when a file is saved into it.
- A nested `Directory` class is a child type; the parent gets an accessor named in snake_case
  (`Track` becomes `.track`). `archive` is reserved: archived children live in `archive/`.
- Files are read and written by suffix (`load`, `save`); see `tensordir.register_suffix`.
"""

import contextlib
import contextvars
import json
import os
import re
import time
import types
import typing
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Literal, NewType, Self

from ._files import read_file, write_file
from ._lease import Lease
from .codecs import atomic_path

__all__ = ["Directory", "Level", "NodeId", "active_producer"]

META = "_meta.yaml"
ARCHIVE = "archive"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")

#: A metadata field holding node IDs (`builds_on: list[td.NodeId]`): each value must resolve
#: from the node's tree root when it is written.
NodeId = NewType("NodeId", str)

_DIRECTORIES: dict[str, type["Directory"]] = {}
_ACTIVE: contextvars.ContextVar["Directory | None"] = contextvars.ContextVar("tensordir_active", default=None)


def _snake(name: str) -> str:
    """`StudyVersion` -> `study_version`."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _check_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME.match(name) or name == ARCHIVE:
        raise ValueError(f"invalid directory name {name!r}: letters, digits, '_', '.', '-', and not {ARCHIVE!r}")
    return name


def _read_meta(path: Path) -> dict | None:
    import yaml

    try:
        return yaml.safe_load((path / META).read_text()) or {}
    except (FileNotFoundError, NotADirectoryError):
        return None


def active_producer() -> dict | None:
    """The directory set by `Directory.active()` in this context or a parent process, as
    `{"id": ..., "path": ...}`; stored in the provenance of records built meanwhile."""
    node = _ACTIVE.get()
    if node is not None:
        return {"id": node.id, "path": str(node.path)}
    env = os.environ.get("TENSORDIR_PRODUCER")
    return json.loads(env) if env else None


def _validate(node: "Directory", name: str, annotation: Any, value: Any) -> None:
    """Raise if `value` does not fit the metadata field's annotation."""
    origin = typing.get_origin(annotation)
    if annotation is Any:
        return
    if origin is typing.Annotated:
        return _validate(node, name, typing.get_args(annotation)[0], value)
    if origin in (typing.Union, types.UnionType):
        errors = []
        for option in typing.get_args(annotation):
            try:
                return _validate(node, name, option, value)
            except (TypeError, ValueError, KeyError) as exc:
                errors.append(exc)
        raise TypeError(f"{name}: {value!r} matches none of {annotation}")
    if origin is Literal:
        if value not in typing.get_args(annotation):
            raise ValueError(f"{name}: {value!r} is not one of {typing.get_args(annotation)}")
        return
    if annotation is type(None):
        if value is not None:
            raise TypeError(f"{name}: expected None")
        return
    if annotation is NodeId:
        if not isinstance(value, str):
            raise TypeError(f"{name}: expected a node ID, got {value!r}")
        node.origin.resolve(value)  # raises KeyError if it names no node
        return
    if origin in (list, tuple, set, frozenset) or annotation in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{name}: expected a list, got {type(value).__qualname__}")
        (item,) = typing.get_args(annotation)[:1] or (Any,)
        for v in value:
            _validate(node, name, item, v)
        return
    if annotation is float and isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(annotation, type) and not isinstance(value, annotation):
        raise TypeError(f"{name}: expected {annotation.__qualname__}, got {type(value).__qualname__}")


class _DirectoryType(type):
    """Calling a directory type opens a directory: `Dev("dev")`. Record types override it."""

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        return cls._construct(*args, **kwargs)


class _MetaField:
    """A typed metadata field of a directory type, stored in its `_meta.yaml`."""

    def __init__(self, name: str, annotation: Any, default: Any):
        self.name, self.annotation, self.default = name, annotation, default

    def __get__(self, node: "Directory | None", owner: type | None = None) -> Any:
        if node is None:
            return self
        meta = node._metadata()
        return meta[self.name] if self.name in meta else _copy(self.default)

    def __set__(self, node: "Directory", value: Any) -> None:
        node.update(**{self.name: value})


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value)) if isinstance(value, (list, dict)) else value


class _Children:
    """The accessor of a child type on its parent type: `Dev.track`, `dev.track`."""

    def __init__(self, name: str, cls: type["Directory"]):
        self.name, self.cls = name, cls

    def __get__(self, node: "Directory | None", owner: type | None = None) -> Any:
        if node is None:
            return self
        return Level(self.cls, [node], node.origin)


class Level:
    """The children of one type under one or more parent directories.

    Call it with a name (or a position in `keys()`) for one child, `keys()` to list names, an
    attribute for the next level down across all of them, and `new(...)` to create a child.
    """

    def __init__(self, cls: type["Directory"], parents: list["Directory"], origin: "Directory", *, across: bool = False):
        self.cls, self.parents, self.origin, self.across = cls, parents, origin, across

    def nodes(self, *, archived: bool = False) -> list["Directory"]:
        """The child directories, sorted by name within each parent."""
        out = []
        for parent in self.parents:
            out.extend(parent._children_of(self.cls, archived=archived))
        return out

    def keys(self, *, archived: bool = False) -> list[str]:
        """Child names under one parent; IDs relative to where the chain started when the level
        spans several parents (`dev.track.study.keys()`)."""
        if not self.across:
            return [n.name for n in self.nodes(archived=archived)]
        return [n._id_from(self.origin) for n in self.nodes(archived=archived)]

    def __call__(self, key: str | int, *, archived: bool = True) -> "Directory":
        """The child named `key` (archived children included), or at position `key` in `keys()`."""
        if isinstance(key, int):
            return self.nodes()[key]
        matches = [n for n in self.nodes(archived=archived) if key in (n.name, n._id_from(self.origin))]
        if not matches:
            raise KeyError(f"no {self.cls.__qualname__} {key!r}; available: {self.keys()}")
        if len(matches) > 1:
            raise KeyError(f"{key!r} names several: {[n._id_from(self.origin) for n in matches]}; give the ID")
        return matches[0]

    __getitem__ = __call__

    def __iter__(self) -> Iterator["Directory"]:
        return iter(self.nodes())

    def __len__(self) -> int:
        return len(self.nodes())

    def __getattr__(self, name: str) -> "Level":
        child = self.cls._child_types.get(name)
        if child is None:
            raise AttributeError(f"{self.cls.__qualname__} has no child type {name!r}; children: {sorted(self.cls._child_types)}")
        return Level(child, self.nodes(), self.origin, across=True)

    def new(self, name: str | None = None, **fields: Any) -> "Directory":
        """Create a child, named `name` or by the child type's `names` pattern (`"v{n}"`: one more
        than the highest number taken by a child, an archived child or an alias)."""
        if self.across or len(self.parents) != 1:
            raise TypeError("new() needs one parent: choose it first, e.g. track(...).study.new(...)")
        parent = self.parents[0]
        taken = parent._taken_names()
        if name is None:
            if not self.cls.names:
                raise TypeError(f"{self.cls.__qualname__} has no `names` pattern; pass name=")
            head, _, tail = self.cls.names.partition("{n}")
            pattern = re.compile(f"^{re.escape(head)}(\\d+){re.escape(tail)}$")
            numbers = [int(m.group(1)) for m in map(pattern.match, taken) if m]
            name = self.cls.names.format(n=max(numbers, default=0) + 1)
        elif name in taken:
            raise FileExistsError(f"{parent!r} already has, archived or aliases a child {name!r}")
        child = self.cls._at(parent.path / _check_name(name), parent=parent)
        child._write_meta({**{f: _copy(d.default) for f, d in self.cls._meta_fields.items()}, **fields}, create=True)
        return child

    def __repr__(self) -> str:
        return f"<{self.cls.__qualname__} level: {self.keys()}>"


class Directory(metaclass=_DirectoryType):
    """A folder addressed by its path. `td.Directory(path)` views any folder; subclasses declare
    metadata fields (annotations with defaults) and child types (nested classes)."""

    #: Type id stored in `_meta.yaml`; defaults to the class's qualified name.
    __type_name__: ClassVar[str]
    #: Root of this directory type, joined onto the base root like a record type's `root`.
    root: ClassVar[str | os.PathLike | None] = None
    #: Name pattern for `Level.new()` without a name, with `{n}` for the number: `"v{n}"`.
    names: ClassVar[str | None] = None

    _is_record: ClassVar[bool] = False
    _meta_fields: ClassVar[dict[str, _MetaField]] = {}
    _child_types: ClassVar[dict[str, type["Directory"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls._is_record:
            return
        hints = typing.get_type_hints(cls, include_extras=True) if cls.__dict__.get("__annotations__") else {}
        fields = dict(cls._meta_fields)
        for name, annotation in hints.items():
            if name.startswith("_") or typing.get_origin(annotation) is ClassVar or name not in cls.__dict__.get("__annotations__", {}):
                continue
            if name in ("aliases", "type"):
                raise TypeError(f"{cls.__qualname__}: {name!r} is reserved")
            if name not in cls.__dict__:
                raise TypeError(f"{cls.__qualname__}.{name}: metadata fields need a default")
            field = _MetaField(name, annotation, cls.__dict__[name])
            setattr(cls, name, field)
            fields[name] = field
        cls._meta_fields = fields
        children = dict(cls._child_types)
        for value in list(cls.__dict__.values()):
            if isinstance(value, type) and issubclass(value, Directory) and not value._is_record:
                attr = _snake(value.__name__)
                setattr(cls, attr, _Children(attr, value))
                children[attr] = value
        cls._child_types = children
        _DIRECTORIES[cls.type_id()] = cls

    # ----- opening -------------------------------------------------------------------------

    @classmethod
    def type_id(cls) -> str:
        """The type stored in `_meta.yaml`."""
        return cls.__dict__.get("__type_name__") or cls.__qualname__

    @classmethod
    def _construct(cls, path: str | os.PathLike = ".") -> Self:
        return cls.open(path)

    @classmethod
    def open(cls, path: str | os.PathLike = ".") -> Self:
        """The directory at `path`, without writing anything. A relative path is joined onto the
        type's root (see `tensordir.get_root`); `td.Directory.open` returns a typed directory as
        its registered class."""
        from ._root import get_root

        path = Path(path).expanduser()
        if not path.is_absolute():
            path = get_root(cls) / path
        path = Path(os.path.normpath(path))  # `..` is collapsed, symlinks are kept
        meta = _read_meta(path)
        klass = cls
        if meta is not None and meta.get("type"):
            stored = _DIRECTORIES.get(meta["type"])
            if cls is Directory and stored is not None:
                klass = stored
            elif cls is not Directory and meta["type"] != cls.type_id():
                raise TypeError(f"{path} holds a {meta['type']}, not a {cls.type_id()}")
        return klass._at(path, parent=None)

    @classmethod
    def _at(cls, path: Path, *, parent: "Directory | None") -> Self:
        self = object.__new__(cls)
        self.__dict__.update(_path=Path(path), _parent=parent, _meta_cache=None)
        return self

    # ----- identity --------------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def name(self) -> str:
        return self._path.name

    @property
    def parent(self) -> "Directory | None":
        """The directory this one was reached from, if any."""
        return self._parent

    @property
    def origin(self) -> "Directory":
        """The directory where navigation started (the tree root of IDs)."""
        node = self
        while node._parent is not None:
            node = node._parent
        return node

    @property
    def id(self) -> str:
        """The path from `origin` to this directory, without `archive` folders."""
        return self._id_from(self.origin)

    def _id_from(self, origin: "Directory") -> str:
        relative = self._path.relative_to(origin.path) if self._path != origin.path else Path()
        return "/".join(p for p in relative.parts if p != ARCHIVE)

    @property
    def archived(self) -> bool:
        """Whether this directory lies in an `archive/` folder below its origin."""
        return self._path != self.origin.path and ARCHIVE in self._path.relative_to(self.origin.path).parts

    @property
    def exists(self) -> bool:
        return self._path.is_dir()

    # ----- metadata --------------------------------------------------------------------------

    def _metadata(self) -> dict:
        try:
            mtime = (self._path / META).stat().st_mtime_ns
        except (FileNotFoundError, NotADirectoryError):
            return {}
        cache = self._meta_cache
        if cache is None or cache[0] != mtime:
            cache = (mtime, _read_meta(self._path) or {})
            self.__dict__["_meta_cache"] = cache
        return cache[1]

    @property
    def meta(self) -> dict:
        """The contents of `_meta.yaml` (empty if there is none)."""
        return dict(self._metadata())

    @property
    def aliases(self) -> list[str]:
        """Other IDs that resolve to this directory (old IDs after a move, legacy names)."""
        return list(self._metadata().get("aliases", []))

    @aliases.setter
    def aliases(self, value: list[str]) -> None:
        self._write_meta({"aliases": list(value)})

    def update(self, **fields: Any) -> Self:
        """Set metadata fields, validated against their annotations; returns `self`."""
        self._write_meta(fields)
        return self

    def _write_meta(self, changes: dict, *, create: bool = False) -> None:
        cls = type(self)
        if cls is Directory or cls._is_record:
            raise TypeError(f"{cls.__qualname__} has no metadata fields; declare them on a Directory subclass")
        for name, value in changes.items():
            if name == "aliases":
                if not all(isinstance(a, str) for a in value):
                    raise TypeError("aliases are strings")
                continue
            field = cls._meta_fields.get(name)
            if field is None:
                raise AttributeError(f"{cls.__qualname__} has no metadata field {name!r}; fields: {sorted(cls._meta_fields)}")
            _validate(self, name, field.annotation, value)
        if self._parent is not None:
            self._parent._materialize()
        self._path.mkdir(parents=True, exist_ok=True)
        lock = self._path.with_name(f".{self._path.name}.lock")
        lease = None
        deadline = time.monotonic() + 30
        while lease is None:
            lease = Lease.acquire(lock, ttl=30.0, heartbeat=0)
            if lease is None:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{self!r}: metadata is locked by another process")
                time.sleep(0.01)
        try:
            current = _read_meta(self._path)
            if current is None:
                current = {f: _copy(d.default) for f, d in cls._meta_fields.items()}
            elif create:
                raise FileExistsError(f"{self!r} exists already")
            current.update(changes)
            ordered = {"type": cls.type_id()}
            ordered.update({f: current[f] for f in cls._meta_fields if f in current})
            if current.get("aliases"):
                ordered["aliases"] = current["aliases"]
            import yaml

            with atomic_path(self._path / META) as tmp:
                tmp.write_text(yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True))
            self.__dict__["_meta_cache"] = None
        finally:
            lease.release()

    def _materialize(self) -> None:
        """Create this typed directory and its typed ancestors if they have no `_meta.yaml` yet."""
        cls = type(self)
        if cls is Directory or cls._is_record or (self._path / META).exists():
            return
        self._write_meta({})

    # ----- children --------------------------------------------------------------------------

    def _children_of(self, cls: type["Directory"], *, archived: bool) -> list["Directory"]:
        out = []
        folders = [self._path] + ([self._path / ARCHIVE] if archived else [])
        for folder in folders:
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if path.name.startswith((".", "_")) or path.name == ARCHIVE or not path.is_dir():
                    continue
                meta = _read_meta(path)
                if meta is not None and meta.get("type") == cls.type_id():
                    out.append(cls._at(path, parent=self))
        return sorted(out, key=lambda n: (n.archived, n.name))

    def _child_named(self, name: str) -> "Directory | None":
        for folder in (self._path, self._path / ARCHIVE):
            path = folder / name
            meta = _read_meta(path)
            if meta is None:
                continue
            for cls in type(self)._child_types.values():
                if meta.get("type") == cls.type_id():
                    return cls._at(path, parent=self)
        return None

    def _taken_names(self) -> set[str]:
        """Child names in use: children, archived children, and the last part of aliases that
        name a child of this directory."""
        taken = set()
        for folder in (self._path, self._path / ARCHIVE):
            if folder.is_dir():
                taken.update(p.name for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))
        own = self.id
        for node in self.origin.walk():
            for alias in node.aliases:
                head, _, last = alias.rpartition("/")
                if head == own and head:
                    taken.add(last)
        taken.discard(ARCHIVE)
        return taken

    def walk(self) -> Iterator["Directory"]:
        """Every typed directory below this one, archived ones included, parents first."""
        for cls in type(self)._child_types.values():
            for child in self._children_of(cls, archived=True):
                yield child
                yield from child.walk()

    def resolve(self, id: str) -> "Directory":
        """The directory with `id`, a path of child names relative to this directory (archived
        children included), else the directory listing `id` among its aliases."""
        parts = [p for p in id.split("/") if p]
        node: Directory | None = self
        for part in parts:
            node = node._child_named(part)
            if node is None:
                break
        if node is not None and parts:
            return node
        full = "/".join(p for p in (self.id, id) if p)
        matches = [n for n in self.walk() if id in n.aliases or full in n.aliases]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise KeyError(f"{id!r} is an alias of several directories: {[m.id for m in matches]}")
        raise KeyError(f"{self!r} has no directory {id!r}")

    def move(self, id: str) -> Self:
        """Move this directory to `id` (relative to `origin`, whose parent must exist) and add the
        old ID to its aliases; returns `self`, now at the new path."""
        origin = self.origin
        head, _, name = id.strip("/").rpartition("/")
        parent = origin.resolve(head) if head else origin
        target = parent._path / _check_name(name)
        if target.exists():
            raise FileExistsError(f"{target} exists")
        old = self.id
        os.rename(self._path, target)
        self.__dict__.update(_path=target, _parent=parent, _meta_cache=None)
        if old and old not in self.aliases:
            self.aliases = [*self.aliases, old]
        return self

    def archive(self) -> Self:
        """Move this directory into its parent's `archive/` folder; its ID stays the same."""
        if self._parent is None:
            raise TypeError(f"{self!r} was not reached from a parent; open its parent and navigate to it")
        if self.archived:
            return self
        target = self._parent._path / ARCHIVE / self.name
        target.parent.mkdir(exist_ok=True)
        os.rename(self._path, target)
        self.__dict__.update(_path=target, _meta_cache=None)
        return self

    @contextlib.contextmanager
    def active(self) -> Iterator[Self]:
        """Within the block, records built (in this process and in subprocesses started inside
        it) store this directory as their `producer` in provenance."""
        token = _ACTIVE.set(self)
        previous = os.environ.get("TENSORDIR_PRODUCER")
        os.environ["TENSORDIR_PRODUCER"] = json.dumps({"id": self.id, "path": str(self._path)})
        try:
            yield self
        finally:
            _ACTIVE.reset(token)
            if previous is None:
                os.environ.pop("TENSORDIR_PRODUCER", None)
            else:
                os.environ["TENSORDIR_PRODUCER"] = previous

    # ----- files -----------------------------------------------------------------------------

    def keys(self) -> list[str]:
        """The names of the entries in this directory, without hidden files and `_meta.yaml`."""
        if not self._path.is_dir():
            return []
        return sorted(p.name for p in self._path.iterdir() if not p.name.startswith(".") and p.name != META)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and (self._path / name).exists()

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __truediv__(self, name: str) -> Path:
        return self._path / name

    def __getitem__(self, name: str) -> Any:
        """A subfolder as a directory, a file by its suffix."""
        path = self._path / name
        if path.is_dir() and not name.endswith(".zarr"):
            child = self._child_named(name) if "/" not in name else None
            return child if child is not None else Directory._at(path, parent=self)
        return read_file(path)

    def load(self, name: str) -> Any:
        """The file `name` (a path relative to this directory), read by its suffix: images and
        videos as displayable `Image`/`Video`, `.json`, `.yaml`, `.csv`, `.parquet`, `.npy`, ..."""
        return read_file(self._path / name)

    def save(self, name: str, value: Any) -> Path:
        """Write `value` to the file `name` by its suffix (a matplotlib figure or PIL image as
        `.png`/`.pdf`/`.svg`, a DataFrame as `.csv`/`.parquet`, ...), atomically; returns the path."""
        self._materialize()
        path = self._path / name
        write_file(path, value)
        return path

    # ----- display ---------------------------------------------------------------------------

    def _tree(self, depth: int = 2) -> str:
        lines = [repr(self) if self._parent is None else f"{self!r} {self._path}"]

        def visit(path: Path, prefix: str, level: int) -> None:
            if not path.is_dir() or level >= depth:
                return
            entries = sorted(p for p in path.iterdir() if not p.name.startswith(".") and p.name != META)
            for i, entry in enumerate(entries):
                last = i == len(entries) - 1
                meta = _read_meta(entry) if entry.is_dir() else None
                label = entry.name + ("/" if entry.is_dir() else "")
                if meta and meta.get("type"):
                    label += f"  {meta['type']}"
                lines.append(f"{prefix}{'└── ' if last else '├── '}{label}")
                if entry.is_dir() and not entry.name.endswith(".zarr"):
                    visit(entry, prefix + ("    " if last else "│   "), level + 1)

        visit(self._path, "", 0)
        return "\n".join(lines)

    def tree(self, depth: int = 2) -> None:
        """Print the directory's entries, `depth` levels deep, with the types of typed folders."""
        print(self._tree(depth))

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text(repr(self) if cycle else self._tree())

    def __repr__(self) -> str:
        label = self.id if self._parent is not None else str(self._path)
        return f"<{type(self).__qualname__} {label}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Directory) and type(self) is type(other) and self._path == other._path

    def __hash__(self) -> int:
        return hash((type(self).__qualname__, self._path))

    def __reduce__(self):
        return _reopen_directory, (type(self).type_id(), str(self._path))


def _reopen_directory(type_id: str, path: str) -> Directory:
    cls = _DIRECTORIES.get(type_id, Directory)
    return cls._at(Path(path), parent=None)


Directory._meta_fields = {}
Directory._child_types = {}
