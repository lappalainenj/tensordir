"""Packs: many small records of one type stored as one record.

A record costs at least a directory, `_meta.json` and a zarr store, and opening it costs as many
reads. When records are small and read together, store them in a pack instead:

```python
calls = [Segment.call(seg_id, mesh, lod=0) for seg_id in seg_ids]   # keys only; nothing runs
pack = td.Pack(Segment)(calls)          # builds each member once into one record, or reopens it
member = pack.member(calls[3])           # a read-only Segment, read from the pack's arrays
member.key == Segment.compute_key(seg_ids[3], mesh, lod=0)   # members keep their own keys
```

- Each array field of the members is concatenated along axis 0 into one pack field of the same
  name, with row offsets in `<name>.offsets` (`int64[M + 1]`, members in key order). JSON fields,
  specs, inputs and per-member field info go into the `members` field.
- The pack's key is the digest of its sorted member keys, so the same members give the same pack
  whatever their order, and a different grouping is a different pack.
- A member already built as a standalone record (status `done` under its type's root) is read
  instead of being rebuilt. The others are built one at a time in a temporary directory.
- Members may hold numpy arrays (any dtype, 0-d included) and JSON values; trailing shapes and
  dtypes of a field must agree across members. Other codecs raise.
- `td.Pack(T)` is a record type with the type id `Pack[<T's type id>]`, stored under `T`'s root;
  its `layouts` are `T`'s.
"""

import dataclasses
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

import numpy as np

from ._record import Record, _REGISTRY
from ._signature import transient
from .errors import ReadOnlyError
from .spec import Spec

__all__ = ["Call", "Pack", "PackRecord"]

_ARRAY_CODECS = ("ndarray", "hdf5")
_PACKS: dict[type, type] = {}
_VIEWS: dict[type, type] = {}


@dataclasses.dataclass(frozen=True, eq=False)
class Call:
    """A bound call of a record type: its inputs, spec and key, without building anything.

    Make one with `RecordType.call(*inputs, **settings)`; `td.Pack(RecordType)(calls)` builds
    the calls' records into one pack.
    """

    type: type
    inputs: tuple
    spec: Spec
    transients: Mapping[str, Any]
    key: str

    def settings(self) -> dict[str, Any]:
        """The keyword arguments that reproduce this call's spec."""
        return {name: getattr(self.spec, name) for name in self.type._interface.settings}

    def __tensordir_canonical__(self, *, full: bool = False) -> dict:
        return {"__call__": self.type.type_id(), "key": self.key}

    def __repr__(self) -> str:
        return f"<Call {self.type.__qualname__} {self.key[:16]}>"


def _key_rows(keys: Iterable[str]) -> np.ndarray:
    data = b"".join(bytes.fromhex(k) for k in keys)
    return np.frombuffer(data, dtype=np.uint8).reshape(-1, 32).copy()


class PackRecord(Record):
    """Base of the record types made by `Pack(member_type)`; see `tensordir.pack`."""

    __record_name__ = "PackRecord"
    #: The record type of the members.
    member_type: type[Record]

    def __init__(self, keys: np.ndarray, /, *, calls: Annotated[Sequence[Call] | None, transient] = None):
        cls = type(self)
        if calls is None:
            raise ValueError(f"building {cls.__qualname__} needs its calls: td.Pack({cls.member_type.__qualname__})(calls)")
        by_key = {call.key: call for call in calls}
        hex_keys = [bytes(row).hex() for row in np.asarray(keys, dtype=np.uint8)]
        arrays: dict[str, list[tuple[int, np.ndarray]]] = {}
        table = []
        scratch_root = self._tmp if self._tmp.is_dir() else None
        with tempfile.TemporaryDirectory(prefix=".pack-", dir=scratch_root) as scratch:
            for index, key in enumerate(hex_keys):
                member = _member_record(by_key[key], Path(scratch))
                table.append(_collect(member, index, arrays))
                if member.path.is_relative_to(scratch):
                    _remove(member.path)
        for name, parts in arrays.items():
            values, offsets = _concatenate(name, parts, len(hex_keys))
            self.write(name, values)
            self.write(f"{name}.offsets", offsets)
        self.write("members", table)

    @classmethod
    def _construct(cls, calls: Iterable[Call], /, **options: Any) -> "PackRecord":
        """`Pack(T)(calls, root=..., block=..., timeout=..., retry_failed=...)`."""
        calls = list(calls)
        return cls.get(cls.keys_of(calls), calls=calls, **options)

    @classmethod
    def get(cls, *inputs: Any, root: Any = None, block: bool = True, timeout: float | None = None,
            retry_failed: bool = False, **settings: Any) -> "PackRecord":
        """`Record.get`; without `calls=`, only a finished pack is returned (it cannot be built)."""
        if settings.get("calls") is None:
            record = cls.locate(*inputs, root=root, **settings)
            if record.status == "done":
                return record
            raise ValueError(
                f"{record!r} is {record.status}; building it needs its calls: td.Pack({cls.member_type.__qualname__})(calls)"
            )
        return super().get(*inputs, root=root, block=block, timeout=timeout, retry_failed=retry_failed, **settings)

    @classmethod
    def keys_of(cls, calls: Iterable[Call]) -> np.ndarray:
        """The pack input for `calls`: their sorted, distinct keys as a `uint8[M, 32]` array."""
        keys = set()
        for call in calls:
            if not isinstance(call, Call) or call.type.type_id() != cls.member_type.type_id():
                raise TypeError(f"{cls.__qualname__} packs calls of {cls.member_type.__qualname__}, got {call!r}")
            keys.add(call.key)
        return _key_rows(sorted(keys))

    # ----- members -----------------------------------------------------------------------

    def _table(self) -> list[dict]:
        cached = self.__dict__.get("_members_cache")
        if cached is None:
            table = self.load("members")
            cached = self.__dict__["_members_cache"] = (table, {entry["key"]: i for i, entry in enumerate(table)})
        return cached[0]

    def member_keys(self) -> list[str]:
        """The members' keys, sorted."""
        return [entry["key"] for entry in self._table()]

    def member(self, key: "str | Call") -> Record:
        """The member with this key (or of this call) as a read-only record of the member type."""
        key = key.key if isinstance(key, Call) else key
        self._table()
        index = self.__dict__["_members_cache"][1].get(key)
        if index is None:
            raise KeyError(f"{self!r} has no member {key[:16]}")
        return _view_type(type(self).member_type)._of(self, index)

    def __contains__(self, name: object) -> bool:
        if isinstance(name, Call):
            name = name.key
        if isinstance(name, str) and len(name) == 64 and name not in self._manifest():
            self._table()
            return name in self.__dict__["_members_cache"][1]
        return super().__contains__(name)

    @property
    def members(self):
        """Every member as a read-only record, in key order, as a `Collection`."""
        from .collection import Collection

        view = _view_type(type(self).member_type)
        return Collection([view._of(self, i) for i in range(len(self._table()))])


def _member_record(call: Call, scratch: Path) -> Record:
    """The finished record of `call`: standalone if it exists, else built in `scratch`."""
    member_type = call.type
    base = member_type.resolve_root()
    standalone = member_type._new(base / member_type.dirname() / call.key[:16], key=call.key, managed_root=base)
    if standalone.status == "done":
        return standalone
    try:
        record = member_type.at(scratch / call.key[:16], *call.inputs, **call.settings(), **call.transients)
    except Exception as exc:
        raise RuntimeError(f"member {call.key[:16]} of {member_type.__qualname__} failed: {exc}") from exc
    if record.key != call.key:
        raise RuntimeError(f"member {call.key[:16]} was built with key {record.key[:16]}")
    return record


def _collect(member: Record, index: int, arrays: dict[str, list]) -> dict:
    """One member's table entry; its arrays are added to `arrays` as `(index, rows)`."""
    meta = member.meta
    entry = {
        "key": member.key,
        "spec": meta["spec"],
        "inputs": meta["inputs"],
        "upstream": meta.get("upstream", {}),
        "duration_s": meta.get("duration_s"),
        "fields": {},
        "values": {},
    }
    for name, info in meta["fields"].items():
        if name == "members" or name.endswith(".offsets"):
            raise ValueError(f"member field {name!r} clashes with the pack's own fields")
        if info["codec"] in _ARRAY_CODECS:
            value = np.asarray(member.load(name))
            entry["fields"][name] = {
                "codec": "ndarray",
                "dims": info.get("dims"),
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "packed": True,
            }
            arrays.setdefault(name, []).append((index, value.reshape(1) if value.ndim == 0 else value))
        elif info["codec"] == "json":
            entry["fields"][name] = {"codec": "json", "packed": True}
            entry["values"][name] = member.load(name)
        else:
            raise TypeError(
                f"member {member.key[:16]} field {name!r} uses codec {info['codec']!r}; packs hold arrays and JSON values"
            )
    return entry


def _concatenate(name: str, parts: list[tuple[int, np.ndarray]], n_members: int) -> tuple[np.ndarray, np.ndarray]:
    first = parts[0][1]
    for index, rows in parts:
        if rows.shape[1:] != first.shape[1:] or rows.dtype != first.dtype:
            raise TypeError(
                f"field {name!r}: members differ in trailing shape or dtype "
                f"({first.dtype} {first.shape[1:]} vs {rows.dtype} {rows.shape[1:]})"
            )
    counts = np.zeros(n_members, dtype=np.int64)
    for index, rows in parts:
        counts[index] = len(rows)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return np.concatenate([rows for _, rows in parts]), offsets


def _remove(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def Pack(member_type: type[Record]) -> type[PackRecord]:
    """The pack record type of `member_type` (made once per type); see `tensordir.pack`."""
    if not (isinstance(member_type, type) and issubclass(member_type, Record)) or issubclass(member_type, PackRecord):
        raise TypeError(f"Pack takes a record type, got {member_type!r}")
    if member_type.__dict__.get("_tensordir_view"):
        member_type = member_type.__mro__[2]
    pack = _PACKS.get(member_type)
    if pack is None:
        name = f"Pack[{member_type.type_id()}]"
        pack = type(PackRecord)(
            name,
            (PackRecord,),
            {
                "__record_name__": name,
                "__module__": member_type.__module__,
                "__qualname__": name,
                "member_type": member_type,
                "layouts": dict(member_type.layouts),
                "root": member_type.root,
            },
        )
        _PACKS[member_type] = pack
    return pack


def pack_type_for(type_id: str) -> type[PackRecord] | None:
    """`Pack(T)` for the type id `Pack[<T's type id>]` if `T` is registered, else None."""
    if not (type_id.startswith("Pack[") and type_id.endswith("]")):
        return None
    member_type = _REGISTRY.get(type_id[5:-1])
    return Pack(member_type) if member_type is not None else None


# ----- member views ------------------------------------------------------------------------


class _MemberView:
    """Read side of a record whose fields live in a pack; mixed in before the member type."""

    _tensordir_view = True

    @classmethod
    def _of(cls, pack: PackRecord, index: int) -> Record:
        entry = pack._table()[index]
        self = cls._new(pack.path, key=entry["key"], managed_root=pack._root)
        self.__dict__.update(_pack=pack, _index=index, _entry=entry)
        return self

    def _disk_meta(self) -> dict:
        pack, entry = self.__dict__["_pack"], self.__dict__["_entry"]
        meta = pack.meta
        return {
            "tensordir": meta.get("tensordir"),
            "type": type(self).type_id(),
            "schema_version": type(self).schema_version,
            "key": entry["key"],
            "spec": entry["spec"],
            "inputs": entry["inputs"],
            "upstream": entry["upstream"],
            "status": "done",
            "fields": entry["fields"],
            "created": meta.get("created"),
            "started": meta.get("started"),
            "finished": meta.get("finished"),
            "duration_s": entry["duration_s"],
            "modified": None,
            "error": None,
            "pack": {"type": type(pack).type_id(), "key": pack.key, "path": str(pack.path)},
        }

    @property
    def status(self) -> str:
        return "done"

    @property
    def pack(self) -> PackRecord:
        """The pack holding this member."""
        return self.__dict__["_pack"]

    def _rows(self, name: str) -> tuple[int, int]:
        offsets = self.__dict__["_pack"]._zarr_array(f"{name}.offsets")
        index = self.__dict__["_index"]
        a, b = offsets[index : index + 2].tolist()
        return a, b

    def _read(self, name: str, lazy: bool) -> Any:
        info = self._manifest().get(name)
        if info is None:
            raise KeyError(f"{self!r} has no field {name!r}; fields: {self.keys()}")
        if info["codec"] == "json":
            return self.__dict__["_entry"]["values"][name]
        a, b = self._rows(name)
        rows = self.__dict__["_pack"]._zarr_array(name)[a:b]
        return rows.reshape(()) if info["shape"] == [] else rows

    def read(self, request: Mapping[str, Any], *, engine: str | None = None) -> dict[str, np.ndarray]:
        """`Record.read` on the member's rows of the pack's arrays."""
        from ._read import _parse

        shifted = {}
        for name, value in request.items():
            info = self._manifest().get(name)
            if info is None:
                raise KeyError(f"{self!r} has no field {name!r}; fields: {self.keys()}")
            if info["codec"] != "ndarray" or info["shape"] == []:
                raise TypeError(f"{name}: read() needs an array field with rows")
            a, b = self._rows(name)
            kind, first, second = _parse(value, b - a)
            if kind == "ranges":
                shifted[name] = np.stack([first + a, second + a], axis=1) if first.size else []
            else:
                shifted[name] = first + a
        return self.__dict__["_pack"].read(shifted, engine=engine)

    def _refuse(self, *args: Any, **kwargs: Any) -> None:
        raise ReadOnlyError(f"{self!r} is a member of {self.__dict__['_pack']!r} and cannot change")

    write = append = delete_field = save = delete = _refuse

    def _edit(self):
        self._refuse()

    def publish(self, root: Any, *, upstream: bool = False) -> Any:
        raise TypeError(f"{self!r} lives in a pack; publish the pack ({self.__dict__['_pack']!r})")

    def __repr__(self) -> str:
        return f"<{type(self).__qualname__} {self.key[:16]} in {self.__dict__['_pack']!r}>"

    def __reduce__(self):
        pack = self.__dict__["_pack"]
        return _reopen_member, (type(pack).type_id(), pack.key, str(pack.path), self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _MemberView) and (self.key, self.path) == (other.key, other.path)

    def __hash__(self) -> int:
        return hash((self.key, self.path))


def _view_type(member_type: type[Record]) -> type:
    view = _VIEWS.get(member_type)
    if view is None:
        view = type(member_type)(
            member_type.__qualname__,
            (_MemberView, member_type),
            {"__module__": member_type.__module__, "__qualname__": member_type.__qualname__, "_tensordir_view": True},
        )
        _VIEWS[member_type] = view
    return view


def _reopen_member(pack_type_id: str, pack_key: str, pack_path: str, key: str) -> Record:
    from ._record import _reopen

    return _reopen(pack_type_id, pack_key, pack_path).member(key)
