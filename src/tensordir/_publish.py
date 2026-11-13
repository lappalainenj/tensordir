"""`Record.publish`: copy finished records to another root with sequential, write-once files.

Object-store mounts such as mountpoint-s3 accept each file as one sequential write and support
neither rename nor overwrite, so records cannot be built there. Build in a local root and
publish: every file is streamed once, data first and `_meta.json` last, so a reader of the
destination sees the record as `absent` until it is complete.
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .errors import KeyCollision, ReadOnlyError

__all__ = ["publish", "PUBLISHED", "check_buildable"]

#: Marker in `<root>/.tensordir/` of a root created by `publish`: records there are only read.
PUBLISHED = "published"
_TEMPORARY = re.compile(r"^\..*\.(tmp|trash)-[0-9a-f]+$")
_BUFFER = 8 << 20


def check_buildable(root: Path | None) -> None:
    """Raise `ReadOnlyError` if `root` was created by `publish` (builds need rename and leases)."""
    from ._record import INTERNAL

    if root is not None and (root / INTERNAL / PUBLISHED).exists():
        raise ReadOnlyError(
            f"{root} is a publish target (it holds {INTERNAL}/{PUBLISHED}); build in a local root "
            "and publish the finished records here"
        )


def publish(record: Any, root: str | os.PathLike, *, upstream: bool = False) -> Any:
    """Copy `record` (status `done`) to `root`; with `upstream`, also every record it was built
    from (`lineage()`), upstream first. Returns the record in `root`. See `Record.publish`."""
    from ._record import INTERNAL, Record

    root = Path(root).expanduser().absolute()
    if record.status != "done":
        raise ValueError(f"cannot publish {record!r}: only done records are published")
    if not (root / INTERNAL).is_dir():
        root.mkdir(parents=True, exist_ok=True)
        _write_once(root / INTERNAL / PUBLISHED, json.dumps({"tensordir": 1}).encode())
    for source in [*record.lineage(), record] if upstream else [record]:
        _publish_one(source, root)
    return Record.at(_destination(record, root))


def _destination(record: Any, root: Path) -> Path:
    from ._record import _safe

    return root / _safe(record.meta["type"]) / record.key[:16]


def _publish_one(record: Any, root: Path) -> None:
    from ._record import META, _read_json

    if record.status != "done":
        raise ValueError(f"cannot publish {record!r}: it is {record.status}")
    destination = _destination(record, root)
    if destination.resolve() == record.path.resolve():
        return
    existing = _read_json(destination / META)
    if existing is not None:
        if existing.get("key") != record.key:
            raise KeyCollision(f"{destination} holds key {existing.get('key')}, not {record.key}")
        if existing.get("status") == "done" and not type(record).mutable:
            return  # same key, same contents
    lease = record._edit_lease() if type(record).mutable else None
    try:
        meta = (record.path / META).read_bytes()  # under the lease for a workspace record
        if existing is not None or destination.exists():
            _clear(destination)  # a partial or older copy; files are removed, not overwritten
        for path in _files(record.path):
            _copy(path, destination / path.relative_to(record.path))
        _write_once(destination / META, meta)
    finally:
        if lease is not None:
            lease.release()


def _files(directory: Path) -> list[Path]:
    """The record's files except `_meta.json` and leftovers of interrupted atomic writes."""
    from ._record import META

    out = []
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = sorted(d for d in dirnames if not _TEMPORARY.match(d))
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if not _TEMPORARY.match(name) and not (Path(dirpath) == directory and name == META):
                out.append(path)
    return out


def _clear(directory: Path) -> None:
    """Remove a directory's files, `_meta.json` first, so a half-removed copy reads as absent."""
    from ._record import META

    (directory / META).unlink(missing_ok=True)
    for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
        for name in filenames:
            os.unlink(os.path.join(dirpath, name))
        for name in dirnames:
            try:
                os.rmdir(os.path.join(dirpath, name))
            except OSError:
                pass


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(source, "rb") as src, open(target, "xb") as dst:
        shutil.copyfileobj(src, dst, _BUFFER)


def _write_once(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "xb") as f:
        f.write(data)
