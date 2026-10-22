"""`<root>/.tensordir/index.parquet`: one row per record, rebuilt from the `_meta.json` files.

A type directory's rows are trusted only while the directory's mtime equals the mtime recorded at
reindex time. Records enter and leave a type directory by rename, which updates that mtime, so a
stale index falls back to scanning. Rows of records that were not `done` are re-read from disk.
"""

import json
import os
from collections.abc import Iterator
from pathlib import Path

from ._root import get_root
from .codecs import atomic_path
from ._record import INTERNAL, META, _read_json

__all__ = ["reindex", "records", "index_path"]

COLUMNS = ["dir", "name", "type", "key", "status", "created", "finished", "dir_mtime_ns", "meta"]


def index_path(root: Path) -> Path:
    return root / INTERNAL / "index.parquet"


def _type_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def _scan(type_dir: Path) -> Iterator[tuple[Path, dict]]:
    for path in sorted(type_dir.iterdir()):
        if path.is_dir() and not path.name.startswith("."):
            meta = _read_json(path / META)
            if meta is not None:
                yield path, meta


def reindex(root: str | os.PathLike | type | None = None) -> Path:
    """Rebuild the index of all records under `root` (needs pandas and pyarrow).

    `root` may be a record class (`get_root(cls)` is reindexed) or a path, which is joined onto
    the base root like a `root=` argument; `None` reindexes the base root.
    """
    import pandas as pd

    root = get_root(root) if isinstance(root, type) else get_root(root=root)
    rows = []
    for type_dir in _type_dirs(root):
        mtime = type_dir.stat().st_mtime_ns
        for path, meta in _scan(type_dir):
            rows.append(
                {
                    "dir": type_dir.name,
                    "name": path.name,
                    "type": meta.get("type"),
                    "key": meta.get("key"),
                    "status": meta.get("status"),
                    "created": meta.get("created"),
                    "finished": meta.get("finished"),
                    "dir_mtime_ns": mtime,
                    "meta": json.dumps(meta),
                }
            )
    target = index_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(target) as tmp:
        pd.DataFrame(rows, columns=COLUMNS).to_parquet(tmp)
    return target


def records(root: Path, dirname: str | None = None) -> list[tuple[Path, dict]]:
    """`(path, meta)` of every record under `root` (or in one type directory), via the index when fresh."""
    type_dirs = _type_dirs(root) if dirname is None else [root / dirname]
    rows_by_dir = _index_rows(root)
    out: list[tuple[Path, dict]] = []
    for type_dir in type_dirs:
        if not type_dir.is_dir():
            continue
        rows = rows_by_dir.get(type_dir.name)
        if rows and rows[0]["dir_mtime_ns"] == type_dir.stat().st_mtime_ns:
            for row in rows:
                path, meta = type_dir / row["name"], json.loads(row["meta"])
                if meta.get("status") != "done":
                    meta = _read_json(path / META)
                if meta is not None:
                    out.append((path, meta))
        else:
            out.extend(_scan(type_dir))
    return out


def _index_rows(root: Path) -> dict[str, list[dict]]:
    path = index_path(root)
    if not path.exists():
        return {}
    try:
        import pandas as pd
    except ImportError:
        return {}
    grouped: dict[str, list[dict]] = {}
    for row in pd.read_parquet(path).to_dict("records"):
        grouped.setdefault(row["dir"], []).append(row)
    return grouped
