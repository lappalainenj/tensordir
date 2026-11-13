"""Command line: `tensordir [--root R] ls|status|reindex|gc|publish`."""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from . import _lease, index
from ._root import get_root
from ._record import INTERNAL

__all__ = ["main"]

DEFAULT_TTL = 300.0


def _lock(root: Path, path: Path) -> Path:
    return root / INTERNAL / "locks" / path.parent.name / path.name


def _status(root: Path, path: Path, meta: dict) -> str:
    if meta.get("status") == "done":
        return "done"
    fresh = _lease.lease_state(_lock(root, path), DEFAULT_TTL) == "fresh"
    if meta.get("status") == "failed":
        return "running" if fresh else "failed"
    return "running" if fresh else "stale"


def _ls(root: Path, args: argparse.Namespace) -> int:
    for path, meta in index.records(root, args.type):
        status = _status(root, path, meta)
        if args.status and status != args.status:
            continue
        spec = json.dumps(meta.get("spec", {}), separators=(",", ":"))
        if len(spec) > 80:
            spec = spec[:77] + "..."
        print(f"{path.parent.name}/{path.name}  {status:<7}  {meta.get('finished') or '-':<32}  {spec}")
    return 0


def _summary(root: Path, args: argparse.Namespace) -> int:
    counts = Counter((path.parent.name, _status(root, path, meta)) for path, meta in index.records(root))
    for (dirname, status), n in sorted(counts.items()):
        print(f"{dirname:<30} {status:<8} {n}")
    return 0


def _reindex(root: Path, args: argparse.Namespace) -> int:
    print(index.reindex(root))
    return 0


def _remove(path: Path, dry_run: bool) -> None:
    print(f"{'would remove' if dry_run else 'removed'} {path}")
    if not dry_run:
        shutil.rmtree(path, ignore_errors=True)


def _gc(root: Path, args: argparse.Namespace) -> int:
    """Remove stale leases, temporary build directories without a live lease, and trash."""
    internal = root / INTERNAL
    locks = internal / "locks"
    for lock in sorted(locks.glob("*/*")) if locks.is_dir() else []:
        if ".broken-" in lock.name or ".released-" in lock.name:
            _remove(lock, args.dry_run)
        elif _lease.lease_state(lock, DEFAULT_TTL) == "stale":
            _remove(lock, args.dry_run)
    tmp = internal / "tmp"
    for build in sorted(tmp.iterdir()) if tmp.is_dir() else []:
        parts = build.name[1:].rsplit("-", 2)
        if len(parts) != 3:
            continue
        dirname, key16, _ = parts
        if _lease.lease_state(locks / dirname / key16, DEFAULT_TTL) != "fresh":
            _remove(build, args.dry_run)
    for type_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")) if root.is_dir() else []:
        for trash in sorted(type_dir.glob(".*.trash-*")):
            _remove(trash, args.dry_run)
    return 0


def _publish(root: Path, args: argparse.Namespace) -> int:
    """Copy every done record (of one type directory, if given) to the destination root."""
    from ._record import Record

    destination = Path(args.destination).expanduser().absolute()
    count = 0
    for path, meta in index.records(root, args.type):
        if meta.get("status") != "done":
            continue
        Record.at(path).publish(destination, upstream=args.upstream)
        count += 1
        print(f"published {path.parent.name}/{path.name}")
    print(f"{count} records published to {destination}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tensordir", description="Inspect and maintain a tensordir root.")
    parser.add_argument(
        "--root",
        help="storage root, joined onto the base root ($TENSORDIR_ROOT, else the working directory); "
        "absolute paths are used as given (default: the base root)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ls = commands.add_parser("ls", help="list records, optionally of one record type directory")
    ls.add_argument("type", nargs="?", help="record type directory name, e.g. Fit")
    ls.add_argument("--status", choices=["running", "done", "failed", "stale"])
    ls.set_defaults(run=_ls)
    commands.add_parser("status", help="count records by type and status").set_defaults(run=_summary)
    commands.add_parser("reindex", help="rebuild <root>/.tensordir/index.parquet").set_defaults(run=_reindex)
    gc = commands.add_parser("gc", help="remove stale leases, orphaned temporary builds and trash")
    gc.add_argument("--dry-run", action="store_true")
    gc.set_defaults(run=_gc)
    publish = commands.add_parser(
        "publish", help="copy done records to another root with write-once files (e.g. a mountpoint-s3 mount)"
    )
    publish.add_argument("destination", help="destination root")
    publish.add_argument("type", nargs="?", help="record type directory name, e.g. Fit")
    publish.add_argument("--upstream", action="store_true", help="also publish the records each was built from")
    publish.set_defaults(run=_publish)
    args = parser.parse_args(argv)
    return args.run(get_root(root=args.root), args)


if __name__ == "__main__":
    sys.exit(main())
