import time

import pytest

import tensordir as td
from tensordir import _lease
from tensordir._lease import Lease
from tensordir.cli import main
from _records import Square


class Broken(td.Record):
    __record_name__ = "Broken"

    def __init__(self):
        raise RuntimeError("no")


def test_ls_status_reindex(root, capsys):
    Square.get(n=2)
    Square.get(n=3)
    with pytest.raises(td.BuildFailed):
        Broken.get()
    assert main(["ls"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and sum("failed" in line for line in lines) == 1
    main(["ls", "Square", "--status", "done"])
    assert len(capsys.readouterr().out.splitlines()) == 2
    main(["status"])
    out = capsys.readouterr().out
    assert "Square" in out and "done" in out and "Broken" in out
    main(["--root", str(root), "reindex"])
    assert capsys.readouterr().out.strip().endswith("index.parquet")


def test_gc_removes_stale_leases_and_orphaned_builds(root, capsys, monkeypatch):
    rec = Square.locate(n=4)
    lease = Lease.acquire(rec._lock, ttl=60, heartbeat=0)
    orphan = root / ".tensordir" / "tmp" / f".Square-{rec.path.name}-deadbeef"
    orphan.mkdir(parents=True)
    trash = root / "Square" / ".abc.trash-1"
    trash.mkdir(parents=True)
    main(["gc"])
    out = capsys.readouterr().out
    assert "deadbeef" not in out and orphan.exists() and rec._lock.exists()
    assert not trash.exists()
    real = time.time
    monkeypatch.setattr(_lease, "now", lambda: real() + 3600)
    main(["gc", "--dry-run"])
    assert "would remove" in capsys.readouterr().out and orphan.exists()
    main(["gc"])
    assert not orphan.exists() and not rec._lock.exists()
    assert not lease.held
