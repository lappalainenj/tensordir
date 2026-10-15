"""Expiring build leases held as directories.

`mkdir` is atomic on local filesystems, NFS and Lustre/GPFS, unlike `flock`, whose behaviour
across nodes depends on mount options. The owner refreshes `owner.json` from a heartbeat thread;
a lease whose heartbeat is older than its TTL is stale and may be broken by anyone.

The lease only prevents duplicate work. Integrity does not depend on it: builds go to a temporary
directory that is renamed into place, so a lost lease can at worst cause a second, discarded build.
"""

import json
import os
import shutil
import socket
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

__all__ = ["Lease", "lease_state", "job_id"]

Clock = Callable[[], float]


def now() -> float:
    """The lease clock (seconds since the epoch). Tests monkeypatch this."""
    return time.time()


def job_id() -> dict | None:
    """The batch job running this process, from Slurm or LSF environment variables."""
    env = os.environ
    if "SLURM_JOB_ID" in env:
        task = env.get("SLURM_ARRAY_TASK_ID")
        return {"scheduler": "slurm", "id": env["SLURM_JOB_ID"] + (f"_{task}" if task else "")}
    if "LSB_JOBID" in env:
        index = env.get("LSB_JOBINDEX", "0")
        return {"scheduler": "lsf", "id": env["LSB_JOBID"] + (f"[{index}]" if index != "0" else "")}
    return None


def _read_owner(lock: Path) -> dict | None:
    try:
        return json.loads((lock / "owner.json").read_text())
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
        return None


def _age(lock: Path, owner: dict | None, clock: Clock) -> float | None:
    """Seconds since the lease was last refreshed; None if the lock is gone."""
    if owner is not None:
        return clock() - float(owner["heartbeat"])
    try:
        return clock() - lock.stat().st_mtime  # between mkdir and the first owner.json write
    except FileNotFoundError:
        return None


def lease_state(lock: Path, ttl: float, clock: Clock | None = None) -> Literal["absent", "fresh", "stale"]:
    """State of the lease at `lock`. The owner's declared TTL takes precedence over `ttl`."""
    clock = clock or now
    owner = _read_owner(lock)
    age = _age(lock, owner, clock)
    if age is None:
        return "absent"
    return "fresh" if age <= _ttl(owner, ttl) else "stale"


def _ttl(owner: dict | None, default: float) -> float:
    return float(owner.get("ttl", default)) if owner else default


def _break_if_stale(lock: Path, ttl: float, clock: Clock) -> bool:
    """Remove `lock` if it is stale. True if the lock is now gone."""
    owner = _read_owner(lock)
    age = _age(lock, owner, clock)
    if age is None:
        return True
    if age <= _ttl(owner, ttl):
        return False
    grave = lock.with_name(f"{lock.name}.broken-{uuid.uuid4().hex}")
    try:
        os.rename(lock, grave)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    victim = _read_owner(grave)
    stale_token = owner.get("token") if owner else None
    if victim is not None and victim.get("token") != stale_token and clock() - float(victim["heartbeat"]) <= _ttl(victim, ttl):
        # another process replaced the stale lease between our check and the rename: give it back
        try:
            os.rename(grave, lock)
            return False
        except OSError:
            pass  # its heartbeat thread will notice the loss
    shutil.rmtree(grave, ignore_errors=True)
    return True


class Lease:
    """An exclusive claim on `lock` (a directory path) that expires `ttl` seconds after its last
    heartbeat. Acquire with `Lease.acquire`; release with `release()` or a `with` block."""

    def __init__(self, lock: Path, ttl: float, heartbeat: float | None, clock: Clock | None):
        self.lock = lock
        self.ttl = ttl
        self.token = uuid.uuid4().hex
        self._clock = clock
        self._interval = heartbeat
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.lost = False

    @property
    def clock(self) -> Clock:
        return self._clock or now

    @classmethod
    def acquire(
        cls, lock: str | os.PathLike, *, ttl: float, heartbeat: float | None = None, clock: Clock | None = None
    ) -> "Lease | None":
        """Try once to take the lease, breaking it first if stale. None if someone else holds it.

        `heartbeat` is the refresh interval in seconds (default `ttl / 4`; 0 disables the thread).
        """
        lock = Path(lock)
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(lock)
        except FileExistsError:
            if not _break_if_stale(lock, ttl, clock or now):
                return None
            try:
                os.mkdir(lock)
            except FileExistsError:
                return None
        lease = cls(lock, ttl, ttl / 4 if heartbeat is None else heartbeat, clock)
        lease._write_owner(acquired=True)
        if lease._interval:
            lease._thread = threading.Thread(target=lease._beat, name=f"tensordir-lease-{lock.name}", daemon=True)
            lease._thread.start()
        return lease

    def _write_owner(self, acquired: bool = False) -> None:
        t = self.clock()
        if acquired:
            self._acquired = t
        data = {
            "token": self.token,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "job": job_id(),
            "acquired": self._acquired,
            "heartbeat": t,
            "ttl": self.ttl,
        }
        tmp = self.lock / f".owner.{self.token}.tmp"
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self.lock / "owner.json")

    @property
    def held(self) -> bool:
        """Whether `owner.json` still names this lease (False once broken or released)."""
        if self.lost:
            return False
        owner = _read_owner(self.lock)
        return owner is not None and owner.get("token") == self.token

    def refresh(self) -> bool:
        """Write a heartbeat now. False (and `lost` set) if the lease was taken away."""
        if not self.held:
            self.lost = True
            return False
        try:
            self._write_owner()
        except OSError:
            self.lost = True
            return False
        return True

    def _beat(self) -> None:
        while not self._stop.wait(self._interval):
            if not self.refresh():
                return

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self.held:
            grave = self.lock.with_name(f"{self.lock.name}.released-{self.token}")
            try:
                os.rename(self.lock, grave)
                shutil.rmtree(grave, ignore_errors=True)
            except OSError:
                pass
        self.lost = True

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *exc) -> None:
        self.release()
