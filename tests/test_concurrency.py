import multiprocessing
import os
import signal
import time

import pytest

import tensordir as td
from tensordir import _lease
from tensordir._lease import Lease, lease_state
from _records import Counted, Hangs, SlowWorkspace, Square, get_counted, get_hangs, make_slow_workspace


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_lease_is_exclusive_until_it_expires(tmp_path):
    clock = FakeClock()
    lock = tmp_path / "locks" / "x"
    first = Lease.acquire(lock, ttl=10, heartbeat=0, clock=clock)
    assert first is not None and first.held
    assert Lease.acquire(lock, ttl=10, heartbeat=0, clock=clock) is None
    assert lease_state(lock, 10, clock) == "fresh"
    clock.t += 9
    assert Lease.acquire(lock, ttl=10, heartbeat=0, clock=clock) is None
    clock.t += 2
    assert lease_state(lock, 10, clock) == "stale"
    second = Lease.acquire(lock, ttl=10, heartbeat=0, clock=clock)
    assert second is not None and second.held and not first.held
    assert not first.refresh() and first.lost
    first.release()
    assert lock.exists() and second.held
    second.release()
    assert lease_state(lock, 10, clock) == "absent" and not lock.exists()
    assert not list(lock.parent.iterdir())


def test_refresh_keeps_lease_fresh(tmp_path):
    clock = FakeClock()
    lease = Lease.acquire(tmp_path / "x", ttl=10, heartbeat=0, clock=clock)
    clock.t += 8
    assert lease.refresh()
    clock.t += 8
    assert lease_state(lease.lock, 10, clock) == "fresh"
    lease.release()


def test_owner_ttl_wins_over_reader_ttl(tmp_path):
    clock = FakeClock()
    lease = Lease.acquire(tmp_path / "x", ttl=100, heartbeat=0, clock=clock)
    clock.t += 50
    assert lease_state(lease.lock, 10, clock) == "fresh"
    assert Lease.acquire(tmp_path / "x", ttl=10, heartbeat=0, clock=clock) is None
    lease.release()


def test_heartbeat_thread(tmp_path):
    lease = Lease.acquire(tmp_path / "x", ttl=0.6, heartbeat=0.1)
    time.sleep(1.2)
    assert lease_state(lease.lock, 0.6) == "fresh"
    lease.release()
    assert lease_state(lease.lock, 0.6) == "absent"


def test_owner_records_job_and_host(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    assert _lease.job_id() == {"scheduler": "slurm", "id": "123_7"}
    monkeypatch.delenv("SLURM_JOB_ID")
    monkeypatch.setenv("LSB_JOBID", "55")
    monkeypatch.setenv("LSB_JOBINDEX", "3")
    assert _lease.job_id() == {"scheduler": "lsf", "id": "55[3]"}


def test_block_false_and_timeout(root):
    rec = Square.locate(n=3)
    held = Lease.acquire(rec._lock, ttl=60, heartbeat=0)
    assert rec.status == "running"
    with pytest.raises(td.RecordBusy):
        Square.get(n=3, block=False)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        Square.get(n=3, timeout=0.3)
    assert time.monotonic() - t0 < 5
    held.release()
    assert Square.get(n=3, block=False).status == "done"


def test_stale_lease_is_broken_with_fake_clock(root, monkeypatch):
    rec = Square.locate(n=2)
    abandoned = Lease.acquire(rec._lock, ttl=Square.lease_ttl, heartbeat=0)
    assert rec.status == "running"
    real = time.time
    monkeypatch.setattr(_lease, "now", lambda: real() + Square.lease_ttl + 1)
    assert rec.status == "stale"
    assert Square.get(n=2).status == "done"
    assert not abandoned.held


def test_processes_racing_build_exactly_once(root, tmp_path):
    counter = tmp_path / "counter.txt"
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(6) as pool:
        paths = pool.starmap(get_counted, [(str(root), str(counter))] * 6)
    assert len(set(paths)) == 1
    assert len(counter.read_text().split()) == 1
    rec = Counted.get(counter=str(counter))
    assert rec.status == "done" and str(rec.path) == paths[0]


def test_sigkill_mid_build_becomes_stale_and_recovers(root, tmp_path):
    marker = tmp_path / "started"
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=get_hangs, args=(str(root), str(marker)))
    proc.start()
    deadline = time.monotonic() + 60
    while not marker.exists():
        assert time.monotonic() < deadline and proc.is_alive()
        time.sleep(0.05)
    rec = Hangs.locate(marker=str(marker))
    assert rec.status == "running"
    os.kill(proc.pid, signal.SIGKILL)
    proc.join()
    deadline = time.monotonic() + 10
    while rec.status != "stale":
        assert time.monotonic() < deadline
        time.sleep(0.1)
    recovered = Hangs.get(marker=str(marker), hang=False)
    assert recovered.status == "done" and recovered.ok is True
    assert recovered.path == rec.path


def test_processes_constructing_a_workspace_record_build_it_once(root, tmp_path):
    marker = tmp_path / "started"
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(3) as pool:
        results = pool.starmap(make_slow_workspace, [(str(root), str(marker), 0.5)] * 3)
    assert {status for _, status in results} == {"done"}
    assert len({pid for pid, _ in results}) == 1  # one process ran __init__; the others waited
    record = SlowWorkspace(marker=str(marker))
    record.note = "edited after the build"
    assert record.status == "done" and record.note == "edited after the build"
