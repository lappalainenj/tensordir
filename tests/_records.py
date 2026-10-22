"""Record classes shared by tests and by subprocesses spawned from them (must be importable)."""

import os
import time
from pathlib import Path
from typing import Annotated

import numpy as np
from jaxtyping import Float

import tensordir as td


class Square(td.Record):
    __record_name__ = "Square"

    values: Float[np.ndarray, "n_points"]

    def __init__(self, /, *, n: int, scale: float = 1.0, device: Annotated[str, td.field(hashed=False)] = "cpu"):
        self.values = np.arange(n, dtype=float) ** 2 * scale
        self["summary"] = {"n": n}


class Counted(td.Record):
    """Appends its pid to a counter file, then sleeps to widen the race window."""

    __record_name__ = "Counted"
    lease_ttl = 30.0

    def __init__(self, /, *, counter: Annotated[str, td.field(hashed=False)], sleep: float = 0.5):
        with open(counter, "a") as f:
            f.write(f"{os.getpid()}\n")
        time.sleep(sleep)
        self["pid"] = os.getpid()


class Hangs(td.Record):
    """Touches a marker, then hangs until killed (or finishes when hang=False)."""

    __record_name__ = "Hangs"
    lease_ttl = 1.0
    heartbeat_interval = 0.2

    def __init__(
        self,
        /,
        *,
        marker: Annotated[str, td.field(hashed=False)],
        hang: Annotated[bool, td.field(hashed=False)] = True,
    ):
        Path(marker).touch()
        while hang:
            time.sleep(0.05)
        self["ok"] = True


class SlowWorkspace(td.Record):
    """A workspace record whose __init__ (the build) touches a marker and sleeps."""

    __record_name__ = "SlowWorkspace"
    mutable = True

    def __init__(
        self,
        /,
        *,
        n: int = 1,
        marker: Annotated[str, td.field(hashed=False)] = "",
        sleep: Annotated[float, td.field(hashed=False)] = 0.0,
    ):
        Path(marker).touch()
        time.sleep(sleep)
        self.pid = os.getpid()


def get_counted(root: str, counter: str) -> str:
    with td.use_root(root):
        return str(Counted(counter=counter).path)


def make_slow_workspace(root: str, marker: str, sleep: float) -> tuple[int, str]:
    with td.use_root(root):
        record = SlowWorkspace(marker=marker, sleep=sleep)
        return record.pid, record.status


def get_hangs(root: str, marker: str) -> None:
    with td.use_root(root):
        Hangs(marker=marker)
