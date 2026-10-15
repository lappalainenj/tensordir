"""Record classes shared by tests and by subprocesses spawned from them (must be importable)."""

import os
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float

import tensordir as td


class Square(td.Record):
    __record_name__ = "Square"

    class Spec(td.Spec):
        n: int
        scale: float = 1.0
        device: str = td.field("cpu", hashed=False)

    values: Float[np.ndarray, "n_points"]

    def build(self, spec):
        self.values = np.arange(spec.n, dtype=float) ** 2 * spec.scale
        self["summary"] = {"n": spec.n}


class Counted(td.Record):
    """Appends its pid to a counter file, then sleeps to widen the race window."""

    __record_name__ = "Counted"
    lease_ttl = 30.0

    class Spec(td.Spec):
        counter: str = td.field(hashed=False)
        sleep: float = 0.5

    def build(self, spec):
        with open(spec.counter, "a") as f:
            f.write(f"{os.getpid()}\n")
        time.sleep(spec.sleep)
        self["pid"] = os.getpid()


class Hangs(td.Record):
    """Touches a marker, then hangs until killed (or finishes when hang=False)."""

    __record_name__ = "Hangs"
    lease_ttl = 1.0
    heartbeat_interval = 0.2

    class Spec(td.Spec):
        marker: str = td.field(hashed=False)
        hang: bool = td.field(True, hashed=False)

    def build(self, spec):
        Path(spec.marker).touch()
        while spec.hang:
            time.sleep(0.05)
        self["ok"] = True


def get_counted(root: str, counter: str) -> str:
    with td.use_root(root):
        return str(Counted.get({"counter": counter}).path)


def get_hangs(root: str, marker: str) -> None:
    with td.use_root(root):
        Hangs.get({"marker": marker})
