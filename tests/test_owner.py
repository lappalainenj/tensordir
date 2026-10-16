"""`@td.record`: regular classes owning a workspace record keyed by their `__init__` arguments."""

import multiprocessing
import pickle
import time
from typing import Annotated, Any

import numpy as np
import pytest
from jaxtyping import Float

import tensordir as td
from tensordir import _record
from _records import SlowOwner, make_slow_owner


@td.record
class Trainer:
    def __init__(
        self,
        lr: float = 0.1,
        layers: list[int] = [8, 8],  # noqa: B006 - copied per spec by the decorator
        device: Annotated[str, td.field(hashed=False)] = "cpu",
        seed: int = td.field(0, since=1),
        logger: Annotated[Any, td.transient] = None,
    ):
        self.logger = logger
        self.steps = 0

    def step(self, loss: float) -> None:
        self.steps += 1
        self.record.append("loss", [loss])


class LossLog(td.Record):
    losses: Float[np.ndarray, "n_steps"]


@td.record(name="TypedRun", base=LossLog)
class TypedRun:
    def __init__(self, n_steps: int):
        self.n_steps = n_steps  # same value: the spec does not change


def test_spec_from_init_arguments(root):
    trainer = Trainer(lr=1, logger=print)
    assert trainer.spec == Trainer.Spec(lr=1.0, layers=[8, 8], device="cpu", seed=0)
    assert isinstance(trainer.lr, float) and trainer.logger is print and trainer.steps == 0
    assert set(Trainer.Spec.__dataclass_fields__) == {"lr", "layers", "device", "seed"}
    assert Trainer.Spec().hashed_dict() == {"layers": [8, 8], "lr": 0.1}  # device unhashed, seed since=1
    assert Trainer.Record.type_id() == "Trainer" and Trainer.Record.mutable
    with pytest.raises(td.CanonicalizationError):
        Trainer(layers=object())  # spec values must have a canonical form


def test_record_is_built_by_init_and_reopens(root):
    trainer = Trainer()
    record = trainer.record
    assert record.status == "done" and record.keys() == []  # built by __init__, nothing written
    assert Trainer.Record.lookup(trainer.spec) == record
    trainer.step(3.0)
    trainer.step(2.0)
    assert record.status == "done" and record.spec == trainer.spec
    np.testing.assert_array_equal(record.load("loss"), [3.0, 2.0])
    again = Trainer(device="cuda")  # unhashed: same key
    assert again.record.key == record.key
    np.testing.assert_array_equal(again.record.load("loss"), [3.0, 2.0])


def test_init_is_the_build(root, monkeypatch):
    clock = iter(f"2026-01-01T00:00:{s:02d}+00:00" for s in range(60))
    monkeypatch.setattr(_record, "_now", lambda: next(clock))
    seen = []
    plan = {"fail": True}

    @td.record
    class Probe:
        def __init__(self, x: int = 0):
            other = Probe.Record.locate(self.spec)  # a second handle, as another process sees it
            seen.append((other.status, self.record.status, "x" in self.record))
            if "x" not in self.record:
                self.record.x = x  # inside the build: no `modified`
            if plan["fail"]:
                raise ValueError("bad x")

    with pytest.raises(ValueError, match="bad x"):  # the caller's exception, not BuildFailed
        Probe()
    failed = Probe.Record.locate(Probe.Spec())
    assert failed.status == "failed" and "bad x" in failed.error["traceback"]
    plan["fail"] = False
    probe = Probe()  # retried
    record = probe.record
    assert seen == [("running", "running", False), ("running", "running", False)]
    assert record.status == "done" and record.x == 0 and record.meta["modified"] is None
    record.y = 1  # after __init__: an edit, status unchanged
    first = record.meta["modified"]
    record.append("loss", [1.0])
    assert record.status == "done" and first is not None and record.meta["modified"] > first
    second = Probe()  # done: __init__ reruns without the lease, on the same record
    assert seen[-1] == ("done", "done", True) and second.record == record


def test_concurrent_construction_waits_for_the_build(root, tmp_path):
    marker = tmp_path / "started"
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(2) as pool:
        first = pool.apply_async(make_slow_owner, (str(root), str(marker), 3.0))
        deadline = time.monotonic() + 60
        while not marker.exists():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert SlowOwner.Record.locate(SlowOwner.Spec()).status == "running"
        second = pool.apply_async(make_slow_owner, (str(root), str(tmp_path / "unused"), 0.0))
        (resumed_a, pid_a, status_a), (resumed_b, pid_b, status_b) = first.get(60), second.get(60)
    assert (resumed_a, resumed_b) == (False, True)  # the second waited, then reopened the build
    assert pid_a == pid_b and status_a == status_b == "done"
    assert not (tmp_path / "unused").exists()


def test_changing_spec_attributes_switches_records(root):
    trainer = Trainer()
    first = trainer.record
    trainer.record.note = "first"
    trainer.device = "cuda"  # unhashed: same record, spec updated
    assert trainer.record is first and trainer.record.spec.device == "cuda"
    trainer.lr = 0.2
    second = trainer.record
    assert second.key != first.key and second.spec.lr == 0.2
    second.note = "second"
    assert first.note == "first" and second.note == "second"
    trainer.layers = [4]
    assert trainer.record.spec.layers == [4] and trainer.record.key not in (first.key, second.key)
    trainer.lr, trainer.layers = 0.1, [8, 8]
    assert trainer.record.key == first.key  # back to the first record
    assert len(Trainer.Record.query()) == 3
    assert sorted(r.spec.layers[0] for r in Trainer.Record.query(lr=0.2)) == [4, 8]


def test_base_record_types_fields(root):
    run = TypedRun(4)
    run.record.losses = np.linspace(1, 0, 4)
    assert isinstance(run.record.losses, np.ndarray)
    assert run.record.meta["fields"]["losses"]["dims"] == ["step"]
    assert isinstance(td.Record.at(run.record.path), TypedRun.Record)
    with pytest.raises(TypeError):
        run.record.losses = [1, 2]  # the base annotation is checked


def test_root_options(tmp_path):
    @td.record(root=tmp_path / "a")
    class A:
        def __init__(self, x: int = 0):
            pass

    @td.root(tmp_path / "b")
    @td.record
    class B:
        def __init__(self, x: int = 0):
            pass

    assert A().record.path.parents[1] == tmp_path / "a"
    assert B().record.path.parents[1] == tmp_path / "b"


def test_pickle_keeps_the_spec(root):
    trainer = Trainer(lr=0.3)
    trainer.record.note = "x"
    clone = pickle.loads(pickle.dumps(trainer))
    assert clone.spec == trainer.spec and clone.record.key == trainer.record.key


def test_subclass_with_own_decorated_init(root):
    @td.record
    class Base:
        def __init__(self, a: int = 1):
            self.extra = "base"

    @td.record
    class Child(Base):
        def __init__(self, a: int = 1, b: int = 2):
            super().__init__(a)

    child = Child(b=3)
    assert child.spec == Child.Spec(a=1, b=3) and child.extra == "base"
    assert child.record.type_id() == "test_subclass_with_own_decorated_init.<locals>.Child"


def test_invalid_classes():
    with pytest.raises(TypeError, match=r"\*args"):
        @td.record
        class Star:
            def __init__(self, *args):
                pass

    with pytest.raises(TypeError, match="reserved"):
        @td.record
        class Reserved:
            def __init__(self, record: int = 0):
                pass

    with pytest.raises(TypeError, match="already exists"):
        @td.record
        class Clash:
            lr = 0.1

            def __init__(self, lr: float = 0.1):
                pass

    with pytest.raises(TypeError, match="__init__"):
        @td.record
        class NoInit:
            pass
