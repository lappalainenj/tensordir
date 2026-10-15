"""Attribute-style writes, field deletion and workspace (mutable) records."""

import threading

import numpy as np
import pytest
from jaxtyping import Float

import tensordir as td
from _records import Square


class Run(td.Record):
    """A workspace record without a build: a training loop fills it from outside."""

    __record_name__ = "WorkspaceRun"
    mutable = True

    class Spec(td.Spec):
        lr: float = 0.1

    losses: Float[np.ndarray, "n_steps"]


class Trainer:
    def __init__(self, lr: float):
        self.record = Run.get({"lr": lr})

    def step(self, losses):
        self.record.losses = np.asarray(losses)


def test_immutable_records_reject_attribute_writes(root):
    record = Square.get({"n": 3})
    with pytest.raises(td.ReadOnlyError, match="mutable = True"):
        record.extra = 1  # used to set a plain Python attribute silently
    with pytest.raises(td.ReadOnlyError):
        del record.values
    with pytest.raises(td.ReadOnlyError):
        del record["summary"]
    assert "extra" not in record and "values" in record
    with pytest.raises(AttributeError):
        record.path = root  # properties stay read-only


def test_build_stores_untyped_attributes_and_deletes(root):
    class Scratch(td.Record):
        __record_name__ = "WorkspaceScratch"

        def build(self, spec):
            self.notes = {"a": 1}  # no annotation: still a field
            self.tmp = np.zeros(3)
            del self.tmp

    record = Scratch.get()
    assert record.notes == {"a": 1} and record.keys() == ["notes"]


def test_trainer_writes_through_held_record(root):
    trainer = Trainer(lr=0.1)
    record = trainer.record
    assert record.status == "done" and record.keys() == []
    meta_before = record.meta
    trainer.step([3.0, 2.0])
    np.testing.assert_array_equal(record.losses, [3.0, 2.0])  # typed numpy field reads as ndarray
    other = td.Record.at(record.path)  # another handle sees the edit
    trainer.step([3.0, 2.0, 1.0])
    np.testing.assert_array_equal(other.losses, [3.0, 2.0, 1.0])
    assert other.meta["modified"] is not None and meta_before["modified"] is None
    assert Trainer(lr=0.1).record.key == record.key  # reopened, not rebuilt
    np.testing.assert_array_equal(Trainer(lr=0.1).record.losses, [3.0, 2.0, 1.0])
    del record.losses
    assert "losses" not in other
    with pytest.raises(AttributeError):
        del record.losses


def test_workspace_append_and_overwrite_keep_appendable(root):
    record = Run.get({"lr": 0.2})
    record.append("step", [0, 1])
    record.append("step", [2])
    np.testing.assert_array_equal(record.load("step"), [0, 1, 2])
    record.step = np.array([5])  # replaces the rows, stays appendable
    record.append("step", [6])
    np.testing.assert_array_equal(record.load("step"), [5, 6])
    record["table"] = {"x": [1, 2]}
    assert record.table == {"x": [1, 2]}
    record.table = np.ones(2)  # a different codec replaces the old file
    assert record.meta["fields"]["table"]["codec"] == "ndarray"
    assert not (record.path / "table.json").exists()


def test_concurrent_edits_keep_every_field(root):
    record = Run.get({"lr": 0.3})

    def writer(i):
        handle = Run.locate({"lr": 0.3}, root=root)  # use_root does not reach new threads
        for j in range(5):
            handle[f"w{i}_{j}"] = j

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(record.keys()) == 20


def test_workspace_edits_need_a_finished_build(root):
    seen = {}

    class Slow(td.Record):
        __record_name__ = "WorkspaceSlow"
        mutable = True
        in_place = True

        def build(self, spec):
            self["first"] = 1
            try:
                Slow.locate().other = 2  # another handle while the build runs
            except td.ReadOnlyError as exc:
                seen["error"] = str(exc)

    record = Slow.get()
    assert "running" in seen["error"]
    record.other = 2
    assert record.keys() == ["first", "other"]

    class Broken(td.Record):
        __record_name__ = "WorkspaceBroken"
        mutable = True

        def build(self, spec):
            raise RuntimeError("boom")

    with pytest.raises(td.BuildFailed):
        Broken.get()
    with pytest.raises(td.ReadOnlyError, match="failed"):
        Broken.locate().x = 1
