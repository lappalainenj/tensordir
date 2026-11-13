"""`Record.publish`: write-once copies of finished records to another root."""

import json

import numpy as np
import pytest

import tensordir as td
from tensordir import _publish
from tensordir.cli import main


class Source(td.Record):
    __record_name__ = "PublishSource"

    def __init__(self, /, *, n: int):
        self.write("x", np.arange(n), layout=td.Layout(chunk_bytes=16, shard_bytes=64))
        self.write("note", {"n": n})


class Derived(td.Record):
    __record_name__ = "PublishDerived"

    def __init__(self, source: Source, /, *, scale: int = 2):
        self.write("y", source.load("x") * scale)


class Notebook(td.Record):
    __record_name__ = "PublishNotebook"
    mutable = True


@pytest.fixture
def destination(tmp_path):
    return tmp_path / "published"


def _record_writes(monkeypatch):
    order = []
    copy, write_once = _publish._copy, _publish._write_once

    def recording_copy(source, target):
        order.append(target)
        copy(source, target)

    def recording_write_once(target, data):
        order.append(target)
        write_once(target, data)

    monkeypatch.setattr(_publish, "_copy", recording_copy)
    monkeypatch.setattr(_publish, "_write_once", recording_write_once)
    return order


def test_publish_copies_files_once_with_meta_last(root, destination, monkeypatch):
    record = Source(n=40)
    order = _record_writes(monkeypatch)
    copy = record.publish(destination)
    names = [p.name for p in order]
    assert names[0] == "published" and names[-1] == "_meta.json" and names.count("_meta.json") == 1
    assert len(order) == len(set(order))  # no file is written twice
    assert copy.path == destination / "PublishSource" / record.key[:16]
    assert copy.key == record.key and copy.status == "done" and isinstance(copy, Source)
    np.testing.assert_array_equal(copy.load("x"), np.arange(40))
    assert copy.load("note") == {"n": 40}
    assert Source.lookup(n=40, root=destination).path == copy.path  # found by its key, nothing built


def test_republishing_keeps_the_copy(root, destination, monkeypatch):
    record = Source(n=5)
    record.publish(destination)
    order = _record_writes(monkeypatch)
    record.publish(destination)
    assert order == []


def test_a_partial_copy_is_redone(root, destination):
    record = Source(n=5)
    target = destination / "PublishSource" / record.key[:16]
    (target / "data.zarr").mkdir(parents=True)
    (target / "data.zarr" / "stray").write_text("half a copy")
    record.publish(destination)
    assert not (target / "data.zarr" / "stray").exists()
    np.testing.assert_array_equal(td.Record.at(target).load("x"), np.arange(5))


def test_a_different_record_at_the_destination_raises(root, destination):
    record = Source(n=5)
    target = destination / "PublishSource" / record.key[:16]
    target.mkdir(parents=True)
    (target / "_meta.json").write_text(json.dumps({"key": "0" * 64, "status": "done"}))
    with pytest.raises(td.KeyCollision):
        record.publish(destination)


def test_only_done_records_are_published(root, destination):
    with pytest.raises(ValueError, match="only done"):
        Source.locate(n=3).publish(destination)


def test_upstream_records_are_published_first(root, destination):
    source = Source(n=4)
    derived = Derived(source)
    copy = derived.publish(destination, upstream=True)
    assert (destination / "PublishSource" / source.key[:16] / "_meta.json").exists()
    (upstream,) = copy.inputs
    assert upstream.path == destination / "PublishSource" / source.key[:16]
    np.testing.assert_array_equal(copy.load("y"), np.arange(4) * 2)


def test_builds_and_edits_are_refused_in_a_publish_root(root, destination):
    Source(n=4).publish(destination)
    with pytest.raises(td.ReadOnlyError, match="publish target"):
        Source.get(n=5, root=destination)
    notebook = Notebook.get()
    notebook.write("loss", np.array([1.0]))
    copy = notebook.publish(destination)
    with pytest.raises(td.ReadOnlyError, match="publish target"):
        copy.write("loss", np.array([2.0]))


def test_workspace_records_are_copied_again(root, destination):
    notebook = Notebook.get()
    notebook.write("loss", np.array([1.0]))
    notebook.publish(destination)
    notebook.append("steps", [1, 2])
    copy = notebook.publish(destination)
    assert copy.keys() == ["loss", "steps"]


def test_publishing_into_an_existing_root_keeps_it_buildable(root, tmp_path):
    other = tmp_path / "other"
    Source.get(n=1, root=other)  # an ordinary root
    Source(n=2).publish(other)
    assert Source.get(n=3, root=other).status == "done"


def test_cli_publishes_done_records(root, destination, capsys):
    Source(n=1)
    Source(n=2)
    assert main(["--root", str(root), "publish", str(destination), "PublishSource"]) == 0
    assert "2 records published" in capsys.readouterr().out
    assert len(list((destination / "PublishSource").iterdir())) == 2
