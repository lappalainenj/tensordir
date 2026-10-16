"""Root resolution: a base root joined with a class root or a `root=` argument (pathlib `/`)."""

import contextvars
import threading

import numpy as np
import pytest

import tensordir as td
from tensordir import _root
from tensordir._record import INTERNAL
from tensordir.cli import main


@pytest.fixture
def clean_roots(monkeypatch, tmp_path):
    monkeypatch.setattr(_root, "_default", None)
    monkeypatch.delenv("TENSORDIR_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)


class Plain(td.Record):
    __record_name__ = "RootsPlain"

    class Spec(td.Spec):
        n: int = 3

    def build(self, spec):
        self["values"] = np.arange(spec.n)


def _classes(tmp_path):
    class Connectome(td.Record):
        __record_name__ = "RootsConnectome"
        root = "connectomes"  # relative: joined onto the base root on every access

        class Spec(td.Spec):
            n: int = 3

        def build(self, spec):
            lock = self.resolve_root() / INTERNAL / "locks" / self.path.parent.name / self.path.name
            self["lock_seen"] = lock.is_dir()
            self["values"] = np.arange(spec.n)

    @td.root(tmp_path / "datasets")  # absolute: the base root does not matter
    class Dataset(td.Record):
        __record_name__ = "RootsDataset"

        class Spec(td.Spec):
            n: int = 3

        def build(self, spec):
            self["values"] = np.arange(spec.n)

    return Connectome, Dataset


def test_unset_base_is_the_working_directory_at_resolution(tmp_path, clean_roots, monkeypatch):
    Connectome, Dataset = _classes(tmp_path)
    assert td.get_root() == tmp_path and Plain.resolve_root() == tmp_path
    assert td.get_root(Connectome) == tmp_path / "connectomes"
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    assert td.get_root(Connectome) == tmp_path / "elsewhere" / "connectomes"
    assert td.get_root(Dataset) == tmp_path / "datasets"
    record = Plain.get()
    assert record.path.parents[1] == tmp_path / "elsewhere"


def test_examples_of_the_join_rule(tmp_path, clean_roots, monkeypatch):
    Connectome, Dataset = _classes(tmp_path)
    monkeypatch.setenv("TENSORDIR_ROOT", str(tmp_path / "data"))
    assert td.get_root(Connectome) == tmp_path / "data" / "connectomes"
    assert td.get_root(Plain) == tmp_path / "data"
    with td.use_root(tmp_path / "ctx"):
        assert td.get_root(Connectome) == tmp_path / "ctx" / "connectomes"
        assert td.get_root(Dataset) == tmp_path / "datasets"  # absolute class root: unaffected
        assert td.get_root(Connectome, root="arg") == tmp_path / "ctx" / "arg"  # replaces the class root
        assert td.get_root(Dataset, root="arg") == tmp_path / "ctx" / "arg"
        assert td.get_root(Connectome, root=tmp_path / "abs") == tmp_path / "abs"
        assert Connectome.locate(root="arg").path.parents[1] == tmp_path / "ctx" / "arg"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert td.get_root(Plain, root="~/runs") == tmp_path / "home" / "runs"
    td.root("~/connectomes")(Connectome)
    assert td.get_root(Connectome) == tmp_path / "home" / "connectomes"


def test_base_root_precedence_and_relative_values(tmp_path, clean_roots, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    monkeypatch.chdir(tmp_path / "a")
    monkeypatch.setenv("TENSORDIR_ROOT", "env")  # relative: read at resolution time
    assert td.get_root() == tmp_path / "a" / "env"
    monkeypatch.chdir(tmp_path / "b")
    assert td.get_root() == tmp_path / "b" / "env"
    td.set_root("default")  # relative: resolved now, against b
    monkeypatch.chdir(tmp_path / "a")
    assert td.get_root() == tmp_path / "b" / "default"
    with td.use_root("ctx") as ctx:  # resolved now, against a
        assert ctx == tmp_path / "a" / "ctx" == td.get_root()
        with td.use_root(tmp_path / "inner"):
            assert td.get_root() == tmp_path / "inner"
        assert td.get_root() == ctx
    td.set_root(None)
    assert td.get_root() == tmp_path / "a" / "env"


def test_relative_class_root_follows_use_root(tmp_path, clean_roots):
    Connectome, Dataset = _classes(tmp_path)
    for base in ("one", "two"):
        with td.use_root(tmp_path / base):
            record = Connectome.get({"n": 2})
            assert record.path.parents[1] == tmp_path / base / "connectomes"
            assert record.lock_seen  # the lease lives under the class's resolved root
            assert [r.spec.n for r in Connectome.query()] == [2]
            assert td.reindex(Connectome) == tmp_path / base / "connectomes" / INTERNAL / "index.parquet"
            assert len(Connectome.query(n=2)) == 1  # served from that root's index
            Dataset.get({"n": 5})
    assert len(Dataset.query()) == 1  # one absolute root for both blocks
    assert len(Connectome.query()) == 0  # cwd/connectomes holds nothing
    assert len(Connectome.query(root=tmp_path / "one" / "connectomes")) == 1
    record = td.Record.at((tmp_path / "two" / "connectomes" / "RootsConnectome").iterdir().__next__())
    assert isinstance(record, Connectome) and record.status == "done"


def test_use_root_is_context_local(tmp_path, clean_roots):
    Connectome, _ = _classes(tmp_path)
    td.set_root(tmp_path / "default")
    seen = {}
    barrier = threading.Barrier(2)

    def worker(name):
        seen[f"{name}-start"] = td.get_root(Connectome)  # a new thread starts without the block
        with td.use_root(tmp_path / name):
            barrier.wait(5)
            seen[name] = td.get_root(Connectome)
            barrier.wait(5)

    with td.use_root(tmp_path / "main"):
        threads = [threading.Thread(target=worker, args=(name,)) for name in ("x", "y")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        copied = threading.Thread(target=contextvars.copy_context().run, args=(lambda: seen.setdefault("copied", td.get_root()),))
        copied.start()
        copied.join(10)
        assert td.get_root(Connectome) == tmp_path / "main" / "connectomes"
    assert seen["x-start"] == seen["y-start"] == tmp_path / "default" / "connectomes"
    assert seen["x"] == tmp_path / "x" / "connectomes" and seen["y"] == tmp_path / "y" / "connectomes"
    assert seen["copied"] == tmp_path / "main"


def test_decorator_inheritance_and_removal(tmp_path, clean_roots):
    _, Dataset = _classes(tmp_path)

    class Movie(Dataset):
        __record_name__ = "RootsMovie"

    assert Movie.resolve_root() == tmp_path / "datasets"
    td.root(None)(Movie)
    assert Movie.resolve_root() == tmp_path  # the base root
    assert Dataset.resolve_root() == tmp_path / "datasets"
    td.root("movies")(Movie)
    assert Movie.root == "movies"  # stored as given
    with pytest.raises(TypeError):
        td.root(tmp_path)(object)
    with pytest.raises(TypeError):
        td.get_root(object)


def test_owner_classes_and_cli(tmp_path, clean_roots, monkeypatch, capsys):
    @td.record(root="runs")
    class Trainer:
        def __init__(self, lr: float = 0.1):
            pass

    monkeypatch.setenv("TENSORDIR_ROOT", str(tmp_path / "data"))
    assert td.get_root(Trainer) == td.get_root(Trainer.Record) == tmp_path / "data" / "runs"
    assert Trainer().record.path.parents[1] == tmp_path / "data" / "runs"
    assert len(Trainer.Record.query()) == 1
    assert td.reindex(Trainer).parents[1] == tmp_path / "data" / "runs"
    main(["--root", "runs", "ls"])  # joined onto $TENSORDIR_ROOT like a class root
    assert len(capsys.readouterr().out.splitlines()) == 1
    main(["ls"])
    assert capsys.readouterr().out == ""
