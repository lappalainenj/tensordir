"""Per-class default roots and the root resolution order."""

import numpy as np
import pytest

import tensordir as td
from tensordir import _root
from tensordir._record import INTERNAL


@pytest.fixture
def clean_roots(monkeypatch):
    monkeypatch.setattr(_root, "_default", None)
    monkeypatch.delenv("TENSORDIR_ROOT", raising=False)


class Plain(td.Record):
    __record_name__ = "RootsPlain"

    class Spec(td.Spec):
        n: int = 3

    def build(self, spec):
        self["values"] = np.arange(spec.n)


def _classes(tmp_path):
    class Connectome(td.Record):
        __record_name__ = "RootsConnectome"
        root = tmp_path / "connectomes"

        class Spec(td.Spec):
            n: int = 3

        def build(self, spec):
            lock = self.resolve_root() / INTERNAL / "locks" / self.path.parent.name / self.path.name
            self["lock_seen"] = lock.is_dir()
            self["values"] = np.arange(spec.n)

    @td.root(tmp_path / "datasets")
    class Dataset(td.Record):
        __record_name__ = "RootsDataset"

        class Spec(td.Spec):
            n: int = 3

        def build(self, spec):
            self["values"] = np.arange(spec.n)

    return Connectome, Dataset


def test_class_roots_separate_record_types(tmp_path, clean_roots):
    Connectome, Dataset = _classes(tmp_path)
    connectome, dataset = Connectome.get(), Dataset.get()
    assert connectome.path.parents[1] == tmp_path / "connectomes"
    assert dataset.path.parents[1] == tmp_path / "datasets"
    assert connectome.lock_seen  # the lease lives under the class root
    with pytest.raises(td.RootNotSet):
        Plain.get()  # no class root and no default


def test_resolution_order(tmp_path, clean_roots, monkeypatch):
    Connectome, _ = _classes(tmp_path)
    monkeypatch.setenv("TENSORDIR_ROOT", str(tmp_path / "env"))
    assert Plain.resolve_root() == tmp_path / "env"
    td.set_root(tmp_path / "default")
    assert Plain.resolve_root() == tmp_path / "default"
    assert Connectome.resolve_root() == tmp_path / "connectomes"  # class root beats set_root
    with td.use_root(tmp_path / "context"):
        assert Connectome.resolve_root() == tmp_path / "context"  # use_root beats the class root
        assert Connectome.resolve_root(tmp_path / "arg") == tmp_path / "arg"  # root= beats all
        assert Connectome.locate().path.parents[1] == tmp_path / "context"
    assert Connectome.locate(root=tmp_path / "arg").path.parents[1] == tmp_path / "arg"


def test_decorator_inheritance_and_removal(tmp_path, clean_roots):
    _, Dataset = _classes(tmp_path)

    class Movie(Dataset):
        __record_name__ = "RootsMovie"

    assert Movie.resolve_root() == tmp_path / "datasets"
    td.root(None)(Movie)
    with pytest.raises(td.RootNotSet):
        Movie.resolve_root()
    assert Dataset.resolve_root() == tmp_path / "datasets"
    with pytest.raises(TypeError):
        td.root(tmp_path)(object)


def test_query_reindex_and_reopen_per_root(tmp_path, clean_roots):
    Connectome, Dataset = _classes(tmp_path)
    for n in (1, 2):
        Connectome.get({"n": n})
    Dataset.get({"n": 5})
    td.set_root(tmp_path / "default")  # the class roots still win
    assert sorted(r.spec.n for r in Connectome.query()) == [1, 2]
    assert [r.spec.n for r in Dataset.query()] == [5]
    assert len(Connectome.query(root=tmp_path / "datasets")) == 0
    index = td.reindex(Connectome)
    assert index == tmp_path / "connectomes" / INTERNAL / "index.parquet"
    assert len(Connectome.query(n=2)) == 1  # served from the fresh index
    record = td.Record.at(Connectome.locate({"n": 1}).path)
    assert isinstance(record, Connectome) and record.status == "done"
