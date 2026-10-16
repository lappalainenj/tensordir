import pickle
import re

import joblib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import tensordir as td
from tensordir import index


class Model(td.Spec):
    lr: float = 0.1
    layers: tuple[int, ...] = (8, 8)


class Sweep(td.Record):
    __record_name__ = "Sweep"

    class Spec(td.Spec):
        seed: int
        model: Model = td.field(default_factory=Model)
        tag: str = "base"
        fail: bool = False

    response: xr.DataArray

    def build(self, spec):
        if spec.fail:
            raise RuntimeError("diverged")
        t = np.arange(4)
        self.response = xr.DataArray(spec.seed * spec.model.lr * t, dims="time", coords={"time": t})


@pytest.fixture
def sweep(root):
    for seed in range(3):
        Sweep.get({"seed": seed})
    Sweep.get({"seed": 0, "model": Model(lr=0.5)})
    with pytest.raises(td.BuildFailed):
        Sweep.get({"seed": 9, "fail": True})
    return root


def test_query_filters(sweep):
    assert len(Sweep.query()) == 4
    assert len(Sweep.query(status=None)) == 5
    assert [r.spec.seed for r in Sweep.query(status="failed")] == [9]
    assert len(Sweep.query(seed=0)) == 2
    (fast,) = Sweep.query(model__lr=0.5)
    assert fast.spec.model.lr == 0.5
    assert len(Sweep.query(seed=lambda s: s >= 1)) == 2
    assert len(Sweep.query(model__layers=(8, 8))) == 4
    assert len(Sweep.query(missing=1)) == 0
    assert len(Sweep.query().where(lambda r: r.spec.seed == 2)) == 1
    assert len(Sweep.query().where(tag="base")) == 4


def test_query_and_where_share_the_condition_grammar(sweep):
    assert sorted(r.spec.seed for r in Sweep.query(seed=[1, 2])) == [1, 2]
    assert [r.spec.model.lr for r in Sweep.query(model__lr=">0.2")] == [0.5]
    runs = Sweep.query()
    assert len(runs.where({"model.lr": 0.5})) == 1
    assert len(runs.where(lambda r: r.spec.seed == 0, model__lr=0.1)) == 1
    assert len(td.where(runs, seed=">=1")) == 2
    assert len(runs.where(tag=re.compile("^ba"))) == 4
    table = runs.table(sizes=False)
    assert sorted(td.where(table, seed=[1, 2])["key"]) == sorted(r.key[:16] for r in runs.where(seed=[1, 2]))


def test_queries_see_defaults_of_fields_added_later(root):
    class Grown(td.Record):
        __record_name__ = "Grown"

        class Spec(td.Spec):
            n: int

        def build(self, spec):
            self["n"] = spec.n

    old = Grown.get({"n": 1})

    class Grown(td.Record):  # the same record type after adding a field
        __record_name__ = "Grown"

        class Spec(td.Spec):
            n: int
            prior: float = td.field(1.0, since=2)

        def build(self, spec):
            self["n"] = spec.n

    assert Grown.get({"n": 1}).key == old.key
    assert len(Grown.query(prior=1.0)) == 1
    assert len(Grown.query().where(prior=1.0)) == 1
    assert Grown.query().table(sizes=False)["spec.prior"].tolist() == [1.0]


def test_table_varying_only(sweep):
    table = Sweep.query().table(sizes=False, varying_only=True)
    assert [c for c in table.columns if c.startswith("spec.")] == ["spec.seed", "spec.model.lr"]
    assert "status" in table and "key" in table


def test_collection_repr_and_status_counts(sweep):
    runs = Sweep.query(status=None)
    assert runs.status_counts() == {"done": 4, "failed": 1}
    assert repr(runs) == "Collection(5 records of Sweep: 4 done, 1 failed)"
    assert repr(td.Collection()) == "Collection(0 records of no type)"
    page = runs._repr_html_()
    assert "<table" in page and "spec.seed" in page and "spec.tag" not in page and "path" not in page


def test_collection_delete(sweep):
    assert Sweep.query(status="failed").delete() == 1
    assert len(Sweep.query(status=None)) == 4
    with pytest.raises(td.BuildFailed):
        Sweep.get({"seed": 9, "fail": True})  # rebuilt from scratch, fails again


def test_table(sweep):
    table = Sweep.query(status=None).table()
    assert list(table.columns[:9]) == ["type", "key", "status", "created", "finished", "duration_s", "size_mb", "n_fields", "path"]
    spec_columns = list(table.columns[9:])
    assert spec_columns == ["spec.fail", "spec.seed", "spec.tag", "spec.model.layers", "spec.model.lr"]
    assert sorted(table["status"]) == ["done", "done", "done", "done", "failed"]
    assert str(table["spec.seed"].dtype) == "Int64" and str(table["spec.fail"].dtype) == "boolean"
    assert table["spec.model.layers"].iloc[0] == "[8, 8]"
    assert table["created"].dt.tz is not None and (table["size_mb"] > 0).all()
    shallow = Sweep.query().table(depth=1, sizes=False)
    assert "spec.model.lr" not in shallow and "size_mb" not in shallow
    assert "spec.model.lr" in Sweep.query().table(depth=1, include=["lr$"])


def test_info_on_class_and_record(sweep):
    pd.testing.assert_frame_equal(
        Sweep.info(depth=1, include=["lr$"], sizes=False),
        Sweep.query(status=None).table(depth=1, include=["lr$"], sizes=False),
    )
    assert len(Sweep.info(sizes=False)) == 5  # all statuses
    failed = Sweep.info(status="failed", seed=9, sizes=False)
    assert failed["status"].tolist() == ["failed"]
    assert len(Sweep.info(seed=[1, 2], varying_only=True, sizes=False)) == 2
    record = Sweep.get({"seed": 1})
    row = record.info(depth=1, sizes=False)
    assert len(row) == 1 and row["key"].iloc[0] == record.key[:16]
    assert "spec.seed" in row and "spec.model.lr" not in row
    pd.testing.assert_frame_equal(row, td.Collection([record]).table(depth=1, sizes=False))
    with pytest.raises(TypeError):
        record.info(seed=1)  # a record's table takes no filters


def test_map_concatenates_with_spec_coords(sweep):
    runs = Sweep.query(model__lr=0.1)
    result = runs.map(lambda r: r.response, dim="run", coords=["seed", "model__lr"])
    assert result.dims == ("run", "time") and result.sizes["run"] == 3
    assert list(result["seed"].values) == [r.spec.seed for r in runs]
    assert (result["model.lr"] == 0.1).all()
    assert list(result["run"].values) == [r.key[:16] for r in runs]
    with pytest.raises(TypeError, match="xarray"):
        runs.map(lambda r: 1.0)
    with pytest.raises(ValueError):
        td.Collection().map(lambda r: r.response)


def test_map_with_memory(sweep, tmp_path):
    calls = []

    def peak(record):
        calls.append(record.key)
        return record.response.max().rename("peak")

    memory = td.memory(tmp_path / "cache", verbose=0)
    runs = Sweep.query()
    first = runs.map(peak, memory=memory)
    second = runs.map(peak, memory=memory)
    assert len(calls) == 4
    xr.testing.assert_identical(first, second.load())


def test_reindex_is_used_while_fresh(sweep, monkeypatch):
    path = td.reindex()
    rows = pd.read_parquet(path)
    assert len(rows) == 5 and set(rows["status"]) == {"done", "failed"}

    def no_scan(type_dir):
        raise AssertionError("scanned although the index is fresh")

    monkeypatch.setattr(index, "_scan", no_scan)
    assert len(Sweep.query()) == 4
    monkeypatch.undo()
    Sweep.get({"seed": 7})
    monkeypatch.setattr(index, "_scan", no_scan)
    with pytest.raises(AssertionError, match="scanned"):
        Sweep.query()
    monkeypatch.undo()
    assert len(Sweep.query()) == 5
    Sweep.query(seed=7)[0].delete()
    assert len(Sweep.query()) == 4


class Analysis(td.View):
    calls = 0

    @td.cached_method(maxsize=2)
    def scaled(self, factor, weights=None, offset=0.0):
        type(self).calls += 1
        w = 1.0 if weights is None else weights.sum()
        return float(self.response.sum()) * factor * w + offset


def test_view_and_cached_method(sweep):
    record = Sweep.query(seed=2, model__lr=0.1)[0]
    view = Analysis(record)
    assert view.spec.seed == 2 and view.response.dims == ("time",)
    a = view.scaled(2)
    assert view.scaled(2) == view.scaled(factor=2) == view.scaled(2, None, 0.0) == a
    assert Analysis.calls == 1
    view.scaled(2, weights=np.ones(3))
    view.scaled(2, weights=np.ones(3))
    assert Analysis.calls == 2
    view.scaled(3)
    view.scaled(2)
    assert Analysis.calls == 4  # maxsize=2 evicted factor=2
    Analysis.scaled.cache_clear(view)
    view.scaled(3)
    assert Analysis.calls == 5
    assert Analysis(record).scaled(3) and Analysis.calls == 6  # caches are per instance


def test_view_pickles_as_its_record(sweep):
    record = Sweep.query(seed=1)[0]
    view = Analysis(record)
    view.scaled(1)
    back = pickle.loads(pickle.dumps(view))
    assert back.record == record and "__tensordir_cache_scaled" not in back.__dict__
    assert joblib.hash(view) == joblib.hash(Analysis(Sweep.query(seed=1)[0]))
    assert view.memory.location == str(record.path / "__cache__")
    assert view.needs_compute(view.memory.cache(lambda x: x), 1)
