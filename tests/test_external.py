"""Pinned references to Delta tables and parquet directories."""

import importlib.util
import re

import numpy as np
import pytest

import tensordir as td

pa = pytest.importorskip("pyarrow")
import pyarrow.dataset as ds  # noqa: E402

needs_delta = pytest.mark.skipif(importlib.util.find_spec("deltalake") is None, reason="needs deltalake")


def _synapses(n, offset=0):
    rng = np.random.default_rng(offset)
    return pa.table(
        {
            "pre": pa.array(rng.integers(0, 5, n), pa.int64()),
            "post": pa.array(rng.integers(0, 5, n), pa.int64()),
            "size": pa.array(rng.uniform(0, 1, n)),
            "part": pa.array([i % 2 for i in range(n)], pa.int64()),
        }
    )


@pytest.fixture
def delta_table(tmp_path):
    from deltalake import write_deltalake

    path = tmp_path / "synapses"
    write_deltalake(str(path), _synapses(40), partition_by=["part"])  # version 0
    write_deltalake(str(path), _synapses(10, offset=1), mode="append", partition_by=["part"])  # version 1
    return path


class Edges(td.Record):
    __record_name__ = "ExternalEdges"

    def __init__(self, /, *, synapses: td.DeltaRef, min_size: float = 0.0):
        synapses.materialize(self, "synapses", columns=["pre", "post", "size"], filter={"size": f">{min_size}"})
        frame = synapses.to_pandas(columns=["pre", "post"], filter={"size": f">{min_size}"})
        self["edges"] = frame.groupby(["pre", "post"]).size().rename("n_syn").reset_index()


@needs_delta
def test_delta_ref_pins_versions(delta_table):
    latest = td.DeltaRef(delta_table)
    first = td.DeltaRef(str(delta_table), version=0, storage_options={"secret": "x"})
    assert latest.version == 1 and first.version == 0
    assert first.to_table().num_rows == 40 and latest.to_table().num_rows == 50
    assert first == td.DeltaRef(delta_table, version=0)  # storage options are not identity
    assert "secret" not in repr(first)
    table = first.to_table(columns=["pre", "size"], filter={"pre": [1, 2], "size": ">0.5"})
    assert table.column_names == ["pre", "size"]
    assert set(table["pre"].to_pylist()) <= {1, 2} and min(table["size"].to_pylist()) > 0.5
    assert sum(b.num_rows for b in first.to_batches(filter={"part": 0}, batch_size=7)) == 20
    assert first.files() and isinstance(first.dataset(), ds.Dataset)
    with pytest.raises(TypeError):
        td.DeltaRef(delta_table, version="0")


@needs_delta
def test_delta_ref_in_specs_and_records(root, delta_table):
    first = td.DeltaRef(delta_table, version=0, storage_options={"token": "secret"})
    spec = Edges.Spec(synapses=first)
    assert spec.to_dict()["synapses"] == {"__delta__": first.uri, "version": 0}
    assert Edges.Spec.from_dict(spec.to_dict()).synapses == first  # revived as a DeltaRef
    assert Edges.compute_key(**spec) != Edges.compute_key(**spec.replace(synapses=td.DeltaRef(delta_table, version=1)))
    record = Edges(**spec)
    assert "secret" not in (record.path / "_meta.json").read_text()
    stored = record["synapses"]
    assert isinstance(stored, ds.Dataset) and stored.schema.names == ["pre", "post", "size"]
    assert record.load("synapses").num_rows == first.to_table(filter={"size": ">0.0"}).num_rows
    assert record.meta["fields"]["synapses"] == {"codec": "arrow", "rows": record.load("synapses").num_rows}
    assert record.edges["n_syn"].sum() == record.load("synapses").num_rows
    table = Edges.query().table(sizes=False)
    assert table["spec.synapses"].iloc[0] == f"delta:{first.uri}@0"
    assert len(Edges.query(synapses=re.compile("@0$"))) == 1
    reopened = td.Record.at(record.path)
    assert reopened.spec.synapses.version == 0 and reopened.spec.synapses.storage_options is None


@needs_delta
def test_delta_ref_engines(delta_table):
    ref = td.DeltaRef(delta_table, version=1)
    if importlib.util.find_spec("duckdb") is not None:
        relation = ref.duckdb()
        assert relation.filter("part = 1").aggregate("count(*)").fetchone()[0] == 25
    if importlib.util.find_spec("polars") is not None:
        assert ref.to_polars().collect().height == 50


def test_parquet_ref_pins_listing(root, tmp_path):
    path = tmp_path / "parquet"
    ds.write_dataset(_synapses(30), path, format="parquet", partitioning=["part"], partitioning_flavor="hive")
    (path / "_SUCCESS").write_text("")  # ignored, as pyarrow ignores it
    ref = td.ParquetRef(path)
    assert len(ref.version) == 64 and len(ref.files()) == 2
    assert ref.to_table(filter={"part": 1}).num_rows == 15
    assert td.ParquetRef(path, version=ref.version) == ref

    class Summary(td.Record):
        __record_name__ = "ExternalSummary"

        def __init__(self, /, *, table: td.ParquetRef):
            self["n"] = table.to_table().num_rows

    assert Summary.get(table=ref).n == 30
    extra = _synapses(4, offset=2).drop_columns(["part"])
    ds.write_dataset(extra, path / "part=2", format="parquet", basename_template="extra-{i}.parquet")
    with pytest.raises(td.SnapshotChanged):
        ref.to_table()
    newer = td.ParquetRef(path)
    assert newer.version != ref.version and newer.to_table().num_rows == 34
    assert Summary.compute_key(table=newer) != Summary.compute_key(table=ref)
    with pytest.raises(FileNotFoundError):
        td.ParquetRef(tmp_path / "empty-or-missing")


def test_filter_mappings():
    from tensordir.external import expression

    pc = pytest.importorskip("pyarrow.compute")
    table = pa.table({"a": [1, 2, 3, None], "b": ["x", "y", "x", "z"]})

    def rows(condition):
        return table.filter(expression(condition))["a"].to_pylist()

    assert rows({"a": ">=2, <3"}) == [2]
    assert rows({"b": "x"}) == [1, 3]
    assert rows({"b": ("y", "z")}) == [2, None]
    assert rows({"a": None}) == [None]
    assert rows({"b": td.any_of("y", "z")}) == [2, None]
    assert rows({"a": td.exactly(2)}) == [2]
    assert rows({"b": td.exactly(">a")}) == []  # a literal string, not a comparison
    assert expression(pc.field("a") > 1).equals(pc.field("a") > 1)
    with pytest.raises(TypeError):
        expression({"b": re.compile("x")})
    with pytest.raises(TypeError):
        expression({"b": td.exactly(["x"])})
