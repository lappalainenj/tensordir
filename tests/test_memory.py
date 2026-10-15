import importlib.metadata
import inspect
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from joblib._store_backends import FileSystemStoreBackend

import tensordir as td
from tensordir import needs_compute
from tensordir._memory import TensordirStoreBackend
from _records import Square


def test_joblib_version_is_in_the_pinned_range():
    major, minor = (int(p) for p in joblib.__version__.split(".")[:2])
    assert (1, 4) <= (major, minor) < (1, 6)
    requirement = next(r for r in importlib.metadata.requires("tensordir") if r.startswith("joblib"))
    assert "<1.6" in requirement


@pytest.mark.parametrize(
    "method, params",
    [
        ("dump_item", ["self", "call_id", "item", "verbose"]),
        ("load_item", ["self", "call_id", "verbose", "timestamp", "metadata"]),
        ("contains_item", ["self", "call_id"]),
        ("configure", ["self", "location", "verbose", "backend_options"]),
        ("create_location", ["self", "location"]),
        ("_item_exists", ["path"]),
    ],
)
def test_private_backend_contract(method, params):
    """The joblib internals the backend overrides or calls keep their signatures."""
    signature = inspect.signature(getattr(FileSystemStoreBackend, method))
    assert list(signature.parameters) == params


@pytest.fixture
def cache(tmp_path):
    return td.memory(tmp_path / "cache", verbose=0)


def labeled(n, name="rate"):
    t = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    return xr.DataArray(
        np.arange(2.0 * n).reshape(n, 2),
        dims=("cell", "time"),
        coords={"cell": [f"c{i}" for i in range(n)], "time": t},
        attrs={"units": "Hz"},
        name=name,
    )


def test_xarray_results_are_zarr_and_lazy(cache, tmp_path):
    calls = []

    @cache.cache
    def f(n):
        calls.append(n)
        return labeled(n).to_dataset()

    first = f(3)
    second = f(3)
    assert calls == [3]
    assert not second["rate"].variable._in_memory
    xr.testing.assert_identical(second.load(), first)
    outputs = list((tmp_path / "cache").rglob("output.*"))
    assert [p.name for p in outputs] == ["output.zarr"]
    assert not list((tmp_path / "cache").rglob(".output.zarr.tmp-*"))


@pytest.mark.parametrize("name", ["rate", None])
def test_dataarray_round_trip(cache, name):
    @cache.cache
    def f(n):
        return labeled(n, name)

    first, second = f(2), f(2)
    assert isinstance(second, xr.DataArray) and second.name == name
    xr.testing.assert_identical(second.load(), first)


def test_datatree_round_trip(cache):
    @cache.cache
    def f():
        return xr.DataTree.from_dict({"/": labeled(2).to_dataset(), "/a": xr.Dataset({"w": ("k", [1, 2])})})

    first, second = f(), f()
    xr.testing.assert_identical(second.load(), first)


def test_eager_option(tmp_path):
    cache = td.memory(tmp_path / "c", lazy=False, verbose=0)
    f = cache.cache(lambda n: labeled(n).to_dataset())
    f(2)
    assert f(2)["rate"].variable._in_memory


def test_dataframes_are_parquet_and_others_pickle(cache, tmp_path):
    @cache.cache
    def table(n):
        return pd.DataFrame({"x": np.arange(n), "s": list("abc"[:n])})

    @cache.cache
    def array(n):
        return np.arange(n)

    @cache.cache
    def mapping(n):
        return {"n": n}

    pd.testing.assert_frame_equal(table(3), table(3))
    np.testing.assert_array_equal(array(3), array(3))
    assert mapping(3) == mapping(3) == {"n": 3}
    names = sorted(p.name for p in (tmp_path / "cache").rglob("output.*"))
    assert names == ["output.parquet", "output.pkl", "output.pkl"]


def test_recompute_replaces_output(cache):
    @cache.cache
    def f(n):
        return labeled(n).to_dataset()

    f(2)
    assert f.call(2)[0]["rate"].shape == (2, 2)
    assert f(2)["rate"].shape == (2, 2)


def test_legacy_h5_outputs_are_read(cache):
    """A cache slot written by the flyvis / another project netCDF backend loads as before."""

    @cache.cache
    def f(n):
        return labeled(n).to_dataset()

    f(2)
    (slot,) = [p.parent for p in Path(cache.location).rglob("output.zarr")]
    shutil.rmtree(slot / "output.zarr")
    legacy = (labeled(2) * 10).to_dataset(name="rate")
    legacy.attrs["__microns_columns_dataarray_name__"] = "rate"
    legacy.to_netcdf(slot / "output.h5", engine="h5netcdf")
    out = f(2)
    assert isinstance(out, xr.DataArray) and out.name == "rate"
    xr.testing.assert_identical(out, labeled(2) * 10)


def test_records_hash_by_identity(root, cache):
    calls = []

    @cache.cache
    def total(rec):
        calls.append(rec.key)
        return float(rec.values.sum())

    a = Square.get({"n": 3})
    assert total(a) == 5.0
    assert total(Square.locate({"n": 3})) == 5.0
    assert total(Square.get({"n": 4})) == 14.0
    assert len(calls) == 2
    assert joblib.hash(a) == joblib.hash(Square.locate(Square.Spec(n=3, device="cuda")))
    assert needs_compute(total, Square.locate({"n": 5}))
    assert not needs_compute(total, a)


def test_specs_hash_canonically():
    assert joblib.hash(Square.Spec(n=np.int64(2), scale=np.float32(1.0))) == joblib.hash(Square.Spec(n=2, scale=1))


def test_record_memory_lives_in_the_record(root):
    rec = Square.get({"n": 2})
    mem = rec.memory
    assert mem.location == str(rec.path / "__cache__")
    assert isinstance(mem.store_backend, TensordirStoreBackend)
    assert mem.cache(lambda x: x + 1)(1) == 2
    assert "__cache__" not in rec.keys()
