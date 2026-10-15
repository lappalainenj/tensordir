import importlib.metadata
import importlib.util
import inspect
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from joblib._store_backends import FileSystemStoreBackend

import tensordir as td
from tensordir import codecs, needs_compute
from tensordir.codecs import FileCodec, get_codec
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
    assert [p.name for p in outputs] == ["output.td"]
    assert json.loads((outputs[0] / "codec.json").read_text())["codec"] == "xarray"
    assert not list((tmp_path / "cache").rglob(".output.td.tmp-*"))


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
    names = sorted(p.name for p in (tmp_path / "cache").rglob("output.*") if p.parent.name != "output.td")
    assert names == ["output.pkl", "output.pkl", "output.td"]
    (stored,) = (tmp_path / "cache").rglob("output.td")
    assert json.loads((stored / "codec.json").read_text())["codec"] == "parquet"


def test_registered_codecs_store_results(cache, tmp_path, monkeypatch):
    monkeypatch.setattr(codecs, "_CODECS", dict(codecs._CODECS))

    class Point:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class PointCodec(FileCodec):
        name = "point"
        suffix = ".point"

        def accepts(self, value):
            return isinstance(value, Point)

        def write(self, directory, name, value, *, dims=None):
            self.path(directory, name).write_text(f"{value.x},{value.y}")
            return {}

        def read(self, directory, name, info, *, lazy=True):
            return Point(*map(float, self.path(directory, name).read_text().split(",")))

    td.register_codec(PointCodec())
    calls = []

    @cache.cache
    def make(x):
        calls.append(x)
        return Point(x, 2 * x)

    first, second = make(1.5), make(1.5)
    assert calls == [1.5] and isinstance(second, Point) and (second.x, second.y) == (1.5, 3.0)
    assert next((tmp_path / "cache").rglob("output.point"))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch")
def test_tensor_dicts_are_safetensors_modules_pickle(cache, tmp_path):
    import torch

    @cache.cache
    def state(n):
        return {"w": torch.ones(n)}

    @cache.cache
    def module(n):
        return torch.nn.Linear(n, 1)

    assert torch.equal(state(2)["w"], state(2)["w"])
    assert isinstance(module(2), torch.nn.Linear) and isinstance(module(2), torch.nn.Linear)
    assert next((tmp_path / "cache").rglob("output.td/output.safetensors"))
    assert sorted(p.name for p in (tmp_path / "cache").rglob("output.*") if p.parent.name != "output.td") == [
        "output.pkl", "output.td"
    ]


class BigArraysAsZarr(td.StoreBackend):
    threshold = 1000

    def select_codec(self, item):
        if isinstance(item, np.ndarray) and item.nbytes > self.threshold:
            return get_codec("ndarray")
        return super().select_codec(item)


def test_backend_subclass(tmp_path):
    cache = td.memory(tmp_path / "c", backend=BigArraysAsZarr, verbose=0)
    assert isinstance(cache.store_backend, BigArraysAsZarr)
    f = cache.cache(lambda n: np.arange(n, dtype=float))
    f(10), f(1000)
    small, big = f(10), f(1000)
    assert isinstance(small, np.ndarray) and isinstance(big, zarr.Array) and big[999] == 999
    assert len(list((tmp_path / "c").rglob("output.td"))) == 1
    with pytest.raises(TypeError):
        td.memory(tmp_path / "d", backend=FileSystemStoreBackend)


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
    (slot,) = [p.parent for p in Path(cache.location).rglob("output.td")]
    shutil.rmtree(slot / "output.td")
    legacy = (labeled(2) * 10).to_dataset(name="rate")
    legacy.attrs["__microns_columns_dataarray_name__"] = "rate"
    legacy.to_netcdf(slot / "output.h5", engine="h5netcdf")
    out = f(2)
    assert isinstance(out, xr.DataArray) and out.name == "rate"
    xr.testing.assert_identical(out, labeled(2) * 10)


def test_earlier_tensordir_outputs_are_read(cache):
    """`output.zarr` and `output.parquet` slots written before the codec-based backend."""

    @cache.cache
    def f(n):
        return labeled(n)

    @cache.cache
    def g(n):
        return pd.DataFrame({"x": range(n)})

    f(2), g(2)
    slot_f, slot_g = (next(Path(cache.location).glob(f"**/{fn.__name__}/*/output.td")).parent for fn in (f, g))
    shutil.rmtree(slot_f / "output.td")
    shutil.rmtree(slot_g / "output.td")
    old = (labeled(2) * 3).to_dataset(name="rate")
    old.attrs = {"__tensordir_kind__": "dataarray", "__tensordir_dataarray__": "rate"}
    old.to_zarr(slot_f / "output.zarr", zarr_format=3, consolidated=False)
    pd.DataFrame({"x": [7, 8]}).to_parquet(slot_g / "output.parquet")
    xr.testing.assert_identical(f(2).load(), labeled(2) * 3)
    assert list(g(2).x) == [7, 8]
    assert f.call(2)[0].name == "rate"  # recomputing replaces the old slot
    assert not (slot_f / "output.zarr").exists() and (slot_f / "output.td").exists()


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
    assert isinstance(mem.store_backend, td.StoreBackend)
    assert mem.cache(lambda x: x + 1)(1) == 2
    assert "__cache__" not in rec.keys()
