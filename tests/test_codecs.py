import pickle

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

import tensordir as td
from tensordir.codecs import Codec, codec_for, get_codec

VALUES: dict = {}
CODECS: dict = {}


class Holder(td.Record):
    """Stores VALUES[case] as field `value` (with CODECS[case] if given)."""

    __record_name__ = "Holder"

    def __init__(self, /, *, case: str):
        self.write("value", VALUES[case], codec=CODECS.get(case))


def store(case, value, codec=None):
    VALUES[case] = value
    if codec:
        CODECS[case] = codec
    return Holder.get(case=case)


ARRAYS = {
    "float2d": np.arange(6.0).reshape(2, 3),
    "bool": np.array([True, False, True]),
    "unicode": np.array(["a", "βγ", ""]),
    "bytes": np.array([b"x", b"yz"]),
    "datetime": np.array(["2020-01-01T00:00", "2021-06-01T12:30"], dtype="datetime64[ns]"),
    "timedelta": np.array([1, 2], dtype="timedelta64[s]"),
    "nan": np.array([np.nan, 1.0, np.inf]),
    "empty": np.zeros((0, 3), dtype="f4"),
    "scalar": np.array(3.5),
    "int8": np.array([-1, 2], dtype=np.int8),
    "bigendian": np.array([1.0, 2.0], dtype=">f8"),
}


@pytest.mark.filterwarnings("ignore:The data type:zarr.errors.UnstableSpecificationWarning")
@pytest.mark.parametrize("case", sorted(ARRAYS))
def test_ndarray_round_trip(root, case):
    value = ARRAYS[case]
    rec = store(f"nd-{case}", value)
    assert rec.meta["fields"]["value"]["codec"] == "ndarray"
    back = rec.load("value")
    assert back.shape == value.shape and back.dtype.kind == value.dtype.kind
    np.testing.assert_array_equal(back, value)
    assert isinstance(rec["value"], zarr.Array)


def dataset():
    t = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
    return xr.Dataset(
        {
            "rate": (("cell", "time"), np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]]), {"units": "Hz"}),
            "active": ("cell", np.array([True, False])),
            "label": ("cell", np.array(["exc", "inh"])),
            "empty": (("cell", "none"), np.zeros((2, 0))),
        },
        coords={"cell": ["c0", "c1"], "time": t, "depth": ("cell", [10.5, 20.0])},
        attrs={"source": "test", "n": 3, "scales": [1, 2]},
    )


def test_dataset_round_trip_is_lazy(root):
    ds = dataset()
    rec = store("ds", ds)
    lazy = rec["value"]
    assert not lazy["rate"].variable._in_memory
    xr.testing.assert_identical(lazy.load(), ds)
    xr.testing.assert_identical(rec.load("value"), ds)
    assert rec.value.attrs == ds.attrs


@pytest.mark.parametrize("name", ["rate", None])
def test_dataarray_round_trip(root, name):
    da = dataset()["rate"].rename(name)
    rec = store(f"da-{name}", da)
    back = rec.load("value")
    assert isinstance(back, xr.DataArray) and back.name == name
    xr.testing.assert_identical(back, da)


def test_datatree_round_trip(root):
    tree = xr.DataTree.from_dict({"/": dataset(), "/fits/a": xr.Dataset({"w": ("k", [1, 2])}), "/fits/b": xr.Dataset({"w": ("k", [3, 4])})})
    rec = store("tree", tree)
    back = rec.load("value")
    assert isinstance(back, xr.DataTree)
    xr.testing.assert_identical(back, tree)


def test_dataframe_round_trip(root):
    df = pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.0],
            "n": np.array([1, 2, 3], dtype="int32"),
            "flag": [True, False, True],
            "name": pd.array(["a", None, "c"], dtype="string"),
            "when": pd.to_datetime(["2020-01-01", "2020-01-02", None]),
            "kind": pd.Categorical(["u", "v", "u"]),
        },
        index=pd.Index([10, 20, 30], name="cell_id"),
    )
    rec = store("df", df)
    assert rec.meta["fields"]["value"]["codec"] == "parquet"
    pd.testing.assert_frame_equal(rec.value, df)
    empty = store("df-empty", df.iloc[:0])
    # parquet does not keep the categories of an empty categorical column
    pd.testing.assert_frame_equal(empty.value.drop(columns="kind"), df.iloc[:0].drop(columns="kind"))
    assert isinstance(empty.value["kind"].dtype, pd.CategoricalDtype)


def test_json_values(root):
    value = {"a": np.int64(3), "b": [np.float32(0.5), None, float("nan")], "c": {"d": "e"}, "t": (1, 2), "ok": np.bool_(True)}
    back = store("json", value).value
    assert back["a"] == 3 and back["b"][0] == 0.5 and np.isnan(back["b"][2])
    assert back["t"] == [1, 2] and back["ok"] is True and back["c"] == {"d": "e"}
    assert store("json-scalar", 2.5).value == 2.5


class Opaque:
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return isinstance(other, Opaque) and other.x == self.x


def test_pickle_is_opt_in(root):
    with pytest.raises(TypeError, match="codec='pickle'"):
        codec_for(Opaque(1))
    with pytest.raises(TypeError):
        codec_for(np.array([Opaque(1)], dtype=object))
    rec = store("pickle", Opaque(2), codec="pickle")
    assert rec.meta["fields"]["value"]["codec"] == "pickle" and rec.value == Opaque(2)


def test_safetensors_state_dict(root):
    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(3, 2)
    rec = store("torch", module)
    assert rec.meta["fields"]["value"] == {"codec": "safetensors", "state_dict_of_module": True}
    state = rec.value
    assert set(state) == {"weight", "bias"}
    torch.testing.assert_close(state["weight"], module.weight.detach())
    plain = store("torch-dict", {"a": torch.arange(3)})
    torch.testing.assert_close(plain.value["a"], torch.arange(3))


class Rewrites(td.Record):
    __record_name__ = "Rewrites"

    def __init__(self):
        self["x"] = np.arange(3)
        self["x"] = {"now": "json"}
        self["y"] = {"first": 1}
        self["y"] = np.ones(2)


def test_overwrite_with_another_codec_removes_old_storage(root):
    rec = Rewrites.get()
    assert rec.x == {"now": "json"} and "x" not in zarr.open_group(rec.path / "data.zarr", mode="r")
    np.testing.assert_array_equal(rec.y, [1, 1])
    assert not (rec.path / "y.json").exists()


class Upper(Codec):
    name = "upper"

    def accepts(self, value):
        return isinstance(value, str)

    def write(self, directory, name, value, *, dims=None):
        (directory / f"{name}.txt").write_text(value.upper())
        return {}

    def read(self, directory, name, info, *, lazy=True):
        return (directory / f"{name}.txt").read_text()

    def delete(self, directory, name, info):
        (directory / f"{name}.txt").unlink(missing_ok=True)


def test_registered_codecs_take_precedence(root):
    from tensordir import codecs

    td.register_codec(Upper())
    try:
        with pytest.raises(ValueError, match="already registered"):
            td.register_codec(Upper())
        assert codec_for("abc").name == "upper"
        assert store("upper", "abc").value == "ABC"
        assert get_codec("json").accepts("abc")
    finally:
        codecs._CODECS.pop("upper")
    assert codec_for("abc").name == "json"


def test_records_pickle_without_data(root):
    rec = store("big", np.zeros(10_000))
    assert len(pickle.dumps(rec)) < 1000
