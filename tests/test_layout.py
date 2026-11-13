"""Storage layouts (chunks, shards), consolidated metadata, cached groups and `Record.read`."""

import json
from typing import Annotated

import numpy as np
import pytest
import zarr
from zarr.abc.store import SuffixByteRequest
from zarr.storage import LocalStore

import tensordir as td
from tensordir import codecs
from tensordir.layout import APPEND_DEFAULT, DEFAULT, Layout

# ----- Layout.resolve ---------------------------------------------------------------------------


def test_byte_targets_become_rows_along_axis_0():
    spec = Layout(chunk_bytes=1200, shard_bytes=12_000).resolve((1000, 3), "f4")
    assert spec == {"chunks": (100, 3), "shards": (1000, 3)}  # 12 bytes per row; shard capped at the array
    spec = Layout(chunk_bytes=1200, shard_bytes=3_600).resolve((1000, 3), "f4")
    assert spec == {"chunks": (100, 3), "shards": (300, 3)}


def test_small_arrays_are_one_chunk_without_shards():
    assert DEFAULT.resolve((10,), "f8") == {"chunks": (10,), "shards": None}
    assert DEFAULT.resolve((0, 3), "f4") == {"chunks": (1, 3), "shards": None}
    assert DEFAULT.resolve((), "f4") == {"chunks": None, "shards": None}
    assert Layout().resolve((50,), "i1") == {"chunks": None, "shards": None}  # zarr decides


def test_large_arrays_get_bounded_shards():
    spec = DEFAULT.resolve((10**8,), "i4")
    assert spec["chunks"] == (32768,) and spec["shards"] == (32768 * 1024,)


def test_growing_arrays_keep_full_sizes():
    assert APPEND_DEFAULT.resolve((5, 2), "i8", growing=True) == {"chunks": (8192, 2), "shards": None}
    spec = Layout(chunk_bytes=64, shard_bytes=640).resolve((1,), "i8", growing=True)
    assert spec == {"chunks": (8,), "shards": (80,)}


def test_explicit_shapes_are_checked():
    assert Layout(chunks=(10, 2), shards=(20, 2)).resolve((40, 2), "i8") == {"chunks": (10, 2), "shards": (20, 2)}
    with pytest.raises(ValueError, match="multiple"):
        Layout(chunks=(10,), shards=(15,)).resolve((40,), "i8")
    with pytest.raises(ValueError, match="1-d"):
        Layout(chunks=(10, 2)).resolve((40,), "i8")
    with pytest.raises(ValueError, match="positive"):
        Layout(chunk_bytes=0)


# ----- records ----------------------------------------------------------------------------------


class Layered(td.Record):
    __record_name__ = "Layered"
    layouts = {"*": Layout(chunk_bytes=256, shard_bytes=4096), "special": Layout(chunk_bytes=64)}

    def __init__(self, /, *, n: int, layout: Annotated[Layout | None, td.field(hashed=False)] = None):
        self.write("plain", np.arange(n, dtype="i8"), layout=layout)
        self.write("special", np.arange(n, dtype="i8"))
        self.write("given", np.arange(n, dtype="i8"), layout=Layout(chunk_bytes=128, shard_bytes=1024))
        self.write("note", {"n": n})


def test_layout_order_argument_field_star(root):
    fields = Layered(n=1000).meta["fields"]
    assert fields["plain"]["layout"] == {"chunks": [32], "shards": [512]}  # "*"
    assert fields["special"]["layout"] == {"chunks": [8], "shards": None}  # field entry
    assert fields["given"]["layout"] == {"chunks": [16], "shards": [128]}  # argument
    assert "layout" not in fields["note"]


def test_layout_does_not_change_the_key(root):
    a = Layered.get(n=300)
    b = Layered.locate(n=300, layout=Layout(chunk_bytes=8))
    assert a.key == b.key == Layered.compute_key(n=300, layout=Layout(chunks=(7,)))
    assert a.meta["fields"]["plain"]["layout"]["chunks"] == [32]


def test_layout_for_a_codec_without_chunks_raises(root):
    class JsonLayout(td.Record):
        def __init__(self, /):
            self.write("note", {"a": 1}, layout=Layout(chunk_bytes=8))

    with pytest.raises(td.BuildFailed, match="stores no chunks"):
        JsonLayout.get()


def test_class_layouts_must_be_layouts():
    with pytest.raises(TypeError, match="must be a tensordir.Layout"):

        class Bad(td.Record):
            layouts = {"*": {"chunk_bytes": 8}}


DTYPES = ["int16", "uint8", "float16", "int32", "int64", "bool", "float32"]


class Typed(td.Record):
    __record_name__ = "LayoutTyped"

    def __init__(self, /, *, dtype: str, shape: tuple):
        rng = np.random.default_rng(0)
        self.write("x", (rng.random(shape) * 100).astype(dtype), layout=Layout(chunk_bytes=40, shard_bytes=200))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(103,), (37, 3), (0,), (0, 2)])
def test_sharded_round_trip(root, dtype, shape):
    record = Typed(dtype=dtype, shape=shape)
    expected = (np.random.default_rng(0).random(shape) * 100).astype(dtype)
    np.testing.assert_array_equal(record.load("x"), expected)
    np.testing.assert_array_equal(td.Record.at(record.path).load("x"), expected)


class Grows(td.Record):
    __record_name__ = "LayoutGrows"
    mutable = True


def test_append_keeps_the_layout_it_was_created_with(root):
    record = Grows.get()
    record.append("rows", np.arange(6).reshape(3, 2), layout=Layout(chunks=(2, 2), shards=(4, 2)))
    for i in range(5):
        record.append("rows", np.full((1, 2), 10 + i))
    info = record.meta["fields"]["rows"]
    assert info["layout"] == {"chunks": [2, 2], "shards": [4, 2]} and info["shape"] == [8, 2]
    np.testing.assert_array_equal(record.load("rows")[3:, 0], [10, 11, 12, 13, 14])
    record.append("log", [1.0])
    assert record.meta["fields"]["log"]["layout"] == {"chunks": [16384], "shards": None}


# ----- consolidated metadata and cached groups ---------------------------------------------------


class Three(td.Record):
    __record_name__ = "LayoutThree"

    def __init__(self, /, *, n: int):
        for name in ("a", "b", "c"):
            self.write(name, np.arange(n, dtype="i4"))


def _root_metadata(record):
    return json.loads((record.path / "data.zarr" / "zarr.json").read_text())


def test_finished_records_have_consolidated_metadata(root):
    record = Three(n=10)
    consolidated = _root_metadata(record)["consolidated_metadata"]["metadata"]
    assert set(consolidated) == {"a", "b", "c"}


def test_a_handle_opens_its_group_once(root, monkeypatch):
    record = Three(n=10)
    reopened = td.Record.at(record.path)
    calls = []
    original = codecs.open_store_group

    def counting(directory, *, immutable):
        calls.append(immutable)
        return original(directory, immutable=immutable)

    monkeypatch.setattr(td._record, "open_store_group", counting)
    for _ in range(3):
        for name in ("a", "b", "c"):
            reopened[name][:2]
        reopened.read({"a": slice(0, 3), "b": [1, 2]})
    assert calls == [True]


def test_records_without_consolidated_metadata_still_open(root):
    record = Three(n=10)
    path = record.path / "data.zarr" / "zarr.json"
    meta = json.loads(path.read_text())
    meta.pop("consolidated_metadata")
    path.write_text(json.dumps(meta))
    reopened = td.Record.at(record.path)
    np.testing.assert_array_equal(reopened.load("b"), np.arange(10))
    np.testing.assert_array_equal(reopened.read({"c": [(2, 4)]})["c"], [2, 3])


def test_a_member_newer_than_consolidated_metadata_is_found(tmp_path):
    group = zarr.open_group(tmp_path / "data.zarr", mode="w", zarr_format=3)
    group.create_array("old", shape=(2,), dtype="i4")[...] = [1, 2]
    codecs.consolidate(tmp_path)
    zarr.open_group(tmp_path / "data.zarr", mode="a", zarr_format=3, use_consolidated=False).create_array(
        "new", shape=(1,), dtype="i4"
    )[...] = [7]
    opened = codecs.open_store_group(tmp_path, immutable=True)
    assert codecs.group_member(opened, "new")[...].tolist() == [7]


class Workspace(td.Record):
    __record_name__ = "LayoutWorkspace"
    mutable = True


def test_workspace_edits_reconsolidate_and_refresh_handles(root):
    record = Workspace.get()
    reader = td.Record.at(record.path)
    record.write("x", np.arange(3))
    assert reader.load("x").tolist() == [0, 1, 2]
    writer = td.Record.at(record.path)
    writer.write("x", np.arange(5))
    writer.write("y", np.ones(2))
    assert reader["x"].shape == (5,) and reader.load("y").tolist() == [1.0, 1.0]
    assert set(_root_metadata(record)["consolidated_metadata"]["metadata"]) == {"x", "y"}


def _count_index_reads(monkeypatch):
    counts = {"index": 0}
    get, get_sync = LocalStore.get, LocalStore.get_sync

    async def counting_get(self, key, prototype, byte_range=None):
        counts["index"] += isinstance(byte_range, SuffixByteRequest)
        return await get(self, key, prototype, byte_range)

    def counting_get_sync(self, key, *, prototype=None, byte_range=None):
        counts["index"] += isinstance(byte_range, SuffixByteRequest)
        return get_sync(self, key, prototype=prototype, byte_range=byte_range)

    monkeypatch.setattr(LocalStore, "get", counting_get)
    monkeypatch.setattr(LocalStore, "get_sync", counting_get_sync)
    return counts


class Sharded(td.Record):
    __record_name__ = "LayoutSharded"

    def __init__(self, /, *, n: int):
        self.write("x", np.arange(n, dtype="i8"), layout=Layout(chunk_bytes=80, shard_bytes=800))


def test_shard_indexes_are_read_once_per_handle(root, monkeypatch):
    record = Sharded(n=1000)
    reopened = td.Record.at(record.path)
    counts = _count_index_reads(monkeypatch)
    reopened["x"][5:15]
    first = counts["index"]
    assert first >= 1
    reopened["x"][5:15]
    reopened.read({"x": [(5, 15), (20, 30)]})
    np.testing.assert_array_equal(reopened.read({"x": slice(990, 1000)})["x"], np.arange(990, 1000))
    assert counts["index"] == first + 1  # only the newly touched last shard


# ----- Record.read ------------------------------------------------------------------------------


class Readable(td.Record):
    __record_name__ = "LayoutReadable"

    def __init__(self, /, *, n: int):
        x = np.arange(n * 2, dtype="i4").reshape(n, 2)
        self.write("unsharded", x, layout=Layout(chunk_bytes=64))
        self.write("sharded", x, layout=Layout(chunk_bytes=64, shard_bytes=1024))
        self.write("h5", x, codec="hdf5", layout=Layout(chunk_bytes=64))
        self.write("note", {"a": 1})


ENGINES = ["zarr", pytest.param("tensorstore", marks=pytest.mark.skipif(
    __import__("importlib").util.find_spec("tensorstore") is None, reason="needs tensorstore"
))]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("field", ["unsharded", "sharded", "h5"])
@pytest.mark.parametrize("seed", range(3))
def test_read_equals_slicing(root, field, seed, engine, monkeypatch):
    monkeypatch.setenv("TENSORDIR_READ_ENGINE", engine)
    n = 500
    record = Readable(n=n)
    x = np.arange(n * 2, dtype="i4").reshape(n, 2)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, 20)
    ranges = [(int(a), int(min(n, a + rng.integers(0, 40)))) for a in starts]  # unsorted, overlapping, empty
    index = rng.integers(-n, n, 300)
    out = record.read({field: ranges})
    np.testing.assert_array_equal(out[field], np.concatenate([x[a:b] for a, b in ranges]))
    np.testing.assert_array_equal(record.read({field: index})[field], x[index])
    np.testing.assert_array_equal(record.read({field: index.tolist()})[field], x[index])
    np.testing.assert_array_equal(record.read({field: np.array(ranges)})[field], out[field])
    mask = rng.random(n) < 0.1
    np.testing.assert_array_equal(record.read({field: mask})[field], x[mask])
    np.testing.assert_array_equal(record.read({field: slice(-7, None)})[field], x[-7:])


def test_read_empty_requests(root):
    record = Readable(n=20)
    out = record.read({"sharded": [], "unsharded": np.array([], dtype=int), "h5": [(3, 3)]})
    for name, value in out.items():
        assert value.shape == (0, 2) and value.dtype == np.int32, name


def test_read_several_fields_in_one_call(root):
    record = Readable(n=100)
    out = record.read({"sharded": [(0, 2)], "unsharded": [5, 1], "h5": slice(98, 100)})
    assert out["sharded"].tolist() == [[0, 1], [2, 3]]
    assert out["unsharded"].tolist() == [[10, 11], [2, 3]]
    assert out["h5"].tolist() == [[196, 197], [198, 199]]


def test_read_errors(root):
    record = Readable(n=10)
    with pytest.raises(IndexError):
        record.read({"sharded": [10]})
    with pytest.raises(ValueError, match="step"):
        record.read({"sharded": slice(0, 10, 2)})
    with pytest.raises(TypeError, match="array field"):
        record.read({"note": slice(0, 1)})
    with pytest.raises(KeyError):
        record.read({"missing": slice(0, 1)})


@pytest.mark.parametrize("engine", ENGINES)
def test_read_engines_follow_workspace_edits(root, engine):
    record = Workspace.get()
    record.write("x", np.arange(4))
    reader = td.Record.at(record.path)
    assert reader.read({"x": [1, 2]}, engine=engine)["x"].tolist() == [1, 2]
    td.Record.at(record.path).write("x", np.arange(10, 20))
    assert reader.read({"x": [1, 9]}, engine=engine)["x"].tolist() == [11, 19]


@pytest.mark.skipif(__import__("importlib").util.find_spec("tensorstore") is None, reason="needs tensorstore")
def test_tensorstore_falls_back_to_zarr_for_unsupported_fields(root):
    class Strings(td.Record):
        __record_name__ = "LayoutStrings"

        def __init__(self, /):
            self.write("s", np.array(["a", "bc", "def"]))
            self.write("i", np.arange(3))

    out = Strings.get().read({"s": [2, 0], "i": slice(1, None)}, engine="tensorstore")
    assert out["s"].tolist() == ["def", "a"] and out["i"].tolist() == [1, 2]


def test_unknown_engine_raises(root):
    with pytest.raises(ValueError, match="engine"):
        Readable(n=5).read({"sharded": slice(0, 1)}, engine="numpy")
