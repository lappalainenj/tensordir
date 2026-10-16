import json
import pickle
import threading

import numpy as np
import pytest
import zarr
from jaxtyping import Float, Int

import tensordir as td
from _records import Square


def test_get_builds_once_then_reopens(root, monkeypatch):
    spec = Square.Spec(n=4)
    assert Square.locate(spec).status == "absent"
    assert Square.lookup(spec) is None
    rec = Square.get(spec)
    assert rec.status == "done"
    np.testing.assert_array_equal(rec.values, [0, 1, 4, 9])

    def boom(self, spec):
        raise AssertionError("rebuilt")

    monkeypatch.setattr(Square, "build", boom)
    again = Square.get({"n": 4})
    assert again == rec and again.path == rec.path
    assert Square.lookup(spec) == rec


def test_layout_and_metadata(root):
    rec = Square.get(Square.Spec(n=3, scale=2.0))
    assert rec.path == root / "Square" / rec.key[:16]
    meta = json.loads((rec.path / "_record.json").read_text())
    assert meta["key"] == rec.key and len(rec.key) == 64
    assert meta["type"] == "Square" and meta["status"] == "done" and meta["schema_version"] == 0
    assert meta["spec"] == {"n": 3, "scale": 2.0, "device": "cpu"}
    assert meta["fields"]["values"] == {"codec": "ndarray", "dims": ["point"], "shape": [3], "dtype": "<f8"}
    assert meta["fields"]["summary"]["codec"] == "json"
    prov = meta["provenance"]
    assert prov["pid"] > 0 and prov["host"] and "numpy" in prov["packages"]
    assert prov["git"] is None or set(prov["git"]) == {"repo", "commit", "dirty"}
    assert meta["duration_s"] >= 0 and meta["finished"] >= meta["created"]
    assert not list((root / ".tensordir" / "tmp").iterdir())
    assert rec.keys() == ["summary", "values"] and "values" in rec


def test_unhashed_fields_share_the_record(root):
    a = Square.get(Square.Spec(n=2, device="cpu"))
    b = Square.get(Square.Spec(n=2, device="cuda"))
    assert a.path == b.path
    assert b.spec.device == "cuda" and b.meta["spec"]["device"] == "cpu"


def test_lazy_and_eager_reads(root):
    rec = Square.get({"n": 5})
    handle = rec["values"]
    assert isinstance(handle, zarr.Array) and handle.metadata.dimension_names == ("point",)
    assert isinstance(rec.values, np.ndarray) and isinstance(rec.load("values"), np.ndarray)
    assert rec.summary == {"n": 5}
    assert rec.to_xarray("values").dims == ("point",)
    with pytest.raises(KeyError):
        rec["missing"]
    with pytest.raises(AttributeError):
        rec.missing


def test_writes_outside_build_are_rejected(root):
    rec = Square.get({"n": 2})
    with pytest.raises(td.ReadOnlyError):
        rec["x"] = 1
    with pytest.raises(td.ReadOnlyError):
        rec.values = np.zeros(2)


class Flaky(td.Record):
    __record_name__ = "Flaky"
    calls = 0
    fail = True

    def build(self, spec):
        type(self).calls += 1
        self["partial"] = 1
        if type(self).fail:
            raise ValueError("bad input")
        self["ok"] = True


def test_failure_is_recorded_and_retried_on_request(root):
    Flaky.calls, Flaky.fail = 0, True
    with pytest.raises(td.BuildFailed, match="ValueError: bad input") as info:
        Flaky.get()
    assert isinstance(info.value.__cause__, ValueError)
    rec = Flaky.locate()
    assert rec.status == "failed"
    assert "Traceback" in rec.error["traceback"] and rec.keys() == []
    with pytest.raises(td.BuildFailed):
        Flaky.get()
    assert Flaky.calls == 1
    Flaky.fail = False
    rec = Flaky.get(retry_failed=True)
    assert rec.status == "done" and rec.ok is True and Flaky.calls == 2 and rec.error is None


class Interrupted(td.Record):
    __record_name__ = "Interrupted"

    def build(self, spec):
        self["x"] = 1
        raise KeyboardInterrupt


def test_interrupt_leaves_no_record(root):
    with pytest.raises(KeyboardInterrupt):
        Interrupted.get()
    assert Interrupted.locate().status == "absent"
    assert not list((root / ".tensordir" / "tmp").iterdir())


def test_hand_named_directories(tmp_path):
    path = tmp_path / "my-run"
    rec = Square.at(path, Square.Spec(n=3))
    assert rec.path == path and rec.status == "done"
    reopened = td.Record.at(path)
    assert type(reopened) is Square and reopened.key == rec.key
    np.testing.assert_array_equal(reopened.values, [0, 1, 4])
    assert Square.at(path, {"n": 3}) == rec
    with pytest.raises(td.KeyCollision):
        Square.at(path, {"n": 4})
    with pytest.raises(FileNotFoundError):
        td.Record.at(tmp_path / "nothing")


class Downstream(td.Record):
    __record_name__ = "Downstream"

    class Spec(td.Spec):
        offset: float = 0.0

    total: float

    def build(self, spec, source, weights):
        self["total"] = float(source.values.sum() + weights.sum() + spec.offset)


def test_inputs_enter_the_key(root):
    source = Square.get({"n": 3})
    w = np.ones(2)
    rec = Downstream.get(source=source, weights=w)
    assert rec.total == 0 + 1 + 4 + 2
    assert rec.meta["inputs"]["source"] == {"__record__": "Square", "key": source.key}
    assert rec.meta["inputs"]["weights"] == {"__digest__": td.Digest.of(w).sha256}
    assert Downstream.lookup(source=source, weights=w * 2) is None
    assert Downstream.lookup(source=Square.get({"n": 4}), weights=w) is None
    assert Downstream.lookup(source=source, weights=w.copy()) == rec


def test_schema_version_changes_keys(root):
    class A(td.Record):
        __record_name__ = "A"

    class B(td.Record):
        __record_name__ = "A"
        schema_version = 1

    assert A.compute_key() != B.compute_key()


def test_type_id_defaults_to_module_qualname():
    class Local(td.Record):
        pass

    assert Local.type_id() == f"{__name__}:test_type_id_defaults_to_module_qualname.<locals>.Local"
    assert Local.dirname() == "test_type_id_defaults_to_module_qualname._locals_.Local"
    assert Square.type_id() == "Square"


def test_pickle_reopens_by_identity(root):
    rec = Square.get({"n": 3})
    back = pickle.loads(pickle.dumps(rec))
    assert back == rec and type(back) is Square and back.status == "done"
    np.testing.assert_array_equal(back.values, rec.values)


def test_root_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("TENSORDIR_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert Square.locate({"n": 1}).path.parent.parent == tmp_path  # the working directory
    monkeypatch.setenv("TENSORDIR_ROOT", str(tmp_path / "env"))
    assert Square.locate({"n": 1}).path.parent.parent == tmp_path / "env"
    td.set_root(tmp_path / "default")
    try:
        assert td.get_root() == tmp_path / "default"
        with td.use_root(tmp_path / "ctx"):
            assert Square.locate({"n": 1}).path.parent.parent == tmp_path / "ctx"
        assert Square.locate({"n": 1}, root=tmp_path / "arg").path.parent.parent == tmp_path / "arg"
    finally:
        td.set_root(None)


def test_golden_record_key():
    key = Square.compute_key({"n": 3, "scale": 0.5})
    assert key == GOLDEN_SQUARE
    assert Downstream.compute_key(source=None, weights=np.arange(4, dtype="<f8")) == GOLDEN_DOWNSTREAM


# sha256(b'{"inputs":{},"schema_version":0,"spec":{"n":3,"scale":0.5},"type":"Square"}')
GOLDEN_SQUARE = "c50bf504a8e3912e0b13741d84245a9eedbb048f0ef657c0872d6c73fd206f9b"
GOLDEN_DOWNSTREAM = "fd41344e64ecb991831c17d0c05069dc988282cc6670ef9ba8caaf3506728fc8"


class Typed(td.Record):
    __record_name__ = "Typed"

    class Spec(td.Spec):
        n_a: int = 2
        n_b: int = 2
        dtype: str = "float64"

    x: Float[np.ndarray, "n_cells n_dims"]
    y: Int[np.ndarray, "n_cells"]

    def build(self, spec):
        self.x = np.zeros((spec.n_a, 3), dtype=spec.dtype)
        self.y = np.arange(spec.n_b)


def test_typed_fields_are_checked(root):
    rec = Typed.get()
    assert rec.meta["fields"]["x"]["dims"] == ["cell", "dim"]
    with pytest.raises(td.BuildFailed, match="DimsError: dimension 'cell'"):
        Typed.get({"n_b": 3})
    with pytest.raises(td.BuildFailed, match="TypeError"):
        Typed.get({"dtype": "int64"})


def test_invalid_field_names(root):
    class Bad(td.Record):
        __record_name__ = "Bad"

        class Spec(td.Spec):
            name: str

        def build(self, spec):
            self[spec.name] = 1

    for name in ["../x", "_hidden", ".dot", "a/b", "x.lock"]:
        with pytest.raises(td.BuildFailed, match="invalid field name"):
            Bad.get({"name": name})


class Training(td.Record):
    __record_name__ = "Training"
    in_place = True
    started = threading.Event()
    proceed = threading.Event()

    loss: Float[np.ndarray, "n_iterations"]

    def build(self, spec):
        for i in range(3):
            self.append("loss", np.array([1.0 / (i + 1)]))
            if i == 1:
                type(self).started.set()
                type(self).proceed.wait(10)
        self["final"] = 0.5


def test_in_place_builds_expose_partial_logs(root):
    result = {}
    # context variables do not propagate into new threads, so pass the root explicitly
    thread = threading.Thread(target=lambda: result.setdefault("rec", Training.get(root=root)))
    thread.start()
    assert Training.started.wait(10)
    reader = Training.locate()
    assert reader.status == "running"
    np.testing.assert_allclose(reader["loss"][...], [1.0, 0.5])
    assert reader.meta["fields"]["loss"]["dims"] == ["iteration"]
    Training.proceed.set()
    thread.join(10)
    rec = result["rec"]
    assert rec.status == "done" and rec.loss.shape == (3,) and rec.final == 0.5


def test_delete(root):
    rec = Square.get({"n": 2})
    rec.delete()
    assert rec.status == "absent" and not rec.path.exists()
