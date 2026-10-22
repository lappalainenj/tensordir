import hashlib
import json
import pickle
import threading
from typing import Annotated

import numpy as np
import pytest
import zarr
from jaxtyping import Float, Int

import tensordir as td
from _records import Square


def test_get_builds_once_then_reopens(root, monkeypatch):
    assert Square.locate(n=4).status == "absent"
    assert Square.lookup(n=4) is None
    rec = Square(n=4)  # calling the type is get()
    assert rec.status == "done"
    np.testing.assert_array_equal(rec.values, [0, 1, 4, 9])

    def boom(self, **settings):
        raise AssertionError("rebuilt")

    monkeypatch.setattr(Square, "__init__", boom)
    again = Square.get(n=4)
    assert again == rec and again.path == rec.path
    assert Square.lookup(**Square.Spec(n=4)) == rec  # a spec passes its settings with **


def test_init_signature_is_the_interface():
    class Fit(td.Record):
        def __init__(
            self,
            points: np.ndarray,
            weights: np.ndarray | None = None,
            /,
            *,
            k: int,
            lr: float = 0.1,
            device: Annotated[str, td.field(hashed=False)] = "cpu",
            log: Annotated[list | None, td.transient] = None,
        ):
            pass

    assert [f for f in Fit.Spec.__dataclass_fields__] == ["k", "lr", "device"]
    assert Fit._interface.inputs == ("points", "weights") and Fit._interface.transients == ("log",)
    x = np.ones(3)
    assert Fit.compute_key(x, k=2) == Fit.compute_key(x, None, k=2, lr=0.1, device="gpu", log=[])
    assert Fit.compute_key(x, k=2) != Fit.compute_key(x, x, k=2)
    with pytest.raises(TypeError):
        Fit.compute_key(points=x, k=2)  # inputs are positional-only
    with pytest.raises(TypeError):
        Fit.compute_key(x, 2)  # settings are keyword-only

    for bad in ("def __init__(self, x): pass", "def __init__(self, *, root=None): pass",
                "def __init__(self, **kwargs): pass"):
        namespace = {}
        exec(bad, namespace)
        with pytest.raises(TypeError):
            type("Bad", (td.Record,), {"__init__": namespace["__init__"]})
    with pytest.raises(TypeError, match="Spec is generated"):
        type("Declared", (td.Record,), {"Spec": type("Spec", (td.Spec,), {})})


def test_renaming_an_input_keeps_the_key():
    class Before(td.Record):
        __record_name__ = "Renamed"

        def __init__(self, points, /, *, k: int = 1):
            pass

    class After(td.Record):
        __record_name__ = "Renamed"

        def __init__(self, xy, /, *, k: int = 1):
            pass

    assert Before.compute_key([1, 2]) == After.compute_key([1, 2])


def test_layout_and_metadata(root):
    rec = Square(n=3, scale=2.0)
    assert rec.path == root / "Square" / rec.key[:16]
    meta = json.loads((rec.path / "_meta.json").read_text())
    assert meta["key"] == rec.key and len(rec.key) == 64
    assert meta["type"] == "Square" and meta["status"] == "done" and meta["schema_version"] == 0
    assert meta["spec"] == {"n": 3, "scale": 2.0, "device": "cpu"}
    assert meta["fields"]["values"] == {"codec": "ndarray", "dims": ["point"], "shape": [3], "dtype": "<f8"}
    assert meta["fields"]["summary"]["codec"] == "json"
    prov = meta["provenance"]
    assert prov["pid"] > 0 and prov["host"] and "numpy" in prov["packages"]
    assert prov["git"] is None or set(prov["git"]) == {"repo", "commit", "dirty"}
    assert prov["tensordir_git"] is None or set(prov["tensordir_git"]) == {"repo", "commit", "dirty"}
    assert meta["duration_s"] >= 0 and meta["finished"] >= meta["created"]
    assert not list((root / ".tensordir" / "tmp").iterdir())
    assert rec.keys() == ["summary", "values"] and "values" in rec


def test_builds_classes_defined_in_a_notebook(root, monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))  # a kernel's __main__ has no file

    class Notebook(td.Record):
        __record_name__ = "Notebook"
        __module__ = "__main__"

        def __init__(self):
            self.x = np.arange(3)

    rec = Notebook()
    assert rec.status == "done" and rec.meta["provenance"]["git"] is None


def test_unhashed_fields_share_the_record(root):
    a = Square(n=2, device="cpu")
    b = Square(n=2, device="cuda")
    assert a.path == b.path
    assert b.spec.device == "cuda" and b.meta["spec"]["device"] == "cpu"


def test_lazy_and_eager_reads(root):
    rec = Square(n=5)
    handle = rec["values"]
    assert isinstance(handle, zarr.Array) and handle.metadata.dimension_names == ("point",)
    assert isinstance(rec.values, np.ndarray) and isinstance(rec.load("values"), np.ndarray)
    assert rec.summary == {"n": 5}
    assert rec.to_xarray("values").dims == ("point",)
    with pytest.raises(KeyError):
        rec["missing"]
    with pytest.raises(AttributeError):
        rec.missing


def test_writes_outside_the_build_are_rejected(root):
    rec = Square(n=2)
    with pytest.raises(td.ReadOnlyError):
        rec["x"] = 1
    with pytest.raises(td.ReadOnlyError):
        rec.values = np.zeros(2)


class Flaky(td.Record):
    __record_name__ = "Flaky"
    calls = 0
    fail = True

    def __init__(self):
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

    def __init__(self):
        self["x"] = 1
        raise KeyboardInterrupt


def test_interrupt_leaves_no_record(root):
    with pytest.raises(KeyboardInterrupt):
        Interrupted.get()
    assert Interrupted.locate().status == "absent"
    assert not list((root / ".tensordir" / "tmp").iterdir())


def test_hand_named_directories(tmp_path):
    path = tmp_path / "my-run"
    rec = Square.at(path, n=3)
    assert rec.path == path and rec.status == "done"
    reopened = td.Record.at(path)
    assert type(reopened) is Square and reopened.key == rec.key
    np.testing.assert_array_equal(reopened.values, [0, 1, 4])
    assert Square.at(path, n=3) == rec
    with pytest.raises(td.KeyCollision):
        Square.at(path, n=4)
    with pytest.raises(FileNotFoundError):
        td.Record.at(tmp_path / "nothing")


def test_at_on_a_managed_path_shares_the_keyed_lease(root):
    rec = Square.locate(n=3)
    assert Square.at(Square(n=3).path, n=3)._lock == rec._lock  # one lease per directory


class Downstream(td.Record):
    __record_name__ = "Downstream"

    total: float

    def __init__(self, source, weights, /, *, offset: float = 0.0):
        self["total"] = float(source.values.sum() + weights.sum() + offset)


def test_inputs_enter_the_key(root):
    source = Square(n=3)
    w = np.ones(2)
    rec = Downstream(source, w)
    assert rec.total == 0 + 1 + 4 + 2
    assert rec.meta["inputs"][0] == {"__record__": "Square", "key": source.key}
    assert rec.meta["inputs"][1] == {"__digest__": td.Digest.of(w).sha256}
    assert Downstream.lookup(source, w * 2) is None
    assert Downstream.lookup(Square(n=4), w) is None
    assert Downstream.lookup(source, w.copy()) == rec


def test_schema_version_changes_keys(root):
    class A(td.Record):
        __record_name__ = "A"

    class B(td.Record):
        __record_name__ = "A"
        schema_version = 1

    assert A.compute_key() != B.compute_key()


def test_type_id_defaults_to_qualname():
    class Local(td.Record):
        pass

    assert Local.type_id() == "test_type_id_defaults_to_qualname.<locals>.Local"
    assert Local.dirname() == "test_type_id_defaults_to_qualname._locals_.Local"
    assert Square.type_id() == "Square"


def test_implicit_type_ids_collide_across_modules():
    first = type("Twin", (td.Record,), {"__module__": "pkg.a", "__qualname__": "Twin"})
    with pytest.raises(TypeError, match="set __record_name__"):
        type("Twin", (td.Record,), {"__module__": "pkg.b", "__qualname__": "Twin"})
    again = type("Twin", (td.Record,), {"__module__": "pkg.a", "__qualname__": "Twin"})  # a redefinition
    named = type("Twin", (td.Record,), {"__module__": "pkg.c", "__qualname__": "Twin", "__record_name__": "Twin"})
    assert first.type_id() == again.type_id() == named.type_id() == "Twin"


def test_pickle_reopens_by_identity(root):
    rec = Square(n=3)
    back = pickle.loads(pickle.dumps(rec))
    assert back == rec and type(back) is Square and back.status == "done"
    np.testing.assert_array_equal(back.values, rec.values)


def test_root_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("TENSORDIR_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert Square.locate(n=1).path.parent.parent == tmp_path  # the working directory
    monkeypatch.setenv("TENSORDIR_ROOT", str(tmp_path / "env"))
    assert Square.locate(n=1).path.parent.parent == tmp_path / "env"
    td.set_root(tmp_path / "default")
    try:
        assert td.get_root() == tmp_path / "default"
        with td.use_root(tmp_path / "ctx"):
            assert Square.locate(n=1).path.parent.parent == tmp_path / "ctx"
        assert Square.locate(n=1, root=tmp_path / "arg").path.parent.parent == tmp_path / "arg"
    finally:
        td.set_root(None)


def test_golden_record_key():
    payload = b'{"inputs":[],"schema_version":0,"spec":{"n":3,"scale":0.5},"type":"Square"}'
    assert GOLDEN_SQUARE == hashlib.sha256(payload).hexdigest()
    assert Square.compute_key(n=3, scale=0.5) == GOLDEN_SQUARE
    assert Downstream.compute_key(None, np.arange(4, dtype="<f8")) == GOLDEN_DOWNSTREAM


GOLDEN_SQUARE = "fd11f33c10a909bffc78c1ba9022781923e3b12cd661386b0daa2097326367c7"
GOLDEN_DOWNSTREAM = "1024655e4800dbe1fec680f3eb9598ae199b02a9eccef31fdc323bffed327465"


class Typed(td.Record):
    __record_name__ = "Typed"

    x: Float[np.ndarray, "n_cells n_dims"]
    y: Int[np.ndarray, "n_cells"]

    def __init__(self, /, *, n_a: int = 2, n_b: int = 2, dtype: str = "float64"):
        self.x = np.zeros((n_a, 3), dtype=dtype)
        self.y = np.arange(n_b)


def test_typed_fields_are_checked(root):
    rec = Typed()
    assert rec.meta["fields"]["x"]["dims"] == ["cell", "dim"]
    with pytest.raises(td.BuildFailed, match="DimsError: dimension 'cell'"):
        Typed(n_b=3)
    with pytest.raises(td.BuildFailed, match="TypeError"):
        Typed(dtype="int64")


def test_invalid_field_names(root):
    class Bad(td.Record):
        __record_name__ = "Bad"

        def __init__(self, /, *, name: str):
            self[name] = 1

    for name in ["../x", "_hidden", ".dot", "a/b", "x.lock"]:
        with pytest.raises(td.BuildFailed, match="invalid field name"):
            Bad(name=name)


class Training(td.Record):
    __record_name__ = "Training"
    in_place = True
    started = threading.Event()
    proceed = threading.Event()

    loss: Float[np.ndarray, "n_iterations"]

    def __init__(self):
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
    rec = Square(n=2)
    rec.delete()
    assert rec.status == "absent" and not rec.path.exists()
