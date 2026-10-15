import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from tensordir.legacy import DatamateDirectory, open_datamate


def write_h5(path, value, extendable=False):
    with h5py.File(path, "w", libver="latest") as f:
        if extendable:
            f.create_dataset("data", data=value, maxshape=(None, *np.shape(value)[1:]), chunks=True)
        else:
            f["data"] = value


@pytest.fixture
def legacy_dir(tmp_path):
    """A directory laid out as datamate 0.2-1.0 writes it (meta yaml, one h5 per array, csv)."""
    d = tmp_path / "NetworkDir_0000"
    d.mkdir()
    (d / "_meta.yaml").write_text(
        yaml.safe_dump({"config": {"type": "NetworkDir", "lr": 0.1, "layers": [3, 4], "nested": {"a": 1}}, "status": "done", "run_complete": True})
    )
    write_h5(d / "loss.h5", np.array([1.0, 0.5, 0.25]), extendable=True)
    write_h5(d / "weights.h5", np.arange(6.0).reshape(2, 3))
    write_h5(d / "names.h5", np.array(["exc", "inh"], dtype="S"))
    write_h5(d / "scalar.h5", np.float64(2.5))
    write_h5(d / "a.b.h5", np.array([1, 2]))
    pd.DataFrame({"x": [1, 2], "y": ["u", "v"]}).to_csv(d / "table.csv", index=False)
    (d / "notes.txt").write_text("hello")
    sub = d / "chkpts"
    sub.mkdir()
    write_h5(sub / "epoch.h5", np.array([0, 1]))
    (d / ".hidden").write_text("x")
    return d


def test_meta_and_entries(legacy_dir):
    d = open_datamate(legacy_dir)
    assert d.status == "done" and d.type == "NetworkDir"
    assert d.config["layers"] == [3, 4] and d.meta["run_complete"] is True
    assert sorted(d) == ["a.b", "chkpts", "loss", "names", "notes.txt", "scalar", "table", "weights"]
    assert len(d) == 8 and "loss" in d


def test_values(legacy_dir):
    d = open_datamate(legacy_dir)
    np.testing.assert_array_equal(d["loss"], [1.0, 0.5, 0.25])
    np.testing.assert_array_equal(d.weights, np.arange(6.0).reshape(2, 3))
    assert d["names"].dtype.kind == "U" and list(d["names"]) == ["exc", "inh"]
    assert d["scalar"] == 2.5
    np.testing.assert_array_equal(d.a__b, [1, 2])
    pd.testing.assert_frame_equal(d["table"], pd.DataFrame({"x": [1, 2], "y": ["u", "v"]}))
    assert d["notes.txt"].read_text() == "hello"
    sub = d.chkpts
    assert isinstance(sub, DatamateDirectory) and sub.config is None and sub.status == "done"
    np.testing.assert_array_equal(sub["epoch"], [0, 1])
    with pytest.raises(KeyError):
        d["missing"]
    with pytest.raises(AttributeError):
        d.missing


def test_legacy_spec_key_and_running_status(tmp_path):
    (tmp_path / "_meta.yaml").write_text("spec:\n  n: 3\nstatus: running\n")
    d = open_datamate(tmp_path)
    assert d.config == {"n": 3} and d.status == "running"


def test_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_datamate(tmp_path / "nope")


def test_reads_directories_written_by_datamate(tmp_path):
    """Run with `uv run --with datamate ...` to check against the real writer."""
    datamate = pytest.importorskip("datamate")

    class Run(datamate.Directory):
        def __init__(self, n: int = 3):
            self.x = np.arange(n)
            self.names = np.array(["a", "b"], dtype="S")
            self.df = pd.DataFrame({"a": [1, 2]})

    with datamate.set_root_context(tmp_path):
        run = Run(n=4)
        run.extend("log", np.array([1.0, 2.0]))
        run.extend("log", np.array([3.0]))
    d = open_datamate(run.path)
    assert d.config["n"] == 4 and d.status == "done"
    np.testing.assert_array_equal(d.x, np.arange(4))
    np.testing.assert_array_equal(d.log, [1.0, 2.0, 3.0])
    assert list(d.names) == ["a", "b"]
    pd.testing.assert_frame_equal(d.df, pd.DataFrame({"a": [1, 2]}))
