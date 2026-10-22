"""The opt-in HDF5 codec and per-record codec overrides."""

import numpy as np
import pytest
from jaxtyping import Float

import tensordir as td
from tensordir import legacy

h5py = pytest.importorskip("h5py")


class Responses(td.Record):
    __record_name__ = "Hdf5Responses"
    codecs = {"ndarray": "hdf5"}

    rates: Float[np.ndarray, "n_cells n_frames"]
    gains: Float[np.ndarray, "n_cells"]

    def __init__(self, /, *, n_cells: int = 4, n_frames: int = 3):
        self.rates = np.arange(n_cells * n_frames, dtype=float).reshape(n_cells, n_frames)
        self.gains = np.ones(n_cells)
        self.append("loss", [1.0, 0.5])  # appends follow the record's codecs too
        self.append("loss", [0.25])
        self["summary"] = {"ok": True}  # other codecs unaffected


def test_record_codec_override(root):
    record = Responses.get()
    fields = record.meta["fields"]
    assert fields["rates"]["codec"] == "hdf5" and fields["rates"]["dims"] == ["cell", "frame"]
    assert fields["loss"]["codec"] == "hdf5" and fields["loss"]["append"] and fields["summary"]["codec"] == "json"
    np.testing.assert_array_equal(record.loss, [1.0, 0.5, 0.25])
    assert (record.path / "rates.h5").is_file()
    rates = record.rates
    assert isinstance(rates, np.ndarray) and rates.shape == (4, 3)
    lazy = record["rates"]
    assert isinstance(lazy, h5py.Dataset) and lazy[1, 2] == 5.0
    assert record.to_xarray("rates").dims == ("cell", "frame")
    with h5py.File(record.path / "rates.h5", "r", locking=False) as f:  # what other h5 tools see
        assert [d.label for d in f["data"].dims] == ["cell", "frame"]
    np.testing.assert_array_equal(legacy._read_h5(record.path / "gains.h5"), np.ones(4))  # datamate layout


def test_per_field_choice_and_size_checks(root):
    class Mixed(td.Record):
        __record_name__ = "Hdf5Mixed"
        mutable = True

        def __init__(self):
            self.write("a", np.zeros((2, 3)), codec="hdf5", dims=("x", "y"))
            self.write("b", np.zeros(2), dims=("x",))

    record = Mixed.get()
    assert record.meta["fields"]["a"]["codec"] == "hdf5" and record.meta["fields"]["b"]["codec"] == "ndarray"
    record.a = np.ones(3)  # workspace edit: the default codec replaces the HDF5 file
    assert not (record.path / "a.h5").exists() and record.meta["fields"]["a"]["codec"] == "ndarray"
    record.write("c", np.arange(3), codec="hdf5")
    del record.c
    assert not (record.path / "c.h5").exists()

    class Mismatch(td.Record):
        __record_name__ = "Hdf5Mismatch"

        def __init__(self):
            self.write("a", np.zeros(2), codec="hdf5", dims=("x",))
            self.write("b", np.zeros(3), dims=("x",))

    with pytest.raises(td.BuildFailed, match="dimension 'x'"):
        Mismatch.get()
    with pytest.raises(td.BuildFailed):

        class Objects(td.Record):
            __record_name__ = "Hdf5Objects"

            def __init__(self):
                self.write("a", np.array([object()]), codec="hdf5")

        Objects.get()


class Log(td.Record):
    __record_name__ = "Hdf5Log"
    mutable = True
    codecs = {"ndarray": "hdf5"}

    loss: Float[np.ndarray, "n_steps"]

    def __init__(self, /, *, run: int = 0):
        pass


def test_append_extends_a_resizable_dataset(root):
    record = Log.get()
    record.append("loss", [1.0, 0.5], chunk=4)
    record.append("loss", np.float64(0.25))  # a scalar is one row
    reopened = td.Record.at(record.path)  # another handle, after reopening
    reopened.append("loss", [0.125, 0.0625])
    np.testing.assert_array_equal(record.load("loss"), [1.0, 0.5, 0.25, 0.125, 0.0625])
    info = reopened.meta["fields"]["loss"]
    assert info["codec"] == "hdf5" and info["shape"] == [5] and info["dims"] == ["step"] and info["append"]
    with h5py.File(record.path / "loss.h5", "r", locking=False) as f:
        data = f["data"]
        assert data.maxshape == (None,) and data.chunks == (4,) and data.dims[0].label == "step"
    record.loss = np.array([9.0])  # overwriting keeps it appendable, as with zarr
    record.append("loss", [8])  # ints are cast to the stored dtype
    np.testing.assert_array_equal(record.loss, [9.0, 8.0])
    assert record.meta["fields"]["loss"]["codec"] == "hdf5"
    record.append("grid", np.zeros((2, 3)), codec="ndarray")  # per call: zarr
    record.append("grid", np.ones((1, 3)))  # later appends keep the field's codec
    assert record.meta["fields"]["grid"]["codec"] == "ndarray" and record.load("grid").shape == (3, 3)


@pytest.mark.parametrize("codec", ["ndarray", "hdf5"])
def test_append_errors_match_between_codecs(root, codec):
    record = Log.get(run=1)
    record.append("rows", np.zeros((2, 3)), codec=codec)
    with pytest.raises(ValueError, match="cannot append rows of shape"):
        record.append("rows", np.zeros((1, 4)))
    with pytest.raises(td.DimsError):
        record.append("fresh", np.zeros((2, 3)), dims=("x",), codec=codec)
    record.write("fixed", np.zeros(3), codec=codec)
    with pytest.raises(ValueError, match="append\\(\\) needs a field created by append\\(\\)"):
        record.append("fixed", [1.0])
    other = "hdf5" if codec == "ndarray" else "ndarray"
    with pytest.raises(ValueError, match="appended with codec"):
        record.append("rows", np.zeros((1, 3)), codec=other)
    with pytest.raises(TypeError, match="cannot append"):
        record.append("notes", [1], codec="json")
    assert record.load("rows").shape == (2, 3) and "fresh" not in record and "notes" not in record
