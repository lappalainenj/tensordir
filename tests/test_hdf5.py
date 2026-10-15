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

    class Spec(td.Spec):
        n_cells: int = 4
        n_frames: int = 3

    rates: Float[np.ndarray, "n_cells n_frames"]
    gains: Float[np.ndarray, "n_cells"]

    def build(self, spec):
        self.rates = np.arange(spec.n_cells * spec.n_frames, dtype=float).reshape(spec.n_cells, spec.n_frames)
        self.gains = np.ones(spec.n_cells)
        self.append("loss", [1.0, 0.5])  # append stays zarr
        self["summary"] = {"ok": True}  # other codecs unaffected


def test_record_codec_override(root):
    record = Responses.get()
    fields = record.meta["fields"]
    assert fields["rates"]["codec"] == "hdf5" and fields["rates"]["dims"] == ["cell", "frame"]
    assert fields["loss"]["codec"] == "ndarray" and fields["summary"]["codec"] == "json"
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

        def build(self, spec):
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

        def build(self, spec):
            self.write("a", np.zeros(2), codec="hdf5", dims=("x",))
            self.write("b", np.zeros(3), dims=("x",))

    with pytest.raises(td.BuildFailed, match="dimension 'x'"):
        Mismatch.get()
    with pytest.raises(td.BuildFailed):

        class Objects(td.Record):
            __record_name__ = "Hdf5Objects"

            def build(self, spec):
                self.write("a", np.array([object()]), codec="hdf5")

        Objects.get()
