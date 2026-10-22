from typing import Annotated, Optional

import numpy as np
import pytest
import xarray as xr
from jaxtyping import Bool, Float, Int, Shaped

import tensordir as td
from tensordir import dims as dims_module
from tensordir.dims import dim_name, dims_for


@pytest.mark.parametrize(
    "size, dim",
    [
        ("n_cells", "cell"),
        ("n_timesteps", "timestep"),
        ("n_dims", "dim"),
        ("n_points", "point"),
        ("n_quantiles", "quantile"),
        ("n_vertices", "vertex"),
        ("n_radii", "radius"),
        ("n_classes", "class"),
        ("n_glass", "glass"),
        ("n_basis", "basis"),
        ("n_status", "status"),
        ("n_batches", "batch"),
        ("n_patches", "patch"),
        ("n_meshes", "mesh"),
        ("n_boxes", "box"),
        ("n_entries", "entry"),
        ("n_frequencies", "frequency"),
        ("n_ties", "tie"),
        ("n_cases", "case"),
        ("n_responses", "response"),
        ("n_caches", "cache"),
        ("n_s", "s"),
        ("height", "height"),
        ("rank", "rank"),
        ("x", "x"),
    ],
)
def test_default_rule(size, dim):
    assert dim_name(size) == dim


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (Float[np.ndarray, "n_points n_dims"], ("point", "dim")),
        (Float[np.ndarray, "pre=n_types post=n_types"], ("pre", "post")),
        (Float[np.ndarray, "dz=n_bins dx=n_bins"], ("dz", "dx")),
        (Float[np.ndarray, "#n_cells n_timesteps"], ("cell", "timestep")),
        (Float[np.ndarray, "_n_cells"], ("cell",)),
        (Float[np.ndarray, ""], ()),
        (Optional[Int[np.ndarray, "n_edges"]], ("edge",)),
        (Annotated[Float[np.ndarray, "n_bins+1=_"], td.Dims("bin_edge")], ("bin_edge",)),
        (Annotated[Shaped[np.ndarray, "*batch n_dims"], td.Dims("sample", "dim")], ("sample", "dim")),
        (np.ndarray, None),
        (int, None),
        (xr.DataArray, None),
    ],
)
def test_dims_for(annotation, expected):
    assert dims_for(annotation) == expected


@pytest.mark.parametrize(
    "annotation, message",
    [
        (Float[np.ndarray, "n_bins+1"], "symbolic"),
        (Float[np.ndarray, "n_bins+1=_"], "symbolic"),
        (Int[np.ndarray, "n_edges 2"], "literal"),
        (Float[np.ndarray, "*batch n_dims"], "variadic"),
        (Float[np.ndarray, "..."], "variadic"),
        (Float[np.ndarray, "_ n_dims"], "anonymous"),
        (Float[np.ndarray, "n_types n_types"], "repeated"),
    ],
)
def test_unnamed_dims_are_rejected(annotation, message):
    with pytest.raises(td.DimsError, match=message):
        dims_for(annotation)


def test_registry_and_overrides(monkeypatch):
    monkeypatch.setattr(dims_module, "_REGISTRY", {})
    assert dim_name("n_offsets") == "offset"
    td.register_dims(n_offsets="shift")
    assert dim_name("n_offsets") == "shift"
    td.register_dims({"n_cells": "neuron"})
    assert dims_for(Float[np.ndarray, "n_cells"]) == ("neuron",)
    assert dims_for(Float[np.ndarray, "n_cells"], {"n_cells": "unit"}) == ("unit",)


class Profile(td.Record):
    __record_name__ = "Profile"
    dims = {"n_types": "cell_type"}

    positions: Float[np.ndarray, "n_points n_dims"]
    types: Int[np.ndarray, "n_points"]
    matrix: Float[np.ndarray, "pre=n_types post=n_types"]
    edges: Annotated[Float[np.ndarray, "n_bins+1=_"], td.Dims("bin_edge")]
    counts: Int[np.ndarray, "n_types"]
    mask: Bool[np.ndarray, "n_points"]

    def __init__(self):
        self.positions = np.zeros((4, 3))
        self.types = np.array([0, 1, 1, 0])
        self.matrix = np.eye(2)
        self.edges = np.linspace(0, 1, 6)
        self.counts = np.array([2, 2])
        self.mask = np.array([True, False, True, True])


def test_record_dims_are_stored_and_readable_by_xarray(root):
    rec = Profile.get()
    fields = rec.meta["fields"]
    assert fields["positions"]["dims"] == ["point", "dim"]
    assert fields["matrix"]["dims"] == ["pre", "post"]
    assert fields["edges"]["dims"] == ["bin_edge"]
    assert fields["counts"]["dims"] == ["cell_type"]
    ds = xr.open_zarr(rec.path / "data.zarr", consolidated=False, chunks=None)
    assert ds["positions"].dims == ("point", "dim") and ds.sizes["point"] == 4
    assert ds["mask"].dtype == bool


class Variadic(td.Record):
    __record_name__ = "Variadic"

    x: Float[np.ndarray, "*batch n_dims"]

    def __init__(self):
        self.x = np.zeros((2, 3))


def test_variadic_field_without_dims_fails_the_build(root):
    with pytest.raises(td.BuildFailed, match="variadic"):
        Variadic.get()


class ExplicitWrite(td.Record):
    __record_name__ = "ExplicitWrite"

    x: Float[np.ndarray, "*batch n_dims"]

    def __init__(self):
        self.write("x", np.zeros((2, 3)), dims=("sample", "dim"))
        self.write("free", np.zeros(2), dims=("item",))


def test_explicit_dims_at_write(root):
    rec = ExplicitWrite.get()
    assert rec.meta["fields"]["x"]["dims"] == ["sample", "dim"]
    assert rec.to_xarray("free").dims == ("item",)


class WrongRank(td.Record):
    __record_name__ = "WrongRank"

    def __init__(self):
        self.write("x", np.zeros(3), dims=("a", "b"))


def test_dims_must_match_rank(root):
    with pytest.raises(td.BuildFailed, match="DimsError"):
        WrongRank.get()
