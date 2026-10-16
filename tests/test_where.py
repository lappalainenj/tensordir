import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import tensordir as td
from tensordir._where import mask, matches


@pytest.mark.parametrize(
    ("value", "condition", "expected"),
    [
        (0.1, 0.1, True),
        (0.1 + 0.2, 0.3, True),  # relative tolerance 1e-9
        (1e-9, 2e-9, False),  # no absolute tolerance: small learning rates stay distinct
        (3, 3.0, True),
        ("relu", "relu", True),
        ("relu", "tanh", False),
        (0.05, ">0.01, <0.1", True),
        (0.5, ">0.01, <0.1", False),
        (2, ">=2", True),
        (2, "!=2", False),
        ("b", ">a", True),
        ("relu", ">1", False),  # a string never compares with a number
        ("a,b", "a,b", True),  # commas without operators: plain equality
        ("T4a", ["T4a", "T4b"], True),
        ("T5a", ["T4a", "T4b"], False),
        (3, [1, ">2"], True),
        ("[8, 8]", (8, 8), True),  # list-valued fields (table form) match the whole sequence
        ("[8, 8]", [8, 16], False),
        ("base-1", re.compile(r"^base"), True),
        ("other", re.compile(r"^base"), False),
        (4, lambda v: v % 2 == 0, True),
        (None, None, True),
        (float("nan"), None, True),
        (pd.NA, None, True),
        (pd.NA, 1, False),
        (pd.NA, ">0", False),
        (1, None, False),
        ("[8, 8]", td.any_of(8, 16), False),  # any_of never compares the whole sequence
        (8, td.any_of(8, 16), True),
        (3, td.any_of(1, ">2"), True),  # items are conditions
        ("[8, 8]", td.any_of([8, 8], [4]), True),  # list items compare whole
        (8, td.any_of([8, 8], [4]), False),
        ("[8, 16]", td.exactly([8, 16]), True),
        (8, td.exactly([8, 16]), False),  # exactly never matches an item
        ([8, 16], td.exactly((8, 16)), True),  # raw lists (not in table form) too
        (">2", td.exactly(">2"), True),  # no comparison parsing
        (3, td.exactly(">2"), False),
        (0.1 + 0.2, td.exactly(0.3), True),
    ],
)
def test_matches(value, condition, expected):
    assert matches(value, condition) is expected


def test_numeric_mask_agrees_with_matches():
    values = np.array([0.0, 0.1, 0.30000000000000004, 1.0, np.nan])
    for condition in [0.3, ">0.05", ">=0.1, <1", "!=0.1", [0.0, 1.0], "relu", lambda v: v > 0.5, td.any_of(0.0, 1.0)]:
        expected = [matches(v, condition) for v in values]
        assert mask(values, condition).tolist() == expected, condition
    ints = np.arange(5)
    assert mask(ints, [1, 3]).tolist() == [False, True, False, True, False]
    assert mask(ints, ">=3").tolist() == [False, False, False, True, True]
    assert mask(ints, td.any_of(1, 3)).tolist() == [False, True, False, True, False]
    assert mask(ints, td.exactly(3)).tolist() == [False, False, False, True, False]
    assert not mask(ints, td.exactly([1, 3])).any()
    assert mask(values, td.any_of(0.3, ">0.9")).tolist() == [False, False, True, True, False]


def test_wrappers_everywhere(root):
    class Layered(td.Record):
        __record_name__ = "WhereLayered"

        class Spec(td.Spec):
            layers: list[int]

        def build(self, spec):
            self["n"] = len(spec.layers)

    for layers in ([8], [16], [8, 16]):
        Layered.get({"layers": layers})

    def found(collection):
        return sorted(tuple(r.spec.layers) for r in collection)

    assert found(Layered.query(layers=[8, 16])) == [(8, 16)]  # the whole list (no field equals 8)
    assert found(Layered.query(layers=[[8], [16]])) == [(8,), (16,)]  # any item
    assert found(Layered.query(layers=td.any_of(8, 16))) == []
    assert found(Layered.query(layers=td.exactly([8, 16]))) == [(8, 16)]
    assert found(Layered.query(layers=td.any_of([8], [16]))) == [(8,), (16,)]
    assert found(Layered.query().where(layers=td.exactly([8]))) == [(8,)]
    assert found(td.where(Layered.query(), layers=td.exactly([16]))) == [(16,)]
    table = Layered.info(sizes=False)
    assert len(td.where(table, layers=td.exactly([8, 16]))) == 1
    assert len(table.td.where(layers=td.any_of([8], [8, 16]))) == 2
    series = pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"], name="tag"))
    assert series.td.where(td.any_of(1, 3), tag=td.exactly("c")).tolist() == [3]
    data = xr.DataArray([1, 2, 3], dims="x", coords={"x": [0.1, 0.2, 0.3]})
    assert data.td.where(x=td.any_of(0.1, 0.3)).values.tolist() == [1, 3]
    assert repr(td.any_of(1, "a")) == "tensordir.any_of(1, 'a')" and repr(td.exactly([1])) == "tensordir.exactly([1])"


def test_where_on_frames():
    frame = pd.DataFrame(
        {
            "key": ["a", "b", "c", "d"],
            "spec.seed": pd.array([0, 1, 2, 3], dtype="Int64"),
            "spec.model.lr": [0.1, 0.1, 0.5, None],
            "spec.tag": pd.array(["base", "base", "wide", "base"], dtype="string"),
        }
    )
    assert list(td.where(frame, seed=[0, 2])["key"]) == ["a", "c"]
    assert list(td.where(frame, model__lr=0.1, tag="base")["key"]) == ["a", "b"]
    assert list(td.where(frame, {"spec.model.lr": ">0.2"})["key"]) == ["c"]
    assert list(td.where(frame, model__lr=None)["key"]) == ["d"]
    assert list(frame.pipe(td.where, key=re.compile("[bc]"))["key"]) == ["b", "c"]
    with pytest.raises(KeyError, match="did you mean"):
        td.where(frame, sed=0)


def test_where_on_xarray():
    data = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("cell", "time"),
        coords={"cell_type": ("cell", ["T4a", "T4b", "T5a"]), "time": [-0.5, 0.0, 0.5, 1.0]},
    )
    picked = data.pipe(td.where, cell_type=["T4a", "T4b"], time=">=0, <1")
    assert picked.dims == ("cell", "time")
    assert list(picked.cell_type.values) == ["T4a", "T4b"] and list(picked.time.values) == [0.0, 0.5]
    assert picked.dtype == data.dtype  # 1-d coordinates select with isel: no NaN padding
    grid = data.assign_coords(early=(("cell", "time"), data.values < 5))
    assert int(td.where(grid, early=True).count()) == 5
    with pytest.raises(KeyError, match="coordinate"):
        td.where(data, celltype="T4a")
    with pytest.raises(TypeError):
        td.where([1, 2], x=1)


def test_missing_optional_dependency_names_the_extra():
    from tensordir.collection import require

    with pytest.raises(ImportError, match=r"tensordir\[tables\]"):
        require("no_such_module_xyz", "tables", "Collection.table")
