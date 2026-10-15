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
    ],
)
def test_matches(value, condition, expected):
    assert matches(value, condition) is expected


def test_numeric_mask_agrees_with_matches():
    values = np.array([0.0, 0.1, 0.30000000000000004, 1.0, np.nan])
    for condition in [0.3, ">0.05", ">=0.1, <1", "!=0.1", [0.0, 1.0], "relu", lambda v: v > 0.5]:
        expected = [matches(v, condition) for v in values]
        assert mask(values, condition).tolist() == expected, condition
    ints = np.arange(5)
    assert mask(ints, [1, 3]).tolist() == [False, True, False, True, False]
    assert mask(ints, ">=3").tolist() == [False, False, False, True, True]


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
