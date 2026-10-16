"""`.td` accessors on pandas and xarray objects."""

import importlib
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import tensordir as td
from tensordir import accessors


def test_dataframe_where_and_diff():
    table = pd.DataFrame(
        {"key": ["a", "b", "c"], "spec.seed": [0, 1, 2], "spec.model.lr": [0.1, 0.01, 0.1], "status": ["done"] * 3}
    )
    assert list(table.td.where(seed=[0, 2]).key) == ["a", "c"]
    assert list(table.td.where({"model.lr": 0.1}, seed=">0").key) == ["c"]
    assert table.td.where(model__lr=0.01).equals(td.where(table, model__lr=0.01))
    d = table.td.diff(0, 1)
    assert d.changed == {"model.lr": (0.1, 0.01), "seed": (0, 1)} and not d.added
    assert not table.td.diff(0, 0)
    indexed = table.set_index("key")
    assert list(indexed.td.where(key=["b"])["spec.seed"]) == [1]  # index levels by name


def test_series_where_and_diff():
    losses = pd.Series([0.5, 0.05, 0.01], index=pd.Index(["T4a", "T4b", "T5a"], name="cell_type"))
    assert list(losses.td.where("<0.1")) == [0.05, 0.01]
    assert list(losses.td.where(cell_type=["T4a", "T5a"])) == [0.5, 0.01]
    assert list(losses.td.where("<0.1", cell_type="T5a")) == [0.01]
    with pytest.raises(KeyError):
        losses.td.where(neuron="T4a")
    row = pd.Series({"spec.seed": 0, "spec.tag": "x", "spec.extra": np.nan, "path": "/a"})
    other = pd.Series({"spec.seed": 1, "spec.tag": "x", "spec.extra": 3.0, "path": "/b"})
    d = row.td.diff(other)
    assert d.changed == {"seed": (0, 1)} and d.added == {"extra": 3.0}
    assert row.td.diff({"seed": 0, "tag": "x"}).names() == []


def test_xarray_where():
    responses = xr.DataArray(
        np.arange(6.0).reshape(3, 2),
        dims=("sample", "time"),
        coords={"cell_type": ("sample", ["T4a", "T4b", "T5a"]), "time": [0.0, 1.0]},
    )
    picked = responses.td.where(cell_type=["T4a", "T5a"], time=">0")
    assert picked.shape == (2, 1) and list(picked.cell_type.values) == ["T4a", "T5a"]
    assert responses.to_dataset(name="r").td.where(cell_type="T4b").sizes["sample"] == 1
    assert not hasattr(responses.td, "diff")


def test_registration_is_idempotent_and_does_not_overwrite(monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        accessors.register()  # already registered by tensordir: silent no-op

    class Foreign:
        def __init__(self, obj):
            pass

    monkeypatch.delattr(pd.Series, "td")
    pd.api.extensions.register_series_accessor("td")(Foreign)
    with pytest.warns(UserWarning, match="already defined"):
        accessors.register()
    assert isinstance(pd.Series([1]).td, Foreign)


def test_opt_out_environment_variable():
    code = (
        "import sys, tensordir as td; "
        "assert 'pandas' not in sys.modules and 'xarray' not in sys.modules, 'imported'; "
        "import pandas as pd, xarray as xr; "
        "assert not hasattr(pd.DataFrame, 'td') and not hasattr(xr.DataArray, 'td'); "
        "assert pd.DataFrame({'a': [1, 2]}).pipe(td.where, a=2)['a'].tolist() == [2]; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env={**_env(), "TENSORDIR_NO_ACCESSORS": "1"}
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [sys.executable, "-c", "import tensordir, pandas as pd; assert hasattr(pd.DataFrame, 'td')"],
        capture_output=True, text=True, env=_env(),
    )
    assert result.returncode == 0, result.stderr


def _env():
    import os

    return {k: v for k, v in os.environ.items() if k != "TENSORDIR_NO_ACCESSORS"}


def test_reload_keeps_working():
    importlib.reload(accessors)
    accessors.register()
    assert list(pd.DataFrame({"a": [1, 2]}).td.where(a=2).a) == [2]
