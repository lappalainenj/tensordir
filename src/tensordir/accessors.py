"""`.td` accessors on pandas and xarray objects, registered when tensordir is imported.

```python
table.td.where(seed=[0, 1], lr=">1e-3")        # pandas.DataFrame: columns, `spec.` implied
responses.td.where(cell_type=["T4a", "T4b"])    # xarray: coordinates
losses.td.where("<0.1")                         # pandas.Series: values (and index levels by name)
table.td.diff(0, 1)                             # what differs between two rows' spec columns
```

Importing tensordir imports pandas and xarray if they are installed and registers the accessors
on `DataFrame`, `Series`, `DataArray` and `Dataset`. If another package registered `.td` first,
tensordir warns and leaves it in place; `tensordir.where` works either way. Set the environment
variable `TENSORDIR_NO_ACCESSORS=1` to skip registration (and those imports).
"""

import importlib.util
import os
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np

from ._diff import SpecDiff, diff
from ._where import _missing, conditions, mask, where

__all__ = ["NAME", "FrameAccessor", "SeriesAccessor", "XarrayAccessor", "register"]

NAME = "td"
_UNSET = object()


class _Accessor:
    __tensordir_accessor__ = True

    def __init__(self, obj: Any):
        self._obj = obj

    def where(self, conditions_: Mapping[str, Any] | None = None, /, **keywords: Any) -> Any:
        """`tensordir.where(obj, ...)`: the rows or coordinate entries meeting every condition."""
        return where(self._obj, conditions_, **keywords)


def _row_dict(row: Any) -> dict[str, Any]:
    """A table row as a spec-like dict: its `spec.` columns if any (prefix dropped), else all.
    Missing values are left out; values without a JSON form become strings."""
    items = {}
    for name, value in row.items():
        if _missing(value):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        items[str(name)] = value if isinstance(value, (bool, int, float, str)) else str(value)
    spec = {k[len("spec."):]: v for k, v in items.items() if k.startswith("spec.")}
    return spec or items


class FrameAccessor(_Accessor):
    """`DataFrame.td`: `where` on columns (a name also finds `spec.<name>` and index levels)."""

    def diff(self, a: Any, b: Any) -> SpecDiff:
        """What changes from row `a` to row `b` (index labels), compared on their `spec.` columns."""
        frame = self._obj
        return diff(_row_dict(frame.loc[a]), _row_dict(frame.loc[b]))


class SeriesAccessor(_Accessor):
    """`Series.td`: `where` on values (positional condition) and on index levels (by name)."""

    def where(self, condition: Any = _UNSET, /, **levels: Any) -> Any:
        series = self._obj
        keep = np.ones(len(series), dtype=bool)
        if condition is not _UNSET:
            keep &= mask(series.to_numpy(dtype=object), condition)
        names = [str(n) for n in series.index.names if n is not None]
        for name, wanted in conditions(None, levels).items():
            if name not in names:
                raise KeyError(f"no index level {name!r}; levels: {names}")
            keep &= mask(series.index.get_level_values(name).to_numpy(dtype=object), wanted)
        return series[keep]

    def diff(self, other: Any) -> SpecDiff:
        """What changes from this row (a table row, or any series) to `other` (a series or mapping)."""
        right = other if isinstance(other, Mapping) else _row_dict(other)
        return diff(_row_dict(self._obj), dict(right))


class XarrayAccessor(_Accessor):
    """`DataArray.td` and `Dataset.td`: `where` on coordinates (flyvis's `.custom.where`)."""


def _register(cls: type, register: Any, accessor: type) -> bool:
    existing = getattr(cls, NAME, None)
    if existing is not None:
        if not getattr(existing, "__tensordir_accessor__", False):
            owner = getattr(existing, "__module__", None) or type(existing).__module__
            warnings.warn(
                f"{cls.__module__}.{cls.__qualname__}.{NAME} is already defined (by {owner}); tensordir "
                f"does not replace it. Use tensordir.where(obj, ...) instead.",
                UserWarning,
                stacklevel=3,
            )
        return False
    register(NAME)(accessor)
    return True


def register() -> None:
    """Register the `.td` accessors on the installed libraries (idempotent)."""
    if os.environ.get("TENSORDIR_NO_ACCESSORS"):
        return
    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        _register(pd.DataFrame, pd.api.extensions.register_dataframe_accessor, FrameAccessor)
        _register(pd.Series, pd.api.extensions.register_series_accessor, SeriesAccessor)
    if importlib.util.find_spec("xarray") is not None:
        import xarray as xr

        _register(xr.DataArray, xr.register_dataarray_accessor, XarrayAccessor)
        _register(xr.Dataset, xr.register_dataset_accessor, XarrayAccessor)
