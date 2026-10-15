"""One condition grammar for records, tables and xarray coordinates.

The grammar generalises flyvis's `where_xarray` / `.custom.where` (MIT, Copyright (c) 2023
Janne K. Lappalainen et al.). A condition is

- a value: equality (floats within a relative tolerance of 1e-9; `None` matches missing values),
- a string of comparisons such as `">0.1"` or `">=2, <8"` (comma means AND),
- a list, tuple or set: equal to the whole sequence or to any of its items (OR),
- a compiled regex (`re.compile(...)`): searched in the value's string form,
- a callable: called with the value, truthy means match.

Names use dots or double underscores for nesting: `model.lr` and `model__lr` are the same.
"""

import json
import operator
import re
import sys
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

__all__ = ["where", "matches", "mask", "conditions"]

_OPS: dict[str, Callable[[Any, Any], Any]] = {
    ">=": operator.ge, "<=": operator.le, "==": operator.eq, "!=": operator.ne, ">": operator.gt, "<": operator.lt,
}
_OP_ORDER = sorted(_OPS, key=len, reverse=True)
_RTOL = 1e-9


def conditions(positional: Mapping[str, Any] | None, keywords: Mapping[str, Any]) -> dict[str, Any]:
    """Merge a mapping of (possibly dotted) names and keyword conditions; `__` becomes `.`."""
    merged = dict(positional or {})
    merged.update(keywords)
    return {name.replace("__", "."): value for name, value in merged.items()}


def display(value: Any) -> Any:
    """The form a value takes in tables: plain scalars, JSON strings for lists and dicts."""
    if isinstance(value, dict):
        if "__float__" in value:
            return float(value["__float__"])
        if "__digest__" in value:
            return f"digest:{value['__digest__'][:16]}"
        if "__record__" in value:
            return f"{value['__record__']}:{value['key'][:16]}"
        if "__delta__" in value:
            return f"delta:{value['__delta__']}@{value['version']}"
        if "__parquet__" in value:
            return f"parquet:{value['__parquet__']}@{str(value['version'])[:16]}"
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    return value


def _literal(value: Any) -> Any:
    from ._canonical import canonical

    try:
        return display(canonical(value, full=True))
    except Exception:
        return value


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return type(value).__name__ in ("NAType", "NaTType")


def _number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _equal(value: Any, wanted: Any) -> bool:
    if wanted is None:
        return _missing(value)
    if _missing(value):
        return False
    if _number(value) and _number(wanted):
        return bool(np.isclose(value, wanted, rtol=_RTOL, atol=0.0))
    try:
        return bool(value == wanted)
    except Exception:
        return False


def _comparisons(condition: str) -> list[tuple[str, Any]] | None:
    parts = [p.strip() for p in condition.split(",") if p.strip()]
    parsed = []
    for part in parts:
        op = next((o for o in _OP_ORDER if part.startswith(o)), None)
        if op is None:
            return None
        target: Any = part[len(op):].strip().strip("'\"")
        try:
            target = float(target)
        except ValueError:
            pass
        parsed.append((op, target))
    return parsed or None


def _compare(value: Any, op: str, target: Any) -> bool:
    if _missing(value):
        return False
    if op in ("==", "!="):
        return _equal(value, target) == (op == "==")
    if _number(value) != _number(target):
        return False
    try:
        return bool(_OPS[op](value, target))
    except TypeError:
        return False


def matches(value: Any, condition: Any) -> bool:
    """Whether one value (in table form) satisfies `condition`."""
    if isinstance(condition, re.Pattern):
        return not _missing(value) and condition.search(str(value)) is not None
    if callable(condition) and not isinstance(condition, type):
        return bool(condition(value))
    if isinstance(condition, str):
        parsed = _comparisons(condition)
        if parsed is not None:
            return all(_compare(value, op, target) for op, target in parsed)
        return _equal(value, condition)
    if isinstance(condition, (list, tuple, set, frozenset)):
        if isinstance(condition, (list, tuple)) and _equal(value, _literal(condition)):
            return True
        return any(matches(value, item) for item in condition)
    return _equal(value, _literal(condition))


def mask(values: Any, condition: Any) -> np.ndarray:
    """Elementwise `matches` over an array-like; numeric arrays take a vectorized path."""
    array = np.asarray(values)
    if array.dtype.kind in "iuf":
        fast = _numeric_mask(array, condition)
        if fast is not None:
            return fast
    flat = array.ravel()
    out = np.fromiter((matches(v, condition) for v in flat), dtype=bool, count=flat.size)
    return out.reshape(array.shape)


def _numeric_mask(array: np.ndarray, condition: Any) -> np.ndarray | None:
    def equal(target: float) -> np.ndarray:
        if array.dtype.kind == "f":
            return np.isclose(array, target, rtol=_RTOL, atol=0.0)
        return array == target

    if _number(condition):
        return equal(condition)
    if isinstance(condition, str):
        parsed = _comparisons(condition)
        if parsed is None or not all(_number(t) for _, t in parsed):
            return None
        out = np.ones(array.shape, dtype=bool)
        for op, target in parsed:
            if op == "==":
                out &= equal(target)
            elif op == "!=":
                out &= ~equal(target) & ~np.isnan(array) if array.dtype.kind == "f" else ~equal(target)
            else:
                out &= _OPS[op](array, target)
        return out
    if isinstance(condition, (list, tuple, set, frozenset)) and all(_number(c) for c in condition):
        out = np.zeros(array.shape, dtype=bool)
        for item in condition:
            out |= equal(item)
        return out
    return None


def _close_names(name: str, names: list[str]) -> str:
    import difflib

    close = difflib.get_close_matches(name, names, n=3, cutoff=0.5)
    return f"; did you mean {close}?" if close else f"; available: {names}"


def _frame_where(frame: Any, conds: dict[str, Any]) -> Any:
    keep = np.ones(len(frame), dtype=bool)
    columns = [str(c) for c in frame.columns]
    levels = [str(n) for n in frame.index.names if n is not None]
    for name, condition in conds.items():
        column = name if name in frame.columns else f"spec.{name}"
        if column in frame.columns:
            values = frame[column].to_numpy(dtype=object)
        elif name in levels:
            values = frame.index.get_level_values(name).to_numpy(dtype=object)
        else:
            raise KeyError(f"no column {name!r}{_close_names(name, columns + levels)}")
        keep &= mask(values, condition)
    return frame[keep]


def _xarray_where(obj: Any, conds: dict[str, Any]) -> Any:
    for name, condition in conds.items():
        if name not in obj.coords:
            raise KeyError(f"no coordinate {name!r}{_close_names(name, [str(c) for c in obj.coords])}")
        coord = obj.coords[name]
        selected = mask(coord.values, condition)
        if coord.ndim == 1:
            obj = obj.isel({coord.dims[0]: selected})
        else:
            obj = obj.where(coord.copy(data=selected), drop=True)
    return obj


def where(obj: Any, conditions_: Mapping[str, Any] | None = None, /, **keywords: Any) -> Any:
    """Rows, records or coordinate entries of `obj` that satisfy every condition.

    `obj` is a `Collection` (conditions on the flattened spec), a `pandas.DataFrame` (on columns;
    `name` also finds `spec.name`), or an xarray object (on coordinates). See the module docstring
    for the grammar:

    ```python
    td.where(Fit.query().table(), seed=[0, 1], model__lr=">1e-3, <0.1", tag=re.compile("^base"))
    responses.pipe(td.where, cell_type=["T4a", "T4b"], time=">=0")
    ```
    """
    conds = conditions(conditions_, keywords)
    from .collection import Collection

    if isinstance(obj, Collection):
        return obj.where(conds)
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(obj, pd.DataFrame):
        return _frame_where(obj, conds)
    xr = sys.modules.get("xarray")
    if xr is not None and isinstance(obj, (xr.DataArray, xr.Dataset)):
        return _xarray_where(obj, conds)
    raise TypeError(f"where needs a Collection, DataFrame or xarray object, got {type(obj).__qualname__}")
