"""joblib.Memory with a store backend for labeled results.

xarray results are stored as `output.zarr` (written to a temporary directory, then renamed),
DataFrames as `output.parquet`, everything else through joblib's own pickle path (numpy arrays
natively). Loads are lazy by default. `output.h5` caches written by the flyvis and another project
netCDF backends stay readable.

Adapted from flyvis (`flyvis/utils/xarray_joblib_backend.py`, MIT, Copyright (c) 2023 Janne K.
Lappalainen, Fabian D. Tschopp, Mason McGill, Jakob H. Macke, Srinivas C. Turaga) and its
another project revision (pinned engine, atomic writes, DataArray round-trip).

joblib exposes store backends only through the private `joblib._store_backends` module; the
supported joblib range is pinned in `pyproject.toml` and checked by contract tests.
"""

import os
import shutil
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any

from joblib import Memory, register_store_backend
from joblib._store_backends import CacheWarning, FileSystemStoreBackend

from .codecs import atomic_path

__all__ = ["memory", "TensordirStoreBackend", "BACKEND"]

BACKEND = "tensordir"
ZARR, PARQUET, H5, PICKLE = "output.zarr", "output.parquet", "output.h5", "output.pkl"
_KIND = "__tensordir_kind__"
_NAME = "__tensordir_dataarray__"
_UNNAMED = "__xarray_dataarray_variable__"
_LEGACY_NAME_FLAGS = ("__microns_columns_dataarray_name__",)


def _kind(item: Any) -> str | None:
    xr, pd = sys.modules.get("xarray"), sys.modules.get("pandas")
    if xr is not None and isinstance(item, (xr.Dataset, xr.DataArray, xr.DataTree)):
        return ZARR
    if pd is not None and isinstance(item, pd.DataFrame):
        return PARQUET
    return None


def _write_zarr(item: Any, dest: Path) -> None:
    import xarray as xr

    tmp = dest.with_name(f".{dest.name}.tmp-{uuid.uuid4().hex}")
    try:
        if isinstance(item, xr.DataTree):
            tree = item.copy()
            tree.attrs = {**item.attrs, _KIND: "datatree"}
            tree.to_zarr(tmp, mode="w", zarr_format=3, consolidated=False)
        else:
            if isinstance(item, xr.DataArray):
                variable = item.name if item.name is not None else _UNNAMED
                dataset = item.to_dataset(name=variable)
                dataset.attrs = {_KIND: "dataarray", _NAME: "" if item.name is None else variable}
            else:
                dataset = item.copy(deep=False)
                dataset.attrs = {**item.attrs, _KIND: "dataset"}
            dataset.drop_encoding().to_zarr(tmp, mode="w", zarr_format=3, consolidated=False)
        if dest.exists():
            grave = dest.with_name(f".{dest.name}.old-{uuid.uuid4().hex}")
            os.rename(dest, grave)
            shutil.rmtree(grave, ignore_errors=True)
        os.rename(tmp, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _read_zarr(path: Path, lazy: bool) -> Any:
    import xarray as xr
    import zarr

    kind = zarr.open_group(path, mode="r").attrs.get(_KIND, "dataset")
    if kind == "datatree":
        tree = xr.open_datatree(path, engine="zarr", consolidated=False, chunks=None)
        tree = tree if lazy else tree.load()
        tree.attrs.pop(_KIND, None)
        return tree
    dataset = xr.open_zarr(path, consolidated=False, chunks=None)
    dataset = dataset if lazy else dataset.load()
    dataset.attrs.pop(_KIND, None)
    if kind == "dataarray":
        name = dataset.attrs.pop(_NAME)
        array = dataset[name or _UNNAMED]
        return array if name else array.rename(None)
    return dataset


def _read_h5(path: Path) -> Any:
    import xarray as xr

    with xr.open_dataset(path, engine="h5netcdf") as opened:
        dataset = opened.load()  # then release the handle: HDF5 locks break the next overwrite
    for flag in _LEGACY_NAME_FLAGS:
        name = dataset.attrs.pop(flag, None)
        if name is not None:
            return dataset[name]
    return dataset


class TensordirStoreBackend(FileSystemStoreBackend):
    """A joblib `FileSystemStoreBackend` that stores xarray as zarr and DataFrames as parquet.

    Backend option `lazy` (default True) opens zarr outputs without loading them.
    """

    lazy = True

    def configure(self, location, verbose=1, backend_options=None):
        options = dict(backend_options or {})
        self.lazy = options.pop("lazy", True)
        super().configure(location, verbose=verbose, backend_options=options)

    def dump_item(self, call_id, item, verbose=1):
        kind = _kind(item)
        item_path = Path(self.location, *call_id)
        for other in (ZARR, PARQUET, H5, PICKLE):
            if other != (kind or PICKLE) and (item_path / other).exists():
                target = item_path / other
                shutil.rmtree(target) if target.is_dir() else target.unlink()
        if kind is None:
            return super().dump_item(call_id, item, verbose=verbose)
        try:
            self.create_location(str(item_path))
            if kind == ZARR:
                _write_zarr(item, item_path / ZARR)
            else:
                with atomic_path(item_path / PARQUET) as tmp:
                    item.to_parquet(tmp)
        except Exception as exc:
            warnings.warn(f"Unable to cache {type(item).__name__} to disk: {exc}", CacheWarning, stacklevel=2)

    def load_item(self, call_id, verbose=1, timestamp=None, metadata=None):
        item_path = Path(self.location, *call_id)
        if (item_path / ZARR).exists():
            return _read_zarr(item_path / ZARR, self.lazy)
        if (item_path / PARQUET).exists():
            import pandas as pd

            return pd.read_parquet(item_path / PARQUET)
        if (item_path / H5).exists():
            return _read_h5(item_path / H5)
        return super().load_item(call_id, verbose=verbose, timestamp=timestamp, metadata=metadata)

    def contains_item(self, call_id):
        item_path = os.path.join(self.location, *call_id)
        return any(self._item_exists(os.path.join(item_path, name)) for name in (ZARR, PARQUET, H5, PICKLE))


register_store_backend(BACKEND, TensordirStoreBackend)


def memory(location: str | os.PathLike, *, lazy: bool = True, **kwargs: Any) -> Memory:
    """A `joblib.Memory` at `location` using the tensordir store backend.

    `lazy=False` loads zarr outputs into memory. Other keyword arguments go to `joblib.Memory`.
    """
    options = {"lazy": lazy, **kwargs.pop("backend_options", {})}
    return Memory(location=str(location), backend=BACKEND, backend_options=options, **kwargs)
