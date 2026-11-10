"""joblib.Memory with a store backend that writes results through tensordir's codec registry.

A result whose type a registered codec claims (xarray as zarr, DataFrames and arrow tables as
parquet, `{str: Tensor}` as safetensors, and any codec you register: images, models, ...) is
written by that codec into `output.td/`, which appears by rename. numpy arrays, plain Python
values and everything else take joblib's pickle path. Loads are lazy where the codec supports it.
`output.zarr` and `output.parquet` from earlier tensordir versions and `output.h5` from the
flyvis netCDF backends stay readable.

Adapted from flyvis (`flyvis/utils/xarray_joblib_backend.py`, MIT, Copyright (c) 2023 Janne K.
Lappalainen, Fabian D. Tschopp, Mason McGill, Jakob H. Macke, Srinivas C. Turaga) and a later
revision of it (pinned engine, atomic writes, DataArray round-trip).

joblib exposes store backends only through the private `joblib._store_backends` module; the
supported joblib range is pinned in `pyproject.toml` and checked by contract tests.
"""

import json
import os
import shutil
import uuid
import warnings
from pathlib import Path
from typing import Any, ClassVar

from joblib import Memory, register_store_backend
from joblib._store_backends import CacheWarning, FileSystemStoreBackend

from .codecs import Codec, codec_for, get_codec, remove_tree

__all__ = ["memory", "StoreBackend", "BACKEND"]

BACKEND = "tensordir"
OUTPUT, PICKLE = "output.td", "output.pkl"
ZARR, PARQUET, H5 = "output.zarr", "output.parquet", "output.h5"  # read only (earlier formats)
_CODEC_FILE = "codec.json"
_FIELD = "output"
_KIND = "__tensordir_kind__"
_NAME = "__tensordir_dataarray__"
_UNNAMED = "__xarray_dataarray_variable__"
_LEGACY_NAME_FLAGS = ("__microns_columns_dataarray_name__",)


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


def _remove(path: Path) -> None:
    if path.is_dir():
        remove_tree(path)
    else:
        path.unlink(missing_ok=True)


class StoreBackend(FileSystemStoreBackend):
    """A joblib store backend that stores results with tensordir codecs.

    For each result, `select_codec` picks a registered codec or `None` (joblib's pickle path).
    The default picks `codec_for(item)` unless its name is in `skip_codecs` or the codec says
    it would not restore the value faithfully (`Codec.faithful`). Subclass to change either, then
    pass the class to `tensordir.memory(location, backend=MyBackend)`:

    ```python
    class ZarrArrays(td.StoreBackend):
        skip_codecs = frozenset({"json", "pickle"})  # numpy results as zarr, not joblib pickles
    ```

    **Experimental:** subclassing through `select_codec` and `skip_codecs` may change or be removed.

    Backend option `lazy` (default True) opens results lazily where the codec can.
    """

    #: Codecs never used for cached results: joblib already stores numpy arrays (with memory
    #: mapping) and plain Python values natively, and `pickle` is opt-in.
    skip_codecs: ClassVar[frozenset[str]] = frozenset({"ndarray", "json", "pickle"})
    lazy = True

    def configure(self, location, verbose=1, backend_options=None):
        options = dict(backend_options or {})
        self.lazy = options.pop("lazy", True)
        super().configure(location, verbose=verbose, backend_options=options)

    def select_codec(self, item: Any) -> Codec | None:
        """The codec storing `item`, or None to let joblib pickle it."""
        try:
            codec = codec_for(item)
        except TypeError:
            return None
        if codec.name in self.skip_codecs or not codec.faithful(item):
            return None
        return codec

    def dump_item(self, call_id, item, verbose=1):
        item_path = Path(self.location, *call_id)
        codec = self.select_codec(item)
        keep = OUTPUT if codec is not None else PICKLE
        for other in (OUTPUT, PICKLE, ZARR, PARQUET, H5):
            if other != keep and (item_path / other).exists():
                _remove(item_path / other)
        if codec is None:
            return super().dump_item(call_id, item, verbose=verbose)
        self.create_location(str(item_path))
        tmp = item_path / f".{OUTPUT}.tmp-{uuid.uuid4().hex}"
        try:
            tmp.mkdir()
            info = codec.write(tmp, _FIELD, item, dims=None)
            (tmp / _CODEC_FILE).write_text(json.dumps({"codec": codec.name, "info": info}))
            target = item_path / OUTPUT
            if target.exists():
                _remove(target)
            try:
                os.rename(tmp, target)
            except OSError:
                if not target.exists():  # otherwise a concurrent call stored the same result first
                    raise
        except Exception as exc:
            warnings.warn(f"Unable to cache {type(item).__name__} to disk: {exc}", CacheWarning, stacklevel=2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def load_item(self, call_id, verbose=1, timestamp=None, metadata=None):
        item_path = Path(self.location, *call_id)
        if (item_path / OUTPUT).is_dir():
            stored = json.loads((item_path / OUTPUT / _CODEC_FILE).read_text())
            return get_codec(stored["codec"]).read(item_path / OUTPUT, _FIELD, stored["info"], lazy=self.lazy)
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
        return any(self._item_exists(os.path.join(item_path, name)) for name in (OUTPUT, PICKLE, ZARR, PARQUET, H5))


register_store_backend(BACKEND, StoreBackend)


def memory(
    location: str | os.PathLike, *, lazy: bool = True, backend: type[StoreBackend] = StoreBackend, **kwargs: Any
) -> Memory:
    """A `joblib.Memory` at `location` using `backend` (a `StoreBackend` subclass).

    `lazy=False` loads results into memory. Other keyword arguments go to `joblib.Memory`.
    """
    if not (isinstance(backend, type) and issubclass(backend, StoreBackend)):
        raise TypeError(f"backend must be a tensordir.StoreBackend subclass, got {backend!r}")
    name = BACKEND if backend is StoreBackend else f"{BACKEND}:{backend.__module__}.{backend.__qualname__}"
    register_store_backend(name, backend)
    options = {"lazy": lazy, **kwargs.pop("backend_options", {})}
    return Memory(location=str(location), backend=name, backend_options=options, **kwargs)
