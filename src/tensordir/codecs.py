"""Codecs: how a value is stored inside a record directory, chosen by the value's type.

| value | codec | stored as |
|---|---|---|
| `numpy.ndarray` | `ndarray` | zarr v3 array in `data.zarr/<name>` (with `dimension_names`) |
| `xarray.DataArray` / `Dataset` / `DataTree` | `xarray` | zarr group `data.zarr/<name>` |
| `pandas.DataFrame` | `parquet` | `<name>.parquet` |
| `pyarrow.Table` / `RecordBatchReader` | `arrow` | `<name>.parquet` (streamed) |
| JSON scalars, lists, str-keyed dicts | `json` | `<name>.json` |
| `torch.nn.Module` or `{str: torch.Tensor}` | `safetensors` | `<name>.safetensors` (state dict) |
| anything | `pickle` | `<name>.pkl`, only when asked for with `codec="pickle"` |
| `numpy.ndarray` | `hdf5` | `<name>.h5`, only when asked for (`codec="hdf5"` or a record's `codecs`) |

Register more with `register_codec`; later registrations take precedence.
"""

import json
import os
import pickle
import shutil
import sys
import threading
import uuid
import warnings
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .errors import DimsError
from .layout import APPEND_DEFAULT, DEFAULT, Layout

__all__ = ["Codec", "register_codec", "get_codec", "codec_for", "append_array"]

STORE = "data.zarr"
_DATAARRAY = "__xarray_dataarray_variable__"


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temporary sibling of `path`; on success rename it onto `path`."""
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _group(directory: Path, mode: str = "a") -> "zarr.Group":
    import zarr  # imported on first use: it costs 0.2 s at import tensordir

    return zarr.open_group(directory / STORE, mode=mode, zarr_format=3)


def consolidate(directory: Path) -> None:
    """Write the metadata of every array and group in `data.zarr` into its root `zarr.json`, so
    a reader opens the record's arrays with one metadata read. No-op without a zarr store."""
    if not (directory / STORE / "zarr.json").exists():
        return
    import zarr

    with warnings.catch_warnings():  # zarr warns that v3 consolidated metadata is not in the spec
        warnings.simplefilter("ignore")
        zarr.consolidate_metadata(directory / STORE, zarr_format=3)


#: Bytes of shard indexes kept per open record; an index is a few KB to a few hundred KB.
SHARD_INDEX_CACHE_BYTES = 64 << 20


def _caching_store(path: Path) -> "zarr.abc.store.Store":
    """A read-only `LocalStore` at `path` that keeps the shard indexes it has read.

    zarr reads a shard's index (a suffix of the shard file) before every read from that shard;
    for a finished record the index never changes, so it is read once per handle.
    """
    from zarr.abc.store import SuffixByteRequest
    from zarr.storage import LocalStore, WrapperStore

    class ShardIndexCache(WrapperStore):
        def __init__(self, store: Any) -> None:
            super().__init__(store)
            self._cache: OrderedDict = OrderedDict()
            self._bytes = 0
            self._lock = threading.Lock()

        def _with_store(self, store: Any) -> "ShardIndexCache":
            return type(self)(store)

        def _lookup(self, key: str, byte_range: Any) -> Any:
            if not isinstance(byte_range, SuffixByteRequest):
                return None
            with self._lock:
                found = self._cache.get((key, byte_range.suffix))
                if found is not None:
                    self._cache.move_to_end((key, byte_range.suffix))
                return found

        def _keep(self, key: str, byte_range: Any, value: Any) -> None:
            if value is None or not isinstance(byte_range, SuffixByteRequest):
                return
            with self._lock:
                self._cache[(key, byte_range.suffix)] = value
                self._bytes += len(value)
                while self._bytes > SHARD_INDEX_CACHE_BYTES and len(self._cache) > 1:
                    _, old = self._cache.popitem(last=False)
                    self._bytes -= len(old)

        async def get(self, key: str, prototype: Any, byte_range: Any = None) -> Any:
            found = self._lookup(key, byte_range)
            if found is None:
                found = await self._store.get(key, prototype, byte_range)
                self._keep(key, byte_range, found)
            return found

        def get_sync(self, key: str, *, prototype: Any = None, byte_range: Any = None) -> Any:
            found = self._lookup(key, byte_range)
            if found is None:
                found = super().get_sync(key, prototype=prototype, byte_range=byte_range)
                self._keep(key, byte_range, found)
            return found

    return ShardIndexCache(LocalStore(path, read_only=True))


def open_store_group(directory: Path, *, immutable: bool) -> "zarr.Group":
    """`data.zarr` opened read-only, with consolidated metadata when the record has it; an
    `immutable` (finished, not workspace) record also caches shard indexes."""
    import zarr

    path = directory / STORE
    store = _caching_store(path) if immutable else path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return zarr.open_group(store, mode="r", zarr_format=3, use_consolidated=None)


def group_member(group: "zarr.Group", name: str) -> Any:
    """`group[name]`, falling back to the member's own metadata when consolidated metadata is
    older than the member (a field written after the last consolidation)."""
    try:
        return group[name]
    except KeyError:
        import zarr

        return zarr.open(group.store, path=name, mode="r", zarr_format=3)


def _layout_kwargs(spec: dict, layout: Layout) -> dict:
    kwargs: dict[str, Any] = {}
    if spec["chunks"] is not None:
        kwargs["chunks"] = spec["chunks"]
    if spec["shards"] is not None:
        kwargs["shards"] = spec["shards"]
    if layout.compressors != "auto":
        kwargs["compressors"] = layout.compressors
    return kwargs


def _layout_info(array: Any) -> dict:
    shards = getattr(array, "shards", None)
    return {"chunks": list(array.chunks), "shards": list(shards) if shards else None}


def _delete_member(directory: Path, name: str) -> None:
    if (directory / STORE).exists():
        group = _group(directory)
        if name in group:
            del group[name]


class Codec:
    """Base class. `write` returns JSON-able info stored in the record's field manifest; `read`
    receives that info back.

    A codec that sets `takes_layout = True` receives `layout=` (a `Layout` or None for its own
    default) in `write` and `append`."""

    name: ClassVar[str]
    takes_layout: ClassVar[bool] = False

    def accepts(self, value: Any) -> bool:
        return False

    def faithful(self, value: Any) -> bool:
        """Whether `read` returns `value` as it was (same type, equal content). The joblib store
        backend only uses codecs that are faithful for the result at hand."""
        return True

    def write(self, directory: Path, name: str, value: Any, *, dims: Sequence[str] | None = None) -> dict:
        raise NotImplementedError

    def read(self, directory: Path, name: str, info: dict, *, lazy: bool = True) -> Any:
        raise NotImplementedError

    def delete(self, directory: Path, name: str, info: dict) -> None:
        raise NotImplementedError

    def append(
        self, directory: Path, name: str, values: Any, *, dims: Sequence[str] | None = None, layout: Layout | None = None
    ) -> dict:
        """Append rows along axis 0 of field `name`, creating it on first use; returns its info
        with `"append": True`. Only array codecs implement it."""
        raise TypeError(f"codec {self.name!r} cannot append")


class FileCodec(Codec):
    """A codec storing one file `<name><suffix>` next to `_meta.json`."""

    suffix: ClassVar[str]

    def path(self, directory: Path, name: str) -> Path:
        return directory / f"{name}{self.suffix}"

    def delete(self, directory: Path, name: str, info: dict) -> None:
        self.path(directory, name).unlink(missing_ok=True)


class NdarrayCodec(Codec):
    """numpy arrays as zarr v3 arrays; chunks and shards follow the `Layout` (default:
    `tensordir.layout.DEFAULT`, 128 KB chunks along axis 0 in shards of up to 128 MB)."""

    name = "ndarray"
    takes_layout = True

    def accepts(self, value: Any) -> bool:
        return isinstance(value, np.ndarray) and not value.dtype.hasobject

    def write(self, directory, name, value, *, dims=None, layout=None):
        if dims is not None and len(dims) != value.ndim:
            raise DimsError(f"{name}: {value.ndim}-d array with dims {tuple(dims)}")
        layout = layout or DEFAULT
        spec = layout.resolve(value.shape, value.dtype)
        _delete_member(directory, name)
        array = _group(directory).create_array(
            name,
            shape=value.shape,
            dtype=value.dtype,
            dimension_names=list(dims) if dims else None,
            **_layout_kwargs(spec, layout),
        )
        array[...] = value
        return {
            "dims": list(dims) if dims else None,
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "layout": _layout_info(array),
        }

    def read(self, directory, name, info, *, lazy=True):
        array = _group(directory, "r")[name]
        return array if lazy else array[...]

    def delete(self, directory, name, info):
        _delete_member(directory, name)

    def append(self, directory, name, values, *, dims=None, layout=None):
        return append_array(directory, name, values, dims=dims, layout=layout)


def _rows(name: str, values: Any, dims: Sequence[str] | None) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 0:
        values = values[None]
    if dims is not None and len(dims) != values.ndim:
        raise DimsError(f"{name}: appending {values.ndim}-d values with dims {tuple(dims)}")
    return values


def _check_rows(name: str, values: np.ndarray, shape: tuple[int, ...]) -> None:
    if tuple(shape[1:]) != values.shape[1:]:
        raise ValueError(f"{name}: cannot append rows of shape {values.shape[1:]} to {tuple(shape)}")


def append_array(
    directory: Path, name: str, values: Any, *, dims: Sequence[str] | None = None, layout: Layout | None = None
) -> dict:
    """Append `values` along axis 0 of the zarr array `name`, creating it on first use with
    `layout` (default `APPEND_DEFAULT`: 128 KB chunks, no shards; with shards, each append
    rewrites the last shard). Later appends keep the array's layout.

    Readers may briefly see fill values at the tail while a resize is in flight.
    """
    values = _rows(name, values, dims)
    group = _group(directory)
    if name not in group:
        layout = layout or APPEND_DEFAULT
        spec = layout.resolve(values.shape, values.dtype, growing=True)
        array = group.create_array(
            name,
            shape=values.shape,
            dtype=values.dtype,
            dimension_names=list(dims) if dims else None,
            **_layout_kwargs(spec, layout),
        )
        array[...] = values
    else:
        array = group[name]
        _check_rows(name, values, array.shape)
        n = array.shape[0]
        array.resize((n + len(values), *array.shape[1:]))
        array[n:] = values.astype(array.dtype, copy=False)
    return {
        "dims": list(dims) if dims else None,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "append": True,
        "layout": _layout_info(array),
    }


class XarrayCodec(Codec):
    name = "xarray"

    def accepts(self, value: Any) -> bool:
        xr = sys.modules.get("xarray")
        return xr is not None and isinstance(value, (xr.Dataset, xr.DataArray, xr.DataTree))

    def write(self, directory, name, value, *, dims=None):
        import xarray as xr

        _group(directory)
        _delete_member(directory, name)
        store = directory / STORE
        if isinstance(value, xr.DataTree):
            for node in value.subtree:
                group = name if node.path == "/" else f"{name}{node.path}"
                node.to_dataset(inherit=False).drop_encoding().to_zarr(
                    store, group=group, mode="w", zarr_format=3, consolidated=False
                )
            return {"kind": "datatree"}
        if isinstance(value, xr.DataArray):
            variable = value.name if value.name is not None else _DATAARRAY
            dataset = value.to_dataset(name=variable)
            info = {"kind": "dataarray", "variable": variable, "named": value.name is not None}
        else:
            dataset, info = value, {"kind": "dataset"}
        dataset.drop_encoding().to_zarr(store, group=name, mode="w", zarr_format=3, consolidated=False)
        return info

    def read(self, directory, name, info, *, lazy=True):
        import xarray as xr

        store = directory / STORE
        if info["kind"] == "datatree":
            tree = xr.open_datatree(store, group=name, engine="zarr", consolidated=False, chunks=None)
            return tree if lazy else tree.load()
        dataset = xr.open_zarr(store, group=name, consolidated=False, chunks=None)
        if not lazy:
            dataset = dataset.load()
        if info["kind"] == "dataarray":
            array = dataset[info["variable"]]
            return array if info["named"] else array.rename(None)
        return dataset

    def delete(self, directory, name, info):
        _delete_member(directory, name)


class ParquetCodec(FileCodec):
    name = "parquet"
    suffix = ".parquet"

    def accepts(self, value: Any) -> bool:
        pd = sys.modules.get("pandas")
        return pd is not None and isinstance(value, pd.DataFrame)

    def write(self, directory, name, value, *, dims=None):
        with atomic_path(self.path(directory, name)) as tmp:
            value.to_parquet(tmp)
        return {}

    def read(self, directory, name, info, *, lazy=True):
        import pandas as pd

        return pd.read_parquet(self.path(directory, name))


class ArrowCodec(FileCodec):
    """`pyarrow.Table` and `RecordBatchReader` (streamed) as parquet. `record[name]` opens a lazy
    `pyarrow.dataset.Dataset`; loading gives a `pyarrow.Table`."""

    name = "arrow"
    suffix = ".parquet"

    def accepts(self, value: Any) -> bool:
        pa = sys.modules.get("pyarrow")
        return pa is not None and isinstance(value, (pa.Table, pa.RecordBatchReader))

    def faithful(self, value: Any) -> bool:
        return False  # lazy reads return a dataset, and a reader cannot be replayed

    def write(self, directory, name, value, *, dims=None):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with atomic_path(self.path(directory, name)) as tmp:
            if isinstance(value, pa.Table):
                pq.write_table(value, tmp)
                rows = value.num_rows
            else:
                rows = 0
                with pq.ParquetWriter(tmp, value.schema) as writer:
                    for batch in value:
                        writer.write_batch(batch)
                        rows += batch.num_rows
        return {"rows": rows}

    def read(self, directory, name, info, *, lazy=True):
        if lazy:
            import pyarrow.dataset as ds

            return ds.dataset(self.path(directory, name), format="parquet")
        import pyarrow.parquet as pq

        return pq.read_table(self.path(directory, name), memory_map=True)


def _jsonable(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str, np.bool_, np.integer, np.floating)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_jsonable(v) for v in value)
    if isinstance(value, Mapping):
        return all(isinstance(k, str) and _jsonable(v) for k, v in value.items())
    return False


def _to_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _to_python(v) for k, v in value.items()}
    return value


class JsonCodec(FileCodec):
    name = "json"
    suffix = ".json"

    def accepts(self, value: Any) -> bool:
        return _jsonable(value)

    def write(self, directory, name, value, *, dims=None):
        with atomic_path(self.path(directory, name)) as tmp:
            tmp.write_text(json.dumps(_to_python(value)))
        return {}

    def read(self, directory, name, info, *, lazy=True):
        return json.loads(self.path(directory, name).read_text())


class SafetensorsCodec(FileCodec):
    name = "safetensors"
    suffix = ".safetensors"

    def accepts(self, value: Any) -> bool:
        torch = sys.modules.get("torch")
        if torch is None:
            return False
        if isinstance(value, torch.nn.Module):
            return True
        return isinstance(value, Mapping) and bool(value) and all(isinstance(v, torch.Tensor) for v in value.values())

    def faithful(self, value: Any) -> bool:
        return isinstance(value, Mapping)  # a module comes back as its state dict

    def write(self, directory, name, value, *, dims=None):
        import torch
        from safetensors.torch import save_file

        module = isinstance(value, torch.nn.Module)
        state = value.state_dict() if module else value
        with atomic_path(self.path(directory, name)) as tmp:
            save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(tmp))
        return {"state_dict_of_module": module}

    def read(self, directory, name, info, *, lazy=True):
        from safetensors.torch import load_file

        return load_file(str(self.path(directory, name)))


class PickleCodec(FileCodec):
    """Opt-in only (`codec="pickle"`): pickles execute code on load and break across refactors."""

    name = "pickle"
    suffix = ".pkl"

    def write(self, directory, name, value, *, dims=None):
        with atomic_path(self.path(directory, name)) as tmp:
            with open(tmp, "wb") as f:
                pickle.dump(value, f, protocol=5)
        return {}

    def read(self, directory, name, info, *, lazy=True):
        with open(self.path(directory, name), "rb") as f:
            return pickle.load(f)


class Hdf5Codec(FileCodec):
    """Opt-in: arrays as `<name>.h5` holding one dataset `data` (datamate's layout), with
    dimension labels, for h5-based tools. Choose it per field (`write(..., codec="hdf5")`) or per
    record type (`codecs = {"ndarray": "hdf5"}`); zarr stays the default.

    `append` creates a chunked dataset with `maxshape=(None, ...)` and resizes it along axis 0
    (datamate's `extend`), with the checks of zarr appends.

    Trade-offs against zarr: HDF5 file locking misbehaves on NFS and Lustre (tensordir opens
    files with `locking=False`; other readers may need `HDF5_USE_FILE_LOCKING=FALSE`); a file has
    a single writer and each `write` rewrites the whole file; chunks cannot be written in
    parallel; and a lazy read keeps the file open until the handle is dropped.
    """

    name = "hdf5"
    suffix = ".h5"
    takes_layout = True

    def write(self, directory, name, value, *, dims=None, layout=None):
        import h5py

        value = np.asarray(value)
        if value.dtype.hasobject:
            raise TypeError(f"{name}: object arrays have no HDF5 form")
        if dims is not None and len(dims) != value.ndim:
            raise DimsError(f"{name}: {value.ndim}-d array with dims {tuple(dims)}")
        chunks = layout.resolve(value.shape, value.dtype)["chunks"] if layout is not None else None
        with atomic_path(self.path(directory, name)) as tmp:
            with h5py.File(tmp, "w", locking=False) as f:
                dataset = f.create_dataset("data", data=value, chunks=chunks if value.size else None)
                for axis, label in enumerate(dims or ()):
                    dataset.dims[axis].label = label
                stored = dataset.chunks
        info = {"dims": list(dims) if dims else None, "shape": list(value.shape), "dtype": value.dtype.str}
        return {**info, "layout": {"chunks": list(stored) if stored else None, "shards": None}}

    def read(self, directory, name, info, *, lazy=True):
        import h5py

        f = h5py.File(self.path(directory, name), "r", locking=False)
        if lazy:
            return f["data"]
        with f:
            return f["data"][()]

    def append(self, directory, name, values, *, dims=None, layout=None):
        import h5py

        values = _rows(name, values, dims)
        if values.dtype.hasobject:
            raise TypeError(f"{name}: object arrays have no HDF5 form")
        path = self.path(directory, name)
        if not path.exists():
            with atomic_path(path) as tmp:
                with h5py.File(tmp, "w", locking=False) as f:
                    dataset = f.create_dataset(
                        "data",
                        data=values,
                        maxshape=(None, *values.shape[1:]),
                        chunks=(layout or APPEND_DEFAULT).resolve(values.shape, values.dtype, growing=True)["chunks"],
                    )
                    for axis, label in enumerate(dims or ()):
                        dataset.dims[axis].label = label
            shape, dtype = values.shape, values.dtype
        else:
            with h5py.File(path, "a", locking=False) as f:
                dataset = f["data"]
                _check_rows(name, values, dataset.shape)
                n = dataset.shape[0]
                dataset.resize(n + len(values), axis=0)
                dataset[n:] = values.astype(dataset.dtype, copy=False)
                shape, dtype = dataset.shape, dataset.dtype
        return {"dims": list(dims) if dims else None, "shape": list(shape), "dtype": dtype.str, "append": True}


_CODECS: dict[str, Codec] = {}


def register_codec(codec: Codec, *, replace: bool = False) -> Codec:
    """Register `codec` under `codec.name`; it is tried before all previously registered codecs."""
    if codec.name in _CODECS and not replace:
        raise ValueError(f"codec {codec.name!r} is already registered; pass replace=True")
    _CODECS.pop(codec.name, None)
    _CODECS[codec.name] = codec
    return codec


def get_codec(name: str) -> Codec:
    try:
        return _CODECS[name]
    except KeyError:
        raise KeyError(f"unknown codec {name!r}; registered: {sorted(_CODECS)}") from None


def codec_for(value: Any) -> Codec:
    for codec in reversed(_CODECS.values()):
        if codec.accepts(value):
            return codec
    raise TypeError(
        f"no codec stores {type(value).__module__}.{type(value).__qualname__}; "
        "register one with tensordir.register_codec or pass codec='pickle'"
    )


for _codec in (
    Hdf5Codec(), PickleCodec(), SafetensorsCodec(), JsonCodec(), ParquetCodec(), ArrowCodec(), XarrayCodec(), NdarrayCodec()
):
    register_codec(_codec)


def remove_tree(path: Path) -> None:
    """Remove a directory by renaming it aside first, so it never appears half-deleted."""
    if not path.exists():
        return
    grave = path.with_name(f".{path.name}.trash-{uuid.uuid4().hex}")
    os.rename(path, grave)
    shutil.rmtree(grave, ignore_errors=True)
