"""tensordir: content-addressed records of labeled tensors in directories."""

from ._canonical import Digest
from ._diff import SpecDiff, diff
from ._directory import Directory, Level, NodeId
from ._files import Image, Video, register_suffix
from ._root import get_root, root, set_root, use_root
from ._where import any_of, exactly, where
from .codecs import Codec, register_codec
from .collection import Collection
from .dims import Dims, dim_name, dims_for, register_dims
from .errors import (
    BuildFailed,
    CanonicalizationError,
    DimsError,
    KeyCollision,
    ReadOnlyError,
    RecordBusy,
    TensordirError,
)
from .external import DeltaRef, ParquetRef, SnapshotChanged
from .index import reindex
from ._signature import transient
from ._stages import Stage, StageOrderError, Stages
from ._record import Record
from .spec import Spec, field
from .view import View, cached_method
from . import accessors

accessors.register()  # `.td` on pandas and xarray objects; TENSORDIR_NO_ACCESSORS=1 skips it


def memory(location, *, lazy: bool = True, **kwargs):
    """A `joblib.Memory` whose results are stored by the codec registry (needs joblib).

    xarray results become zarr, DataFrames parquet, tensor dicts safetensors, and values of any
    registered codec use that codec; the rest is pickled by joblib. `backend=` takes a
    `StoreBackend` subclass; other keyword arguments go to `joblib.Memory`.
    """
    from ._memory import memory as _memory

    return _memory(location, lazy=lazy, **kwargs)


def __getattr__(name: str):
    if name == "StoreBackend":  # imported on demand: joblib is optional
        from ._memory import StoreBackend

        return StoreBackend
    raise AttributeError(f"module 'tensordir' has no attribute {name!r}")


def needs_compute(func, *args, **kwargs) -> bool:
    """Whether calling the memoized `func` with these arguments would compute rather than load."""
    return not func.check_call_in_cache(*args, **kwargs)


__all__ = [
    "BuildFailed",
    "CanonicalizationError",
    "Codec",
    "Collection",
    "DeltaRef",
    "Digest",
    "Directory",
    "Dims",
    "DimsError",
    "Image",
    "KeyCollision",
    "Level",
    "NodeId",
    "ParquetRef",
    "ReadOnlyError",
    "Record",
    "RecordBusy",
    "SnapshotChanged",
    "Spec",
    "SpecDiff",
    "Stage",
    "StageOrderError",
    "Stages",
    "StoreBackend",
    "TensordirError",
    "Video",
    "View",
    "any_of",
    "cached_method",
    "diff",
    "dim_name",
    "dims_for",
    "exactly",
    "field",
    "get_root",
    "memory",
    "needs_compute",
    "register_codec",
    "register_dims",
    "register_suffix",
    "reindex",
    "root",
    "set_root",
    "transient",
    "use_root",
    "where",
]
