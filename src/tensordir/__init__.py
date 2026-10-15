"""tensordir: content-addressed records of labeled tensors in directories."""

from ._canonical import Digest
from ._root import get_root, set_root, use_root
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
    RootNotSet,
    TensordirError,
)
from .index import reindex
from .record import Record
from .spec import Spec, field
from .view import View, cached_method


def memory(location, *, lazy: bool = True, **kwargs):
    """A `joblib.Memory` storing xarray results as zarr and DataFrames as parquet (needs joblib)."""
    from ._memory import memory as _memory

    return _memory(location, lazy=lazy, **kwargs)


def needs_compute(func, *args, **kwargs) -> bool:
    """Whether calling the memoized `func` with these arguments would compute rather than load."""
    return not func.check_call_in_cache(*args, **kwargs)


__all__ = [
    "BuildFailed",
    "CanonicalizationError",
    "Codec",
    "Collection",
    "Digest",
    "Dims",
    "DimsError",
    "KeyCollision",
    "ReadOnlyError",
    "Record",
    "RecordBusy",
    "RootNotSet",
    "Spec",
    "TensordirError",
    "View",
    "cached_method",
    "dim_name",
    "dims_for",
    "field",
    "get_root",
    "memory",
    "needs_compute",
    "register_codec",
    "register_dims",
    "reindex",
    "set_root",
    "use_root",
]
