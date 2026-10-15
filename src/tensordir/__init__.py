"""tensordir: content-addressed records of labeled tensors in directories."""

from ._canonical import Digest
from ._root import get_root, set_root, use_root
from .codecs import Codec, register_codec
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
from .record import Record
from .spec import Spec, field

__all__ = [
    "BuildFailed",
    "CanonicalizationError",
    "Codec",
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
    "dim_name",
    "dims_for",
    "field",
    "get_root",
    "register_codec",
    "register_dims",
    "set_root",
    "use_root",
]
