"""Canonical JSON form of spec values, and the sha256 keys derived from it.

Rules: mappings get sorted string keys; numpy scalars become Python scalars; paths become POSIX
strings; floats serialise through `repr` (shortest round-trip form); -0.0 becomes 0.0; infinities
are tagged; NaN is rejected; arrays enter only through a `Digest`.
"""

import dataclasses
import enum
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any

import numpy as np

from .errors import CanonicalizationError

__all__ = ["Digest", "canonical", "dumps", "sha256"]


@dataclasses.dataclass(frozen=True)
class Digest:
    """Content fingerprint of an array: sha256 over dtype, shape and C-order bytes.

    Put a `Digest` in a spec where the record depends on array data that is too large to hash as
    JSON. Byte order is normalised to little-endian, so equal values give equal digests.
    """

    sha256: str
    dtype: str | None = dataclasses.field(default=None, compare=False)
    shape: tuple[int, ...] | None = dataclasses.field(default=None, compare=False)

    @classmethod
    def of(cls, array: Any) -> "Digest":
        a = np.asarray(array)
        if a.dtype.hasobject:
            raise CanonicalizationError("object arrays have no stable digest")
        if a.dtype.byteorder == ">":
            a = a.astype(a.dtype.newbyteorder("<"))
        a = np.ascontiguousarray(a)
        h = hashlib.sha256()
        h.update(json.dumps([a.dtype.str, list(a.shape)]).encode())
        h.update(a.tobytes())
        return cls(h.hexdigest(), a.dtype.str, tuple(a.shape))


def _reserved(key: str) -> bool:
    return key.startswith("__") and key.endswith("__")


def canonical(value: Any, *, full: bool = False, digest_arrays: bool = False, _path: str = "value") -> Any:
    """Return the canonical JSON-compatible form of `value`.

    `full=True` keeps spec fields that are excluded from keys (`hashed=False`, `since=` at default);
    `digest_arrays=True` replaces ndarrays by their `Digest` instead of rejecting them.
    """
    if value is None or isinstance(value, str):
        return str(value) if isinstance(value, str) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if math.isnan(x):
            raise CanonicalizationError(f"{_path}: NaN has no canonical form (equality is undefined)")
        if math.isinf(x):
            return {"__float__": "inf" if x > 0 else "-inf"}
        return 0.0 if x == 0 else x
    if isinstance(value, enum.Enum):
        return canonical(value.value, full=full, digest_arrays=digest_arrays, _path=_path)
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, Digest):
        return {"__digest__": value.sha256}
    hook = getattr(type(value), "__tensordir_canonical__", None)
    if hook is not None:
        return hook(value, full=full)
    if isinstance(value, Mapping):
        out = {}
        for k in value:
            if not isinstance(k, str):
                raise CanonicalizationError(f"{_path}: mapping keys must be str, got {type(k).__name__}")
            if _reserved(k):
                raise CanonicalizationError(f"{_path}: key {k!r} is reserved for tensordir tags")
            out[k] = canonical(value[k], full=full, digest_arrays=digest_arrays, _path=f"{_path}.{k}")
        return dict(sorted(out.items()))
    if isinstance(value, (list, tuple)):
        return [
            canonical(v, full=full, digest_arrays=digest_arrays, _path=f"{_path}[{i}]")
            for i, v in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        items = [canonical(v, full=full, digest_arrays=digest_arrays, _path=f"{_path}{{}}") for v in value]
        return sorted(items, key=dumps)
    if isinstance(value, np.ndarray):
        if digest_arrays:
            return canonical(Digest.of(value))
        raise CanonicalizationError(f"{_path}: wrap arrays in tensordir.Digest(...) to use them in a spec")
    raise CanonicalizationError(f"{_path}: cannot canonicalize {type(value).__module__}.{type(value).__qualname__}")


def dumps(value: Any) -> str:
    """Serialise an already canonical value to its unique JSON string."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256(value: Any) -> str:
    """Hex sha256 of the canonical JSON of an already canonical value."""
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def untag(value: Any) -> Any:
    """Invert the float and digest tags of a canonical value (used when reviving specs)."""
    if isinstance(value, dict):
        if set(value) == {"__float__"}:
            return float(value["__float__"])
        if set(value) == {"__digest__"}:
            return Digest(value["__digest__"])
        return {k: untag(v) for k, v in value.items()}
    if isinstance(value, list):
        return [untag(v) for v in value]
    return value
