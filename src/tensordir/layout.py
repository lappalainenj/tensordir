"""`Layout`: how an array field is cut into chunks and shards on disk.

A layout changes how bytes are stored, never what a reader gets back, so it is a storage option
and not part of a record's key. Give it per call (`record.write(name, value, layout=...)`,
`record.append(..., layout=...)`) or per record type (`layouts = {"*": ..., "field": ...}`).
"""

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["Layout", "DEFAULT", "APPEND_DEFAULT"]


@dataclasses.dataclass(frozen=True)
class Layout:
    """Chunks and shards of an array field, by target bytes or explicit shape.

    - `chunk_bytes`: target uncompressed bytes per chunk. Chunks run along axis 0 and span the
      trailing axes whole, which suits row-range reads.
    - `chunks`: explicit chunk shape; overrides `chunk_bytes`.
    - `shard_bytes`: target uncompressed bytes per shard (zarr's `ShardingCodec`, one file per
      shard holding many chunks); `None` stores each chunk as its own file. An array with a
      single chunk is never sharded.
    - `shards`: explicit shard shape; overrides `shard_bytes`. Must be a multiple of the chunks.
    - `compressors`: passed to zarr; `"auto"` keeps zarr's default (zstd).

    The HDF5 codec uses the chunk shape and ignores shards and compressors.
    """

    chunk_bytes: int | None = None
    chunks: tuple[int, ...] | None = None
    shard_bytes: int | None = None
    shards: tuple[int, ...] | None = None
    compressors: Any = "auto"

    def __post_init__(self) -> None:
        for name in ("chunk_bytes", "shard_bytes"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, np.integer)) or value < 1):
                raise ValueError(f"Layout.{name} must be a positive int, got {value!r}")
        for name in ("chunks", "shards"):
            value = getattr(self, name)
            if value is not None:
                value = tuple(int(v) for v in value)
                if any(v < 1 for v in value):
                    raise ValueError(f"Layout.{name} must be positive, got {value}")
                object.__setattr__(self, name, value)

    def __tensordir_canonical__(self, *, full: bool = False) -> dict:
        """The fields that differ from the defaults, so a layout can be a (`hashed=False`)
        setting; JSON-able compressors only."""
        from ._canonical import canonical

        out = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if value != f.default:
                out[f.name] = canonical(value, full=full, _path=f"Layout.{f.name}")
        return out

    @classmethod
    def __tensordir_revive__(cls, value: Mapping) -> "Layout":
        return cls(**value)

    def resolve(self, shape: tuple[int, ...], dtype: Any, *, growing: bool = False) -> dict:
        """Chunk and shard shapes for an array of `shape` and `dtype`.

        Returns `{"chunks": tuple | None, "shards": tuple | None}`; `None` chunks leave the choice
        to zarr. A `growing` array (one created by `append`) keeps full chunk and shard sizes
        even while it holds fewer rows.
        """
        shape = tuple(int(s) for s in shape)
        if not shape:
            return {"chunks": None, "shards": None}
        itemsize = max(1, np.dtype(dtype).itemsize)
        trailing = tuple(max(s, 1) for s in shape[1:])
        if self.chunks is not None:
            if len(self.chunks) != len(shape):
                raise ValueError(f"Layout.chunks {self.chunks} does not match a {len(shape)}-d array")
            chunks = self.chunks
        elif self.chunk_bytes is not None or growing:
            target = self.chunk_bytes if self.chunk_bytes is not None else APPEND_CHUNK_BYTES
            chunks = (max(1, target // (itemsize * math.prod(trailing))), *trailing)
        else:
            chunks = None
        if chunks is not None and not growing:
            chunks = tuple(max(1, min(c, s)) for c, s in zip(chunks, shape, strict=True))
        return {"chunks": chunks, "shards": self._shards(shape, chunks, itemsize, growing)}

    def _shards(self, shape: tuple[int, ...], chunks: tuple[int, ...] | None, itemsize: int, growing: bool) -> tuple | None:
        if self.shards is not None:
            if chunks is None or len(self.shards) != len(shape) or any(
                s % c for s, c in zip(self.shards, chunks, strict=True)
            ):
                raise ValueError(f"Layout.shards {self.shards} must be a multiple of the chunks {chunks}")
            return self.shards
        if self.shard_bytes is None or chunks is None:
            return None
        if not growing and all(c >= max(s, 1) for c, s in zip(chunks, shape, strict=True)):
            return None  # one chunk: a shard would only add an index
        rows = max(1, self.shard_bytes // (itemsize * math.prod(chunks))) * chunks[0]
        if growing:
            return (rows, *chunks[1:])
        # at most the whole array, rounded up to whole chunks on every axis
        whole = [-(-max(s, 1) // c) * c for s, c in zip(shape, chunks, strict=True)]
        return (min(rows, whole[0]), *whole[1:])


#: Chunk bytes of an appended array when its layout names none.
APPEND_CHUNK_BYTES = 128 << 10
#: Layout of arrays written with `write` unless a call or the record type says otherwise.
DEFAULT = Layout(chunk_bytes=128 << 10, shard_bytes=128 << 20)
#: Layout of arrays created by `append`: chunks only, since every append rewrites the last shard.
APPEND_DEFAULT = Layout(chunk_bytes=128 << 10)


def resolve_layout(name: str, given: "Layout | None", layouts: Mapping[str, "Layout"]) -> "Layout | None":
    """The layout for field `name`: `given`, else `layouts[name]`, else `layouts["*"]`, else
    None (the codec's default: `DEFAULT` for zarr arrays, `APPEND_DEFAULT` for appended ones)."""
    for layout in (given, layouts.get(name), layouts.get("*")):
        if layout is not None:
            if not isinstance(layout, Layout):
                raise TypeError(f"{name}: layout must be a tensordir.Layout, got {type(layout).__qualname__}")
            return layout
    return None
