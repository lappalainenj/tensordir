"""Map jaxtyping shape strings to xarray dimension names.

Sizes are named `n_<things>` (`n_cells`, `n_timesteps`); xarray dimensions are the singular thing
(`cell`, `timestep`). The default rule strips `n_` and one trailing `s`; irregular plurals come
from a registry (`register_dims`) or a record class's `dims` mapping. `name=size` tokens
(`"pre=n_types post=n_types"`) use `name`. Symbolic (`n_bins+1`), literal (`2`), anonymous (`_`)
and variadic (`*batch`, `...`) dimensions have no name and need explicit `Dims(...)`.
"""

import types
import typing
from collections.abc import Mapping
from typing import Any

from .errors import DimsError

__all__ = ["Dims", "dim_name", "dims_for", "register_dims"]

_IRREGULAR = {
    "n_vertices": "vertex",
    "n_indices": "index",
    "n_matrices": "matrix",
    "n_radii": "radius",
    "n_axes": "axis",
    "n_classes": "class",
    "n_aliases": "alias",
    "n_analyses": "analysis",
    "n_series": "series",
    "n_species": "species",
}
_REGISTRY: dict[str, str] = {}


class Dims:
    """Explicit dimension names: `Annotated[Float[np.ndarray, "n_bins+1"], Dims("bin_edge")]`."""

    def __init__(self, *names: str):
        if not all(isinstance(n, str) and n for n in names):
            raise DimsError(f"dimension names must be non-empty strings, got {names!r}")
        self.names = tuple(names)

    def __repr__(self) -> str:
        return f"Dims{self.names!r}"


def register_dims(mapping: Mapping[str, str] | None = None, /, **names: str) -> None:
    """Register size-name → dimension-name overrides for this process (e.g. `n_vertices="vertex"`)."""
    _REGISTRY.update(mapping or {})
    _REGISTRY.update(names)


def dim_name(size: str, overrides: Mapping[str, str] | None = None) -> str:
    """The xarray dimension name for a size name: overrides, registry, irregulars, then the rule."""
    for table in (overrides or {}, _REGISTRY, _IRREGULAR):
        if size in table:
            return table[size]
    if size.startswith("n_") and len(size) > 2:
        base = size[2:]
        if base.endswith("s") and not base.endswith("ss") and len(base) > 1:
            base = base[:-1]
        return base
    return size


def _unwrap(annotation: Any) -> tuple[Any, Dims | None]:
    explicit = None
    while True:
        origin = typing.get_origin(annotation)
        if origin is typing.Annotated:
            base, *extras = typing.get_args(annotation)
            explicit = next((e for e in extras if isinstance(e, Dims)), explicit)
            annotation = base
        elif origin in (typing.Union, types.UnionType):
            args = [a for a in typing.get_args(annotation) if a is not type(None)]
            if len(args) != 1:
                return annotation, explicit
            annotation = args[0]
        else:
            return annotation, explicit


def dims_for(annotation: Any, overrides: Mapping[str, str] | None = None) -> tuple[str, ...] | None:
    """Dimension names declared by a field annotation, or None if it declares no shape."""
    annotation, explicit = _unwrap(annotation)
    if explicit is not None:
        return explicit.names
    dim_str = getattr(annotation, "dim_str", None)
    if not isinstance(dim_str, str):
        return None
    out = []
    for token in dim_str.split():
        if token == "..." or "*" in token:
            raise DimsError(f"variadic dimension {token!r} in {dim_str!r}: give explicit Dims(...)")
        name = token.lstrip("#?").partition("=")[0]
        if name.startswith("_") and name[1:].isidentifier():
            name = name[1:]
        if not name or name == "_":
            raise DimsError(f"anonymous dimension {token!r} in {dim_str!r}: give explicit Dims(...)")
        if name.isdigit():
            raise DimsError(f"literal size {token!r} in {dim_str!r} has no name: write 'name={name}' or give Dims(...)")
        if not name.isidentifier():
            raise DimsError(f"symbolic dimension {token!r} in {dim_str!r}: give explicit Dims(...)")
        out.append(dim_name(name, overrides))
    if len(set(out)) != len(out):
        raise DimsError(f"{dim_str!r} maps to repeated dimensions {out}: name them, e.g. 'pre=n_types post=n_types'")
    return tuple(out)
