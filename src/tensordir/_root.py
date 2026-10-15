"""The storage root: a process-wide default, a context-local override, or `TENSORDIR_ROOT`."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from .errors import RootNotSet

__all__ = ["get_root", "set_root", "use_root"]

_default: Path | None = None
_current: ContextVar[Path | None] = ContextVar("tensordir_root", default=None)


def set_root(path: str | os.PathLike | None) -> None:
    """Set the process-wide default root (`None` unsets it)."""
    global _default
    _default = None if path is None else Path(path).expanduser().absolute()


@contextmanager
def use_root(path: str | os.PathLike) -> Iterator[Path]:
    """Use `path` as the root inside a `with` block (context-local; safe with threads and asyncio)."""
    root = Path(path).expanduser().absolute()
    token = _current.set(root)
    try:
        yield root
    finally:
        _current.reset(token)


def get_root(root: str | os.PathLike | None = None) -> Path:
    """Resolve the root: the argument, then `use_root`, then `set_root`, then `$TENSORDIR_ROOT`."""
    if root is not None:
        return Path(root).expanduser().absolute()
    for candidate in (_current.get(), _default, os.environ.get("TENSORDIR_ROOT")):
        if candidate:
            return Path(candidate).expanduser().absolute()
    raise RootNotSet("no root: pass root=..., call tensordir.set_root(path), or set TENSORDIR_ROOT")
