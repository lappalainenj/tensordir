"""The storage root: an argument, a context-local override, a record class's root, a process-wide
default, or `TENSORDIR_ROOT`."""

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from .errors import RootNotSet

__all__ = ["get_root", "root", "set_root", "use_root"]

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


def get_root(root: str | os.PathLike | None = None, *, class_root: str | os.PathLike | None = None) -> Path:
    """Resolve the root, first match wins:

    1. the `root` argument,
    2. the innermost `use_root` block,
    3. `class_root` (a record class's `root`, set with `@tensordir.root(path)`),
    4. `set_root`,
    5. `$TENSORDIR_ROOT`.
    """
    for candidate in (root, _current.get(), class_root, _default, os.environ.get("TENSORDIR_ROOT")):
        if candidate is not None and str(candidate):
            return Path(candidate).expanduser().absolute()
    raise RootNotSet("no root: pass root=..., call tensordir.set_root(path), or set TENSORDIR_ROOT")


def root(path: str | os.PathLike | None) -> Callable[[type], type]:
    """Class decorator: store records of the decorated `Record` subclass (and its subclasses)
    under `path` unless `root=` or `use_root` say otherwise. `None` removes an inherited root.

    ```python
    @tensordir.root("/data/connectomes")
    class Connectome(tensordir.Record): ...
    ```

    Same as the class attribute `root = "/data/connectomes"`; the decorator also accepts classes
    made with `@tensordir.record`. A relative path is resolved against the working directory now.
    """
    resolved = None if path is None else Path(path).expanduser().absolute()

    def decorate(cls: type) -> type:
        from ._record import Record

        target = getattr(cls, "Record", None) if getattr(cls, "__tensordir_owner__", False) else cls
        if not (isinstance(target, type) and issubclass(target, Record)):
            raise TypeError(f"@root decorates Record subclasses or @record classes, not {cls!r}")
        target.root = resolved
        return cls

    return decorate
