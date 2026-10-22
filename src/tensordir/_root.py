"""Where records are stored: `base / path`, joined with pathlib semantics.

- The **base root** is the innermost `use_root` block, else `set_root`, else `$TENSORDIR_ROOT`,
  else the working directory at the moment of resolution.
- `path` is the `root=` argument of a call, else the record class's `root` (class attribute or
  `@tensordir.root(path)`), else nothing. `root=` replaces the class root; it is not joined onto it.

A relative `path` extends the base root and an absolute one replaces it; `~` is expanded. Every
resolved root has its own `.tensordir/` with leases, temporary builds and index.
"""

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

__all__ = ["get_root", "root", "set_root", "use_root"]

_default: Path | None = None
_current: ContextVar[Path | None] = ContextVar("tensordir_root", default=None)


def _absolute(path: str | os.PathLike) -> Path:
    return Path(path).expanduser().absolute()


def set_root(path: str | os.PathLike | None) -> None:
    """Set the process-wide base root (`None` unsets it).

    A relative path is resolved against the working directory now. Record classes with a relative
    `root` store under `path / root`; see `get_root` for the full rule.
    """
    global _default
    _default = None if path is None else _absolute(path)


@contextmanager
def use_root(path: str | os.PathLike) -> Iterator[Path]:
    """Use `path` as the base root inside a `with` block; yields it as an absolute path.

    A relative path is resolved against the working directory now. Classes with a relative `root`
    follow the block (`path / root`); classes with an absolute `root` do not. The setting is
    context-local: it applies to the current thread or asyncio task, and threads started inside
    the block do not see it (pass `root=` or call `set_root` there).
    """
    resolved = _absolute(path)
    token = _current.set(resolved)
    try:
        yield resolved
    finally:
        _current.reset(token)


def _base() -> Path:
    for candidate in (_current.get(), _default, os.environ.get("TENSORDIR_ROOT")):
        if candidate is not None and str(candidate):
            return _absolute(candidate)
    return Path.cwd()


def _record_class(cls: type) -> type:
    from ._directory import Directory

    if not (isinstance(cls, type) and issubclass(cls, Directory)):
        raise TypeError(f"expected a Record or Directory subclass, got {cls!r}")
    return cls


def get_root(cls: type | None = None, /, *, root: str | os.PathLike | None = None) -> Path:
    """The directory holding records of `cls`, a `Record` subclass (or the base of relative
    paths for a `Directory` subclass).

    The result is `base / path`: `base` is the innermost `use_root`, else `set_root`, else
    `$TENSORDIR_ROOT`, else the current working directory; `path` is `root` if given, else
    `cls.root`. A relative `path` extends `base`, an absolute one replaces it. Without `cls` and
    `root`, the base root itself.

    ```python
    td.get_root(Connectome)      # e.g. /data/connectomes for root = "connectomes"
    ```
    """
    if root is None and cls is not None:
        root = _record_class(cls).root
    base = _base()
    if root is None:
        return base
    return base / Path(root).expanduser()


def root(path: str | os.PathLike | None) -> Callable[[type], type]:
    """Class decorator: store records of the decorated `Record` subclass (and its subclasses),
    or resolve relative paths of a `Directory` subclass,
    under `path`, joined onto the base root. `None` removes an inherited root.

    ```python
    @tensordir.root("connectomes")          # <base root>/connectomes, follows use_root
    class Connectome(tensordir.Record): ...

    @tensordir.root("/data/stimuli")        # absolute: unaffected by the base root
    class Stimulus(tensordir.Record): ...

    @tensordir.root(Path(__file__).parent / "data")   # next to this source file
    class Fixture(tensordir.Record): ...
    ```

    Same as the class attribute `root = "connectomes"`. The path is stored as given and joined on
    every access, so a relative root follows `use_root`, `set_root` and the working directory.
    See `get_root`.
    """

    def decorate(cls: type) -> type:
        try:
            target = _record_class(cls)
        except TypeError:
            raise TypeError(f"@root decorates Record and Directory subclasses, not {cls!r}") from None
        target.root = path
        return cls

    return decorate
