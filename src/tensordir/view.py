"""`View`: an in-memory companion of a record for derived, cached quantities.

A view holds only its record; it pickles (and joblib-hashes) as the record's identity. Domain
code subclasses it, as flyvis's `NetworkView` wraps a network directory.
"""

import functools
import inspect
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from ._canonical import canonical, dumps

__all__ = ["View", "cached_method"]


class View:
    """Wraps `record`; unknown attributes resolve on the record.

    `memory` is a joblib cache inside the record (`<record>/__cache__`).
    """

    def __init__(self, record: Any):
        self.record = record

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or "record" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.record, name)

    @functools.cached_property
    def memory(self):
        return self.record.memory

    def needs_compute(self, func: Any, *args: Any, **kwargs: Any) -> bool:
        """Whether the memoized `func` would compute (not load) for these arguments."""
        return not func.check_call_in_cache(*args, **kwargs)

    def __getstate__(self) -> dict:
        return {"record": self.record}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return f"{type(self).__qualname__}({self.record!r})"

    def _repr_html_(self) -> str:
        import html

        return f"<p><b>{html.escape(type(self).__qualname__)}</b> of</p>" + self.record._repr_html_()


def cached_method(func: Callable | None = None, /, *, maxsize: int | None = 128) -> Any:
    """Cache a method per instance, keyed by the canonical form of its bound arguments.

    Arguments may be anything with a canonical form: JSON-like values, specs, records (by key),
    and arrays (by digest). Unlike `functools.lru_cache`, the cache lives on the instance and list
    order matters. Clear it with `method.cache_clear(instance)`.
    """
    if func is None:
        return lambda f: cached_method(f, maxsize=maxsize)
    signature = inspect.signature(func)
    attr = f"__tensordir_cache_{func.__name__}"

    @functools.wraps(func)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        arguments = dict(list(bound.arguments.items())[1:])
        key = dumps(canonical(arguments, digest_arrays=True, _path=f"{func.__qualname__}()"))
        cache = self.__dict__.setdefault(attr, OrderedDict())
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        value = func(self, *args, **kwargs)
        cache[key] = value
        if maxsize is not None and len(cache) > maxsize:
            cache.popitem(last=False)
        return value

    wrapper.cache_clear = lambda instance: instance.__dict__.pop(attr, None)
    return wrapper
