"""`@record`: a regular class whose `__init__` arguments are the spec of a record it owns.

```python
@tensordir.record
class Trainer:
    def __init__(self, lr: float = 0.1, n_steps: int = 100, device: Annotated[str, td.field(hashed=False)] = "cpu"):
        self.model = make_model()
        self.record.config_note = "..."  # __init__ is the build: the record is `running` meanwhile

    def step(self, losses):
        self.record.losses = losses   # a workspace record keyed by (lr, n_steps), writable any time

trainer = Trainer(lr=0.1)             # builds the record, or reruns __init__ on the finished one
trainer.lr = 0.2                      # the next `trainer.record` is the record for lr=0.2
```
"""

import copy
import dataclasses
import functools
import inspect
import os
import threading
import typing
from typing import Annotated, Any

from ._record import Record
from .errors import BuildFailed
from .spec import FieldOptions, Spec, field

__all__ = ["record", "transient"]

_STATE = "__tensordir_state__"
_OWN = ("Spec", "Record", "record", "spec")


class _Transient:
    def __repr__(self) -> str:
        return "tensordir.transient"


#: Mark an `__init__` argument as not part of the spec: `logger: Annotated[Logger, td.transient]`.
#: Experimental: may change or be removed.
transient = _Transient()


class _State:
    """The spec of an owner instance and the record it currently writes to."""

    def __init__(self, spec: Spec):
        self.spec = spec
        self.record: Record | None = None
        self.lock = threading.RLock()

    def __getstate__(self) -> dict:
        return {"spec": self.spec}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["spec"])


class _SpecAttribute(property):
    """An instance attribute backed by a spec field."""


def _state(obj: Any) -> _State:
    try:
        return obj.__dict__[_STATE]
    except KeyError:
        raise AttributeError(f"{type(obj).__qualname__}.__init__ has not run") from None


def _split(annotation: Any) -> tuple[Any, FieldOptions | None, bool]:
    """The annotation without tensordir markers, its field options, and whether it is transient."""
    if typing.get_origin(annotation) is not Annotated:
        return annotation, None, False
    base, *extras = typing.get_args(annotation)
    options, is_transient, rest = None, False, []
    for extra in extras:
        if extra is transient:
            is_transient = True
        elif isinstance(extra, FieldOptions):
            options = extra
        elif isinstance(extra, dataclasses.Field) and "tensordir" in extra.metadata:
            options = extra.metadata["tensordir"]
        else:
            rest.append(extra)
    return (Annotated[(base, *rest)] if rest else base), options, is_transient


def _default_of(f: dataclasses.Field) -> Any:
    return f.default if f.default is not dataclasses.MISSING else f.default_factory()


def _spec_field(name: str, default: Any, options: FieldOptions | None) -> Any:
    if isinstance(default, dataclasses.Field):
        if options is not None:
            raise TypeError(f"{name}: give tensordir field options in the annotation or the default, not both")
        return copy.copy(default)
    kwargs = {} if options is None else {"hashed": options.hashed, "since": options.since}
    if default is inspect.Parameter.empty:
        return field(**kwargs) if kwargs else dataclasses.MISSING
    if type(default).__hash__ is None:  # lists and dicts need a factory in a dataclass
        return field(default_factory=functools.partial(copy.deepcopy, default), **kwargs)
    return field(default, **kwargs) if kwargs else default


def _hints(func: Any) -> dict[str, Any]:
    try:
        return typing.get_type_hints(func, include_extras=True)
    except Exception:
        return dict(getattr(func, "__annotations__", {}))


def record(
    cls: type | None = None,
    /,
    *,
    name: str | None = None,
    root: str | os.PathLike | None = None,
    base: type[Record] = Record,
    mutable: bool = True,
    schema_version: int = 0,
) -> Any:
    """Class decorator: instances own a record whose spec is their `__init__` arguments.

    - Every named `__init__` argument is a spec field, typed by its annotation (`Any` if none),
      except those annotated `Annotated[T, tensordir.transient]`. Keep an argument out of the key
      with a `tensordir.field(..., hashed=False)` default or `Annotated[T, tensordir.field(hashed=False)]`.
    - Spec fields become instance attributes backed by `self.spec`. Assigning one
      (`obj.lr = 0.2`, also inside `__init__`) replaces the spec; if the key changes, the next
      `obj.record` is the record for the new spec. Changing an unhashed field keeps the record.
    - `__init__` is the build. If the record is not `done`, constructing the object takes the
      record's lease and runs `__init__` as the build: the record is `running` meanwhile, `done`
      when `__init__` returns, `failed` (traceback stored, exception re-raised) if it raises;
      the next construction retries. Another process constructing the same spec waits for the
      lease. If the record is already `done`, `__init__` runs without the lease on that record.
    - `obj.record` is that record: a workspace record (`mutable=True`) that accepts
      `obj.record.x = value` and `append` at any time after the build, without changing its
      status; each edit updates `meta["modified"]`.
    - `Cls.Spec` and `Cls.Record` are the generated classes; `Cls.Record.query(lr=0.1)` finds runs.

    `name` is the record type id (default: the class's qualified name), `root` its class root
    (joined onto the base root, see `tensordir.get_root`), `base` a `Record` subclass providing
    typed fields, `dims`, codecs or `in_place`. `mutable=False` makes the record read-only once
    `__init__` has built it.
    """
    if cls is None:
        return lambda c: record(c, name=name, root=root, base=base, mutable=mutable, schema_version=schema_version)
    if not isinstance(cls, type):
        raise TypeError(f"@record decorates classes, not {cls!r}")
    init = getattr(cls.__init__, "__tensordir_original__", cls.__init__)
    if init is object.__init__:
        raise TypeError(f"{cls.__qualname__} needs an __init__ whose arguments form the spec")
    signature = inspect.signature(init)
    hints = _hints(init)
    annotations: dict[str, Any] = {}
    namespace: dict[str, Any] = {}
    factories: dict[str, dataclasses.Field] = {}
    for parameter in list(signature.parameters.values())[1:]:
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(f"{cls.__qualname__}.__init__: *{parameter.name} cannot be part of a spec")
        annotation, options, is_transient = _split(hints.get(parameter.name, Any))
        if is_transient:
            continue
        if parameter.name in _OWN or parameter.name.startswith("_"):
            raise TypeError(f"{cls.__qualname__}.__init__: argument name {parameter.name!r} is reserved")
        existing = inspect.getattr_static(cls, parameter.name, None)
        if existing is not None and not isinstance(existing, _SpecAttribute):
            raise TypeError(f"{cls.__qualname__}.{parameter.name} already exists; it cannot also be a spec field")
        annotations[parameter.name] = annotation
        default = _spec_field(parameter.name, parameter.default, options)
        if default is not dataclasses.MISSING:
            namespace[parameter.name] = default
        if isinstance(parameter.default, dataclasses.Field):
            factories[parameter.name] = parameter.default
    for attr in ("Spec", "Record", "record", "spec"):
        existing = inspect.getattr_static(cls, attr, None)
        if existing is not None and not getattr(cls, "__tensordir_owner__", False):
            raise TypeError(f"{cls.__qualname__}.{attr} already exists; @record defines it")

    spec_cls = type(
        "Spec",
        (Spec,),
        {"__annotations__": annotations, "__module__": cls.__module__, "__qualname__": f"{cls.__qualname__}.Spec", **namespace},
    )
    record_ns: dict[str, Any] = {
        "__record_name__": name or cls.__qualname__,
        "Spec": spec_cls,
        "mutable": mutable,
        "schema_version": schema_version,
        "__module__": cls.__module__,
        "__qualname__": f"{cls.__qualname__}.Record",
    }
    if root is not None:
        record_ns["root"] = root
    record_cls = type("Record", (base,), record_ns)

    @functools.wraps(init)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _STATE in self.__dict__:  # super().__init__ from a decorated subclass: its spec wins
            init(self, *args, **kwargs)
            return
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        for arg, f in factories.items():
            if bound.arguments[arg] is f:
                bound.arguments[arg] = _default_of(f)
        state = _State(spec_cls(**{arg: bound.arguments[arg] for arg in annotations}))
        self.__dict__[_STATE] = state
        state.record = record_cls.locate(state.spec)
        built = False

        def recipe() -> None:
            nonlocal built
            built = True
            init(*bound.args, **bound.kwargs)

        error: BaseException | None = None
        try:  # __init__ is the build: under the lease, `running` until it returns
            state.record._obtain({}, block=True, timeout=None, retry_failed=True, recipe=recipe)
        except BuildFailed as exc:
            if not built or exc.__cause__ is None:
                raise
            error = exc.__cause__
        if error is not None:
            raise error  # the caller's own exception; the failed record keeps the traceback
        if not built:  # already done: run __init__ again on the existing record
            init(*bound.args, **bound.kwargs)

    __init__.__tensordir_original__ = init
    for arg in annotations:
        setattr(cls, arg, _spec_attribute(arg, record_cls))
    cls.__init__ = __init__
    cls.Spec = spec_cls
    cls.Record = record_cls
    cls.spec = property(lambda self: _state(self).spec, doc="The spec formed by the spec attributes.")
    cls.record = property(_get_record(record_cls), doc="The record for the current spec (created on first use).")
    cls.__tensordir_owner__ = True
    return cls


def _spec_attribute(name: str, record_cls: type[Record]) -> _SpecAttribute:
    def get(self: Any) -> Any:
        return getattr(_state(self).spec, name)

    def set(self: Any, value: Any) -> None:
        state = _state(self)
        with state.lock:
            new = state.spec.replace(**{name: value})
            current = state.record
            if current is not None:
                if record_cls.compute_key(new) == current.key:
                    current._spec = new
                else:
                    state.record = None
            state.spec = new

    return _SpecAttribute(get, set, doc=f"Spec field {name!r}; assigning it switches to the matching record.")


def _get_record(record_cls: type[Record]) -> Any:
    def get(self: Any) -> Record:
        state = _state(self)
        with state.lock:
            if state.record is None:
                state.record = record_cls.get(state.spec)
            return state.record

    return get
