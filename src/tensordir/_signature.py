"""A record's `__init__` signature as its interface: positional-only inputs, keyword-only settings.

```python
class Fit(td.Record):
    def __init__(self, points: Points, /, *, n_columns: int, seed: int = 0,
                 device: Annotated[str, td.field(hashed=False)] = "cpu",
                 log: Annotated[Logger | None, td.transient] = None):
        ...
```

- Positional-only parameters (before `/`) are the **inputs**, the things a record is computed
  from. They enter the key by position, so renaming one keeps every key.
- Keyword-only parameters (after `*`) are the **settings**. They form the generated `Fit.Spec`
  and enter the key by name. `td.field(...)` defaults, or `Annotated[T, td.field(...)]`, give
  field options (`hashed=False`, `since=`).
- `Annotated[T, td.transient]` keyword-only parameters reach `__init__` but neither the spec nor
  the key.

Other parameter kinds are rejected, so every parameter says which kind it is.
"""

import copy
import dataclasses
import functools
import inspect
import typing
from collections.abc import Mapping
from typing import Annotated, Any

from .spec import FieldOptions, Spec, field

__all__ = ["transient", "Interface", "interface_of", "RESERVED"]

#: Settings that would shadow keyword options of `Record.get`.
RESERVED = frozenset({"root", "block", "timeout", "retry_failed"})


class _Transient:
    """The marker type of `transient`."""

    def __repr__(self) -> str:
        return "tensordir.transient"


#: Mark a keyword-only `__init__` parameter as neither a setting nor an input:
#: `log: Annotated[Logger | None, td.transient] = None`.
transient = _Transient()


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


def _spec_default(name: str, default: Any, options: FieldOptions | None) -> Any:
    """The dataclass default of setting `name` from its parameter default and field options."""
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
    """The evaluated annotations of `func`, or its raw annotations if they do not evaluate."""
    try:
        return typing.get_type_hints(func, include_extras=True)
    except Exception:
        return dict(getattr(func, "__annotations__", {}))


@dataclasses.dataclass(frozen=True)
class Interface:
    """The inputs, settings and transient parameters of a record type's `__init__`."""

    signature: inspect.Signature
    inputs: tuple[str, ...]
    settings: tuple[str, ...]
    transients: tuple[str, ...]
    spec: type[Spec]

    def bind(self, args: tuple, kwargs: Mapping[str, Any]) -> tuple[tuple, Spec, dict[str, Any]]:
        """`(inputs, spec, transients)` for a call `Cls(*args, **kwargs)`; a spec given as
        `**spec` passes its fields."""
        bound = self.signature.bind(None, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        parameters = self.signature.parameters
        settings = {name: values[name] for name in self.settings if values[name] is not parameters[name].default}
        return (
            tuple(values[name] for name in self.inputs),
            self.spec(**settings),
            {name: values[name] for name in self.transients},
        )


def interface_of(cls: type, init: Any) -> Interface:
    """Parse `init` (the `__init__` of record type `cls`) into its interface and generated spec."""
    signature = inspect.signature(init)
    hints = _hints(init)
    inputs, settings, transients = [], [], []
    annotations: dict[str, Any] = {}
    namespace: dict[str, Any] = {}
    where = f"{cls.__qualname__}.__init__"
    for parameter in list(signature.parameters.values())[1:]:
        kind = parameter.kind
        if kind is inspect.Parameter.POSITIONAL_ONLY:
            inputs.append(parameter.name)
            continue
        if kind is not inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(
                f"{where}: parameter {parameter.name!r} must be positional-only (an input, before /) or "
                f"keyword-only (a setting, after *)"
            )
        annotation, options, is_transient = _split(hints.get(parameter.name, Any))
        if is_transient:
            transients.append(parameter.name)
            continue
        if parameter.name in RESERVED or parameter.name.startswith("_"):
            raise TypeError(f"{where}: the setting name {parameter.name!r} is reserved")
        settings.append(parameter.name)
        annotations[parameter.name] = annotation
        default = _spec_default(parameter.name, parameter.default, options)
        if default is not dataclasses.MISSING:
            namespace[parameter.name] = default
    spec = type(
        "Spec",
        (Spec,),
        {"__annotations__": annotations, "__module__": cls.__module__, "__qualname__": f"{cls.__qualname__}.Spec", **namespace},
    )
    return Interface(signature, tuple(inputs), tuple(settings), tuple(transients), spec)
