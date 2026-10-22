"""Builds in named stages: order checks, resume after a crash, and optional stages after `done`.

```python
class Fit(td.Record):
    stages = td.Stages(
        load=td.Stage(),
        fit=td.Stage(after="load"),
        epoch=td.Stage(after="fit", repeat=True),
        preview=td.Stage(after="fit", required=False),
    )

    def __init__(self, points, /, *, n_epochs: int = 3):
        self.load(points).fit()
        for _ in range(n_epochs):
            self.epoch()

    @stages
    def load(self, points) -> Self: ...
```

- A stage method returns `self`, so stages chain.
- A stage runs only after the stages named in `after`, or raises `StageOrderError`.
- The build finishes only when every required stage has run.
- A record with stages builds in place by default (`in_place = True`). A failed build is resumed
  by `retry_failed=True`: stages that finished are skipped, and a repeated stage skips as many
  calls as finished before. So stages pass results to later stages through fields
  (`self.x = ...`), not private attributes.
- An optional stage (`required=False`) may run after the record is `done`. It may add fields and
  rewrite the fields it added itself, never the others.
- `record.progress()` lists every stage with its calls, times and fields, also while another
  process builds the record.
"""

import dataclasses
import datetime
import functools
import time
from collections.abc import Callable, Iterable
from typing import Any

from .errors import ReadOnlyError, TensordirError

__all__ = ["Stage", "Stages", "StageOrderError", "StageProgress", "Progress"]


class StageOrderError(TensordirError):
    """A stage was called before the stages it runs after, or a required stage never ran."""


def _now() -> str:
    """The current time in ISO format (UTC)."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@dataclasses.dataclass(frozen=True)
class Stage:
    """One stage of a build: the stages it runs `after`, whether it may `repeat`, and whether the
    build is `required` to run it."""

    after: str | Iterable[str] = ()
    repeat: bool = False
    required: bool = True

    def __post_init__(self) -> None:
        after = (self.after,) if isinstance(self.after, str) else tuple(self.after)
        object.__setattr__(self, "after", after)


class Stages:
    """The stages of a record type, by name; decorate each stage method with the instance."""

    def __init__(self, **stages: Stage):
        for name, stage in stages.items():
            if not isinstance(stage, Stage):
                raise TypeError(f"stage {name!r} must be a td.Stage, got {stage!r}")
            unknown = set(stage.after) - set(stages)
            if unknown:
                raise ValueError(f"stage {name!r} runs after unknown stages {sorted(unknown)}")
        self.stages = dict(stages)
        self.methods: dict[str, Callable] = {}

    def __call__(self, method: Callable) -> Callable:
        """Register `method` as the stage of the same name."""
        name = method.__name__
        if name not in self.stages:
            raise ValueError(f"{method.__qualname__}: no stage named {name!r}; declared: {sorted(self.stages)}")
        self.methods[name] = method

        @functools.wraps(method)
        def run(record: Any, *args: Any, **kwargs: Any) -> Any:
            _run_stage(record, self, name, method, args, kwargs)
            return record

        run.__tensordir_stage__ = name
        return run

    def __repr__(self) -> str:
        return f"Stages({', '.join(self.stages)})"


def _state(record: Any) -> dict:
    """The per-stage progress stored in the record's metadata: calls, finish time, own fields."""
    return record._meta.setdefault("stages", {})


def _check_order(record: Any, stages: Stages, name: str, progress: dict) -> None:
    """Raise unless every stage `name` runs after has finished."""
    missing = [s for s in stages.stages[name].after if progress.get(s, {}).get("calls", 0) == 0]
    if missing:
        raise StageOrderError(f"{record!r}: stage {name!r} runs after {missing}, which have not run")


def _run_stage(record: Any, stages: Stages, name: str, method: Callable, args: tuple, kwargs: dict) -> None:
    """Run stage `name` of `record`: inside the build, resuming where an earlier attempt stopped,
    or afterwards for an optional stage."""
    stage = stages.stages[name]
    if not record._writable:
        if stage.required:
            raise ReadOnlyError(f"{record!r}: stage {name!r} is required, so it runs inside the build only")
        status = record.status
        if status != "done":
            raise ReadOnlyError(f"{record!r} is {status}; optional stages run on finished records")
        with record._stage_edit(name) as progress:
            _check_order(record, stages, name, progress)
            if not stage.repeat and progress.get(name, {}).get("calls", 0):
                return
            _call(record, name, method, args, kwargs, progress)
        return
    progress = _state(record)
    calls = record.__dict__.setdefault("_stage_calls", {})
    calls[name] = calls.get(name, 0) + 1
    done = progress.get(name, {}).get("calls", 0)
    if not stage.repeat and calls[name] > 1:
        raise StageOrderError(f"{record!r}: stage {name!r} ran twice; declare it with repeat=True")
    if calls[name] <= done:  # finished by an earlier attempt of this build
        return
    _check_order(record, stages, name, progress)
    _call(record, name, method, args, kwargs, progress)
    record._write_meta()


def _call(record: Any, name: str, method: Callable, args: tuple, kwargs: dict, progress: dict) -> None:
    """Call the stage method and note its calls, start and finish times, duration and the fields it
    added. Inside a build, the start is written first, so readers see which stage is running."""
    info = progress.setdefault(name, {"calls": 0, "started": None, "finished": None, "duration_s": None, "fields": []})
    info["started"] = _now()
    if record._writable and record.__dict__.get("_stage_owner") is None:
        record._write_meta()
    before = set(record._manifest())
    t0 = time.monotonic()
    method(record, *args, **kwargs)
    info["calls"] += 1
    info["finished"] = _now()
    info["duration_s"] = time.monotonic() - t0
    info["fields"] = sorted(set(info["fields"]) | (set(record._manifest()) - before))


@dataclasses.dataclass(frozen=True)
class StageProgress:
    """One stage of a record: its declaration and what has run (times of the latest call)."""

    name: str
    state: str  # "pending", "running", "failed" or "done"
    calls: int
    started: str | None
    finished: str | None
    duration_s: float | None
    fields: tuple[str, ...]
    after: tuple[str, ...]
    repeat: bool
    required: bool


class Progress(tuple):
    """The stages of a record in declaration order, as `StageProgress` items."""

    @property
    def current(self) -> StageProgress | None:
        """The stage running now, or None."""
        return next((s for s in self if s.state == "running"), None)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            for stage in self:
                if stage.name == key:
                    return stage
            raise KeyError(f"no stage {key!r}; stages: {[s.name for s in self]}")
        return super().__getitem__(key)

    def __repr__(self) -> str:
        if not self:
            return "Progress(no stages)"
        width = max(len(s.name) for s in self)
        lines = [
            (
                f"{s.name:<{width}}  {s.state:<7}  calls {s.calls:<3}"
                + (f"  {'last call' if s.repeat else 'took'} {s.duration_s:.2f} s" if s.duration_s is not None else "")
                + (f"  fields {', '.join(s.fields)}" if s.fields else "")
            ).rstrip()
            for s in self
        ]
        return "\n".join(lines)


def progress_of(record: Any, meta: dict, status: str) -> Progress:
    """The progress of every declared stage of `record`, from its metadata `meta` and `status`."""
    stored = meta.get("stages", {})
    items = []
    for stages in stages_of(type(record)):
        for name, stage in stages.stages.items():
            info = stored.get(name, {})
            started, finished = info.get("started"), info.get("finished")
            if started is not None and (finished is None or started > finished):
                state = "failed" if status in ("failed", "stale") else "running"
            else:
                state = "done" if info.get("calls", 0) else "pending"
            items.append(
                StageProgress(
                    name=name,
                    state=state,
                    calls=info.get("calls", 0),
                    started=started,
                    finished=finished,
                    duration_s=info.get("duration_s"),
                    fields=tuple(info.get("fields", ())),
                    after=stage.after,
                    repeat=stage.repeat,
                    required=stage.required,
                )
            )
    return Progress(items)


def check_required(record: Any) -> None:
    """Raise at the end of a build if a required stage of `record` never ran."""
    for stages in stages_of(type(record)):
        progress = record._meta.get("stages", {})
        missing = [n for n, s in stages.stages.items() if s.required and progress.get(n, {}).get("calls", 0) == 0]
        if missing:
            raise StageOrderError(f"{record!r}: required stages {missing} never ran")


def stages_of(cls: type) -> list[Stages]:
    """The `Stages` declared on `cls` or its bases."""
    found = []
    for klass in cls.__mro__:
        for value in vars(klass).values():
            if isinstance(value, Stages) and value not in found:
                found.append(value)
    return found
