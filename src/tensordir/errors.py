"""Exceptions raised by tensordir."""


class TensordirError(Exception):
    """Base class of tensordir errors."""


class CanonicalizationError(TensordirError, TypeError):
    """A value has no canonical JSON form (NaN, raw arrays, unknown types)."""


class DimsError(TensordirError, ValueError):
    """An annotation does not map to a tuple of xarray dimension names."""


class ReadOnlyError(TensordirError, PermissionError):
    """A write was attempted outside of `Record.build`."""


class KeyCollision(TensordirError, RuntimeError):
    """A record directory holds a record with a different full key (16-hex prefix collision)."""


class RecordBusy(TensordirError, RuntimeError):
    """Another process holds a live lease on the record and `block=False` was given."""

    def __init__(self, record):
        super().__init__(f"{record!r} is being built by another process")
        self.record = record


class BuildFailed(TensordirError, RuntimeError):
    """The record's last build raised; its traceback is stored in `_meta.json`."""

    def __init__(self, record, error: dict):
        self.record = record
        self.error = error
        super().__init__(
            f"{record!r} failed: {error.get('type')}: {error.get('message')}\n"
            f"--- traceback of the failed build ---\n{error.get('traceback', '')}"
            "Pass retry_failed=True to rebuild."
        )
