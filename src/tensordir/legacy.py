"""Read-only access to datamate directories.

A datamate directory holds `_meta.yaml` (`config`, `status`; very old ones use `spec` for
`config`), one HDF5 file per array with a single dataset `data`, CSV tables, other files, and
subdirectories that are themselves directories. Needs the `legacy` extra (h5py, pyyaml) and pandas
for CSV files.
"""

import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["DatamateDirectory", "open_datamate"]

_META = "_meta.yaml"


def _read_h5(path: Path) -> np.ndarray:
    import h5py

    # locking=False: HDF5 file locks misbehave on NFS/Lustre and block concurrent readers
    with h5py.File(path, "r", locking=False) as f:
        data = f["data"][()]
    if isinstance(data, bytes):
        return np.array(data.decode("utf-8"))
    if isinstance(data, np.ndarray) and data.dtype.kind == "S":
        return np.char.decode(data, "utf-8")
    if isinstance(data, np.ndarray) and data.dtype.kind == "O":
        return np.array([v.decode("utf-8") if isinstance(v, bytes) else v for v in data.ravel()]).reshape(data.shape)
    return data


class DatamateDirectory(Mapping):
    """A datamate directory, read-only. Keys are entry names without `.h5`/`.csv` suffixes.

    `d["x"]` returns an `ndarray` for `x.h5`, a `DataFrame` for `x.csv`, a `DatamateDirectory` for
    a subdirectory and a `Path` for any other file. Attribute access maps `__` to `.` as datamate
    did (`d.a__b` is `d["a.b"]`).
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)

    @property
    def meta(self) -> dict:
        """`_meta.yaml` with legacy `spec` renamed to `config`; `{config: None, status: done}` if absent."""
        import yaml

        meta_path = self.path / _META
        if not meta_path.exists():
            return {"config": None, "status": "done"}
        meta = yaml.safe_load(meta_path.read_text()) or {}
        if "spec" in meta and "config" not in meta:
            meta["config"] = meta.pop("spec")
        meta.setdefault("config", None)
        meta.setdefault("status", "done")
        return meta

    @property
    def config(self) -> Any:
        return self.meta["config"]

    @property
    def status(self) -> str:
        return self.meta["status"]

    @property
    def type(self) -> str | None:
        """The `type` entry of the config (the datamate Directory subclass name), if any."""
        config = self.config
        return config.get("type") if isinstance(config, dict) else None

    def _entries(self) -> dict[str, Path]:
        entries = {}
        if not self.path.is_dir():
            return entries
        for p in sorted(self.path.iterdir()):
            if p.name.startswith((".", "_")):
                continue
            key = p.name[: -len(p.suffix)] if p.is_file() and p.suffix in (".h5", ".csv") else p.name
            entries[key] = p
        return entries

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries())

    def __len__(self) -> int:
        return len(self._entries())

    def __getitem__(self, key: str) -> Any:
        path = self._entries().get(key)
        if path is None:
            raise KeyError(f"{self.path} has no entry {key!r}")
        if path.is_dir():
            return DatamateDirectory(path)
        if path.suffix == ".h5":
            return _read_h5(path)
        if path.suffix == ".csv":
            import pandas as pd

            return pd.read_csv(path)
        return path

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name.replace("__", ".")]
        except KeyError:
            raise AttributeError(name) from None

    def __repr__(self) -> str:
        return f"DatamateDirectory({str(self.path)!r})"


def open_datamate(path: str | os.PathLike) -> DatamateDirectory:
    """Open a datamate directory read-only."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"no directory at {path}")
    return DatamateDirectory(path)
