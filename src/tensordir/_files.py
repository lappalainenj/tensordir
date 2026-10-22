"""Read and write any file by its suffix: `directory.load("results/bar.png")`, `directory.save(...)`.

Readers and writers are registered per suffix with `register_suffix`; later registrations win.
Images and videos load as `Image` and `Video` objects that display themselves in Jupyter and
marimo when they end a cell, and keep their path and bytes for other uses.
"""

import base64
import json
import mimetypes
import os
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Image", "Video", "register_suffix", "read_file", "write_file", "suffix_of"]

Reader = Callable[[Path], Any]
Writer = Callable[[Path, Any], None]

_READERS: dict[str, Reader] = {}
_WRITERS: dict[str, Writer] = {}


def register_suffix(suffix: str, *, read: Reader | None = None, write: Writer | None = None) -> None:
    """Read files ending in `suffix` with `read(path)` and write them with `write(path, value)`.

    `write` receives a temporary path with the same suffix, renamed onto the target when it
    returns."""
    suffix = suffix.lower()
    if read is not None:
        _READERS[suffix] = read
    if write is not None:
        _WRITERS[suffix] = write


def suffix_of(path: Path) -> str:
    """The registered suffix of `path`, longest match first (`.ome.zarr` before `.zarr`)."""
    name = path.name.lower()
    for suffix in sorted(set(_READERS) | set(_WRITERS), key=len, reverse=True):
        if name.endswith(suffix):
            return suffix
    return path.suffix.lower()


def read_file(path: Path) -> Any:
    """The contents of `path`, read by its suffix's reader."""
    suffix = suffix_of(path)
    if suffix not in _READERS:
        raise ValueError(f"no reader for {suffix or 'files without a suffix'} ({path.name}); register one with td.register_suffix")
    if not path.exists():
        raise FileNotFoundError(path)
    return _READERS[suffix](path)


def write_file(path: Path, value: Any) -> None:
    """Write `value` to `path` with its suffix's writer, atomically (a temporary file, renamed)."""
    suffix = suffix_of(path)
    if suffix not in _WRITERS:
        raise ValueError(f"no writer for {suffix or 'files without a suffix'} ({path.name}); register one with td.register_suffix")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}{suffix}")
    try:
        _WRITERS[suffix](tmp, value)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class _Media:
    """A file shown inline by notebooks: its path, bytes and MIME type."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.mimetype = mimetypes.guess_type(self.path.name)[0] or "application/octet-stream"

    @property
    def data(self) -> bytes:
        """The file's bytes."""
        return self.path.read_bytes()

    def _data_url(self) -> str:
        return f"data:{self.mimetype};base64,{base64.b64encode(self.data).decode()}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self.path)!r})"


class Image(_Media):
    """An image file; displays inline in Jupyter and marimo."""

    def _repr_html_(self) -> str:
        return f'<img src="{self._data_url()}" alt="{self.path.name}" style="max-width:100%">'

    def _repr_png_(self) -> bytes | None:
        return self.data if self.mimetype == "image/png" else None

    def _mime_(self) -> tuple[str, str]:
        return "text/html", self._repr_html_()

    def to_pil(self) -> Any:
        """The image as a `PIL.Image.Image` (needs pillow)."""
        from PIL import Image as PILImage

        image = PILImage.open(self.path)
        image.load()
        return image


class Video(_Media):
    """A video file; plays inline in Jupyter and marimo."""

    def _repr_html_(self) -> str:
        return f'<video src="{self._data_url()}" controls loop style="max-width:100%"></video>'

    def _mime_(self) -> tuple[str, str]:
        return "text/html", self._repr_html_()


def _write_image(path: Path, value: Any) -> None:
    """Save a matplotlib figure, a PIL image, raw bytes or an `Image` to `path`."""
    fmt = path.suffix[1:].lower()
    if hasattr(value, "savefig"):  # matplotlib Figure, without importing matplotlib
        value.savefig(path, format=fmt, bbox_inches="tight")
    elif hasattr(value, "save") and hasattr(value, "mode"):  # PIL image
        value.save(path, format=fmt.upper().replace("JPG", "JPEG").replace("TIF", "TIFF").replace("TIFFF", "TIFF"))
    else:
        _write_bytes(path, value)


def _write_bytes(path: Path, value: Any) -> None:
    """Write bytes, or the bytes of an `Image`/`Video`, to `path`."""
    if isinstance(value, _Media):
        value = value.data
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"cannot write {type(value).__qualname__} to {path.name}")
    path.write_bytes(value)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    from .codecs import _to_python

    path.write_text(json.dumps(_to_python(value), indent=1))


def _read_yaml(path: Path) -> Any:
    import yaml

    return yaml.safe_load(path.read_text())


def _write_yaml(path: Path, value: Any) -> None:
    import yaml

    from .codecs import _to_python

    path.write_text(yaml.safe_dump(_to_python(value), sort_keys=False))


def _read_text(path: Path) -> str:
    return path.read_text()


def _write_text(path: Path, value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{path.name}: expected str, got {type(value).__qualname__}")
    path.write_text(value)


def _read_csv(path: Path) -> Any:
    import pandas as pd

    return pd.read_csv(path)


def _write_csv(path: Path, value: Any) -> None:
    value.to_csv(path, index=False)


def _read_parquet(path: Path) -> Any:
    import pandas as pd

    return pd.read_parquet(path)


def _write_parquet(path: Path, value: Any) -> None:
    if hasattr(value, "to_parquet"):
        value.to_parquet(path)
    else:
        import pyarrow.parquet as pq

        pq.write_table(value, path)


def _read_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _write_npy(path: Path, value: Any) -> None:
    with open(path, "wb") as f:
        np.save(f, np.asarray(value), allow_pickle=False)


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _write_npz(path: Path, value: Any) -> None:
    with open(path, "wb") as f:
        np.savez(f, **dict(value))


def _read_h5(path: Path) -> Any:
    import h5py

    with h5py.File(path, "r", locking=False) as f:
        if list(f.keys()) == ["data"] and isinstance(f["data"], h5py.Dataset):
            return f["data"][()]  # one dataset `data`: datamate's and tensordir's layout
        return {k: f[k][()] for k in f if isinstance(f[k], h5py.Dataset)}


def _write_h5(path: Path, value: Any) -> None:
    import h5py

    with h5py.File(path, "w", locking=False) as f:
        f.create_dataset("data", data=np.asarray(value))


def _read_zarr(path: Path) -> Any:
    try:
        import xarray as xr

        return xr.open_zarr(path, consolidated=False, chunks=None)
    except Exception:
        import zarr

        return zarr.open(path, mode="r")


def _read_netcdf(path: Path) -> Any:
    import xarray as xr

    with xr.open_dataset(path) as opened:
        return opened.load()


def _write_netcdf(path: Path, value: Any) -> None:
    value.to_netcdf(path)


def _read_safetensors(path: Path) -> dict:
    from safetensors.numpy import load_file

    return load_file(str(path))


for _suffix in (".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".webp", ".tif", ".tiff"):
    register_suffix(_suffix, read=Image, write=_write_image)
for _suffix in (".mp4", ".webm", ".mov", ".m4v"):
    register_suffix(_suffix, read=Video, write=_write_bytes)
register_suffix(".json", read=_read_json, write=_write_json)
register_suffix(".yaml", read=_read_yaml, write=_write_yaml)
register_suffix(".yml", read=_read_yaml, write=_write_yaml)
for _suffix in (".txt", ".md", ".log", ".py", ".tex", ".html"):
    register_suffix(_suffix, read=_read_text, write=_write_text)
register_suffix(".csv", read=_read_csv, write=_write_csv)
register_suffix(".parquet", read=_read_parquet, write=_write_parquet)
register_suffix(".npy", read=_read_npy, write=_write_npy)
register_suffix(".npz", read=_read_npz, write=_write_npz)
register_suffix(".h5", read=_read_h5, write=_write_h5)
register_suffix(".hdf5", read=_read_h5, write=_write_h5)
register_suffix(".zarr", read=_read_zarr)
register_suffix(".nc", read=_read_netcdf, write=_write_netcdf)
register_suffix(".safetensors", read=_read_safetensors)
