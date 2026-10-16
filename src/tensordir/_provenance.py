"""Where, when and with what code a record was built."""

import functools
import importlib.metadata
import inspect
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

from ._lease import job_id

__all__ = ["collect"]


@functools.cache
def _git(repo: Path) -> dict | None:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        return None
    return {"repo": str(repo), "commit": commit, "dirty": dirty}


def git_info(path: Path) -> dict | None:
    """Commit and dirty flag of the git repository containing `path`, if any."""
    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            return _git(parent)
    return None


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


@functools.cache
def _distribution_of(module: str) -> str | None:
    top = module.partition(".")[0]
    names = importlib.metadata.packages_distributions().get(top)
    return names[0] if names else None


def collect(cls: type) -> dict:
    """Provenance of a build of record class `cls` in this process."""
    packages = {name: _version(name) for name in ("tensordir", "numpy", "zarr")}
    for name in ("xarray", "pandas", "torch", "jax"):
        if name in sys.modules:
            packages[name] = _version(name)
    owner = _distribution_of(cls.__module__)
    if owner:
        packages[owner] = _version(owner)
    try:
        source = inspect.getsourcefile(cls)
    except (TypeError, OSError):  # OSError: a class defined in a notebook or REPL (`__main__` without a file)
        source = None
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        "job": job_id(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "packages": {k: v for k, v in packages.items() if v is not None},
        "git": git_info(Path(source).resolve().parent) if source else None,
        # An editable install reports one package version for every commit of tensordir itself.
        "tensordir_git": git_info(Path(__file__).resolve().parent),
    }
