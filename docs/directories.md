# Directories

A **record** is addressed by its key: the directory name is the hash of the type, inputs and
settings that produced it. A **directory** is addressed by its name: you choose where it is, as
with datamate's `Directory`. tensordir uses both:

| | `td.Directory` | `td.Record` (a subclass of `Directory`) |
|---|---|---|
| address | a path you name | `<root>/<Type>/<key[:16]>`, from the inputs and settings |
| metadata | typed fields in `_meta.yaml`, editable, tracked by git | the spec, status, fields and provenance in `_meta.json` |
| contents | any files, read and written by suffix; named child directories | fields written by `__init__` (the build), plus files |
| two with the same settings | two directories: names are identities | one record |
| use it for | organizing work: questions, studies, versions | results: fits, tables, trained networks |

Content-addressed records live in one shared root (`td.set_root`). A directory tree organizes
the work that produces and reads them, without holding them. The scripts on this page run in
order in one working directory; `tests/test_docs_examples.py` executes them.

## Any folder

`td.Directory(path)` views a folder. Opening it writes nothing.

```python
import json
from pathlib import Path

import tensordir as td

folder = Path("downloads")
folder.mkdir(exist_ok=True)
(folder / "notes.txt").write_text("hello")
(folder / "settings.json").write_text(json.dumps({"fps": 30}))

downloads = td.Directory(folder)
print(downloads.keys())                    # ['notes.txt', 'settings.json']
assert downloads.load("settings.json") == {"fps": 30}
assert downloads["notes.txt"] == "hello"   # the same, by indexing
downloads.tree()
assert sorted(p.name for p in folder.iterdir()) == ["notes.txt", "settings.json"]  # nothing added
```

`load` and `save` choose a reader or writer by suffix:

| suffix | `load` returns | `save` accepts |
|---|---|---|
| `.png`, `.jpg`, `.gif`, `.svg`, `.pdf`, `.webp`, `.tif` | `td.Image`: shows itself when it ends a Jupyter or marimo cell | a matplotlib figure, a PIL image, bytes |
| `.mp4`, `.webm`, `.mov` | `td.Video`: plays inline | bytes |
| `.json`, `.yaml` | the parsed value | JSON-like values |
| `.csv`, `.parquet` | a `pandas.DataFrame` | a DataFrame (or an arrow table for parquet) |
| `.npy`, `.npz` | an array, a dict of arrays | an array, a mapping of arrays |
| `.h5` | the array in dataset `data` (datamate's layout), else a dict of datasets | an array |
| `.zarr`, `.nc` | an xarray `Dataset` | an xarray object (`.nc`) |
| `.txt`, `.md`, `.log`, `.py`, `.tex`, `.html` | a string | a string |

`td.register_suffix(".tsv", read=..., write=...)` adds or replaces one. A record reads the same
way: `record.load("figure.png")` reads a file in the record directory that is not a field.

## A typed tree

A `Directory` subclass declares metadata fields (annotations with defaults) and child types
(nested `Directory` classes). Each nested class becomes an accessor on its parent, named in
snake_case. Define the tree in a module that every process imports: reopening a folder returns
its registered type, so a process that has not imported the class sees a plain `Directory`.

```python
# file: layout.py
from typing import Literal

import tensordir as td


class Dev(td.Directory):
    class Track(td.Directory):
        title: str = ""

        class Study(td.Directory):
            title: str = ""

            class Version(td.Directory):
                names = "v{n}"  # how new() names the next version
                title: str = ""
                status: Literal["active", "superseded", "private"] = "active"
                builds_on: list[td.NodeId] = []  # IDs; each must name an existing directory
```

```python
import tensordir as td
from layout import Dev

td.set_root("work")

dev = Dev("dev")  # relative to the root: work/dev
track = dev.track.new("toy-model", title="A toy model")
inverse = track.study.new("inverse")
v1 = inverse.version.new(title="the simplest inverse model")
v2 = inverse.version.new(title="rebuilt on the second data release", builds_on=["toy-model/inverse/v1"])

assert v2.name == "v2" and v2.id == "toy-model/inverse/v2"
assert dev.track("toy-model").study("inverse").version("v2") == v2
assert dev.track(0).study("inverse").version(-1) == v2  # an int is a position in keys()
assert dev.track.study.version.keys() == ["toy-model/inverse/v1", "toy-model/inverse/v2"]
assert dev.resolve("toy-model/inverse/v1") == v1

v1.status = "superseded"  # validated against the annotation, written to _meta.yaml
print((v1.path / "_meta.yaml").read_text())
```

Rules:

- **Writes.** A typed directory writes `_meta.yaml` when a field is set, when `new` creates it,
  or when a file is saved into it; its typed parents are created with it. The file holds the
  type, the fields in declaration order and the aliases, without timestamps or hosts, so it can
  be tracked by git. Edits take a short lease, so processes don't overwrite each other.
- **Listing.** `level.keys()` lists names under one parent. A level reached across several
  parents (`dev.track.study.version`) lists IDs relative to where the chain started. Archived
  directories are left out unless `keys(archived=True)`.
- **Names.** `new()` without a name uses the child type's `names` pattern: one more than the
  highest number used by a child, an archived child or an alias, so numbers are never reused.
  `new(name="v6")` takes a name you checked yourself, for example across git branches.

## Moving, archiving and aliases

```python
import layout  # registers the tree's types
import tensordir as td

dev = td.Directory("work/dev")  # reopens as layout.Dev: _meta.yaml names the type
track = dev.track("toy-model")
track.study.new("ablations")

v2 = dev.resolve("toy-model/inverse/v2")
v2.move("toy-model/ablations/v1")  # the parent must exist
assert v2.aliases == ["toy-model/inverse/v2"]
assert dev.resolve("toy-model/inverse/v2") == v2  # the old ID still resolves
assert track.study("inverse").version.new().name == "v3"  # v2 stays taken by the alias

v1 = dev.resolve("toy-model/inverse/v1").archive()  # moves to inverse/archive/v1
assert v1.id == "toy-model/inverse/v1" and v1.archived
assert track.study("inverse").version.keys() == ["v3"]

v1.aliases = ["v29"]  # a legacy name
assert dev.resolve("v29") == v1
```

- `move(id)` renames the folder and adds the old ID to `aliases`.
- `archive()` moves a directory into its parent's `archive/` folder. Its ID stays the same,
  `resolve` and `level(name)` still find it, and its number stays taken. `archive` is reserved
  as a name.
- `resolve(id)` follows child names from the directory it is called on, else finds the one
  directory whose aliases contain `id`.

## Which work produced a record

Records live in the shared root, so a version does not contain them. A version can be named as
their producer: inside `with version.active():`, every record built stores
`{"id": ..., "path": ...}` as `producer` in its provenance. That includes records built by
subprocesses started inside the block, which inherit `TENSORDIR_PRODUCER`. The producer is not
part of the key, so moving a version changes no key.

```python
import numpy as np

import layout
import tensordir as td

version = td.Directory("work/dev").resolve("toy-model/inverse/v3")
td.set_root("work/records")  # the shared root of content-addressed records


class Fit(td.Record):
    def __init__(self, /, *, seed: int = 0):
        self.x = np.random.default_rng(seed).normal(size=3)


with version.active():
    fit = Fit(seed=4)
assert fit.meta["provenance"]["producer"]["id"] == "toy-model/inverse/v3"
```

## Figures and other outputs

A version's figures and tables are files in its folder, written with `save` and shown with
`load`:

```python
# requires: PIL
from PIL import Image

import layout
import tensordir as td

version = td.Directory("work/dev").resolve("toy-model/inverse/v3")
figure = Image.new("RGB", (64, 48), "white")  # or a matplotlib figure
version.save("results/figures/loss.png", figure)
image = version.load("results/figures/loss.png")  # displays when it ends a notebook cell
assert isinstance(image, td.Image) and image.to_pil().size == (64, 48)
```

They are not records: a figure's key would cover its data but not the plotting code, so a
changed plot would reopen the stale figure.
