# Extending tensordir

Five extension points, each with a complete script. Every script on this page runs as-is in an
empty directory; `tests/test_docs_examples.py` executes them, skipping those whose optional
packages (named in the `# requires:` line) are missing.

| To | Use |
|---|---|
| store a new type of value in a record | a `Codec` passed to `register_codec` |
| name array axes that the default rule gets wrong | `register_dims`, a record's `dims`, `Dims` |
| build a record from other records or arrays | keyword inputs to `Record.get` and `build` |
| rebuild domain objects from a record | a `View` subclass with `cached_method` and `memory` |
| change how joblib stores cached results | a subclass of the tensordir store backend |

## Codecs

A codec stores one field. `accepts(value)` claims values by type; `write` stores the value under
the record directory and returns a JSON-able dict that is saved in the record's field manifest;
`read` gets that dict back. `FileCodec` stores one file `<name><suffix>` and supplies `path()` and
`delete()`.

- Codecs registered later are tried first, so a specific codec can take over values that a
  built-in codec also accepts.
- A codec whose `accepts` returns `False` is opt-in: `self.write(name, value, codec="...")`.
- The manifest stores the codec name. Any process that reads the record must have registered the
  codec, so register it in the module that defines the record class.
- Builds run in a temporary directory that is renamed into place, so `write` needs no atomic
  file handling of its own (except for `in_place = True` records).

### A torch model that rebuilds itself

The built-in `safetensors` codec stores a module's state dict and reads back a dict of tensors.
This codec also stores the module class and its constructor arguments, so reading returns the
module.

```python
# requires: torch safetensors
import importlib

import torch
from safetensors.torch import load_file, save_file

import tensordir as td
from tensordir.codecs import FileCodec

td.set_root("results")


class MLP(torch.nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        super().__init__()
        self.config = {"n_in": n_in, "n_hidden": n_hidden, "n_out": n_out}
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_in, n_hidden), torch.nn.Tanh(), torch.nn.Linear(n_hidden, n_out)
        )

    def forward(self, x):
        return self.layers(x)


class ModuleCodec(FileCodec):
    """Modules with a `config` dict of constructor arguments."""

    name = "torch_module"
    suffix = ".safetensors"

    def accepts(self, value):
        return isinstance(value, torch.nn.Module) and isinstance(getattr(value, "config", None), dict)

    def write(self, directory, name, value, *, dims=None):
        state = {k: v.detach().cpu().contiguous() for k, v in value.state_dict().items()}
        save_file(state, str(self.path(directory, name)))
        cls = type(value)
        return {"class": f"{cls.__module__}:{cls.__qualname__}", "config": value.config}

    def read(self, directory, name, info, *, lazy=True):
        module, _, qualname = info["class"].partition(":")
        cls = importlib.import_module(module)
        for part in qualname.split("."):
            cls = getattr(cls, part)
        model = cls(**info["config"])
        model.load_state_dict(load_file(str(self.path(directory, name))))
        return model.eval()


td.register_codec(ModuleCodec())  # tried before the built-in safetensors codec


class Train(td.Record):
    __record_name__ = "Train"

    class Spec(td.Spec):
        n_hidden: int = 16
        n_steps: int = 50
        seed: int = 0

    model: MLP

    def build(self, spec):
        torch.manual_seed(spec.seed)
        model = MLP(2, spec.n_hidden, 1)
        x = torch.randn(256, 2)
        y = x.prod(dim=1, keepdim=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(spec.n_steps):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            self.append("loss", [loss.item()])
        self.model = model


run = Train.get()
model = run.model  # an MLP with the trained weights
assert isinstance(model, MLP) and model.config == {"n_in": 2, "n_hidden": 16, "n_out": 1}
assert run.meta["fields"]["model"]["codec"] == "torch_module"
run.tree()
```

The stored class path is `__main__:MLP` here. Define model classes in an importable module so
that other processes can rebuild them. To keep only the safetensors file and rebuild the module in
your own code, drop the codec: `self.model = model` then stores the state dict, and a `View`
rebuilds the module (see [Views](#views-that-rebuild-domain-objects)).

### Images

```python
# requires: PIL
import numpy as np
from PIL import Image

import tensordir as td
from tensordir.codecs import FileCodec

td.set_root("results")


class PngCodec(FileCodec):
    """PIL images as lossless PNG files."""

    name = "png"
    suffix = ".png"
    format = "PNG"

    def accepts(self, value):
        return isinstance(value, Image.Image)

    def write(self, directory, name, value, *, dims=None):
        value.save(self.path(directory, name), format=self.format)
        return {"size": list(value.size), "mode": value.mode}

    def read(self, directory, name, info, *, lazy=True):
        image = Image.open(self.path(directory, name))
        image.load()
        return image


class JpegCodec(PngCodec):
    """Lossy JPEG; opt-in with `write(name, image, codec="jpeg")`."""

    name = "jpeg"
    suffix = ".jpg"
    format = "JPEG"

    def accepts(self, value):
        return False


td.register_codec(PngCodec())
td.register_codec(JpegCodec())


class Thumbnail(td.Record):
    __record_name__ = "Thumbnail"

    class Spec(td.Spec):
        size: int = 64

    def build(self, spec):
        y, x = np.mgrid[: spec.size, : spec.size]
        pixels = (np.stack([x, y, x[:, ::-1]], axis=-1) * 255 // (spec.size - 1)).astype(np.uint8)
        self["image"] = Image.fromarray(pixels)
        self.write("preview", Image.fromarray(pixels), codec="jpeg")


thumb = Thumbnail.get()
assert np.asarray(thumb["image"]).shape == (64, 64, 3)
assert thumb.meta["fields"]["preview"]["codec"] == "jpeg"
thumb.tree()
```

### Videos

Frames are arrays, which the `ndarray` codec already claims, so wrap them in a type of your own.

```python
# requires: imageio PIL
import dataclasses

import imageio.v3 as iio
import numpy as np

import tensordir as td
from tensordir.codecs import FileCodec

td.set_root("results")


@dataclasses.dataclass
class Video:
    frames: np.ndarray  # (n_frames, height, width) or (n_frames, height, width, 3), uint8
    fps: float = 10.0


class GifCodec(FileCodec):
    name = "gif"
    suffix = ".gif"

    def accepts(self, value):
        return isinstance(value, Video)

    def write(self, directory, name, value, *, dims=None):
        iio.imwrite(self.path(directory, name), value.frames, duration=1000 / value.fps, loop=0)
        return {"fps": value.fps, "n_frames": len(value.frames)}

    def read(self, directory, name, info, *, lazy=True):
        return Video(iio.imread(self.path(directory, name), index=None), fps=info["fps"])


td.register_codec(GifCodec())


class Stimulus(td.Record):
    __record_name__ = "Stimulus"

    class Spec(td.Spec):
        n_frames: int = 12
        size: int = 32

    def build(self, spec):
        x = np.arange(spec.size)
        frames = [(np.sin(x[None, :] / 3 + t) > 0) * np.ones((spec.size, 1)) for t in range(spec.n_frames)]
        self["movie"] = Video((np.array(frames) * 255).astype(np.uint8), fps=12)


movie = Stimulus.get().movie
assert len(movie.frames) == 12 and movie.fps == 12
```

For MP4, install `imageio[ffmpeg]` (or `av`), use `suffix = ".mp4"` and write with
`iio.imwrite(path, frames, fps=value.fps)`.

### A custom xarray encoding

The built-in `xarray` codec drops encodings and stores values as they are. This opt-in codec
stores every variable as `int16` with a scale factor (CF encoding, a quarter of the size of
float64); xarray decodes it on read.

```python
# requires: xarray
import shutil

import numpy as np
import xarray as xr

import tensordir as td
from tensordir.codecs import FileCodec

td.set_root("results")


class Int16Codec(FileCodec):
    """Datasets quantized to int16 steps of `scale`; opt-in with `codec="int16"`."""

    name = "int16"
    suffix = ".zarr"
    scale = 1e-3

    def write(self, directory, name, value, *, dims=None):
        fill = int(np.iinfo(np.int16).min)
        encoding = {v: {"dtype": "int16", "scale_factor": self.scale, "_FillValue": fill} for v in value.data_vars}
        value.drop_encoding().to_zarr(
            self.path(directory, name), mode="w", zarr_format=3, consolidated=False, encoding=encoding
        )
        return {"scale": self.scale}

    def read(self, directory, name, info, *, lazy=True):
        dataset = xr.open_zarr(self.path(directory, name), consolidated=False, chunks=None)
        return dataset if lazy else dataset.load()

    def delete(self, directory, name, info):
        shutil.rmtree(self.path(directory, name), ignore_errors=True)


td.register_codec(Int16Codec())


class Responses(td.Record):
    __record_name__ = "Responses"

    class Spec(td.Spec):
        n_cells: int = 50

    def build(self, spec):
        t = np.linspace(0, 1, 200)
        gain = np.random.default_rng(0).uniform(0.5, 1.5, (spec.n_cells, 1))
        responses = xr.Dataset({"voltage": (("cell", "time"), gain * np.sin(2 * np.pi * 5 * t))}, coords={"time": t})
        self.write("quantized", responses, codec="int16")
        self["exact"] = responses


record = Responses.get()
quantized, exact = record.load("quantized"), record.load("exact")
assert quantized.voltage.encoding["dtype"] == np.int16
assert float(abs(quantized.voltage - exact.voltage).max()) <= 0.5e-3 + 1e-12
```

## Dimension names

Typed fields name their axes from the jaxtyping shape string: `n_<things>` becomes `<thing>`.
Register irregular plurals for the process, override names for one record class with `dims`, or
name an axis explicitly with `Dims`.

```python
from typing import Annotated

import numpy as np
from jaxtyping import Float

import tensordir as td

td.set_root("results")
td.register_dims(n_ommatidia="ommatidium")  # every record in this process


class Retina(td.Record):
    __record_name__ = "Retina"
    dims = {"n_frames": "time"}  # this record type only

    class Spec(td.Spec):
        n_ommatidia: int = 19
        n_frames: int = 20

    photoreceptors: Float[np.ndarray, "n_frames n_ommatidia"]
    frame_edges: Annotated[Float[np.ndarray, "n_frames+1=_"], td.Dims("time_edge")]
    coupling: Float[np.ndarray, "pre=n_ommatidia post=n_ommatidia"]

    def build(self, spec):
        rng = np.random.default_rng(0)
        self.photoreceptors = rng.normal(size=(spec.n_frames, spec.n_ommatidia))
        self.frame_edges = np.linspace(0, 1, spec.n_frames + 1)
        self.coupling = np.eye(spec.n_ommatidia)


retina = Retina.get()
assert retina.to_xarray("photoreceptors").dims == ("time", "ommatidium")
assert retina.to_xarray("frame_edges").dims == ("time_edge",)
assert retina.to_xarray("coupling").dims == ("pre", "post")
```

## Records built from other records

Keyword arguments to `get` beyond the spec are inputs: they reach `build` and enter the key.
Records enter by type and key, arrays by digest, other values by canonical form. A record that
depends on another record therefore gets a new key when its input does.

```python
import numpy as np
from jaxtyping import Float, Int

import tensordir as td

td.set_root("results")


class Points(td.Record):
    __record_name__ = "Points"

    class Spec(td.Spec):
        n_points: int = 300
        seed: int = 0

    xy: Float[np.ndarray, "n_points n_dims"]

    def build(self, spec):
        self.xy = np.random.default_rng(spec.seed).normal(size=(spec.n_points, 2))


class Cluster(td.Record):
    __record_name__ = "Cluster"

    class Spec(td.Spec):
        n_clusters: int = 3
        n_iterations: int = 10

    centres: Float[np.ndarray, "n_clusters n_dims"]
    labels: Int[np.ndarray, "n_points"]

    def build(self, spec, points: Points, weights: np.ndarray | None = None):
        xy = points.xy
        w = np.ones(len(xy)) if weights is None else weights
        centres = xy[: spec.n_clusters].copy()
        for _ in range(spec.n_iterations):
            labels = np.argmin(((xy[:, None] - centres[None]) ** 2).sum(-1), axis=1)
            centres = np.stack([np.average(xy[labels == k], axis=0, weights=w[labels == k]) for k in range(spec.n_clusters)])
        self.centres, self.labels = centres, labels


points = Points.get()
fit = Cluster.get(Cluster.Spec(n_clusters=3), points=points)  # builds
assert Cluster.get({"n_clusters": 3}, points=Points.get()).key == fit.key  # reopens
assert Cluster.compute_key({"n_clusters": 3}, points=points) == fit.key  # no filesystem access
weighted = Cluster.get({"n_clusters": 3}, points=points, weights=np.linspace(0.5, 1.5, 300))
assert weighted.key != fit.key  # the array entered the key by digest
assert fit.meta["inputs"]["points"] == {"__record__": "Points", "key": points.key}
assert Cluster.lookup({"n_clusters": 3}, points=Points.locate({"seed": 1})) is None  # never built
```

## Views that rebuild domain objects

A `View` holds a record and nothing else, pickles as the record's identity, and offers
`memory`, a joblib cache inside the record. Subclass it to rebuild domain objects, as flyvis's
`NetworkView` and `CheckpointedNetwork` do for trained networks:

- `cached_method` caches per view instance (in memory), keyed by the canonical arguments.
- `self.memory.cache(fn)` caches on disk in `<record>/__cache__`. A view argument hashes by its
  record's key, so the cache never hashes network weights.
- `ignore=[...]` keeps arguments such as batch sizes out of the cache key, and `needs_compute`
  tells whether a call would compute.

```python
# requires: torch safetensors xarray joblib
import numpy as np
import torch
import xarray as xr

import tensordir as td

td.set_root("results")


class Net(td.Spec):
    n_hidden: int = 16


def make_model(net: Net) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(1, net.n_hidden), torch.nn.Tanh(), torch.nn.Linear(net.n_hidden, 1))


class Network(td.Record):
    __record_name__ = "Network"
    in_place = True  # readers see checkpoints while training runs

    class Spec(td.Spec):
        net: Net = td.field(default_factory=Net)
        n_steps: int = 60
        checkpoint_every: int = 20
        seed: int = 0

    def build(self, spec):
        torch.manual_seed(spec.seed)
        model = make_model(spec.net)
        x = torch.linspace(-1, 1, 64)[:, None]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for step in range(1, spec.n_steps + 1):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), torch.sin(3 * x))
            loss.backward()
            optimizer.step()
            self.append("loss", [loss.item()])
            if step % spec.checkpoint_every == 0:
                self[f"chkpt_{step:05d}"] = model  # built-in codec: the state dict as safetensors
                self.append("chkpt_step", [step])


def compute_responses(view, x, step, batch_size):
    model = view.network(step)
    with torch.no_grad():
        batches = np.array_split(x, max(1, len(x) // batch_size))
        y = torch.cat([model(torch.as_tensor(b, dtype=torch.float32)[:, None]) for b in batches])
    return xr.DataArray(y[:, 0].numpy(), dims="stimulus", coords={"x": ("stimulus", x)}, name="response")


class NetworkView(td.View):
    @td.cached_method(maxsize=4)
    def network(self, step: int | None = None) -> torch.nn.Module:
        """The network at checkpoint `step` (default: the last)."""
        step = int(self.record.load("chkpt_step")[-1]) if step is None else step
        model = make_model(self.spec.net)
        model.load_state_dict(self.record.load(f"chkpt_{step:05d}"))
        return model.eval()

    def responses(self, x: np.ndarray, step: int | None = None, batch_size: int = 64) -> xr.DataArray:
        return self.memory.cache(compute_responses, ignore=["batch_size"])(self, x, step, batch_size)


view = NetworkView(Network.get())
x = np.linspace(-1, 1, 101)
cached = view.memory.cache(compute_responses, ignore=["batch_size"])
assert view.needs_compute(cached, view, x, None, 64)
first = view.responses(x)  # computes; stored as zarr in <record>/__cache__
assert not view.needs_compute(cached, view, x, None, 8)  # batch_size is not in the key
assert view.network() is view.network()  # cached per view and arguments
assert not torch.equal(view.network(20)[0].weight, view.network()[0].weight)
assert list(view.record.load("chkpt_step")) == [20, 40, 60]

ensemble = td.Collection(Network.get({"seed": seed}) for seed in range(3))
stacked = ensemble.map(lambda record: NetworkView(record).responses(x), dim="network", coords=["seed"])
assert stacked.dims == ("network", "stimulus") and list(stacked["seed"].values) == [0, 1, 2]
```

## joblib store backend

`td.memory(location)` (and `record.memory`, `view.memory`) returns a `joblib.Memory` whose store
backend writes xarray results as zarr and DataFrames as parquet, and leaves everything else to
joblib. joblib hashes arguments by pickling them; records and views pickle as `(type, key, path)`.
Give your own wrapper types the same identity hashing with `__reduce__`, as flyvis's
`CheckpointedNetwork` does.

To store another type natively, subclass the backend and register it with joblib. joblib exposes
store backends only through private modules; tensordir pins `joblib>=1.4,<1.6` and tests the
signatures it relies on, but a subclass of your own inherits that dependency.

```python
# requires: joblib torch safetensors
from pathlib import Path

import joblib
import torch
from safetensors.torch import load_file, save_file

from tensordir._memory import TensordirStoreBackend

OUTPUT = "output.safetensors"


def _is_state(item):
    return isinstance(item, dict) and bool(item) and all(isinstance(v, torch.Tensor) for v in item.values())


class TorchStoreBackend(TensordirStoreBackend):
    """Dicts of tensors as safetensors; xarray, DataFrames and pickles as in tensordir."""

    def dump_item(self, call_id, item, verbose=1):
        if not _is_state(item):
            return super().dump_item(call_id, item, verbose=verbose)
        path = Path(self.location, *call_id)
        self.create_location(str(path))
        save_file({k: v.detach().cpu().contiguous() for k, v in item.items()}, str(path / OUTPUT))

    def load_item(self, call_id, verbose=1, timestamp=None, metadata=None):
        path = Path(self.location, *call_id, OUTPUT)
        if path.exists():
            return load_file(str(path))
        return super().load_item(call_id, verbose=verbose, timestamp=timestamp, metadata=metadata)

    def contains_item(self, call_id):
        return Path(self.location, *call_id, OUTPUT).exists() or super().contains_item(call_id)


joblib.register_store_backend("tensordir_torch", TorchStoreBackend)
memory = joblib.Memory("cache", backend="tensordir_torch", verbose=0)


class Checkpoint:
    """Hashes as (path, step): joblib never pickles the loaded weights."""

    def __init__(self, path: str, step: int):
        self.path, self.step = path, step

    def __reduce__(self):
        return Checkpoint, (self.path, self.step)


calls = []


@memory.cache
def init_weights(checkpoint: Checkpoint, n_units: int):
    calls.append(checkpoint.step)
    torch.manual_seed(checkpoint.step)
    return {"weight": torch.randn(n_units, n_units), "bias": torch.zeros(n_units)}


a = init_weights(Checkpoint("runs/a", 3), 4)
b = init_weights(Checkpoint("runs/a", 3), 4)  # loaded from cache/.../output.safetensors
assert calls == [3] and torch.equal(a["weight"], b["weight"])
assert next(Path("cache").rglob(OUTPUT))
```
