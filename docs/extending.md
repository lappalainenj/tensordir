# Extending tensordir

The extension points below each come with a complete script. Every script on this page runs as-is in an
empty directory; `tests/test_docs_examples.py` executes them, skipping those whose optional
packages (named in the `# requires:` line) are missing.

| To | Use |
|---|---|
| store a new type of value in a record | a `Codec` passed to `register_codec` |
| name array axes that the default rule gets wrong | `register_dims`, a record's `dims`, `Dims` |
| build a record from other records or arrays | inputs to `Record.get`, or nested specs |
| rebuild domain objects from a record | a `View` subclass with `cached_method` and `memory` |
| let an existing class fill a record over time | `@td.record`, or a `mutable = True` record |
| depend on a Delta table or parquet directory outside the root | `td.DeltaRef`, `td.ParquetRef` |
| change how joblib stores cached results | a registered `Codec`, or a `td.StoreBackend` subclass |

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

Typed fields name their axes from the jaxtyping shape string: `n_<things>` becomes `<thing>`
(`n_batches` → `batch`, `n_basis` → `basis`; [the rule](design.md#dimension-names)).
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

## Composing records

A record can depend on other records in two ways. **Inputs** pass finished records to `get` as
keyword arguments; `build` receives them, and each enters the key by its type and key.
**Nested specs** put the upstream record's spec inside the downstream spec; `build` gets or
builds the upstream record itself, and the upstream spec fields enter the key. Either way, a
change upstream gives the downstream record a new key, at any depth.

| | inputs | nested specs |
|---|---|---|
| upstream enters the key as | its key (opaque) | its hashed spec fields |
| query by upstream settings | no: filter the upstream type, then `get` with its records | `Cluster.query(points__seed=1)` |
| upstream may itself take arrays or records as inputs | yes | no: it must be buildable from its spec alone |
| who builds the upstream record | the caller, before `get` | `build`, on demand |
| navigate from a finished record | `record.inputs`, `record.lineage()` | `record.spec.points`, then `Points.get(...)` |

Use nested specs when the upstream record is cheap to describe by its settings and you want to
query by them; use inputs when the upstream record has inputs of its own, is built elsewhere,
or should stay a separate unit of work. A spec field may also hold a record (`points: Points`):
it enters the key like an input, is stored with its path and reopens as a record from
`_record.json`.

### Inputs

Keyword arguments to `get` beyond the spec are inputs: records enter by type and key, arrays by
digest, other values by canonical form. `_record.json` also stores the path of each input record,
so a finished record can reopen its inputs (`record.inputs`) and walk its whole upstream graph
(`record.lineage()`, each record once, upstream first). An input record that is no longer at its
stored path is looked up under its type's current root, so a copied root keeps its links once
it is the root in use; a deleted one reopens as an `absent` handle. `get` refuses to build when
an input record (or a record in a spec field) is not `done`, and names it.

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


class Score(td.Record):
    __record_name__ = "Score"

    inertia: float

    def build(self, spec, cluster: Cluster, points: Points):
        d = points.xy - cluster.centres[cluster.labels]
        self.inertia = float((d**2).sum())


points = Points.get()
fit = Cluster.get(Cluster.Spec(n_clusters=3), points=points)  # builds
assert Cluster.get({"n_clusters": 3}, points=Points.get()).key == fit.key  # reopens
assert Cluster.compute_key({"n_clusters": 3}, points=points) == fit.key  # no filesystem access
weighted = Cluster.get({"n_clusters": 3}, points=points, weights=np.linspace(0.5, 1.5, 300))
assert weighted.key != fit.key  # the array entered the key by digest
assert fit.meta["inputs"]["points"] == {"__record__": "Points", "key": points.key}
assert Cluster.lookup({"n_clusters": 3}, points=Points.locate({"seed": 1})) is None  # never built

score = Score.get(cluster=fit, points=points)
assert score.inputs["cluster"] == fit and score.inputs["cluster"].centres.shape == (3, 2)
assert weighted.inputs["weights"] == td.Digest.of(np.linspace(0.5, 1.5, 300))
assert score.lineage() == [points, fit]  # points is reached twice, listed once

try:
    Score.get(cluster=fit, points=Points.locate({"seed": 1}))
except ValueError as error:
    assert "input 'points'" in str(error) and "absent" in str(error)
```

### Nested specs

A field annotated with a spec class holds that spec; a dict given for it is turned into one.
`build` calls `get` on the upstream type with the nested spec, which reopens or builds it.

```python
import numpy as np
from jaxtyping import Float

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


class Spread(td.Record):
    __record_name__ = "Spread"

    class Spec(td.Spec):
        points: Points.Spec = Points.Spec()
        quantile: float = 0.9

    radius: float

    def build(self, spec):
        xy = Points.get(spec.points).xy
        self.radius = float(np.quantile(np.linalg.norm(xy, axis=1), spec.quantile))


for seed in (0, 1):
    Spread.get({"points": {"seed": seed}})  # the dict becomes a Points.Spec

spread = Spread.get(Spread.Spec(points=Points.Spec(seed=1)))
assert isinstance(spread.spec.points, Points.Spec) and spread.spec.points.seed == 1
assert len(Spread.query(points__seed=1)) == 1  # queryable by upstream fields
assert Points.lookup(spread.spec.points) is not None  # the upstream record, from the spec
assert spread.lineage() == []  # records a build gets itself are not recorded
```

### Workspace records as inputs

A workspace record (`mutable = True`) can change after records were built from it. Its canonical
form therefore adds the time of its last edit, `modified`: each edit gives records built from it
a new key, so `get` builds afresh instead of returning a result computed from the old contents,
and the old result stays under its old key. The time is read when the key is computed; an edit
made while a downstream build runs is not noticed. The same holds for a workspace record in a
spec field or passed to a `cached_method`.

```python
import numpy as np

import tensordir as td

td.set_root("results")


class Losses(td.Record):
    __record_name__ = "Losses"
    mutable = True


class Best(td.Record):
    __record_name__ = "Best"

    loss: float

    def build(self, spec, losses: Losses):
        self.loss = float(losses.load("loss").min())


losses = Losses.get()
losses.append("loss", np.array([3.0, 2.0]))
assert Best.get(losses=losses).loss == 2.0
losses.append("loss", np.array([1.0]))
assert Best.get(losses=losses).loss == 1.0  # a new key, built from the new contents
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

## Classes that own a record

`@td.record` keeps your class as it is and adds a workspace record keyed by its `__init__`
arguments (see [Design](design.md#classes-that-own-a-record) for the exact rules). The first
construction is the build: `__init__` runs under the record's lease (status `running`; other
processes constructing the same spec wait), and the record is `done` when it returns. Methods
keep writing to it afterwards. Spec fields become attributes; assigning one moves later writes
to the record for the new spec.

```python
# requires: torch safetensors
from typing import Annotated

import torch

import tensordir as td

td.set_root("results")


@td.record
class Trainer:
    def __init__(
        self,
        lr: float = 1e-2,
        n_hidden: int = 16,
        seed: int = 0,
        device: Annotated[str, td.field(hashed=False)] = "cpu",  # not in the key
        verbose: Annotated[bool, td.transient] = False,  # not in the spec at all
    ):
        self.verbose = verbose

    def train(self, n_steps: int = 40) -> float:
        if "summary" in self.record:  # finished earlier, maybe by another process
            return self.record.summary["final_loss"]
        torch.manual_seed(self.seed)
        model = torch.nn.Sequential(torch.nn.Linear(1, self.n_hidden), torch.nn.Tanh(), torch.nn.Linear(self.n_hidden, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        x = torch.linspace(-1, 1, 64)[:, None]
        for _ in range(n_steps):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), torch.sin(3 * x))
            loss.backward()
            optimizer.step()
            self.record.append("loss", [loss.item()])
        self.record.model = model  # state dict as safetensors
        self.record.summary = {"final_loss": loss.item()}
        return loss.item()


trainer = Trainer(verbose=True)
assert trainer.record.status == "done"  # built by __init__
first = trainer.train()  # writes after the build: still done, `modified` updated
assert trainer.record.meta["modified"] is not None
assert Trainer().train() == first  # same spec: reopened, not retrained

trainer.lr = 3e-2  # a new spec: the next writes go to a new record
trainer.train()
assert trainer.record.spec.lr == 3e-2

table = Trainer.Record.query().table(sizes=False, varying_only=True)
assert sorted(table["spec.lr"]) == [1e-2, 3e-2]
assert len(Trainer.Record.query(lr=1e-2).where(lambda r: "model" in r)) == 1
```

The same without the decorator: give an ordinary record class `mutable = True` and hold it.

```python
import numpy as np

import tensordir as td

td.set_root("results")


class Run(td.Record):
    __record_name__ = "Run"
    mutable = True  # fields may change after the (empty) build

    class Spec(td.Spec):
        lr: float = 0.1


class Solver:
    def __init__(self, lr: float):
        self.record = Run.get({"lr": lr})  # or Run.at(path, {"lr": lr}) for a hand-named directory

    def step(self, losses):
        self.record.losses = np.asarray(losses)


solver = Solver(0.1)
solver.step([1.0, 0.5])
assert list(Run.get({"lr": 0.1}).load("losses")) == [1.0, 0.5]
del solver.record.losses
assert "losses" not in solver.record
```

## Records built from external tables

A `DeltaRef` pins a Delta Lake table version, a `ParquetRef` a parquet directory listing; as a
spec field either one keys the record by the exact snapshot (see
[Design](design.md#external-tables)). This script writes a small local Delta table as a stand-in
for a CAVE synapse export on GCS.

```python
# requires: deltalake pyarrow pandas
import numpy as np
import pyarrow as pa
from deltalake import write_deltalake

import tensordir as td

td.set_root("results")


def synapses(n, seed):
    rng = np.random.default_rng(seed)
    return pa.table({
        "pre_pt_root_id": rng.integers(0, 20, n),
        "post_pt_root_id": rng.integers(0, 20, n),
        "size": rng.integers(10, 500, n),
    })


write_deltalake("synapses", synapses(2000, 0))  # e.g. gs://.../synapses_pni_2/post_pt_root_id


class Edges(td.Record):
    __record_name__ = "Edges"

    class Spec(td.Spec):
        synapses: td.DeltaRef
        min_size: int = 0

    def build(self, spec):
        table = spec.synapses.to_table(  # only these columns and rows are read
            columns=["pre_pt_root_id", "post_pt_root_id", "size"], filter={"size": f">={spec.min_size}"}
        )
        self["edges"] = table.group_by(["pre_pt_root_id", "post_pt_root_id"]).aggregate(
            [("size", "count"), ("size", "sum")]
        )
        spec.synapses.materialize(self, "large", filter={"size": ">=400"})  # a local parquet copy


snapshot = td.DeltaRef("synapses")  # pins the current version (0); or version=0 explicitly
edges = Edges.get({"synapses": snapshot, "min_size": 50})
assert edges.spec.synapses.version == 0
assert edges.load("edges").column_names == ["pre_pt_root_id", "post_pt_root_id", "size_count", "size_sum"]
assert edges["large"].count_rows() == snapshot.to_table(filter={"size": ">=400"}).num_rows  # lazy dataset

write_deltalake("synapses", synapses(500, 1), mode="append")  # the export moves on: version 1
assert Edges.lookup({"synapses": td.DeltaRef("synapses"), "min_size": 50}) is None  # new snapshot, new key
assert Edges.get({"synapses": snapshot, "min_size": 50}) == edges  # the pinned snapshot reopens
print(Edges.query().table(sizes=False)[["key", "spec.synapses", "spec.min_size"]])
```

On GCS, pass credentials for reading only:
`td.DeltaRef("gs://bucket/path", version=3, storage_options={"google_service_account_path": ...})`.
They stay out of keys and `_record.json`. For a directory of parquet files, use
`td.ParquetRef(path)`. Its version is a digest: a SHA-256 fingerprint of the sorted list of
`(relative path, size)` pairs of the data files. The digest is stored in the spec of each record
that uses the reference, so a changed directory gives new keys, and opening a reference whose
directory has changed since it was pinned raises `td.SnapshotChanged`. A file rewritten with the
same name and size is not detected; a `DeltaRef` pins the Delta table version instead.

## joblib store backend

`td.memory(location)` (and `record.memory`, `view.memory`) returns a `joblib.Memory` whose store
backend, `td.StoreBackend`, writes each result with the codec registered for its type, the same
registry records use: xarray as zarr, DataFrames as parquet, `{str: Tensor}` as safetensors, and
whatever you register. numpy arrays, plain Python values and anything without a faithful codec
go through joblib's pickle path. joblib hashes arguments by pickling them; records and views
pickle as `(type, key, path)`. Give your own wrapper types the same identity hashing with
`__reduce__`, as flyvis's `CheckpointedNetwork` does.

There are two ways to store another type natively:

- **Register a codec.** Records and caches both pick it up. The backend uses a codec only if
  `codec.faithful(value)` is true, i.e. reading returns an equal value of the same type; the
  built-in `safetensors` codec is not faithful for modules (it returns their state dict), so a
  cached module is pickled unless a codec like `ModuleCodec` above claims it.
- **Subclass `td.StoreBackend`** and override `select_codec(item)` (return a codec or `None`) or
  `skip_codecs` (codec names never used, by default `ndarray`, `json` and `pickle`). Pass the
  class as `td.memory(location, backend=...)`; tensordir registers it with joblib.

Results stored by a codec live in `output.td/` next to joblib's `metadata.json`, written to a
temporary directory and renamed. Caches written by earlier tensordir versions (`output.zarr`,
`output.parquet`) and by the flyvis and another project netCDF backends (`output.h5`) still load.

```python
# requires: joblib torch safetensors
from pathlib import Path

import numpy as np
import torch
import zarr

import tensordir as td
from tensordir.codecs import get_codec


class Checkpoint:
    """Hashes as (path, step): joblib never pickles the loaded weights."""

    def __init__(self, path: str, step: int):
        self.path, self.step = path, step

    def __reduce__(self):
        return Checkpoint, (self.path, self.step)


class LargeArraysAsZarr(td.StoreBackend):
    """Arrays above 1 MB as zarr, reopened lazily as `zarr.Array`; the rest as usual."""

    def select_codec(self, item):
        if isinstance(item, np.ndarray) and item.nbytes > 2**20:
            return get_codec("ndarray")
        return super().select_codec(item)


memory = td.memory("cache", backend=LargeArraysAsZarr, verbose=0)
calls = []


@memory.cache
def init_weights(checkpoint: Checkpoint, n_units: int):
    calls.append(checkpoint.step)
    torch.manual_seed(checkpoint.step)
    return {"weight": torch.randn(n_units, n_units), "bias": torch.zeros(n_units)}


@memory.cache
def activity(n_frames: int):
    return np.random.default_rng(0).normal(size=(n_frames, 1000))


a = init_weights(Checkpoint("runs/a", 3), 4)
b = init_weights(Checkpoint("runs/a", 3), 4)  # the built-in safetensors codec stored it
assert calls == [3] and torch.equal(a["weight"], b["weight"])
assert next(Path("cache").rglob("output.td/output.safetensors"))

activity(500)
assert isinstance(activity(500), zarr.Array)  # 4 MB: zarr, opened lazily
assert isinstance(activity(5), np.ndarray)  # small: joblib's own format
```
