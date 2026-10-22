# tensordir

Research code produces many variants of the same result: a model fit per seed, a simulation per
parameter set, an analysis per dataset. Each takes minutes to days to compute, and weeks later you
need it again, found by the parameters that produced it. File names that encode parameters
(`fit_seed3_lr0.001.npz`) break when a parameter is added. A function cache such as
`joblib.Memory` computes twice when two jobs ask at once and cannot be searched by parameters. An
experiment-tracking server needs setup and stores arrays as files it cannot look into.

tensordir names each result by the parameters that produced it. You write the computation as the
`__init__` of a record class, with its inputs before `/` and its settings after `*`; calling the
class reopens the result if it exists and runs `__init__` otherwise, once, even when many jobs on
a shared filesystem ask at the same time. Arrays are
stored as zarr with named dimensions, so you can query results by their parameters and read a
parameter sweep back as one labeled xarray array.

| Use tensordir when | Use something else when |
|---|---|
| results are expensive and come in many variants of one computation | you track metrics and plots over training time: MLflow, Weights & Biases |
| results are arrays (fits, simulations, embeddings) | your data is relational and shared by many users with integrity rules: DataJoint |
| jobs run in parallel on one machine or a cluster with a shared filesystem | your pipeline chains programs in several languages through files: Snakemake |
| you want to find results later by their parameters, from Python | one process caches a few function calls: `joblib.Memory` |

tensordir succeeds [datamate](https://github.com/flyvis/datamate) (2020–2025), which managed the
models of the [flyvis](https://github.com/TuragaLab/flyvis) project and grew out of
[artisan](https://github.com/MasonMcGill/artisan) (2018). It keeps their idea, a result named by
the configuration that produced it, and replaces the implementation.

## Install

Not yet on PyPI. From a clone:

```bash
uv sync --all-extras          # core: numpy, zarr>=3; extras: xarray, memory, tables, typing, legacy, torch, delta, hdf5
uv run pytest
```

## Quickstart

```python
import numpy as np
import tensordir as td
from jaxtyping import Float

td.set_root("results")  # base root; default: $TENSORDIR_ROOT, else the working directory

class Fit(td.Record):  # the type id, part of every key, is the class name
    centres: Float[np.ndarray, "n_columns n_dims"]   # stored with dims ("column", "dim")

    def __init__(
        self,
        points: np.ndarray,                          # an input: enters the key by its digest
        /,
        *,
        n_columns: int,                              # settings: the generated Fit.Spec
        seed: int = 0,
        device: str = td.field("cpu", hashed=False),  # not part of the key
        prior: float = td.field(1.0, since=2),       # added later: old keys unchanged
    ):
        rng = np.random.default_rng(seed)
        self.centres = points[rng.choice(len(points), n_columns)]
        self["summary"] = {"n_points": len(points)}

points = np.random.default_rng(0).normal(size=(500, 2))
fit = Fit(points, n_columns=8)  # builds
fit = Fit(points, n_columns=8)  # reopens the same record; __init__ does not run
assert Fit.lookup(points, n_columns=9) is None  # reopen only: lookup never builds
print(fit.path.relative_to(td.get_root()), fit.status, fit.centres.shape)

print(Fit.query(n_columns=8).table(depth=1, sizes=False)[["key", "status", "spec.seed"]])

memory = td.memory("cache", verbose=0)  # joblib.Memory: xarray as zarr, records hashed by key

@memory.cache
def spread(fit):
    return float(fit.centres.std())

print(spread(fit))
```

## A record on disk

```
results/
├── Fit/
│   └── 3f2a9c0e1b7d4a55/          # first 16 hex digits of the sha256 key
│       ├── _meta.json           # key, spec, inputs, status, fields, timings, provenance
│       ├── data.zarr/centres      # zarr v3 array, dimension_names ["column", "dim"]
│       └── summary.json
└── .tensordir/                    # leases, temporary builds, index.parquet
```

Status is `absent`, `running`, `done`, `failed` (traceback stored; rebuild with
`retry_failed=True`) or `stale` (the builder stopped heartbeating). Concurrent `get` calls on any
number of processes build once: the first takes an `mkdir` lease, the others wait.

Read on: [example notebooks](docs/examples/index.md) for every use case,
[migrating from datamate, flyvis and another project](docs/migrating.md), [extending](docs/extending.md) and
[design](docs/design.md).

## License

MIT. See [LICENSE](LICENSE) for the notices carried over from datamate, artisan, flyvis and
another project.
