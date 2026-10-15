# tensordir

Content-addressed records of labeled tensors in directories. A record is built once from a typed
spec, stored as zarr, parquet and JSON together with its provenance, and reopened by key from any
process or cluster node.

tensordir succeeds [datamate](https://github.com/flyvis/datamate), which grew out of
[artisan](https://github.com/MasonMcGill/artisan); see [docs/history.md](docs/history.md).

## Install

Not yet on PyPI. From a clone:

```bash
uv sync --all-extras          # core: numpy, zarr>=3; extras: xarray, memory, tables, typing, legacy, torch
uv run pytest
```

## Quickstart

```python
import numpy as np
import tensordir as td
from jaxtyping import Float

td.set_root("results")  # or $TENSORDIR_ROOT, `with td.use_root(path):`, or @td.root(path) per class

class Fit(td.Record):
    __record_name__ = "Fit"  # stable type id: keys survive module moves

    class Spec(td.Spec):
        n_columns: int
        seed: int = 0
        device: str = td.field("cpu", hashed=False)  # not part of the key
        prior: float = td.field(1.0, since=2)        # added later: old keys unchanged

    centres: Float[np.ndarray, "n_columns n_dims"]   # stored with dims ("column", "dim")

    def build(self, spec, points):
        rng = np.random.default_rng(spec.seed)
        self.centres = points[rng.choice(len(points), spec.n_columns)]
        self["summary"] = {"n_points": len(points)}

points = np.random.default_rng(0).normal(size=(500, 2))
fit = Fit.get(Fit.Spec(n_columns=8), points=points)  # builds; the array input is hashed by digest
fit = Fit.get({"n_columns": 8}, points=points)       # reopens the same record
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
│       ├── _record.json           # key, spec, inputs, status, fields, timings, provenance
│       ├── data.zarr/centres      # zarr v3 array, dimension_names ["column", "dim"]
│       └── summary.json
└── .tensordir/                    # leases, temporary builds, index.parquet
```

Status is `absent`, `running`, `done`, `failed` (traceback stored; rebuild with
`retry_failed=True`) or `stale` (the builder stopped heartbeating). Concurrent `get` calls on any
number of processes build once: the first takes an `mkdir` lease, the others wait.

Read on: [migrating from datamate, flyvis and another project](docs/migrating.md),
[extending](docs/extending.md), [design](docs/design.md) and [history](docs/history.md).

## License

MIT. See [LICENSE](LICENSE) for the notices carried over from datamate, artisan, flyvis and
another project.
