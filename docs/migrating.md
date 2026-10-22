# Migrating from datamate, flyvis and another project

## Names

datamate had one class, `Directory`, that was at once a configuration, a build recipe, a
mutable folder and a reader. tensordir splits those roles:

| datamate | tensordir | role |
|---|---|---|
| `Directory` subclass with `__init__(self, ...)` | `Record` subclass with `__init__(self, *inputs, /, *, **settings)` | a recipe and its result |
| a class holding `self.dir = Directory(config)` and writing to it (flyvis `Solver`) | a `mutable = True` record, used directly or held as `self.record` | an object that fills a record over time |
| `config` (`Namespace`), `_meta.yaml` | the generated `Spec` (frozen dataclass), `_meta.json` | what identifies a result |
| `Directory(config)` / `Directory(path, config)` | `Fit(*inputs, **settings)` / `Fit.at(path, *inputs, **settings)` | build or reopen |
| list of directories, `info(Type)` | `Collection` from `Record.query(...)`, `Record.info(...)` | many results |
| `NetworkView`, `CheckpointedNetwork` (flyvis) | `View` | derived objects and caches |
| `Namespace.diff` | `tensordir.diff`, `Spec.diff` | compare configurations |
| `.custom.where` (flyvis), `where_dataframe` | `tensordir.where`, `.td.where` accessors | filter tables, xarray and records |

Paths are content-addressed: `<root>/<RecordName>/<key[:16]>/`. A record is read-only after its
build unless its class is a workspace record (`mutable = True`); to change a result, change its
spec or inputs.

## Use-case map

| Task | datamate / flyvis / another project / joblib | tensordir |
|---|---|---|
| **Configure & build** | `class Fit(Directory): def __init__(self, n=8, seed=0)` | `class Fit(td.Record): def __init__(self, /, *, n=8, seed=0)` |
| | `Fit(n=8)`, `Fit({"n": 8})`, `Fit(Namespace(n=8))` | `Fit(n=8)`, `Fit(**spec)` |
| | `Directory("runs/000", config)` (hand-named, flyvis `NetworkDir`) | `Fit.at("runs/000", n=8)` |
| | `Namespace(a=1, model=Namespace(lr=0.1))` | nested `Spec` classes, `spec.replace(model__lr=0.1)` |
| | config fields that do not change the result (device) | `td.field("cpu", hashed=False)` |
| | adding a config field without orphaning results | `td.field(default, since=2)` |
| | `@root(path)` class decorator | `@td.root(path)` or the class attribute `root = path`; a relative path is joined onto the base root ([rule](design.md#roots)) |
| | `set_root_dir`, `set_root_context` | `td.set_root`, `with td.use_root(path)`, `root=` argument |
| | building from another directory's contents | inputs: `Cluster(points_record, weights_array, n_clusters=3)` |
| | reading a Delta export (another project `export.to_table()`), caching by release version | a `td.DeltaRef(uri, version=...)` spec field; `ref.to_table(columns=, filter=)`, `ref.duckdb()`, `ref.materialize(self, name)` |
| **Open & read** | `Directory(path)` | `td.Record.at(path)` (returns the registered class) |
| | `dir.config.x`, `dir.config.to_dict()` | `record.spec.x`, `record.spec.to_dict()`, `record.spec.flat()` |
| | `dir.meta`, `dir.status` | `record.meta`, `record.status`, `record.error` |
| | `dir.x[:]`, `dir.x[()]`, `dir["x"]` (HDF5 `ArrayFile`) | `record.x` (numpy for typed fields), `record["x"][:]` (lazy zarr), `record.load("x")` |
| | `"x" in dir`, `dir.keys()`, `dir.tree()`, `repr(dir)` | `"x" in record`, `record.keys()`, `record.tree()`; IPython shows the tree for a record that ends a cell, Jupyter an HTML table, `repr` one line |
| | sub-directories (`connectome.nodes.type[:]`) | one DataFrame (`parquet`) or `xarray.Dataset`/`DataTree` field |
| | `nodes.to_df()`, `Directory.from_df(df)` | `self["nodes"] = df` in `__init__`; `record.nodes` is the DataFrame |
| **Labeled arrays** | shapes in docstrings, `xarray` built by hand | jaxtyping annotations store dims; `record.to_xarray("x")`, `xr.open_zarr(record.path / "data.zarr")` |
| **Append logs** | `dir.extend("loss", [value])` | `self.append("loss", [value])` in `__init__` (zarr; HDF5 under `codecs = {"ndarray": "hdf5"}`); `in_place = True` to let readers watch |
| | `self.dir.loss = losses`, `self.dir.extend(...)` from a `Solver` | `self.record.loss = losses`, `self.record.append(...)` on a workspace record |
| | `del dir.x`, `dir.rmtree("x")` | `del record.x` (in `__init__` or on a workspace record) |
| | checkpoints `chkpts/chkpt_00010` + `extend("chkpt_index", ...)` | `self[f"chkpt_{i:05d}"] = model` + `self.append("chkpt_step", [i])` |
| **Query many runs** | `collect_all_of_type(T)`, `dir_walk`, `EnsembleDir` children | `T.query(status=None)` |
| | filtering a list of directories by config | `T.query(model__lr=0.1, seed=[0, 1])`, `collection.where(...)` |
| | `generic_responses` over an ensemble (flyvis) | `collection.map(fn, dim="network", coords=["seed"], memory=...)` |
| **Tables & where** | `info(T, depth=1, include=[...])` (another project) | `T.info(depth=1, include=[...])`; `record.info()` for one record |
| | `namespaces_to_df(configs)` | `collection.table()` (`spec.<dotted name>` columns) |
| | `df.query("model_config.optimizer == 'x'")` | `table.td.where(model_config__optimizer="x")` or `td.where(table, ...)` |
| | `where_dataframe(df, type="T4a", u=2)` | `df.td.where(type="T4a", u=2)` (rows; `.index` for the index) |
| | `responses.custom.where(cell_type=["T4a", "T4b"], time=">0,<1")` | `responses.td.where(cell_type=["T4a", "T4b"], time=">0,<1")` |
| **Diff specs** | `config0.diff(config1, name1=, name2=)` | `td.diff(a, b)`, `a.spec.diff(b.spec)`, `hashed=True` for key-relevant fields |
| | `check_namespaces_match(configs, ignore_keys=["id"])` | `collection.table(varying_only=True)` lists what varies |
| **Cache function results** | `Memory(location, backend="xarray_dataset_h5")` | `td.memory(location)` (zarr/parquet; old `output.h5` still read) |
| | `@memory.cache(ignore=["batch_size"])` | unchanged: it is a `joblib.Memory` |
| | `func.check_call_in_cache(*args)` | `td.needs_compute(func, *args)` (the negation) |
| **Per-record caches** | `Memory(location=dir.path / "__cache__")` in `NetworkView` | `record.memory`, `view.memory` |
| | `CheckpointedNetwork.__reduce__` for identity hashing | records and views pickle as `(type, key, path)` |
| | `FIFOCache` / `lru_cache` on view methods | `@td.cached_method(maxsize=...)` |
| **Status & concurrency** | `running` / `done` / `stopped`, killed jobs stay `running` | `absent`, `running`, `done`, `failed` (traceback stored), `stale` (lease expired) |
| | `_override_status(str(e))`, `set_run_failed` (another project) | automatic: an exception in `__init__` stores a `failed` record |
| | `locked_meta` with `fcntl` (another project), SWMR HDF5 | `mkdir` leases with heartbeats; atomic rename of finished builds |
| | rerun a failure | `Fit.get(**spec, retry_failed=True)` |
| | job status of an ensemble (flyvis `ensemble.status()`) | `collection.status_counts()`, `tensordir status` |
| **Delete & clean** | `dir.rmtree("y")`, `delete_if_exists=True` | `record.delete()`, then `get` rebuilds |
| | `rm_all_of_type(T)`, `clean_root_dir()` | `T.query(status=None).delete()`, `tensordir gc` |
| **Legacy directories** | existing datamate trees with `_meta.yaml` and `.h5` | `tensordir.legacy.open_datamate(path)` (read-only) |

## Before and after

### Configure and build

```python
# datamate
@root(results / "fits")
class Fit(Directory):
    def __init__(self, n_columns: int = 8, seed: int = 0, device: str = "cpu"):
        self.centres = fit(load_points(), n_columns, seed)

fit = Fit(n_columns=8)                 # scans _meta.yaml files to find an existing match
fit.config.n_columns, fit.centres[:]
```

```python
# tensordir
@td.root(results / "fits")
class Fit(td.Record):
    centres: Float[np.ndarray, "n_columns n_dims"]

    def __init__(self, points, /, *, n_columns: int = 8, seed: int = 0,
                 device: str = td.field("cpu", hashed=False)):
        self.centres = fit(points, n_columns, seed)

fit = Fit(load_points(), n_columns=8)  # path from the key, under results/fits
fit.spec.n_columns, fit.centres
```

### Append logs during training

```python
# datamate (flyvis Solver)
self.dir.extend("loss", [loss])
self.dir.extend("chkpt_index", [self._last_chkpt_ind])
```

```python
# tensordir
class Train(td.Record):
    in_place = True  # readers see rows while the build runs

    def __init__(self, /, *, n_steps: int = 1000, checkpoint_every: int = 100):
        for step in range(n_steps):
            ...
            self.append("loss", [loss])
            if step % checkpoint_every == 0:
                self[f"chkpt_{step:05d}"] = model
                self.append("chkpt_step", [step])
```

### Objects that hold a directory

```python
# datamate (flyvis Solver, another project meta runs)
class Solver:
    def __init__(self, name, config):
        self.dir = NetworkDir(name, config)
        self.config = self.dir.config
    def checkpoint(self):
        self.dir.extend("chkpt_index", [self._last_chkpt_ind])
        self.dir.best_chkpt_index = self._last_chkpt_ind
```

```python
# tensordir: a workspace record; its __init__ is the build, its methods write later
class Solver(td.Record):
    mutable = True

    def __init__(self, /, *, network: NetworkSpec, task: TaskSpec, n_iters: int = 250_000,
                 device: Annotated[str, td.field(hashed=False)] = "cuda"):
        self.config_note = "..."                   # written as part of the build

    def checkpoint(self, network, index):
        self[f"chkpt_{index:05d}"] = network         # state dict as safetensors
        self.append("chkpt_index", [index])
        self.best_chkpt_index = index

solver = Solver(network=NetworkSpec(), task=TaskSpec())  # builds the run, or reopens it
Solver.query(n_iters=250_000)                            # every run, by settings
```

To keep a class that is not a record, hold one: `self.record = Network(**spec)`, or
`Network.at(results / name, **spec)` for a hand-named directory. See
[Design](design.md#writes-immutability-and-workspace-records) for the rules.

### Query, tables and where

```python
# another project
runs = info(FitDir, depth=1, include=["model_type"])
runs = runs.query("model_config.optimizer == 'adam'")
```

```python
# tensordir
runs = Fit.query(model_config__optimizer="adam")      # filter on disk
table = Fit.info(depth=1, include=["model_type"])                  # all statuses, as a DataFrame
table.td.where(status="done", model_config__lr=">1e-4, <1e-2")        # filter a table
table.groupby("spec.model_type")["duration_s"].mean()                  # flat columns group
Fit.query().table(varying_only=True)                              # only what differs
```

Conditions are the same for `Record.query`, `Collection.where` and `tensordir.where`: a value
(equality, floats to a relative 1e-9), a list (the whole list or any item; `td.any_of(...)` and
`td.exactly(...)` pick one), comparison strings (`">0.1, <=1"`), a compiled regex, or a
callable; see [Conditions](design.md#conditions). Nested names use `__` or a mapping with dotted
keys.

```python
# flyvis
responses.custom.where(cell_type=["T4a", "T4b"], intensity=1, time="<1.0,>0")
```

```python
# tensordir
responses.td.where(cell_type=["T4a", "T4b"], intensity=1, time="<1.0,>0")
```

### Diff specs

```python
# flyvis Ensemble
diff = config0.diff(self[i].dir.config, name1="first", name2="second").first
```

```python
# tensordir
d = td.diff(first, second)       # specs, records, views or dicts
if d:
    print(d)                     # SpecDiff(  ~ network.lr: 0.1 -> 0.01 ...)
d.changed, d.added, d.removed    # dotted names
td.diff(first, second, hashed=True)   # only fields that change the key
table.td.diff(0, 1)                   # two rows of Collection.table(), on their spec columns
```

### Cache function results per record

```python
# flyvis NetworkView
self.memory = Memory(location=self.dir.path / "__cache__", backend="xarray_dataset_h5", verbose=0)
fn = self.memory.cache(compute_responses, ignore=["batch_size"])
if fn.check_call_in_cache(self._network, ...):
    ...
```

```python
# tensordir
class NetworkView(td.View):
    def responses(self, stimuli, batch_size=4):
        return self.memory.cache(compute_responses, ignore=["batch_size"])(self, stimuli, batch_size)

view = NetworkView(Network(**spec))
td.needs_compute(view.memory.cache(compute_responses, ignore=["batch_size"]), view, stimuli, 4)
```

A complete version, with checkpoints and an ensemble `map`, is in
[Extending](extending.md#views-that-rebuild-domain-objects).

### Status, reruns and cleanup

```python
# another project
try:
    train(model_dir)
except Exception as e:
    model_dir._override_status(str(e))
rm_all_of_type(FitDir, y_n="y")
```

```python
# tensordir
try:
    run = Train(**spec)                  # failures are stored with their traceback
except td.BuildFailed as e:
    print(e)
Train.query(status=None).status_counts() # {"done": 48, "failed": 2}
Train.query(status="failed").delete()    # or Train.get(**spec, retry_failed=True)
```

## Not carried over

| datamate feature | Why not | Instead |
|---|---|---|
| Mutating any directory after it is built (`_modified_past_init`) | a result must match its key; edits would silently invalidate it | opt in per record type with `mutable = True`; `in_place = True` for progressive writes during a build |
| Directory search by config (`_meta.yaml` scan, `Directory(config)` lookup) | slow on large roots and ambiguous | content-addressed keys; `Record.query` and the index for search |
| `move` | the path is a function of the key | `Record.at` for hand-named directories |
| Nested `Directory` trees | one record per result keeps keys and status simple | DataFrame, `xarray.Dataset`/`DataTree` fields, or separate records linked as inputs |
| `Namespace` (mutable attribute dict), `namespacify`, `without`, `walk`, `is_subset` | specs are frozen, typed dataclasses | `Spec`, `spec.replace(...)`, `spec.flat()`, `td.diff` |
| SWMR HDF5, HDF5 by default | locking problems on shared filesystems | zarr v3; opt-in HDF5 output with `codecs = {"ndarray": "hdf5"}` ([trade-offs](design.md#hdf5-output-opt-in)); `tensordir.legacy` reads old HDF5 |
| Equality of directory contents (`assert_equal_directories`, `Directory.__eq__`) | records compare by type, key and path | compare fields explicitly, or keys |
| `delete_if_exists`, `y_n` prompts | implicit deletion during construction is easy to trigger by accident | `record.delete()` / `collection.delete()` explicitly |
| `datamate` import aliases | a different object model; aliases would mislead | this page |
