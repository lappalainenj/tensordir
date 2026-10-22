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

In code:

```python
fit = Fit(points, n_columns=8)                        # build or reopen
fit.centres                                           # numpy array (declared type)
fit["centres"]                                        # lazy zarr.Array
Fit.query(seed=0).table()                             # one row per record
Fit.query().map(response, dim="model", coords=["seed"])  # concatenated xarray results
```

| Need | API |
|---|---|
| Declare what identifies a result | `__init__` inputs (before `/`) and settings (after `*`), `field(hashed=False)`, `field(since=...)`, `transient`, `Digest` |
| Build or reopen | `Fit(...)`, `Record.get`, `lookup`, `locate`, `at` |
| Organize work in named folders | `Directory` subclasses with metadata fields and child types; `resolve`, `new`, `move`, `archive`, `active` |
| Read and write any file by suffix | `load`, `save`, `register_suffix`, `Image`, `Video` |
| Store values | `self.x = ...`, `self["x"] = ...`, `write(codec=...)`, `append` |
| Name array axes | jaxtyping field annotations, `Dims`, `register_dims` |
| Work with many records | `Record.query`, `Record.info`, `Collection.where`, `Collection.table`, `Collection.map`, `reindex` |
| Filter tables, records and xarray results | `where`, `any_of`, `exactly` |
| Compare specs and records | `diff`, `Spec.diff`, `Collection.table(varying_only=True)` |
| Inspect in a notebook | rich reprs of `Spec`, `Record`, `Collection`; `Record.tree()` |
| Memoize functions | `memory`, `View`, `cached_method`, `needs_compute` |
| Read datamate directories | `tensordir.legacy.open_datamate` |
| Maintain a root | `tensordir ls`, `status`, `reindex`, `gc` |

**Where records go.** A record type stores its records under `base / root`: the base root is
the innermost `use_root` block, else `set_root`, else `$TENSORDIR_ROOT`, else the working
directory; `root` is a `root=` argument, else the class's `root` (`@td.root(path)`). A relative
`root` extends the base root, an absolute one replaces it, as with `pathlib`'s `/`.
`td.get_root(Fit)` shows the result; [Design](design.md#roots) has examples.

**Import time.** `import tensordir` registers `.td` accessors on pandas and xarray objects and
therefore imports both libraries, about 0.45 s. Set `TENSORDIR_NO_ACCESSORS=1` to skip that,
and use `df.pipe(td.where, ...)` instead of `df.td.where(...)`; see
[Collections and views](design.md#collections-and-views).

Features marked **experimental** may change or be removed; [Design](design.md#experimental-features)
lists them.

See the quickstart in the repository's `README.md`, the [example notebooks](examples/index.md),
[migrating](migrating.md) from datamate, flyvis and another project, [extending](extending.md)
tensordir and the [design](design.md).

## History

tensordir succeeds [datamate](https://github.com/flyvis/datamate) (2020–2025), which managed the
models of the [flyvis](https://github.com/TuragaLab/flyvis) project and grew out of
[artisan](https://github.com/MasonMcGill/artisan) (2018). It keeps their idea, a result named by
the configuration that produced it, and replaces the implementation.
