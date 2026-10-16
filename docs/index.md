# tensordir

tensordir stores results as **records**: directories whose path is the sha256 key of a typed
spec, the record type and the record's inputs. Asking for a record either reopens it or builds it,
once, across any number of processes.

```python
fit = Fit.get(Fit.Spec(n_columns=8), points=points)   # build or reopen
fit.centres                                           # numpy array (declared type)
fit["centres"]                                        # lazy zarr.Array
Fit.query(seed=0).table()                             # one row per record
Fit.query().map(response, dim="model", coords=["seed"])  # concatenated xarray results
```

| Need | API |
|---|---|
| Declare inputs that identify a result | `Spec`, `field(hashed=False)`, `field(since=...)`, `Digest` |
| Build or reopen | `Record.get`, `lookup`, `locate`, `at` |
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

See the quickstart in the repository's `README.md`, [migrating](migrating.md) from datamate,
flyvis and another project, [extending](extending.md) tensordir, the [design](design.md) and the
[history](history.md) of the package.
