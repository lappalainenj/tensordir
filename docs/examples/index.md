# Examples

Each notebook runs top to bottom against a temporary root, so it can be run anywhere without
touching existing results. The documentation build executes all of them, and
`tests/test_examples.py` runs them as part of the test suite.

| Notebook | Covers |
|---|---|
| [Records](01_records.ipynb) | `Spec` fields (`hashed=False`, `since=`, nested specs), `Record.get`, reopening by key, inputs, `locate`, `lookup`, `compute_key`, `Digest`, read-only records, `schema_version`, `Record.at`, `delete` |
| [Codecs](02_codecs.ipynb) | zarr arrays, xarray `DataArray`/`Dataset`/`DataTree`, parquet, arrow tables and streams, JSON, pickle, HDF5 with `append`, lazy and eager reads, dimension names, a codec of your own |
| [Queries and tables](03_queries_and_tables.ipynb) | `Record.query`, conditions, `td.any_of`, `td.exactly`, `Collection.where`, `table`, `Record.info`, `record.info()`, `td.where` on tables and xarray, `td.diff`, failed records, `reindex`, the `tensordir` command |
| [Collections and views](04_collections_and_views.ipynb) | `td.View`, `td.cached_method`, per-record joblib caches, `needs_compute`, `Collection.map` with spec coordinates and `memory=` |
| [Classes that own a record](05_owner_classes.ipynb) | `@td.record`, the status lifecycle of `__init__` as the build, spec attributes, `td.transient`, failing constructors, workspace records (`mutable = True`), `in_place` builds that resume |
| [Roots](06_roots.ipynb) | `TENSORDIR_ROOT`, `set_root`, `use_root`, `get_root`, class roots, `@td.root`, `root=`, copying records between roots, threads |
| [Concurrent builds](07_concurrency.ipynb) | processes racing for one record, leases and `owner.json`, `RecordBusy`, timeouts, a killed builder going `stale`, `tensordir gc`, concurrent edits of a workspace record |
| [joblib caches](08_joblib_memory.ipynb) | `td.memory`, results stored by codec, records and specs as arguments, `record.memory`, `td.StoreBackend` subclasses |
| [Accessors](09_accessors.ipynb) | `.td.where` on `DataFrame`, `Series`, `DataArray`, `Dataset`; `.td.diff` on table rows; `TENSORDIR_NO_ACCESSORS` |
| [External tables](10_external_tables.ipynb) | `ParquetRef` listing digests, `DeltaRef` versions, column and row selection, DuckDB, `materialize`, `SnapshotChanged` |
| [datamate directories](11_legacy_datamate.ipynb) | `tensordir.legacy.open_datamate`, copying a datamate result into a record |

To run a notebook yourself, install the extras and Jupyter, then open it from `docs/examples/`:

```bash
uv sync --all-extras --group dev
uv run --with jupyterlab jupyter lab docs/examples
```
