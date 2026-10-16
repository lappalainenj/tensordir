# Design

## Keys

A record's key is the sha256 of the canonical JSON of

```json
{"inputs": {...}, "schema_version": 0, "spec": {...}, "type": "Fit"}
```

- `type` is `__record_name__`, or `module:qualname` if unset. Set it: the fallback changes when a
  class moves, and classes defined in notebooks have module `__main__`.
- `schema_version` is a class attribute. Bump it when a recipe change invalidates old results.
- `spec` holds the hashed fields of the spec. Canonical JSON sorts keys, turns numpy scalars into
  Python scalars and paths into POSIX strings, writes floats in their shortest round-trip form,
  maps -0.0 to 0.0, tags infinities, rejects NaN, and accepts arrays only through `Digest`. A float
  field given an int is stored as a float, so `lr=1` and `lr=1.0` share a key.
- `inputs` are the keyword arguments of `get` beyond the spec. Records enter by type and key,
  arrays by digest, and other values by canonical form.

Spec class names don't enter the key, so renaming a spec class is free. Two field options
control the key:

- `field(hashed=False)` keeps a field out of the key (device, worker count, batch size). The
  stored spec records the value used by the build that ran.
- `field(default, since=v)` keeps a field out of the key while it equals its default. Adding
  such a field leaves existing keys unchanged. The cost: if you later change that default, records
  built under the old default still match the new one.

Golden keys in the test suite pin the format. A change that alters them orphans stored records.

## Roots

A record type stores its records under `base / path`, joined with pathlib's `/`:

- The **base root** is the innermost `with td.use_root(path)` block, else `td.set_root(path)`,
  else `$TENSORDIR_ROOT`, else the working directory at the moment the root is resolved.
- `path` is the `root=` argument of `get`, `locate`, `lookup`, `query` or `info`, else the class's
  `root` attribute (set directly or with `@td.root(path)`, inherited by subclasses), else
  nothing. A `root=` argument replaces the class root; it is not joined onto it.
- A relative `path` extends the base root; an absolute `path` replaces it. `~` is expanded.

| base root | class root | `root=` | records go to |
|---|---|---|---|
| unset (working directory `/home/me/project`) | `"connectomes"` | | `/home/me/project/connectomes` |
| `TENSORDIR_ROOT=/data` | `"connectomes"` | | `/data/connectomes` |
| `TENSORDIR_ROOT=/data` | | | `/data` |
| `use_root("/tmp/test")` | `"connectomes"` | | `/tmp/test/connectomes` |
| `use_root("/tmp/test")` | `"/archive/stimuli"` | | `/archive/stimuli` |
| `TENSORDIR_ROOT=/data` | `"connectomes"` | `"scratch"` | `/data/scratch` |
| any | `"connectomes"` | `"/fast/run1"` | `/fast/run1` |

`td.get_root(Fit)` (or `Fit.resolve_root()`) prints where `Fit` records go now; `td.get_root()`
gives the base root. The class root is stored as given and joined on every access, so a
`use_root` block in a test redirects every record type with a relative root, while types with
an absolute root stay where they are. Relative values given to `set_root` and `use_root` are
resolved against the working directory when they are set; a relative `$TENSORDIR_ROOT` is
resolved when it is read. For a root next to a source file, write the absolute path:
`@td.root(Path(__file__).parent / "data")`.

Every resolved root carries its own `.tensordir/` with leases, temporary builds and index, so
`Fit.query()`, `Fit.info()` and `td.reindex(Fit)` work on `Fit`'s resolved root, and
`tensordir --root connectomes ls` joins `--root` onto the base root the same way. Class roots
let record types live apart, as flyvis keeps renderings and trained networks in separate
trees. The root never enters the key: the same record can be copied between roots.

## Layout and lifecycle

```
<root>/<RecordName>/<key[:16]>/_record.json   # key, spec, inputs, status, fields, provenance
<root>/<RecordName>/<key[:16]>/data.zarr/     # arrays and xarray groups (zarr v3)
<root>/<RecordName>/<key[:16]>/<name>.json|.parquet|.safetensors|.h5|.pkl
<root>/.tensordir/locks/  tmp/  index.parquet
```

`_record.json` stores the full key, so a 16-hex prefix collision raises `KeyCollision` instead of
returning the wrong record. Provenance covers the host, pid, user, Slurm/LSF job id, Python and
package versions, argv, and the git commit and dirty flag of the repository defining the record
class.

`Record.get` loops until it can return a `done` record:

1. Read `_record.json`. If the status is `done`, return the record. If it is `failed`, raise
   `BuildFailed` with the stored traceback, unless `retry_failed=True`.
2. Try to take the lease (`mkdir` of the lock directory, then `owner.json`). On success, check
   again, run `build` in `.tensordir/tmp/...`, write `_record.json`, and rename the directory
   into place.
3. Otherwise wait with exponential backoff (capped at 2 s), up to `timeout`. With `block=False`,
   raise `RecordBusy` instead.

An exception in `build` stores a `failed` record with the traceback; while `retry_failed=True`
rebuilds it, its status is `running`. `KeyboardInterrupt` and `SystemExit` leave no record behind.

`in_place = True` builds in the final directory, so readers see fields and `append`ed rows while
the build runs (training logs). A rebuild after a crash keeps what the earlier attempt wrote,
which lets a recipe resume from checkpoints.

## Writes, immutability and workspace records

`record.name = value`, `record["name"] = value` and `record.write(name, value)` store a field with
the codec chosen by the value's type; `record.append(name, rows)` grows an array along axis 0;
`del record.name`, `del record["name"]` and `record.delete_field(name)` remove one. Private names
(`_x`) and class attributes (`path`, `spec`, methods) are not fields; assigning them follows normal
Python rules. Reading `record.name` returns the stored value (numpy for typed array fields).

Who may write:

| record | inside `build` | after the build is `done` |
|---|---|---|
| default | yes | no: `ReadOnlyError` |
| `in_place = True` | yes; readers see each write | no |
| `mutable = True` (workspace) | yes | yes, from any handle and process |

A finished record is immutable by default because its key promises its contents: the same spec
and inputs give the same fields. A **workspace record** (`mutable = True`) gives up that promise
for the use datamate served: an object (a trainer, a solver) holds a record and writes losses,
checkpoints and summaries into it over time. Rules:

- Without a `build`, `get` creates an empty `done` record; edits start right away.
- Each edit takes the record's lease, re-reads `_record.json`, writes the field, stores the new
  manifest atomically with a `modified` timestamp, and releases the lease. Edits from many
  handles, threads or processes therefore never lose each other's fields; the last write to the
  same field wins.
- Edits need status `done`. A record that is `running` (being built elsewhere), `failed` or
  `stale` raises `ReadOnlyError`.
- Handles notice edits by the mtime of `_record.json`; immutable records cache their manifest
  forever.
- Writing an array over a field created by `append` replaces its rows and keeps it appendable.
- Dimension sizes are checked at the end of a build, not on later edits.
- Array fields are written in place, so a reader can see a partly written array during an edit.
  File codecs (json, parquet, safetensors) replace their file atomically.

## Classes that own a record

`@td.record` turns a regular class into one whose `__init__` arguments are a spec and whose
`self.record` is a workspace record for that spec. As with datamate's `Directory.__init__`, the
first construction is the build; unlike it, the record stays writable afterwards.

```python
@td.record
class Trainer:
    def __init__(self, lr: float = 0.1, n_steps: int = 100,
                 device: Annotated[str, td.field(hashed=False)] = "cpu",
                 logger: Annotated[Logger | None, td.transient] = None):
        self.model = make_model()
```

- **Spec attributes.** Every named `__init__` argument is a field of the generated `Trainer.Spec`,
  typed by its annotation (`Any` without one), with its default. Exceptions:
  `Annotated[T, td.transient]` arguments stay plain arguments. `*args` and `**kwargs`, and the
  names `record`, `spec`, `Spec`, `Record`, are rejected.
- **Key options.** `td.field(..., hashed=False)` or `since=` as the default, or inside
  `Annotated[T, td.field(hashed=False)]`, work as in a `Spec`.
- **Attributes.** Each spec field is a property backed by `self.spec`. `trainer.lr` reads it;
  `trainer.lr = 0.2` (also inside `__init__`) replaces the spec with `spec.replace(lr=0.2)`. If
  the key changes, the next `trainer.record` is the record for the new spec; earlier handles keep
  pointing at the old record. Changing an unhashed field keeps the record.
- **Records.** `trainer.record` is the record for `trainer.spec`, a `mutable` record type named
  `Trainer` (or `name=`), with `root=`, `schema_version=` and a `base=` record class for typed
  fields, `dims`, codecs or `in_place`. Its root is resolved when the object is constructed.
- **Lifecycle.** `__init__` is the build:

    | moment | status | writes through `self.record` |
    |---|---|---|
    | construction, record not `done` | takes the lease, then `running` | go into the build, as in `Record.build` |
    | `__init__` returns | `done` | |
    | `__init__` raises | `failed`, traceback stored; the exception propagates | discarded |
    | construction, record `done` | stays `done`; `__init__` runs again without the lease | edits |
    | any time after the build | stays `done` | edits; each updates `meta["modified"]` |

    A process constructing the same spec while another runs `__init__` waits for the lease, like
    `Record.get`, and then runs its own `__init__` on the finished record; so a training loop can
    resume or skip finished work (`"loss" in self.record`). A `failed` record is rebuilt by the
    next construction. Edits on the finished record (`obj.record.x = v`, `append`, `del`) take the
    lease briefly and never change the status. With `mutable=False` the record becomes read-only
    once `__init__` has built it.
- **Threads.** Creating and switching the record is locked per instance, and record edits are
  serialized by the lease. Changing spec attributes while other threads write through
  `obj.record` is still a race: their writes go to whichever record they fetched.
- Instances pickle with their spec; unpickling does not run `__init__`, and the record is
  reopened on first use.
  Decorated subclasses get their own spec from their own `__init__`.

## Concurrency on shared filesystems

- **Leases.** A lease is a directory created with `mkdir`, which is atomic on local disks, NFS and
  Lustre/GPFS; tensordir does not rely on `flock`. The owner rewrites `owner.json` from a
  heartbeat thread every `heartbeat_interval` seconds (default `lease_ttl / 4`). A lease whose
  heartbeat is older than its TTL is stale: anyone may break it, by renaming it away and then
  re-checking ownership.
- **Integrity.** Integrity doesn't depend on the lease. Builds write into a temporary directory
  in the same root and appear by `rename`, so readers never see half-written records. If two
  builders race after a lease is broken, the first rename wins and the second build is discarded.
- **Clocks.** Staleness compares heartbeat timestamps from different hosts, so node clocks must
  agree to well within the TTL.
- **Same filesystem.** The root, including `.tensordir/`, must be on one filesystem for `rename`
  to be atomic.

Tests cover six processes racing for one record (exactly one build), SIGKILL during a build
(`running`, then `stale`, then recovery), and lease expiry under a fake clock.

## Codecs

| value | codec | stored as |
|---|---|---|
| `numpy.ndarray` | `ndarray` | zarr v3 array in `data.zarr/<name>` with `dimension_names` |
| `xarray.DataArray`/`Dataset`/`DataTree` | `xarray` | zarr group `data.zarr/<name>` |
| `pandas.DataFrame` | `parquet` | `<name>.parquet` |
| `pyarrow.Table`, `RecordBatchReader` | `arrow` | `<name>.parquet`, streamed batch by batch |
| JSON scalars, lists, str-keyed dicts | `json` | `<name>.json` |
| `torch.nn.Module`, `{str: Tensor}` | `safetensors` | `<name>.safetensors` (state dict) |
| anything | `pickle` | `<name>.pkl`, only with `write(name, value, codec="pickle")` |
| `numpy.ndarray` | `hdf5` | `<name>.h5`, dataset `data` with dimension labels; opt-in |

`record["x"]` returns a lazy handle: a `zarr.Array`, an xarray object opened without loading, or
a `pyarrow.dataset.Dataset` for `arrow` fields. `record.x` returns the declared type: an ndarray
for jaxtyping numpy fields. `record.load("x")` always loads. `register_codec` adds codecs, and
later registrations are tried first.

### HDF5 output (opt-in)

For h5-based tools, a record type can store its arrays as HDF5 with
`codecs = {"ndarray": "hdf5"}`, or one field with `write(name, array, codec="hdf5")`
(`pip install 'tensordir[hdf5]'`). Each field is `<name>.h5` with a single dataset `data`, the
layout datamate wrote, and its dimension names as HDF5 dimension labels. Zarr stays the
default because HDF5 trades away what matters on clusters:

- **File locking.** HDF5 locks files with `flock`/`fcntl`, which is unreliable or disabled on
  NFS and Lustre and can hang or fail readers. tensordir opens files with `locking=False`; other
  tools may need `HDF5_USE_FILE_LOCKING=FALSE`.
- **Single writer.** One process writes a file at a time, and every `write` rewrites the whole
  file (atomically, by rename). Zarr writes each chunk as its own object.
- **No chunk-parallel writes.** Many workers cannot fill one HDF5 array concurrently; with zarr
  they can write disjoint chunks.
- **Appends in place.** Under `codecs = {"ndarray": "hdf5"}` (or `append(..., codec="hdf5")`),
  `append` creates a chunked dataset with `maxshape=(None, ...)` and resizes it along axis 0,
  datamate's `extend`. The contract matches zarr appends: rows must match the trailing shape,
  values are cast to the stored dtype, and a field written with `write` cannot be appended to.
  The file is modified in place, so a reader can see a partly appended tail.
- **Open handles.** `record["x"]` (and `record.x` for fields not typed as numpy arrays) returns
  an `h5py.Dataset` whose file stays open until the handle is dropped; `record.load("x")` and
  typed numpy fields read into memory and close the file.

## External tables

Some inputs live outside any root and are too large to copy: another project reads CAVE synapse
exports, Delta Lake tables on GCS, through `deltalake` and duckdb. `td.DeltaRef` and
`td.ParquetRef` name one snapshot of such a table so that records can depend on it:

| | `DeltaRef(uri, version=None, storage_options=None)` | `ParquetRef(uri, version=None)` |
|---|---|---|
| table | a Delta Lake table (local, `gs://`, `s3://`, ...) | a directory of parquet files, flat or hive-partitioned |
| version | the Delta version; `None` pins the current one when the reference is created | the listing digest (below); `None` computes it |
| canonical form | `{"__delta__": uri, "version": n}` | `{"__parquet__": uri, "version": digest}` |
| drift | Delta keeps old versions until they are vacuumed | reading re-lists the files and raises `SnapshotChanged` on a mismatch |

**The parquet digest.** A parquet directory has no version number, so `ParquetRef` fingerprints
its file listing: it lists the data files (skipping names that start with `.` or `_`, as
pyarrow does), sorts the `(relative path, size in bytes)` pairs, and takes the SHA-256 of their
JSON form. File contents are not read. The digest is the reference's version, so it is stored in
the spec of every record that uses the reference and enters that record's key. Adding,
removing, renaming or resizing a file gives a new `ParquetRef(path)` a new digest and so new
keys; opening an old reference to the changed directory recomputes the digest and raises
`SnapshotChanged` instead of reading different data. A file rewritten with the same name and
size keeps the digest and is not detected. A `DeltaRef` does not need a fingerprint: it pins the
Delta table version, which Delta can still read until it is vacuumed.

- A reference can be a spec field (typed `td.DeltaRef`) or a `get` input. Its identity is the URI
  and version, so a new table version gives new keys, and old records stay valid for the old
  snapshot. Local paths become absolute URIs.
- `storage_options` (credentials) and the parquet `filesystem` are not identity: they never enter
  keys, `_record.json` or reprs. A reference revived from a stored spec has none; pass them again
  with `dataclasses.replace(ref, storage_options=...)`, or rely on ambient credentials.
- Reads are lazy: `dataset()` gives a `pyarrow.dataset.Dataset`; `to_table`, `to_pandas`,
  `to_batches`, `to_reader` and `scanner` take `columns=` and `filter=`, and `duckdb()` and
  `to_polars()` hand the dataset to those engines, which push down their own projections and
  filters. `filter` is a pyarrow expression or a mapping in the pushdown-able part of the
  `where` grammar: a value, `None` (null), a list or `td.any_of` of values (any of),
  `td.exactly` of a scalar, or comparisons like `">=2, <8"`.
- `ref.materialize(record, name, columns=, filter=)` streams a selection into a record as an
  `arrow` field. Use it when a build needs a stable local copy, or when the source may be
  vacuumed.

## Dimension names

Typed fields use jaxtyping annotations. Their shape strings become zarr `dimension_names`, so
`xr.open_zarr(record.path / "data.zarr")` gives a labeled Dataset.

- A size `n_<things>` becomes the dimension `<thing>`: drop `n_` and one trailing `s`.
- Irregular plurals (`n_vertices` → `vertex`, `n_radii` → `radius`, ...) come from a built-in
  table, `register_dims`, or a record's `dims` mapping.
- `name=size` tokens use `name`: `"pre=n_types post=n_types"` gives `("pre", "post")`.
- Symbolic (`n_bins+1`), literal (`2`), anonymous (`_`), variadic (`*batch`, `...`) and repeated
  dimensions are rejected. Write `Annotated[..., Dims("bin_edge")]`, or pass `dims=` to `write`.
- When a build finishes, every field using the same dimension name must have the same size along
  it.

## joblib

`tensordir.memory(location)` returns a `joblib.Memory` whose store backend, `td.StoreBackend`,
uses the codec registry:

- A result is written by `codec_for(result)` into `output.td/` (a temporary directory, then
  renamed), with `codec.json` naming the codec, and read back lazily where the codec supports it.
  xarray goes to zarr, DataFrames to parquet, `{str: Tensor}` to safetensors, and values of a
  registered codec (images, models) use that codec.
- numpy arrays and plain Python values keep joblib's pickle path (`skip_codecs`: `ndarray`,
  `json`, `pickle`), which keeps memory mapping and exact Python types. So does anything whose
  codec is not faithful for it (`Codec.faithful`): torch modules under the built-in safetensors
  codec, arrow tables.
- `output.zarr` and `output.parquet` from earlier tensordir versions, and `output.h5` from the
  flyvis and another project netCDF backends, are still read; recomputing replaces them.
- Subclasses override `select_codec(item)` or `skip_codecs` and are passed as
  `td.memory(location, backend=Subclass)`, which registers them with joblib.
- Records pickle as `(type id, key, path)`, so joblib hashes a record's identity, not its data.
  Specs pickle in canonical form.
- The backend subclasses joblib's private `FileSystemStoreBackend`. joblib is pinned to
  `>=1.4,<1.6`, and contract tests check the private signatures the backend relies on.
  Subclasses of `td.StoreBackend` that only override `select_codec` or `skip_codecs` do not
  touch joblib internals.

## Collections and views

- `Record.query(**filters)` filters on the flattened spec (`model__lr=0.1`, `seed=[0, 1]`,
  `lr=">1e-3"`, a regex or a predicate) and on status. Stored specs are revived through the
  class's current `Spec`, so fields added later with `since=` show their default for old records.
- `tensordir.where` applies the same conditions to `Collection`s, DataFrames (columns, with
  `spec.` implied, then index levels) and xarray objects (coordinates). It generalises flyvis's
  `.custom.where`.
- **Accessors.** `import tensordir` imports pandas and xarray when they are installed and
  registers a `.td` accessor on `DataFrame`, `Series`, `DataArray` and `Dataset`:
  `table.td.where(...)` and `responses.td.where(...)` call `tensordir.where`;
  `series.td.where("<0.1", cell_type="T4a")` filters a Series by value and by index level;
  `table.td.diff(a, b)` and `row.td.diff(other)` diff two table rows on their `spec.` columns.
  xarray accessors have no `diff`: coordinates are not specs. If another package has already
  registered `.td`, tensordir warns and keeps the other accessor.
- **Import time.** Registering the accessors imports pandas and xarray, which makes
  `import tensordir` about 0.45 s slower: 0.79 s instead of 0.32 s (median of ten fresh
  processes on a laptop; `python -X importtime` attributes 0.28 s to pandas and 0.05 s to
  xarray). `TENSORDIR_NO_ACCESSORS=1` in the environment skips registration and both imports;
  `table.pipe(td.where, seed=[0, 1])` and `td.where(responses, cell_type="T4a")` do the same
  filtering without accessors.
- `Collection.table()` gives one row per record; `varying_only=True` keeps the spec columns that
  differ. `Fit.info(depth=, include=, **filters)` is `Fit.query(status=None, **filters).table(...)`
  in one call (another project's `info`), and `record.info()` is a record's one-row table.
  `tensordir.diff` compares two specs, records or dicts by dotted name.
- `Collection.map(fn, dim=, coords=)` concatenates per-record xarray results, optionally
  memoized.
- `tensordir.reindex()` writes `.tensordir/index.parquet`. Queries trust a type directory's rows
  while its mtime is unchanged, because records enter and leave by rename, which updates that
  mtime.
- `View` wraps a record with a joblib cache in `<record>/__cache__`. `cached_method` caches per
  instance, keyed on the canonical bound arguments.

### Conditions

`Record.query`, `Record.info`, `Collection.where`, `tensordir.where` and the `.td.where`
accessors take one condition per name:

| condition | matches |
|---|---|
| `0.1`, `"relu"` | equal values; floats within a relative tolerance of 1e-9 |
| `None` | missing values (None, NaN, NA) |
| `">1e-3, <0.1"` | values meeting every comparison (comma means AND) |
| `[0, 1]`, `(8, 16)` | values equal to the whole list, or to any of its items |
| `{0, 1}` | values equal to any item |
| `td.any_of(0, 1)` | values meeting any item; items are conditions; never the whole list |
| `td.exactly([8, 16])` | values equal to the whole list only; strings are not parsed as comparisons |
| `re.compile("^base")` | values whose string form contains a match |
| `lambda v: v > 0` | values for which the callable returns a truthy result |

A bare list means "any of" for scalar fields (`seed=[0, 1]`) and "equal" for list-valued fields
(`layers=[8, 16]`), and `layers=[[8], [16]]` finds either list. The rule tries the whole list
first, then each item, so one condition can match both ways when a field holds scalars in some
records and lists in others. `td.exactly([8, 16])` allows only the whole list and
`td.any_of([8], [16])` only the items; inside `any_of`, list items compare whole. Filters on
external tables (`DeltaRef.to_table(filter=...)`) accept the pushdown-able subset: values,
`None`, lists and `any_of` of values, `exactly` of a scalar, and comparisons.

## Legacy data

`tensordir.legacy.open_datamate(path)` reads datamate directories without writing to them:
`_meta.yaml`, single-dataset `.h5` arrays (opened with HDF5 file locking disabled), CSV files,
other files and subdirectories. There are no import aliases for datamate's API.

## Known limitations

- `bytes` arrays (`dtype="S"`) have no stable zarr v3 specification yet, and zarr warns about
  them.
- Parquet drops the categories of empty categorical columns.
- `since=` doesn't support changing a field's default later (see Keys).
- Context roots set with `use_root` don't propagate into new threads, which resolve against
  `set_root`, `$TENSORDIR_ROOT` or the working directory instead. Pass `root=`, call `set_root`,
  or start the thread with `contextvars.copy_context().run`.
