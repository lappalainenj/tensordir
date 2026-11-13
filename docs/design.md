# Design

## Keys

A record's key is the sha256 of the canonical JSON of

```json
{"inputs": [...], "schema_version": 0, "spec": {...}, "type": "Fit"}
```

- `type` is `__record_name__`, else the class's qualified name without its module, so moving a
  class between modules keeps its keys. Two classes in different modules with the same name and
  no `__record_name__` raise `TypeError`; set `__record_name__` on one of them, or to keep keys
  across a class rename.
- `schema_version` is a class attribute. Bump it when a recipe change invalidates old results.
- `spec` holds the hashed settings: the keyword-only `__init__` parameters, as the generated
  `Fit.Spec`. Canonical JSON sorts keys, turns numpy scalars into
  Python scalars and paths into POSIX strings, writes floats in their shortest round-trip form,
  maps -0.0 to 0.0, tags infinities, rejects NaN, and accepts arrays only through `Digest`. A float
  field given an int is stored as a float, so `lr=1` and `lr=1.0` share a key.
- `inputs` are the positional-only `__init__` arguments, by position: renaming an input
  parameter keeps every key, reordering inputs changes them. Records enter by type and key,
  arrays by digest, and other values by canonical form. A workspace record also enters with the
  time of its last edit (see [Workspace records as inputs](extending.md#workspace-records-as-inputs)).
  Paths never enter the key; `_meta.json` stores the path of every input record under
  `upstream`, and a record among the settings with its path, so both can be reopened.

Parameter names of inputs and the name of a record's class (when `__record_name__` is set) don't
enter the key. Two field options control the key, given as a parameter default
(`device: str = td.field("cpu", hashed=False)`) or in the annotation
(`device: Annotated[str, td.field(hashed=False)] = "cpu"`):

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
<root>/<RecordName>/<key[:16]>/_meta.json   # key, spec, inputs, status, fields, provenance
<root>/<RecordName>/<key[:16]>/data.zarr/     # arrays and xarray groups (zarr v3)
<root>/<RecordName>/<key[:16]>/<name>.json|.parquet|.safetensors|.h5|.pkl
<root>/.tensordir/locks/  tmp/  index.parquet
```

`_meta.json` stores the full key, so a 16-hex prefix collision raises `KeyCollision` instead of
returning the wrong record. Provenance covers the host, pid, user, Slurm/LSF job id, Python and
package versions, argv, the git commit and dirty flag of the repository defining the record
class, and those of tensordir's own repository when it is installed from a checkout.

`Record.get` loops until it can return a `done` record:

1. Read `_meta.json`. If the status is `done`, return the record. If it is `failed`, raise
   `BuildFailed` with the stored traceback, unless `retry_failed=True`.
2. Try to take the lease (`mkdir` of the lock directory, then `owner.json`). On success, check
   again, run `__init__` in `.tensordir/tmp/...`, write `_meta.json`, and rename the directory
   into place.
3. Otherwise wait with exponential backoff (capped at 2 s), up to `timeout`. With `block=False`,
   raise `RecordBusy` instead.

An exception in `__init__` stores a `failed` record with the traceback; while `retry_failed=True`
rebuilds it, its status is `running`. `KeyboardInterrupt` and `SystemExit` leave no record behind.

A build that is killed outright (SIGKILL, an out-of-memory kill, a lost node) leaves no record
either. What it wrote stays in `.tensordir/tmp/`, and its lease turns `stale` once its heartbeat
is older than `lease_ttl`; the next `get` of the same key breaks the lease and builds again.
`tensordir gc` removes the leftovers: temporary directories without a live lease, stale leases,
and trash. Run it after a job array with killed tasks, and in particular after changing the spec
in response, because the old keys will not be asked for again and nothing else removes their
temporary directories. `tensordir --root <root> gc --dry-run` lists what it would remove. An `in_place` build is
the exception: it was writing in the final directory, which stays as a `stale` record until the
next `get` resumes it.

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

| record | inside `__init__` (the build) | after the build is `done` |
|---|---|---|
| default | yes | no: `ReadOnlyError` |
| `in_place = True` | yes; readers see each write | no |
| `mutable = True` (workspace) | yes | yes, from any handle and process |

A finished record is immutable by default because its key promises its contents: the same spec
and inputs give the same fields. A **workspace record** (`mutable = True`) gives up that promise
for the use datamate served: an object (a trainer, a solver) holds a record and writes losses,
checkpoints and summaries into it over time. Rules:

- Without an `__init__`, the first call creates an empty `done` record; edits start right away.
- Calling the class with the same settings reopens the record without running `__init__`.
- Each edit takes the record's lease, re-reads `_meta.json`, writes the field, stores the new
  manifest atomically with a `modified` timestamp, and releases the lease. Edits from many
  handles, threads or processes therefore never lose each other's fields; the last write to the
  same field wins.
- Edits need status `done`. A record that is `running` (being built elsewhere), `failed` or
  `stale` raises `ReadOnlyError`.
- Handles notice edits by the mtime of `_meta.json`; immutable records cache their manifest
  forever.
- Writing an array over a field created by `append` replaces its rows and keeps it appendable.
- Dimension sizes are checked at the end of a build, not on later edits.
- Array fields are written in place, so a reader can see a partly written array during an edit.
  File codecs (json, parquet, safetensors) replace their file atomically.

## Storage layout and multi-range reads

An array field is a zarr v3 array in `data.zarr/<name>`. Its **layout** (chunk and shard shapes)
changes how the bytes sit on disk, never what a reader gets back, so it is a storage option and
never enters the key:

```python
class Mesh(td.Record):
    layouts = {
        "*": td.Layout(chunk_bytes=128 << 10, shard_bytes=128 << 20),
        "neighbor_indices": td.Layout(chunk_bytes=32 << 10, shard_bytes=128 << 20),
    }

mesh.write("faces", faces, layout=td.Layout(chunk_bytes=1 << 20))  # per call
```

- The layout of a field is the `layout=` argument of `write` or `append`, else the type's
  `layouts[name]`, else `layouts["*"]`, else the default. The manifest stores the resulting
  `{"chunks": ..., "shards": ...}`.
- `chunk_bytes` and `shard_bytes` are targets in uncompressed bytes. Chunks run along axis 0 and
  span the trailing axes whole, the shape row-range reads want; `chunks=` and `shards=` give
  explicit shapes. A shard is one file holding many chunks and an index of their byte ranges, so
  sharding bounds the files per array without making reads coarser.
- Default for `write`: 128 KB chunks in shards of up to 128 MB. An array of one chunk (up to
  128 KB) is never sharded, and arrays up to 128 MB are one file. Default for `append`: 128 KB
  chunks without shards, because each append rewrites the last shard. A field keeps the layout it
  was created with.
- HDF5 fields take the chunk shape of an explicit layout and ignore shards.
- A layout can also be a `td.field(hashed=False)` setting that the build forwards to `write`.

When a build finishes, tensordir writes the metadata of every zarr array into the root
`data.zarr/zarr.json` (zarr's consolidated metadata); workspace edits rewrite it before
`_meta.json`. A record handle opens the zarr group once and keeps its arrays, so `record[name]`
costs no metadata read after the first; a workspace handle reopens the group when `_meta.json`
changes. Handles of finished, immutable records also keep each shard's index after its first
read, which zarr would otherwise read again before every read from that shard. Records written
before this keep opening, with one metadata read per array.

`record.read({name: rows, ...})` reads rows of several array fields in one call. `rows` is a
slice, a list of slices or `(start, stop)` pairs (concatenated in request order), or an integer
or boolean index array (gathered in the given order). Ranges less than a chunk apart are merged,
and every read of the call runs concurrently through zarr's async API, so a query step of many
small ranges costs about one round trip on object storage instead of one per range.

`record.read(..., engine="tensorstore")` (or `$TENSORDIR_READ_ENGINE=tensorstore`) reads zarr
fields with tensorstore instead, from the `tensorstore` extra. It reads the same files in fewer,
larger reads and was about twice as fast as zarr-python on local disk in tensordir's benchmark;
fields it cannot open (e.g. string arrays) fall back to zarr. zarr stays the default.

## Publishing to write-once storage

Object-store mounts such as mountpoint-s3 accept each file as one sequential write, without
rename or overwrite, so a record cannot be built there: builds rename a temporary directory into
place and leases rewrite their heartbeat. Build in a local root and publish:

```python
fit.publish("/mnt/s3/results")               # one record; returns it in the new root
fit.publish("/mnt/s3/results", upstream=True)  # and every record it was built from
```

```bash
tensordir --root /scratch/run publish /mnt/s3/results Fit
```

Each file is streamed once, data first and `_meta.json` last, so a reader of the destination
sees `absent` until the copy is complete. A record already there with the same key is kept; a
partial copy (no `_meta.json`) is removed file by file and redone. A root created by `publish`
carries `.tensordir/published`, and builds and workspace edits there raise `ReadOnlyError`.

## Packs of small records

Each record costs a directory, `_meta.json`, a zarr store and at least two reads to open, which
is too much for millions of small records that are read together. `td.Pack(T)` stores many
records of type `T` as one record:

```python
calls = [Segment.call(i, mesh, lod=0) for i in ids]   # bound and keyed; nothing runs
pack = td.Pack(Segment)(calls)                        # builds each member once, or reopens
member = pack.member(calls[0])                        # a read-only Segment
member.key == Segment.compute_key(ids[0], mesh, lod=0)
```

- `T.call(*inputs, **settings)` binds and keys a call without touching the filesystem.
- The pack's key is the digest of its sorted, distinct member keys; order and duplicates do not
  matter. Its type id is `Pack[<T's type id>]` and it lives under `T`'s root with `T`'s layouts.
- The build runs each member's `__init__` in a temporary directory (or reads a finished
  standalone record of the same key), then concatenates each array field along axis 0 into one
  pack field with row offsets in `<name>.offsets`, members in key order. JSON fields, specs and
  inputs go into the `members` field. Members may hold numpy arrays and JSON values; a field's
  trailing shape and dtype must agree across members. A failing member fails the pack.
- A member is an instance of `T` that reads its rows from the pack: fields, `load`, `read`,
  `spec`, `inputs`, `tree` and `info` work as on a standalone record, writes raise
  `ReadOnlyError`. `pack.members` is a `Collection` of all of them.
- `T.get(...)` and `T.lookup(...)` still find standalone records only; finding a packed member
  by its call needs the pack.
- A pack builds only from its calls: `td.Pack(T)(calls)`. `td.Pack(T).get(keys)` reopens a
  finished pack and raises for one that does not exist.
- All member arrays are held in memory during the pack build.

## The `__init__` interface

`__init__` is the build, and its signature is the record type's interface, as datamate's
`Directory.__init__` was. Calling the class gets the record: `Fit(points, n_columns=8)` is
`Fit.get(points, n_columns=8)`, which reopens a finished record without running `__init__` and
otherwise runs it once under the lease. `Fit.get` also takes the options `root`, `block`,
`timeout` and `retry_failed`, so those names cannot be settings.

```python
class Fit(td.Record):
    def __init__(self, points: Points, weights: np.ndarray | None = None, /, *,
                 n_columns: int, seed: int = 0,
                 device: Annotated[str, td.field(hashed=False)] = "cpu",
                 logger: Annotated[Logger | None, td.transient] = None):
        ...
```

| parameter | role | enters the key |
|---|---|---|
| positional-only (before `/`) | an **input**: what the result is computed from | by position: records by key, arrays by digest |
| keyword-only (after `*`) | a **setting**: a field of the generated `Fit.Spec` | by name, unless `hashed=False` or `since=` |
| keyword-only, `Annotated[T, td.transient]` | passed to `__init__` only | no |

Other parameter kinds (positional-or-keyword, `*args`, `**kwargs`) raise `TypeError` when the
class is defined, so every parameter states its role. The split follows the usual rule for stable
interfaces: the nouns a computation acts on are positional, so they can be renamed; the settings
are keyword-only, so they can be added and reordered. Prefer abstract types for inputs
(`Sequence[Points]`, `Mapping[str, Points]`); a list of records enters the key as the list of
their keys.

- `Fit.Spec` is generated; declaring a `Spec` on a record class raises `TypeError`. A spec passes
  its settings with `**`: `Fit(points, **spec)`, `spec.replace(seed=1)`.
- Inside `__init__`, `self.spec` is the spec and every non-private attribute written is a stored
  field; private names (`self._model`) are plain attributes.
- A subclass without its own `__init__` inherits the interface; one with its own `__init__` has
  its own.

## Directories

`Record` subclasses `Directory`. A directory is addressed by its path; a record by its key.

- **Opening** (`Cls(path)`, `Cls.open(path)`) reads `_meta.yaml` if there is one and writes
  nothing. A relative path is joined onto the type's root, as for records. `td.Directory.open`
  returns a typed directory as its registered class; a typed class refuses a folder of another
  type.
- **`_meta.yaml`** holds `type`, the metadata fields in declaration order and `aliases`. It has
  no timestamps, hosts or build state, so tracking it in git shows only edits. Edits take a lease
  on `.<name>.lock` next to the folder and replace the file atomically.
- **Child types** are nested `Directory` classes; children are subfolders whose `_meta.yaml`
  names the child type. A level spans one parent (`study.version`) or, reached through another
  level, all their children (`dev.track.study.version`).
- **IDs** are paths of child names from the directory navigation started at (`origin`), without
  `archive` folders. `resolve(id)` walks child names, then looks for the one directory listing
  `id` among its aliases (as given, or prefixed with the ID of the directory `resolve` is called
  on).
- **New names** come from the child type's `names` pattern: the highest number used by a child,
  an archived child, or an alias whose parent part is this directory's ID, plus one.
- **Metadata validation** covers `str`, `int`, `float`, `bool`, `Literal`, unions, lists and
  `td.NodeId` (must resolve from the origin); other annotations are not checked.
- **Producers.** `with directory.active():` sets a context variable and `$TENSORDIR_PRODUCER`;
  records built meanwhile store `{"id", "path"}` as `provenance["producer"]`.

## Stages

`td.Stages(name=td.Stage(after=..., repeat=..., required=...))` declares the steps of a build;
the instance decorates each stage method. `_meta.json` stores, per stage, how often it ran, when
it last finished and which fields it added.

- A stage before the stages it runs after raises `StageOrderError`; a non-repeating stage called
  twice in one build raises too. A build that ends without every required stage fails.
- A record type with stages builds in place unless it sets `in_place = False`, so a failed build
  keeps its stage progress. `retry_failed=True` reruns `__init__`, and stage calls that finished
  in an earlier attempt return at once: the first `calls` calls of each stage are skipped.
- An optional stage called on a finished record takes the lease, may add fields and rewrite the
  fields it added before, and stamps `modified`; it never changes the status. A required stage
  called after the build raises `ReadOnlyError`.

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
a `pyarrow.dataset.Dataset` for `arrow` fields. `record.x` returns the same lazy handle, except
for fields typed as numpy arrays (jaxtyping), which it reads into an ndarray. `record.load("x")`
always loads. `register_codec` adds codecs, and
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

Some inputs live outside any root and are too large to copy, such as a Delta Lake table in a cloud
bucket read through `deltalake` and duckdb. `td.DeltaRef` and
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
  keys, `_meta.json` or reprs. A reference revived from a stored spec has none; pass them again
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

- A size `n_<things>` becomes the dimension `<thing>`: drop `n_`, then singularize by suffix.
  Words ending in `ss`, `is` or `us` stay (`n_basis` → `basis`); `ies` becomes `y`
  (`n_entries` → `entry`); `sses`, `shes`, `ches`, `xes` and `zzes` drop `es`
  (`n_batches` → `batch`); any other trailing `s` is dropped (`n_cases` → `case`).
- Irregular plurals (`n_vertices` → `vertex`, `n_radii` → `radius`, `n_caches` → `cache`, ...)
  come from a built-in table, `register_dims`, or a record's `dims` mapping.
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
  flyvis netCDF backends, are still read; recomputing replaces them.
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
- **Accessors.** tensordir registers a `.td` accessor on `DataFrame`, `Series`, `DataArray` and
  `Dataset`:
  `table.td.where(...)` and `responses.td.where(...)` call `tensordir.where`;
  `series.td.where("<0.1", cell_type="T4a")` filters a Series by value and by index level;
  `table.td.diff(a, b)` and `row.td.diff(other)` diff two table rows on their `spec.` columns.
  xarray accessors have no `diff`: coordinates are not specs. If another package has already
  registered `.td`, tensordir warns and keeps the other accessor.
- **Import time.** `import tensordir` imports numpy but not pandas, xarray or zarr, which
  together cost about 0.5 s; each is imported on first use. The accessors are registered at
  `import tensordir` for libraries already imported, and otherwise by an import hook right after
  pandas or xarray finishes importing. `TENSORDIR_NO_ACCESSORS=1` in the environment skips
  registration; `table.pipe(td.where, seed=[0, 1])` and `td.where(responses, cell_type="T4a")`
  do the same filtering without accessors.
- `Collection.table()` gives one row per record; `varying_only=True` keeps the spec columns that
  differ. `Fit.info(depth=, include=, **filters)` is `Fit.query(status=None, **filters).table(...)`
  in one call, and `record.info()` is a record's one-row table.
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

## Experimental features

These features were added without an existing workload that needs them. Each stays until a real
workload shows whether it is needed; until then its behaviour may change and it may be removed.

| Feature | Where |
|---|---|
| `ParquetRef`, its listing digest and `SnapshotChanged` | External tables |
| `TableRef.duckdb`, `TableRef.materialize`, `tensordir.external.expression` | External tables |
| `td.any_of`, `td.exactly` | Collections and views |
| `td.transient` | The `__init__` interface |
| `StoreBackend` subclasses through `select_codec` and `skip_codecs` | joblib |
| `Record.inputs`, `Record.lineage` | [Composing records](extending.md#composing-records) |

`DeltaRef`, workspace records, `in_place` builds and the `.td` accessors are not experimental:
each replaces something datamate or flyvis already did.

## Known limitations

- `bytes` arrays (`dtype="S"`) have no stable zarr v3 specification yet, and zarr warns about
  them.
- Parquet drops the categories of empty categorical columns.
- `since=` doesn't support changing a field's default later (see Keys).
- Context roots set with `use_root` don't propagate into new threads, which resolve against
  `set_root`, `$TENSORDIR_ROOT` or the working directory instead. Pass `root=`, call `set_root`,
  or start the thread with `contextvars.copy_context().run`.
