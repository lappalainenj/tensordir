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

## Layout and lifecycle

```
<root>/<RecordName>/<key[:16]>/_record.json   # key, spec, inputs, status, fields, provenance
<root>/<RecordName>/<key[:16]>/data.zarr/     # arrays and xarray groups (zarr v3)
<root>/<RecordName>/<key[:16]>/<name>.json|.parquet|.safetensors|.pkl
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

An exception in `build` stores a `failed` record with the traceback. `KeyboardInterrupt` and
`SystemExit` leave no record behind.

`in_place = True` builds in the final directory, so readers see fields and `append`ed rows while
the build runs (training logs). A rebuild after a crash keeps what the earlier attempt wrote,
which lets a recipe resume from checkpoints.

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
| JSON scalars, lists, str-keyed dicts | `json` | `<name>.json` |
| `torch.nn.Module`, `{str: Tensor}` | `safetensors` | `<name>.safetensors` (state dict) |
| anything | `pickle` | `<name>.pkl`, only with `write(name, value, codec="pickle")` |

`record["x"]` returns a lazy handle: a `zarr.Array`, or an xarray object opened without loading.
`record.x` returns the declared type: an ndarray for jaxtyping numpy fields. `record.load("x")`
always loads. `register_codec` adds codecs, and later registrations are tried first.

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

`tensordir.memory(location)` returns a `joblib.Memory` with a store backend for labeled results:

- xarray results go to `output.zarr`: written to a temporary directory, then renamed, and opened
  lazily on load.
- DataFrames go to parquet; everything else takes joblib's pickle path.
- `output.h5` caches written by the flyvis and another project netCDF backends are still read.
- Records pickle as `(type id, key, path)`, so joblib hashes a record's identity, not its data.
  Specs pickle in canonical form.
- The backend subclasses joblib's private `FileSystemStoreBackend`. joblib is pinned to
  `>=1.4,<1.6`, and contract tests check the private signatures the backend relies on.

## Collections and views

- `Record.query(**filters)` filters on the flattened spec (`model__lr=0.1`, `seed=[0, 1]`,
  `lr=">1e-3"`, a regex or a predicate) and on status. Stored specs are revived through the
  class's current `Spec`, so fields added later with `since=` show their default for old records.
- `tensordir.where` applies the same conditions to `Collection`s, DataFrames (columns, with
  `spec.` implied) and xarray objects (coordinates). It generalises flyvis's `.custom.where`.
- `Collection.table()` gives one row per record; `varying_only=True` keeps the spec columns that
  differ. `tensordir.diff` compares two specs, records or dicts by dotted name.
- `Collection.map(fn, dim=, coords=)` concatenates per-record xarray results, optionally
  memoized.
- `tensordir.reindex()` writes `.tensordir/index.parquet`. Queries trust a type directory's rows
  while its mtime is unchanged, because records enter and leave by rename, which updates that
  mtime.
- `View` wraps a record with a joblib cache in `<record>/__cache__`. `cached_method` caches per
  instance, keyed on the canonical bound arguments.

## Legacy data

`tensordir.legacy.open_datamate(path)` reads datamate directories without writing to them:
`_meta.yaml`, single-dataset `.h5` arrays (opened with HDF5 file locking disabled), CSV files,
other files and subdirectories. There are no import aliases for datamate's API.

## Known limitations

- `bytes` arrays (`dtype="S"`) have no stable zarr v3 specification yet, and zarr warns about
  them.
- Parquet drops the categories of empty categorical columns.
- `since=` doesn't support changing a field's default later (see Keys).
- Context roots set with `use_root` don't propagate into new threads. Pass `root=` or call
  `set_root` instead.
