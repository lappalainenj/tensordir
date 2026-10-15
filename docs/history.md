# History

tensordir is the third generation of one idea: name a result by the configuration that produced
it, and let the constructor build it or reopen it.

## artisan (2018)

Mason McGill wrote [artisan](https://github.com/MasonMcGill/artisan), a build system whose
artifacts were directories created from a configuration. It was published under the MIT license,
"Copyright (c) 2018 Mason McGill", and installed as `artisan-builder`. Janne Lappalainen
contributed to it. The repository was archived in September 2026.

## datamate (2020–2025)

Janne Lappalainen started datamate in 2020 from artisan's design, in a private repository whose
first commit is from 17 April 2020. The public repository was created on 28 February 2023 under
his GitHub account and later transferred to the `flyvis` organization (`flyvis/datamate`).
Releases 0.1 (28 February 2023) through 1.0.0 (23 January 2025) appeared on PyPI.

datamate added:

- a `Directory` class whose constructor built or reopened a directory from a config
- `_meta.yaml` metadata
- one HDF5 file per array, with SWMR reads
- context-scoped roots

The flyvis connectome-constrained visual system models (`flyvis` package) and the another project
experiments used it. Contributors besides Janne Lappalainen include Felix Pei, Jaivardhan Kapoor
and Linda Ulmer. The license reads "Copyright (c) 2022 Janne Lappalainen" and "Copyright (c) 2018
Mason McGill".

## tensordir (2026–)

Janne Lappalainen started tensordir in 2026 as a successor, not a fork. The repository was
created by cloning datamate's history up to commit `63af784`, so datamate's commits and
authorship are preserved. The code was then replaced in new commits.

The design comes from problems seen in datamate, another project and another project:

- **Lookup.** Finding a directory by config meant scanning every `_meta.yaml` under the root. Its
  replacement is content-addressed paths with sha256 keys, plus `since=` fields so that adding a
  field doesn't orphan existing records.
- **Status.** A killed job left `running` forever. Leases with heartbeats and expiry fix this, as
  do atomic finalization and stored failures.
- **HDF5 on shared filesystems.** Locking and SWMR caused trouble there. zarr v3 is now the
  default, and HDF5 is only read, for legacy data.
- **Types.** Fields were dynamic attributes with shapes documented in docstrings. Fields are now
  typed with jaxtyping annotations, and their dimension names are stored.
- **Memoization.** A joblib store backend for xarray had been written separately in flyvis and in
  another project. It now ships in the package, with identity hashing of records.

Some code was adapted rather than written fresh. The notices are in `LICENSE`.

- **From flyvis** (MIT, "Copyright (c) 2023 Janne K. Lappalainen, Fabian D. Tschopp, Mason McGill,
  Jakob H. Macke, Srinivas C. Turaga"): the joblib xarray backend, the identity-hashing pattern of
  `CheckpointedNetwork.__reduce__`, and the `generic_responses` concatenation behind
  `Collection.map`.
- **From another project** (MIT, "Copyright (c) 2024 mackelab"): the `info` table behind
  `Collection.table`, and the acquire/complete/fail run lifecycle that the lease and `failed`
  status generalize.

The joblib backend (adapted from flyvis) and the code generalized from another project are tested in
this repository.
