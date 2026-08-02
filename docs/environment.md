# Environment and reproducibility contract

Review items G1-08, R-01, R-02.

## Installing

```bash
conda create -n bouldernet python=3.10
conda activate bouldernet
pip install torch --index-url https://download.pytorch.org/whl/cu128   # cu128 maxes out at 2.11
pip install -e .[dev]
pip install -e path/to/rastertools_BOULDERING path/to/shptools_BOULDERING
pytest -m "not slow"            # 394 passed, ~75 s
```

`rastertools_BOULDERING` and `shptools_BOULDERING` are declared in `pyproject.toml` but are not on
PyPI — they install from the maintainer's clones. They were previously imported and *undeclared*,
so a clean environment installed cleanly and then failed on the first import of `predict.py`.

`sahi` and `ultralytics` carry upper bounds. This pipeline bypasses sahi's model wrapper and calls
ultralytics directly, so it depends on calling *behaviour* that a lower bound does not pin: the
`conf=` contract (review X-01), `slice_image`'s return shape, and the fact that
`results.masks.data` comes back at `imgsz` rather than at the prototype size (review G3-10). A
major release could move any of them, and two of the three would move silently.

## Where a result's provenance lives

Not in this file. Bounds keep a fresh install from being obviously wrong; they do not record what
produced a given number. That is per-run, in the manifest written beside every artifact by
`presets.write_manifest`:

- resolved configuration and its hash;
- code SHA **and dirty flag**;
- sha256 of the input raster and the model weights;
- versions of ultralytics, sahi, torch, geopandas, shapely, rasterio, opencv;
- CRS, row counts, the per-reason discard tally, wall time, host and platform.

`runstate.StageLedger` adds the run-level record: append-only JSONL, one entry per stage attempt,
with artifact checksums and dependency keys, so resume is based on evidence rather than on a file
existing.

## Where drivers live (R-01)

**Tracked** — anything that produces a result:

| path | purpose |
|---|---|
| `scripts/benchmark_driver.py` | repeated, hashed inference benchmark; `--summarise` rebuilds the table from raw JSON |
| `scripts/run_multiscale_arms.py` | full-scene multiscale, one subprocess per arm, ledger-backed resume |
| `scripts/review/measure_contract_impact.py` | measures the blast radius of X-01/X-03/G1-06 |
| `scripts/review/measure_batch_neutrality.py` | batch-1 vs batch-8 geometry, and `compare_catalogs`, the template for every old/new catalog diff |
| `scripts/review/measure_operating_point.py` | the conf 0.10 vs 0.25 decision, through the full production path against human labels |
| `scripts/review/profile_production_path.py` | per-stage breakdown of `predict_production`, every stage wrapped in place |
| `scripts/review/probe_gpu_mask_downscale.py` | evidence that the device-side mask downscale is bit-identical |
| `scripts/review/audit_headline_numbers.py` | re-derives headline numbers from saved tables and fails on disagreement |
| `scripts/build_obb_labels.py` | Solution H OBB labels with an audited edge policy |

**Gitignored** (`data/`) — inputs and outputs only: rasters, weights, catalogs, benchmark records,
run ledgers. Nothing that *computes* a published number should live there, because the `data/` tree
is not in version control and several drivers in it hard-coded `C:\Users\brian\...` paths, so they
could not be run by anyone else or re-run after a move.

Analysis drivers still under `data/` are the remaining half of R-01. They are listed in the backlog
rather than moved wholesale, because moving one without re-running it produces a tracked script
nobody has executed — worse than an untracked one that works.

## Reproducing a table

```bash
python scripts/benchmark_driver.py --raster data/raster/crop.tif --repeats 5 --out data/benchmark
python scripts/benchmark_driver.py --summarise data/benchmark
```

The summary is a *view* over the saved records, not a transcription, so a second run adds to the
evidence instead of replacing it. Repetitions are reported as a median with the full range: the
first run of a session carries CUDA and model warm-up and on this machine came in at 31.9 s against
11.4 s for the second, a **94.5 % spread**. A single timing is not a measurement, which is why the
original single-run tables could not be checked.
