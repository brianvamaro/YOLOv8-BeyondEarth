# Multi-scene campaign on Sherlock (hirise2ctx PLAN_v3 P3)

Runs BoulderNet over a scene manifest: PDS fetch -> 8-bit prep -> detection -> verified hand-off, one
directory per ObsId. Built 2026-10-14 as **plumbing only**. Every scientific choice listed under
"Open decisions" is still open, and nothing here has run on the cluster yet: the scripts are tested
locally, and the cluster specifics are copied from jobs that have run (`hirise_arms.sbatch`,
`fuse_only.sbatch`), per the standing lesson (copy the artifact that ran, not the flag that looks right).

## What is fixed, and why

| choice | value | evidence |
|---|---|---|
| georeferencing | the PDS **label** (GDAL's PDS driver), never the JP2's GeoJP2 box | the box's origin is 1.5 px off in x and y, and its CRS has lat_ts/lat_0 transposed; the label route reproduces the Prieur training rasters' origins to 0.00 px on all 3 PDS training scenes |
| 10 -> 8 bit | `min((DN + 2) >> 2, 255)` | reproduces two training rasters **pixel for pixel**: `ESP_017355_2260` (0.5 m/px, 0 of 1,932,385,780 differ) and `ESP_028537_2270` (0.25 m/px, 0 of 1,875,276,700). The clip matters: 12,467 and 6,475 px at DN 1023 would otherwise wrap to nodata |
| grid | native, no resampling or reprojection, nodata 0 | any resampling moves predictions |
| input guard | prediction refuses anything but single-band uint8 | a 10-bit raster used to reach the model clipped at DN 255, silently |
| fill slices | skipped (lever N1), byte-identical | `data/review/n1n2_identity.json`: old code vs new on real 0.25 m/px ground, 256/512/SR arms |
| whole-raster copies | gone (lever N2), byte-identical | same file; RAM coefficients in `data/review/n2_memory.json` |
| `max_det` | 1100, and a truncated arm **fails** its scene | truncation undercounts the densest ground and inflates fabric (`max-det-inflates-fabric`) |

## 0. Stage the code (laptop)

```bash
conda run -n bouldernet python scripts/pack_for_cluster.py --out ../sherlock_upload/bnet_code.tgz \
    --weights data/weights/yolov8-m-boulder-detection-tmp.pt
```
It refuses a dirty tree and writes `CODE_STAMP.json` (commit + content digests) into the tarball, so
every manifest written on the cluster names its commit. Extract into a **clean** directory, never over
an old tree (inference-env gotcha 5: a stale module stays importable):
```bash
mkdir -p ~/BoulderNet/releases/<tag> && tar xzf ~/bnet_code.tgz -C ~/BoulderNet/releases/<tag>
export REPO=~/BoulderNet/releases/<tag>/YOLOv8-BeyondEarth
```

## 1. The scene manifest

A CSV, one row per scene, row 0 = first data row = array task 0. Needs an observation-id column;
URLs are optional (derived from the id) and column names are matched loosely, so hirise2ctx's
`obs_id, product_id, jp2_url, lbl_url` and `ObsId, ProductId, JP2 URL, LBL URL` both work.

## 2. Fetch + prep (CPU)

**First measure the PDS host from Sherlock.** On 2026-10-14 it served ~50 KB/s to the laptop, and
parallel connections did not help (8 got 29 KB/s between them). At that rate a 1 GB JP2 takes ~6 h.
```bash
curl -o /dev/null -w '%{speed_download}\n' --max-time 60 \
  https://hirise-pds.lpl.arizona.edu/PDS/RDR/ESP/ORB_028500_028599/ESP_028537_2270/ESP_028537_2270_RED.JP2
```
- If compute nodes reach it at a usable rate: `FETCH=yes` (default) in the array job.
- If not (or GPU-style no-internet): fetch on the DTN (`dtn.sherlock.stanford.edu`), e.g. in `tmux`:
  `for i in $(seq 0 62); do python scripts/prep_pds_scene.py --manifest $M --row $i --out-root $SCENES --fetch-only; done`,
  then run the array with `FETCH=no`. The fetch resumes `.part` files and renames only complete ones.

```bash
mkdir -p ~/BoulderNet/logs && cd ~/BoulderNet
M=$HOME/campaign/scenes.csv
PREP=$(sbatch --parsable --array=0-62%8 --export=ALL,MANIFEST=$M,REPO=$REPO,FETCH=yes \
       $REPO/sherlock/campaign_prep.sbatch)
```
Each scene gets `$SCENES/<obs>/<product>.tif` + `<product>.prep.json` (rule, frame check, JP2 and
output sha256, DN counts including the saturation and collision counts) + `fetched.json`.

## 3. Detect + verify (GPU)

```bash
sbatch --array=0-62 --dependency=aftercorr:$PREP \
       --export=ALL,MANIFEST=$M,REPO=$REPO,ARMS=256,DEST=<durable dir> $REPO/sherlock/campaign_detect.sbatch
```
`aftercorr` releases task *i* as soon as prep task *i* succeeds. Resume = resubmit: arms are
ledger-keyed on content (`run_ledger.jsonl`), so finished arms are skipped.

`verify` (run by every detect task, or alone: `scripts/campaign.py verify ...`) fails the scene
unless every arm reports `"truncation": {}` and ran on the prepared raster (sha256 match), then writes
`SHA256SUMS` and `handoff.json`.

**Getting the catalogs off the cluster (maintainer's choice, 2026-10-15: transfer from `$SCRATCH`, no
`DEST`).** Pull each verified scene through the DTN as it completes, and re-check it on arrival:
```bash
rsync -a bamaro@dtn.sherlock.stanford.edu:'$SCRATCH/campaign/out/<ObsId>/' <local>/<ObsId>/
(cd <local>/<ObsId> && sha256sum -c SHA256SUMS)
```
`$SCRATCH` purges on inactivity, so do not leave a finished campaign there. `DEST` remains available
if a durable cluster directory is ever wanted: the scene is then rsynced there and `sha256sum -c`'d.

Progress: `python scripts/campaign.py status --manifest $M --scenes $SCENES --out $OUT`.

## 4. Memory accounting

Every arm manifest records `peak_rss_bytes` (the arm runs in its own process), the ledger copies it,
and each detect task prints `sacct ... MaxRSS`. After the fact:
`sacct -j <jobid> --format=JobID,MaxRSS,Elapsed,State --units=G`.
The same works retroactively for the 2026-08-19 HiRISE jobs (pre-N2 RAM at 1.93 Gpx).

## Hand-off layout (D10, proposed)

```
<OUT or DEST>/<ObsId>/
    <ProductId>-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg   (one per arm)
    <same>-manifest.json      config, code identity (commit via CODE_STAMP), inputs, truncation, peak RSS
    run_ledger.jsonl          per-arm attempts, keys, artifact digests
    handoff.json              the verify verdict + per-layer summary + the input raster's provenance
    SHA256SUMS                sha256sum -c format (transfer integrity; GeoPackage bytes are not
                              reproducible, so this is not a reproducibility check)
```
Geometry is in the raster's CRS as the PDS label defines it (lat_ts = CENTER_LATITUDE, lat_0 = 0,
the label frame). hirise2ctx's SP1 correction should find nothing to correct and must stay a
relabel (`set_crs`), never a reprojection.

## Open decisions (not settled by this tooling)

- **D9 configuration.** `ARMS=256` = configuration 1. `ARMS="512 256"` runs configuration 2's arms
  unfused: fusing a `{512,256}` ladder needs a named preset (the runner refuses any set but the
  production three). Configuration 3 (SAM2 re-mask) on a ladder needs phase 5a, which is unbuilt;
  "256 + SAM2" is buildable but SAM2 has never run on this stack, so it needs a parity check first.
- **D36 score floor** is downstream: catalogs are the conf-0.10 superset.
- **When `max_det` 1100 binds** at 0.25 m/px (likeliest on the 512 arm): today the scene fails.
  Retry at a higher cap with a smaller batch, or stop and report? A non-binding run does not depend
  on the cap (279 tiles verified inert), so retrying until nothing binds is well defined.
- **Durable storage** for `DEST`.
- **Mixed-binning scenes.** Many "0.25 m/px" RED products bin some CCDs 2x (e.g. ESP_028537_2270:
  `MRO:BINNING = (2,2,2,1,1,1,1,2,2,...)`), so strips of the scene are 0.5 m/px resampled onto the
  0.25 grid and have a coarser completeness floor. The prep sidecar records the binning per scene.
