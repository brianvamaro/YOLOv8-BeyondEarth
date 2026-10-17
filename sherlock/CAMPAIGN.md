# Multi-scene campaign on Sherlock (hirise2ctx PLAN_v3 P3)

Runs BoulderNet over a scene manifest: PDS fetch -> 8-bit prep -> detection -> verified hand-off, one
directory per ObsId. Built 2026-10-14 as **plumbing only**. Every scientific choice listed under
"Open decisions" is still open, and nothing here has run on the cluster yet: the scripts are tested
locally, and the cluster specifics are copied from jobs that have run (`hirise_arms.sbatch`,
`fuse_only.sbatch`), per the standing lesson (copy the artifact that ran, not the flag that looks right).
Two exceptions are new and have run nowhere: the `srun` step in `campaign_detect.sbatch` (no job that
has run used srun; it now passes `--cpus-per-task` and logs the CPUs the step got), and PDS prep on the
cluster's GDAL (pre-flight in §2).

## What is fixed, and why

| choice | value | evidence |
|---|---|---|
| georeferencing | the PDS **label** (GDAL's PDS driver), never the JP2's GeoJP2 box | the box's origin is 1.5 px off in x and y, and its CRS has lat_ts/lat_0 transposed; the label route reproduces the Prieur training rasters' origins to 0.00 px on all 3 PDS training scenes |
| 10 -> 8 bit | `min((DN + 2) >> 2, 255)` | reproduces two training rasters **pixel for pixel**: `ESP_017355_2260` (0.5 m/px, 0 of 1,932,385,780 differ) and `ESP_028537_2270` (0.25 m/px, 0 of 1,875,276,700). The clip matters: 12,467 and 6,475 px at DN 1023 would otherwise wrap to nodata |
| grid | native, no resampling or reprojection, nodata 0 | any resampling moves predictions |
| input guard | prediction refuses anything but single-band uint8 | a 10-bit raster used to reach the model clipped at DN 255, silently |
| fill slices | skipped (lever N1), byte-identical | `data/review/n1n2_identity.json`: old code vs new on real 0.25 m/px ground, 256/512/SR arms |
| whole-raster copies | gone (lever N2), byte-identical | same file; RAM coefficients in `data/review/n2_memory.json` |
| `max_det` | 1100 to start, then **escalation**: an arm with any slice within 10% of its cap is re-run at 2x the cap and 1/2 the batch until none is (maintainer decision 2026-10-15). `ON_MAX_DET=error` restores fail-the-scene | truncation undercounts the densest ground and inflates fabric (`max-det-inflates-fabric`). Real-inference check, `data/review/escalation_e2e.json`: from a cap of 25 a ROM tile climbs 25/8 -> 50/4 -> 100/2 -> 200/1, and the result is **identical** to an uncapped run at batch 1 (1,632 of 1,632, all geometry, max \|dscore\| 0) |

## 0. Stage the code (laptop)

```bash
conda run -n bouldernet python scripts/pack_for_cluster.py --out ../sherlock_upload/bnet_code.tgz \
    --weights data/weights/yolov8-m-boulder-detection-tmp.pt
```
It refuses a dirty tree and writes `CODE_STAMP.json` (commit + content digests) into the tarball, so
every manifest written on the cluster names its commit. Extract into a **clean** directory, never over
an old tree (inference-env gotcha 5: a stale module stays importable), so give every upload a new `TAG`:
```bash
TAG=pilot-1
mkdir -p ~/BoulderNet/releases/$TAG && tar xzf ~/bnet_code.tgz -C ~/BoulderNet/releases/$TAG
```

## 1. The scene manifest

A CSV, one row per scene, row 0 = first data row = array task 0. Needs an observation-id column;
URLs are optional (derived from the id) and column names are matched loosely, so hirise2ctx's
`obs_id, product_id, jp2_url, lbl_url` and `ObsId, ProductId, JP2 URL, LBL URL` both work.

**Shell setup.** The sbatch scripts set these for themselves. Every command below that is typed by hand
(login node, dev node, DTN) needs them, so run this first in every new shell:
```bash
TAG=pilot-1                                                   # the release extracted in §0
ml python/3.12.1
export ENV_DIR=/home/groups/mlapotre/bamaro/envs/bouldernet-infer
source "$ENV_DIR/bin/activate"
export REPO=~/BoulderNet/releases/$TAG/YOLOv8-BeyondEarth
export M=$HOME/campaign/scenes.csv
export SCENES=$SCRATCH/campaign/scenes OUT=$SCRATCH/campaign/out  # the sbatch scripts' defaults
N=$(python "$REPO/scripts/campaign.py" count --manifest "$M")    # array tasks 0..N-1
echo "$N scenes in $M"
mkdir -p ~/BoulderNet/logs && cd ~/BoulderNet   # the sbatch logs go to logs/ under the submit directory
```
Submit every `sbatch` below from `~/BoulderNet`: the scripts write `logs/<name>-%A_%a.out` relative to
the directory they are submitted from, and a job whose log directory does not exist fails at start
without writing a log.
Array ranges below are derived from `N`, never typed: a typed `0-62` launched 63 tasks for the 62-row
v3 manifest, and the extra task's detect job pended forever behind its failed prep.

## 2. Fetch + prep (CPU)

**Pre-flight: one conversion on the cluster's GDAL.** PDS prep has only run on the laptop's conda GDAL.
Here it runs on the GDAL bundled in the rasterio wheel (the venv has no `osgeo` bindings), and the label
route needs that build's PDS driver to follow the label into the JP2, so it needs a JPEG 2000 driver as
well. A mismatch fails loudly, since prep refuses rather than write a raster in another frame, but it
would fail every prep task and every detect task waiting on them. So, on a dev node
(`sh_dev -c 8 -m 16GB -t 2:00:00`) after the shell setup:
```bash
python -c "import rasterio; e = rasterio.Env(); e.__enter__(); d = e.drivers(); print('GDAL', rasterio.__gdal_version__, '| PDS', 'PDS' in d, '| JP2OpenJPEG', 'JP2OpenJPEG' in d)"
python "$REPO/scripts/prep_pds_scene.py" --lbl "$SCRATCH/campaign/preflight/ESP_028537_2270_RED.LBL" \
    --compare-to "$SCRATCH/campaign/preflight/reference/ESP_028537_2270_RED.tif"
```
First upload, from the laptop's `data/pds_scenes/ESP_028537_2270/`, that product's `.LBL` and `.JP2`
to `$SCRATCH/campaign/preflight/`, and its converted `.tif` (pixel-identical to Prieur's training
raster) to `$SCRATCH/campaign/preflight/reference/`, so the conversion cannot overwrite it. Both drivers
must print `True`, and the conversion must print `IDENTICAL -- 0 of 1875276700 px differ` with transform
and CRS equal, and exit 0. Its sidecar then records the cluster's GDAL version and `frame_check`
dx/dy 0.0. Anything else: fix the environment before submitting the array.

**Then measure the PDS host from Sherlock.** On 2026-10-14 it served ~50 KB/s to the laptop, and
parallel connections did not help (8 got 29 KB/s between them). At that rate a 1 GB JP2 takes ~6 h.
```bash
curl -o /dev/null -w '%{speed_download}\n' --max-time 60 \
  https://hirise-pds.lpl.arizona.edu/PDS/RDR/ESP/ORB_028500_028599/ESP_028537_2270/ESP_028537_2270_RED.JP2
```
- If compute nodes reach it at a usable rate: `FETCH=yes` (default) in the array job.
- If not (or GPU-style no-internet): fetch on the DTN (`dtn.sherlock.stanford.edu`), e.g. in `tmux`,
  then run the array with `FETCH=no`. After the shell setup:
  ```bash
  for i in $(seq 0 $((N - 1))); do
      python "$REPO/scripts/prep_pds_scene.py" --manifest "$M" --row "$i" --out-root "$SCENES" --fetch-only
  done
  ```
  The fetch resumes `.part` files and renames only complete ones. The setup needs `ml` and the venv;
  if the DTN does not have them, run the loop from a node that does.

```bash
mkdir -p ~/BoulderNet/logs && cd ~/BoulderNet
PREP=$(sbatch --parsable --array=0-$((N - 1))%8 --export=ALL,MANIFEST=$M,REPO=$REPO,FETCH=yes \
       $REPO/sherlock/campaign_prep.sbatch)
```
Each scene gets `$SCENES/<obs>/<product>.tif` + `<product>.prep.json` (rule, frame check, JP2 and
output sha256, DN counts including the saturation and collision counts) + `fetched.json`.

## 3. Detect + verify (GPU)

```bash
sbatch --array=0-$((N - 1)) --dependency=aftercorr:$PREP --kill-on-invalid-dep=yes \
       --export=ALL,MANIFEST=$M,REPO=$REPO,ARMS=256 $REPO/sherlock/campaign_detect.sbatch
```
`aftercorr` releases task *i* as soon as prep task *i* succeeds, and `--kill-on-invalid-dep=yes`
cancels it if prep task *i* failed, instead of leaving it pending forever. Resume = resubmit: arms are
ledger-keyed on content (`run_ledger.jsonl`), so finished arms are skipped. To resubmit one scene, pass
its row instead of the range (`--array=7`, say) and no dependency.

⚠ **Resume holds only while the code is unchanged.** Every stage key embeds the package's source
digest, so a new `TAG` with any change under `src/YOLOv8BeyondEarth` re-runs every arm from the first
cap and overwrites its layer. Finish a campaign on one `TAG`, or accept the full re-run.

**When an arm escalates.**
- **What happens.** Each failed attempt runs until its first near-cap batch in slice order. The next
  attempt then reruns the whole arm at twice the cap and half the batch. On the 25-slice tile of the
  end-to-end check a failed attempt took seconds; on a scene it lasts until the run reaches the first
  dense batch, which can be most of a pass.
- **Walltime.** A dense scene can need more than the 12 h limit. Resubmitting resumes at the attempt
  the ledger says it was on, not at 1100, so attempts that already bound are not paid for again (on
  the same code; see the warning above). The
  attempt in progress is, since an attempt has no checkpoint: one whose single pass takes longer than
  `--time` restarts from zero on every resubmit and never finishes. Resubmit that row with a longer
  `--time`, within the partition's limit.
- **The trace.** `handoff.json` lists the attempts behind each layer (cap, batch, why it stopped).
- **What the halved batch costs.** It is the one thing escalation changes in the result. Measured
  against batch 8 on the same tile: 1 detection in 1,632, 98.5% of geometry identical, max
  |dscore| 3.7e-4. That is G1-02's accepted batch 1-vs-8 variation (99.3%, 3.6e-4), the bar the
  stack-parity check already uses.
- **Out of memory.** A CUDA OOM halves the batch at the same cap. At batch 1 the arm stops and says
  it needs a larger card.
  - **Why that can happen, and gets likelier at every step.** Halving the batch keeps the retained
    masks (~1 MiB per detection x batch) where they were. But ultralytics also holds ~5 MiB per
    detection of the slice being processed while it builds masks at imgsz 1024, and that follows the
    cap alone: ~5.4 / 10.7 / 21.5 / 43 GiB worst case at caps 1,100 / 2,200 / 4,400 / 8,800. A 24 GB
    card therefore runs out somewhere around a cap of 4,400, so an arm that keeps escalating will
    normally stop at batch 1 asking for a larger card rather than reach the model's 21,504 ceiling.
    On the densest HiRISE ground the 512 arm can get there: 1,100 bound on 124 of 153 whole 512-px
    tiles of `ESP_030222_1220` (backlog §"The control-site null is provisional").
  - **What to do.** Resubmit that scene on a card with more memory. The ledger resumes it at the
    attempt it stopped on.
- **An arm finished under `ON_MAX_DET=error`**, or the runner's `warn`, can complete with a slice
  near its cap (under `warn`, at it), and verify refuses that layer. Resubmitting with escalation (the
  default) does not accept that complete entry: it re-runs the arm at the next cap.

**NMS time limit.** Ultralytics abandons the rest of a batch once NMS on it takes longer than
2.0 + 0.05 x batch seconds, and those slices return nothing. The manifest records it
(`nms_time_limit_batches`, and `slices_zeroed_after_nms_time_limit` as an upper bound). Verify refuses
the layer only when slices can have been lost: an abandoned slice comes back empty, so a record with no
empty slice in its timed-out batches, or one at batch 1 (whose one slice is written before the limit is
checked), loses nothing and passes. No cap fixes it, so escalation accepts the arm as complete and a
plain resubmit changes nothing. To retry at a smaller batch, which gives each slice more of the limit,
take the arm's entries out of the scene's ledger. This keeps the old ledger beside it, under a new name
each time; `OBS` is the scene, and `256` the arm verify named:
```bash
( cd "$OUT/${OBS:?set OBS to the scene}" && B=run_ledger.before-retry.$(date +%Y%m%dT%H%M%S).jsonl && \
  mv run_ledger.jsonl "$B" && { grep -v '"stage": "arm:256"' "$B" > run_ledger.jsonl || true; } )
```
Then resubmit that row, from `~/BoulderNet`, with the `BATCH_SIZE` and `MAX_DET` verify's message
gives in `--export`: half the batch the layer's manifest records, and the cap it records.
`BATCH_SIZE` reaches an arm only when the ledger has no entry for it, and without `MAX_DET` the arm
climbs the ladder again from 1,100. Deleting the layer instead re-runs the same attempt at the same
batch; the limit is wall-clock time, so that helps only if the slowdown was transient.

**Host RAM.** `--mem=64G` is unmeasured at scene scale: the largest v3 scene is 6.31 Gpx, 3.3x the
1.93 Gpx scene that ran at 128G under the old whole-raster path. A detect task killed for host RAM
ends `OUT_OF_MEMORY` or with exit 137, prints sacct's MaxRSS, and then a `HOST RAM:` line. Its ledger
holds a failed attempt, or only a `running` marker, with no next attempt, so a resubmit at the same
`--mem` repeats the same attempt and dies the same way. Resubmit that row with more (`--mem=128G` on
the `sbatch` line overrides the script's 64G). Put the largest and the densest scenes in the pilot, so
this turns up there rather than across the whole array.

`verify` runs in every detect task; alone, after the shell setup, it is
`python "$REPO/scripts/campaign.py" verify --manifest "$M" --row 7 --scenes "$SCENES" --out "$OUT" --arms 256`.
It fails the scene unless the latest ledger entry of every arm is complete and the layer still
checksums to what that entry recorded, every arm reports `"truncation": {}`, and every arm ran on the
prepared raster (sha256 match). It then writes `SHA256SUMS` and `handoff.json` for those layers. Any
other layer in the directory (from a trial of another configuration, say) is listed in `handoff.json`
under `not_delivered` and left out of `SHA256SUMS`.

**Getting the catalogs off the cluster (maintainer's choice, 2026-10-15: transfer from `$SCRATCH`, no
`DEST`).** Pull each verified scene through the DTN as it completes, and re-check it on arrival. On the
receiving machine, with `OBS` the scene and `LOCAL` where the catalogs go:
```bash
R=$(ssh bamaro@dtn.sherlock.stanford.edu 'echo "$SCRATCH"') && test -n "$R"   # resolve it there
rsync -a bamaro@dtn.sherlock.stanford.edu:"$R/campaign/out/${OBS:?}/" "${LOCAL:?}/$OBS/"
(cd "$LOCAL/$OBS" && sha256sum -c SHA256SUMS)
```
`$SCRATCH` is resolved by a separate `ssh` because rsync 3.2.4 and later escapes `$` in remote paths,
so a literal `\$SCRATCH` in the rsync line would name a directory called `$SCRATCH`. If the `ssh` line
prints nothing (the variable may be unset in a non-interactive DTN shell; not yet checked), set `R` to
the absolute path that `echo $SCRATCH` shows on a login node.
`$SCRATCH` purges on inactivity, so do not leave a finished campaign there. `DEST` remains available
if a durable cluster directory is ever wanted: the scene is then rsynced there and `sha256sum -c`'d.

Progress, after the shell setup (it only reads):
```bash
python "$REPO/scripts/campaign.py" status --manifest "$M" --scenes "$SCENES" --out "$OUT"
```
`VERIFY-STALE` means the ledger moved on after the last `verify`, e.g. a re-run that failed or was
killed: the arm states on that line are current, and the scene's `handoff.json` is not.

## 4. Memory accounting

Every arm manifest records `peak_rss_bytes` (the arm runs in its own process), the ledger copies it,
and each detect task prints `sacct ... MaxRSS`, whatever the detection's exit. After the fact:
`sacct -j <jobid> --format=JobID,MaxRSS,Elapsed,State --units=G`.
The same works retroactively for the 2026-08-19 HiRISE jobs (pre-N2 RAM at 1.93 Gpx).

## Hand-off layout (D10, proposed)

```
<OUT or DEST>/<ObsId>/
    <ProductId>-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg   (one per arm)
    <same>-manifest.json      config, code identity (commit via CODE_STAMP), inputs, truncation, peak RSS
    run_ledger.jsonl          per-arm attempts, keys, artifact digests
    handoff.json              the verify verdict + per-layer summary + the input raster's provenance
                              (+ any layer in the directory it does not deliver, and why)
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
- ~~When `max_det` 1100 binds~~ **decided 2026-10-15: escalate** (see "What is fixed" and "When an
  arm escalates"). A non-binding run does not depend on the cap (279 tiles verified inert), so
  retrying until nothing binds is well defined, and it was checked end to end.
- **Durable storage** for `DEST`.
- **Mixed-binning scenes.** Many "0.25 m/px" RED products bin some CCDs 2x (e.g. ESP_028537_2270:
  `MRO:BINNING = (2,2,2,1,1,1,1,2,2,...)`), so strips of the scene are 0.5 m/px resampled onto the
  0.25 grid and have a coarser completeness floor. The prep sidecar records the binning per scene.
