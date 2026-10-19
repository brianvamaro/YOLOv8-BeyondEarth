# The hand-off: what a delivered scene holds, and how to read it (D10, as built)

For hirise2ctx's Stage 1 (the read-only acceptance check on the pilot, then the P4 port) and for
BoulderNet. It describes what `scripts/campaign.py verify` writes. **D10 itself is Brian's**; this is
the hand-off as implemented, which hirise2ctx proposes to accept with the reading rules in §8. None of
it has been produced on the cluster yet: the pilot (`../../docs/sherlock_pilot_plan.md`) is its first
real test. The runbook is `CAMPAIGN.md`.

## 1. Where it lands

On the cluster, per scene:
- `$OUT/<ObsId>/` is the hand-off: the layers, their manifests, the ledger, `handoff.json` and
  `SHA256SUMS`.
- `$SCENES/<ObsId>/` is the prep's working directory. It holds `<ProductId>.LBL` and `<ProductId>.JP2`
  as PDS names them, the prepared `<ProductId>.tif`, its `<ProductId>.prep.json` sidecar and
  `fetched.json`. **The prep keeps the label beside the raster**, and nothing deletes either.

On the laptop, one stable root per code release (`TAG`), which hirise2ctx-pub can register as a
read-only source:

```
C:\Users\brian\Documents\PhD\BoulderNet\data\deliveries\hirise2ctx_v3\<TAG>\   (TAG = pilot-1 for the pilot)
    CODE_STAMP.json            the release's stamp (§6), copied from the release tree
    out\<ObsId>\               exactly $OUT/<ObsId>/ (§2-§5), checked with sha256sum -c on arrival
    scenes\<ObsId>\            from $SCENES/<ObsId>/: <ProductId>.LBL, <ProductId>.prep.json,
                               fetched.json (absent when the JP2 was staged rather than fetched),
                               and <ProductId>.tif, the prepared raster
```

- **Nothing else writes there.** The top-level BoulderNet repo ignores `data/`. BoulderNet's own
  scripts write under `YOLOv8-BeyondEarth/data/`, not here.
- **One root per `TAG`** keeps the pilot apart from the campaign. Resume holds only on one release
  (`CAMPAIGN.md` §3).
  - **The exception: the v3 campaign runs on `pilot-1` itself** (§9). So its root is
    `...\hirise2ctx_v3\campaign-v3\`, with `pilot-1`'s `CODE_STAMP.json`.
  - The `pilot-1` root keeps the pilot as hirise2ctx accepted it. S1–S3's campaign hand-offs (both layers)
    land only in `campaign-v3`.
- **The JP2 is not pulled by default**, about 0.6 bytes per pixel (~19 GB for the pilot). On the laptop
  the label sits beside the prepared raster, which is already in the label frame. Ask if R11's overlays
  need the native JP2 instead.

**What the pilot actually used (2026-10-18), and it worked.**
1. **Two archives on `$SCRATCH`,** made on a login node:
   ```bash
   cd $SCRATCH/campaign
   tar cf $SCRATCH/pilot-1_catalogs.tar out scenes/*/*_RED.LBL scenes/*/*_RED.prep.json scenes/*/fetched.json
   tar cf $SCRATCH/pilot-1_rasters.tar scenes/*/*_RED.tif
   ```
   They came to 7.4 GB and 14.4 GB.
2. **Downloaded through Sherlock OnDemand's file manager,** then unpacked into the delivery root with
   `tar xf <archive> -C <root>`.
3. **Re-verified on the laptop:** every scene's `SHA256SUMS`, plus each prepared raster against
   `handoff.json`'s `raster.sha256` (`scripts/review/sherlock_pilot_analysis.py --sections run`).

**The direct pull below has not been used.**
- On 2026-10-17 an `scp` to `dtn.sherlock.stanford.edu:~/` reported success, but the file never
  appeared in the login node's home.
- So whether the DTN shares that home, and whether it gives a shell, is unchecked.

The recipe streams `tar` over one `ssh` session in Git Bash, since the laptop has no `rsync`. Pull
after `campaign.py status` reports the scenes `verified`:

```bash
TAG=pilot-1
LOCAL=/c/Users/brian/Documents/PhD/BoulderNet/data/deliveries/hirise2ctx_v3/$TAG   # the v3 campaign: .../campaign-v3 (§9)
mkdir -p "$LOCAL"
R=$(ssh bamaro@dtn.sherlock.stanford.edu 'echo "$SCRATCH"') && test -n "$R"   # resolved there (CAMPAIGN.md §3)
OBS_LIST="PSP_009116_2240 ESP_068483_2280 ESP_045550_2180"                      # the scenes to pull
ssh bamaro@dtn.sherlock.stanford.edu "tar cf - --ignore-failed-read -C '$R/campaign' \
    \$(for o in $OBS_LIST; do echo out/\$o scenes/\$o/\${o}_RED.LBL scenes/\$o/\${o}_RED.prep.json \
       scenes/\$o/fetched.json scenes/\$o/\${o}_RED.tif; done) \
    -C ~/BoulderNet/releases/$TAG/YOLOv8-BeyondEarth CODE_STAMP.json" | tar xf - -C "$LOCAL"
for o in $OBS_LIST; do (cd "$LOCAL/out/$o" && sha256sum -c --quiet SHA256SUMS && echo "$o ok"); done
```

- **`CODE_STAMP.json`** comes from the deployed release, so it cannot be a later pack's.
- **`--ignore-failed-read`** lets a scene without `fetched.json` through. `tar` still names the missing
  file, and that file is the only one allowed to be missing.
- **Two `ssh` sessions,** so expect two Duo prompts.

## 2. `handoff.json`

`verify` writes it whether the scene passes or fails.

| field | type | meaning |
|---|---|---|
| `obs_id`, `product_id` | string | e.g. `ESP_068483_2280`, `ESP_068483_2280_RED` |
| `verified_utc` | string | when `verify` wrote this file: ISO 8601 with offset, e.g. `2026-10-20T14:03:11.512345+00:00` |
| `ok` | bool | `true` exactly when `problems` is empty |
| `problems` | list of string | each refusal, in words; empty when `ok` |
| `stages` | list of string | the stages this hand-off is for, sorted. Configuration 1 is `["arm:256"]`; the v3 campaign delivers `["arm:256", "arm:512"]` (§9) |
| `layers` | list of object | one per delivered layer (below) |
| `not_delivered` | list of `{layer, stage, why}` | any other layer in the directory: a trial of another configuration, a stale attempt. `stage` is `null` when no ledger entry wrote it. Never read these |
| `raster` | object or `null` | the prepared input (below). `null` only when the prep sidecar was missing, and then `ok` is `false` |

Each entry of `layers`:

| field | meaning |
|---|---|
| `layer` | the GeoPackage's file name: `<ProductId>-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg`, and `-ss-512-` for the 512 arm |
| `stage` | `arm:256` or `arm:512` |
| `manifest` | `<layer stem>-manifest.json`, beside it (§4) |
| `config_hash` | the manifest's configuration digest |
| `truncation` | the manifest's record; `{}` when nothing was lost (verify refuses a lossy one) |
| `n_features` | features in the layer (the manifest's `outputs.n_after_nms`) |
| `max_det`, `batch_size` | the cap and batch of the attempt that wrote the layer: 1100 and 8 unless it escalated (2200/4, 4400/2, ...) |
| `attempts` | oldest first, every finished attempt of this layer's configuration: `{state, max_det, batch_size, stopped, worst_slice, wall_seconds, finished_utc}`. `stopped` is `"max_det"` or `"cuda_oom"` on an attempt that escalated, `null` on the one that finished |
| `peak_rss_bytes` | the arm process's peak resident memory |
| `n_slices`, `n_slices_skipped_fill` | slices in the scene; slices skipped as all-fill (lever N1) |
| `code` | `{git_sha, git_dirty, source_sha256, scripts_sha256, host, torch, ultralytics}` from the manifest (§6) |
| `sha256` | of the layer file at verify time; absent if the layer was not on disk (then `ok` is `false`) |

`raster`:

| field | value |
|---|---|
| `path` | the prepared raster's file name, `<ProductId>.tif` |
| `sha256` | of the prepared raster; every layer's manifest `inputs.raster.sha256` equals it, or verify refuses |
| `rule` | `min((DN + 2) >> 2, 255)` |
| `frame` | **always** `PDS label via GDAL's PDS driver (not the JP2's GeoJP2 box)`. The prep refuses to write a raster in any other frame (§7), so no other value can occur. Compare the whole string |
| `jp2_sha256` | of the JP2 it was converted from |

## 3. `SHA256SUMS`

`sha256sum -c` format, one line per delivered layer and one per layer manifest.
- **Not covered:** `handoff.json` itself, `run_ledger.jsonl` and the `scenes/` files. Record
  `handoff.json`'s sha256 on arrival, as Stage 1 plans to.
- **What it certifies:** transfer integrity, not reproducibility. GeoPackage bytes are not
  deterministic (the format stamps its own metadata), so re-running a scene gives new checksums for the
  same geometry. Reproducibility is the manifest's job.

## 4. The layer manifest (`<layer stem>-manifest.json`)

The keys a consumer checks:

| key | value |
|---|---|
| `config.confidence_threshold` | `0.1`: the delivered superset; the score floor (D36) is applied downstream |
| `config.max_det`, `config.batch_size` | as in `handoff.json` |
| `config.inference_contract_version` | `3`, the current inference code (contract 1 was code before 2026-07-30) |
| `config.slice_size`, `config.inference_size`, `config.overlap_height_ratio` | 256 (512 for the 512 arm), 1024, 0.2 |
| `config.contour_method`, `config.remask_fn`, `config.sr_factor` | `skimage`, `null` (no SAM2), `1` |
| `truncation` | `{}` |
| `outputs.n_after_nms` | the layer's feature count |
| `inputs.raster` | `{path, bytes, mtime, sha256}` of the prepared raster; `path` is a cluster path, so match on `sha256` |
| `inputs.weights` | the same for the weights |
| `crs` | the prepared raster's CRS, the label's: `lat_ts` = the label's `CENTER_LATITUDE`, `lat_0` = 0, on the label's own sphere radius (e.g. 3,386,150.747 m near 45° N), as in v2's exports |
| `code` | §6 |
| `n_slices`, `n_slices_skipped_fill`, `peak_rss_bytes`, `wall_seconds`, `written_utc` | run facts |

Each feature is a polygon in the label frame. Its columns are `score`, `nms_prio`, `cat_id`,
`cat_name`, `isin_slice`, `mask_src`, `is_at_edge` and `id`.
- **`score`** is the detector's own confidence, never overwritten since contract 2. Contract 1 wrote a
  literal 0.10 over slice-truncated rows: the `isin_slice = 0` rows in v2's exports.
- **`nms_prio`** is the suppression ordering key.
- **`mask_src`** is `yolo` under configuration 1.

## 5. `run_ledger.jsonl`

One JSON object per line, appended and fsynced, so a kill can tear only the last line. Each attempt
writes two lines.
- **On start, a marker:** `{stage, key, state: "running", started_utc}`.
- **On finish, the outcome:** `{stage, key, state, finished_utc, wall_seconds, base_key, max_det,
  batch_size, on_max_det, command, n, truncation, slices, peak_rss_bytes, ...}`.
  - `state` is `complete`, `failed` or `incomplete`.
  - On a stop, the outcome also has `stopped`, `bound` and `next_attempt`.
  - On a failure, it has `error`.
  - A completed attempt has `artifacts`, `{path: sha256}`.

Paths in the ledger are cluster paths, so match them by file name. A resubmit that finds an arm
complete appends nothing.

## 6. `CODE_STAMP.json` and the `code` block

The cluster runs an extracted tarball, not a checkout. `scripts/pack_for_cluster.py` refuses a dirty
tree and writes `CODE_STAMP.json` at the release root:

| field | meaning |
|---|---|
| `git_sha` | the packed commit (40 hex) |
| `git_describe` | `<short sha> <subject>` |
| `git_dirty`, `dirty_paths` | `false`, `[]` (a dirty pack needs `--allow-dirty` and says so here) |
| `source_sha256` | digest of the package's `.py` sources (`presets.source_digest`, line endings normalised) |
| `scripts_sha256` | the same over `scripts/` |
| `packed_utc` | when it was packed |

Every layer manifest's `code` block records the run's own `source_sha256`. Where git is unusable
(a tarball), it also records:
- `stamp`: the whole `CODE_STAMP.json`;
- `stamp_matches_source`: whether the two digests agree;
- `git_sha` and `git_sha_from: "CODE_STAMP.json"`, only when they agree.

**Trust the commit only when `stamp_matches_source` is `true`.** A stamp beside edited files names a
commit those files are not.

## 7. Frames and sign conventions

**The label frame**, which is the delivered geometry's frame:
- upper-left corner `(-(SAMPLE_PROJECTION_OFFSET + 0.5) * s, (LINE_PROJECTION_OFFSET + 0.5) * s)`,
  with `s` = `MAP_SCALE`;
- CRS `lat_ts = CENTER_LATITUDE`, `lat_0 = 0`.

It is what GDAL's PDS driver gives when it opens the `.LBL`, with the driver's default config options.
`data/review/pds_driver_frames.json` shows it on 12 products (2 PSP v2.0, 10 ESP v1.0, orbits
7,492-76,235, 0.25 and 0.5 m/px): offset 0.000 px, `lat_ts` = `CENTER_LATITUDE`, `lat_0` = 0 on
GDAL 3.10. Setting `PDS_SampleProjOffset_Shift` and `PDS_LineProjOffset_Shift` to -0.5 moves the
origin by a full pixel, so **leave the four `PDS_*ProjOffset_*` options unset**.

**The JP2 box** (GDAL's JP2 driver on the `.JP2`) is another frame:

| products | JP2 origin minus label origin | the box's CRS |
|---|---|---|
| PSP, and ESP through orbit 68,483 at least (S2 included) | (+1.5, +1.5) px | transposed: `lat_ts = 0`, `lat_0 = CENTER_LATITUDE` (X-04) |
| ESP from orbit 69,088 at least | (+1.0, +1.0) px | correct |

The change falls between orbits 68,483 and 69,088, which no checked product separates further.

**Three conventions, and the sign of each.** `s` is the map scale, x is easting and y is northing.

| name | where | definition | +x | +y |
|---|---|---|---|---|
| `tif_minus_label_origin_px` | `data/review/incidence_pairs_label.json`; the same quantity is `minus_label_px` in `pds_driver_frames.json` | `((tif_ulx - label_ulx) / s, (label_uly - tif_uly) / s)` | east (columns) | **south (rows)** |
| `frame_check.dx_px`, `dy_px` | each prep sidecar (`<ProductId>.prep.json`) | `((raster_ulx - label_ulx) / s, (raster_uly - label_uly) / s)` | east | **north** |
| the X-07 shift | `--pred-shift-m`, BoulderNet's re-scoring | metres added to a JP2-frame catalog to bring it into the label frame | east | north |

- **The y signs are opposite.** `tif_minus_label_origin_px` counts rows, so +y is south.
  `frame_check.dy_px` is in map units, so +y is north.
- **Delivered rasters carry zero in both.** The prep refuses any raster whose `frame_check` exceeds
  1e-6 px on either axis.
- **A worked case.** A raster in the JP2 frame has `tif_minus_label_origin_px` (+1.5, +1.5), which is
  `frame_check` (+1.5, -1.5). At 0.5 m/px its catalog sits 0.75 m east and 0.75 m south of the label
  frame. To bring a point into the label frame, subtract `d_x * s` from x and add `d_y * s` to y,
  where `(d_x, d_y)` is `tif_minus_label_origin_px`. That is the X-07 shift `(-0.75, +0.75)` m.

## 8. Reading rules for a consumer

These are Stage 1's six refusals as hirise2ctx proposed them, with what each one reads.
1. **`handoff.json` says `ok: true` and `problems` is empty.**
2. **No ledger entry is newer than `verified_utc`.** A re-run that fails or is killed exits before
   `verify`, so an older `ok: true` survives it.
   - **Apply this test yourself.** In `campaign_detect.sbatch` the detection step's failure ends the
     task, and `campaign.py detect` calls `verify` only after the runner exits 0.
   - **The test** is the one `campaign.py status` applies (`VERIFY-STALE`): refuse if any line of
     `run_ledger.jsonl` has a `finished_utc` or a `started_utc` later than `verified_utc`. All of them
     are timezone-aware UTC, so compare them as datetimes.
3. **Every listed layer and manifest passes `sha256sum -c SHA256SUMS`.**
4. **The delivered arm set is the one D9 rules.** `stages` is `["arm:256"]` for configuration 1. The v3
   campaign delivers both arms, `["arm:256", "arm:512"]`, because it runs before D9 is ruled (§9). Read the
   `arm:256` layer for configuration 1; configuration 2's label is not in the hand-off.
5. **Each layer's manifest** has `truncation` `{}`, `config.confidence_threshold` 0.1, `config.max_det`
   equal to `handoff.json`'s, and `config.inference_contract_version` 3.
6. **`raster.frame` is the label-frame string** (§2). The layer's CRS also has `lat_ts =
   CENTER_LATITUDE`, so v2's SP1 relabel finds nothing; in v3, its firing is an error.

And always:
- **Read only the layers `handoff.json` lists.** Never glob the directory: `not_delivered` names the
  rest.
- **To check a frame directly**, compare `scenes/<ObsId>/<ProductId>.prep.json`'s `output.transform`
  with what your own PDS-driver open of the `.LBL` gives, as `pds_driver_frames.py` does.

## 9. The v3 campaign: two arms, unfused (2026-10-18)

Brian chose to start the campaign before D9 is ruled, with both of configuration 2's arms (`ARMS="256 512"`).
- **What a scene's hand-off holds:** two layers, `arm:256` and `arm:512`, each with its manifest and its own
  checksums. `stages` is `["arm:256", "arm:512"]`.
- **Configuration 1's label is the `arm:256` layer alone,** exactly as in `pilot-1`.
- **Configuration 2's label is the two layers fused under its production rule** (`256pref_engulf`, the rule
  the configuration study scored). The runner does not fuse a two-arm set: it refuses any set but production's
  three, and a `{512, 256}` preset is not built. So that label is not in the hand-off. If D9 rules
  configuration 2, the fusion is a later step on the delivered layers, with its own record.
- **Never fuse the two layers yourself, or count them both.** The 512 arm detects many of the same boulders,
  so their union double-counts.
- **For Stage 1:**
  - **Rule 4's arm set** is `["arm:256", "arm:512"]` on campaign scenes, or `["arm:256"]` alone while a scene's
    512 arm waits for a larger card (below).
  - **Rule 2** applies to every stage the hand-off lists.
  - **Rule 5 applies per layer,** against that layer's own `handoff.json` entry. The 512 manifest differs from
    the 256 one in `config.slice_size` (512), and where that arm escalated, also in `config.max_det` and
    `config.batch_size`.
  - **hirise2ctx's check as built** (one `Release` with `stages ("arm:256",)` and configuration 1's settings)
    refuses these hand-offs. A campaign release needs both stages and a configuration per stage.
- **S1–S3** (delivered in `pilot-1` with `arm:256` only) get their 512 arm in the campaign. Their 256 layer
  is the same file, skipped by the ledger, and the campaign's hand-off lists both layers.
- **If a 512 arm cannot finish** (for example a dense scene at batch 1 on a 24 GB card), that scene's
  two-arm verify fails while its 256 layer stays complete. `verify --arms 256` then delivers configuration 1
  alone, and the 512 arm is resubmitted on a larger card (`CAMPAIGN.md`, "When an arm escalates"). A later
  two-arm verify replaces that hand-off.
- **On the laptop** the campaign lands in its own root, `hirise2ctx_v3\campaign-v3\` (§1), not in `pilot-1`'s.
