"""Re-scan the multi-scene and control-site sets over a FIXED AREA of randomly-ordered tiles.

Why this replaces the earlier sampling. The first pass over these scenes walked 512-px tiles in
**raster order** from a central strip and stopped once it had ~800 *elongated* detections. That has
two problems for a figure whose whole content is per-scene orientation:

* **Raster order takes a contiguous prefix** — the top of a 43 km-long strip. If fabric varies
  down-track (different geological units), the sampled fabric is not the scene's fabric. This was the
  dominant bias, and it is independent of the stopping rule.
* **A detection quota makes `n` incomparable between scenes.** Effective area scanned varied 12x
  (5.0-64.5 Mpx), so "compare scene A's rose to scene B's" was never like-for-like, and the low
  boulder counts on some scenes were indistinguishable from "we happened to scan less ground there".

This pass fixes both:

* **Tile order is shuffled over the scene's whole valid footprint** with a fixed seed. Taking the
  first k tiles of a random permutation is a uniform random sample of the scene at any k, so the
  spatial bias is gone.
* **Every scene gets the same area of valid ground** (`TARGET_KM2`, in km² so it is
  resolution-independent), with no detection quota. `n` is then a *measurement* — boulders per km² —
  which is exactly the quantity the illumination-dependent recall collapse is about, instead of being
  confounded with how many tiles were scanned.

**Known, deliberate caveat.** The downstream figures cut at >= 8 px for angle reliability, and 8 px is
**2 m** on the 0.25 m/px scenes but **4 m** on the 0.5 m/px ones — so the cut selects different
physical populations across scenes. Kept as-is for statistical power (a common 4 m floor costs 10-17x
on the fine-resolution scenes); `dpx_sam * res` is written to the CSV so a metre-based cut can be
applied at any time, and the figure caption states the confound. Resolution is recorded per scene.

Only valid tiles count toward the area target (these products have nodata margins), so "10 km²" means
10 km² of real ground, not of bounding box. Validity is decided by :func:`is_valid`, which tests for
*contiguous* zero regions rather than counting zero pixels — read its docstring before changing it,
because the obvious "fraction of nonzero pixels" test silently throws away deep-shadow ground.

Per-scene CSVs are written incrementally, so a crash or a stop mid-run loses at most one scene, and
re-running skips scenes already complete. GPU. __main__ guard (Windows spawn).

  # cost estimate only, no GPU work:
  ...\\conda.exe run -n bouldernet python data/summary/run_fixed_area_scenes.py --dry-run
  # the real thing (long; run in the background):
  ...\\conda.exe run -n bouldernet --no-capture-output python data/summary/run_fixed_area_scenes.py
  # subsets:
  ... --group=controls --target-km2=40 --scenes=ESP_030222_1220,TRA_000828_2495
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet")
ROOT = BASE / "YOLOv8-BeyondEarth"
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data" / "summary" / "fixed_area"

TARGET_KM2 = 40.0     # valid ground per scene
TILE = 512            # one whole-tile slice, as in the validated methodology
BORDER = 8            # drop detections whose centroid is within BORDER px of a tile edge
MAX_DEAD_BLOCKS = 0.02   # max fraction of 32x32 blocks that may be entirely zero (see `is_valid`)
DEAD_BLOCK = 32
MAX_TILES = 20000     # runaway guard; logged loudly if ever hit
SEED = 20260727

# Detections ultralytics may return per tile (review item X-06). THE 2026-07-27 RUN THAT PRODUCED
# `fixed_area_scenes.csv` DID NOT PASS THIS, so ultralytics' silent 300 applied to all 5,382 of its
# tiles: the per-tile count walls at 295 with a mode of 111 tiles at 278, and the control site
# ESP_030222_1220 averages 275.6 det/tile with 57.4% of its detections inside capped tiles. The cap
# keeps the top N *by confidence*, so the >=8 px / >=4 m tail the orientation figures analyse (5.7%
# of rows) is the least exposed part of it — but every `per_km2_*` number in the existing meta is a
# LOWER BOUND. 600 matches presets.GOAL1_PRODUCTION; the whole-tile TILE=512 slice here is the arm
# X-06 measured returning up to 1013/slice on the densest HiRISE terrain, so check
# `tiles_at_max_det` in the meta rather than assuming 600 clears it.
MAX_DET = 600

WEIGHTS = ROOT / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"
SAM_WEIGHTS = ROOT / "sam2.1_b.pt"

# group -> [(obsid, path relative to YOLOv8-BeyondEarth/)]
GROUPS = {
    "hirise": [
        ("ESP_017355_2260", "data/mars_hirise/raster/ESP_017355_2260_RED.tif"),
        ("ESP_076235_2260", "data/test2_pairs/ESP_076235_2260/ESP_076235_2260_RED.tif"),
        ("ESP_067214_2270", "data/test2_pairs/ESP_067214_2270/ESP_067214_2270_RED.tif"),
        ("ESP_059499_2270", "data/test2_pairs/ESP_059499_2270/ESP_059499_2270_RED.tif"),
        ("ESP_055833_2260", "data/test2_pairs/ESP_055833_2260/ESP_055833_2260_RED.tif"),
        ("ESP_055714_2270", "data/test2_pairs/ESP_055714_2270/ESP_055714_2270_RED.tif"),
        ("PSP_007492_2265", "data/test2_pairs/PSP_007492_2265/PSP_007492_2265_RED.tif"),
        ("PSP_007703_2265", "data/test2_pairs/PSP_007703_2265/PSP_007703_2265_RED.tif"),
        ("ESP_069088_2270", "data/test2_pairs/ESP_069088_2270/ESP_069088_2270_RED.tif"),
        ("ESP_071923_2260", "data/test2_pairs/ESP_071923_2260/ESP_071923_2260_RED.tif"),
        ("ESP_071699_2260", "data/test2_pairs/ESP_071699_2260/ESP_071699_2260_RED.tif"),
    ],
    "controls": [
        ("ESP_030222_1220", "data/test1_controls/ESP_030222_1220/ESP_030222_1220_RED.tif"),
        ("ESP_076499_1160", "data/test1_controls/ESP_076499_1160/ESP_076499_1160_RED.tif"),
        ("ESP_021862_1175", "data/test1_controls/ESP_021862_1175/ESP_021862_1175_RED.tif"),
        ("ESP_013249_1270", "data/test1_controls/ESP_013249_1270/ESP_013249_1270_RED.tif"),
        ("TRA_000828_2495", "data/test1_controls/n_offcoast/TRA_000828_2495_RED.tif"),
    ],
}


def resolve(rel, obsid):
    p = ROOT / rel
    if p.exists():
        return p
    for cand in ROOT.rglob(f"{obsid}_RED.tif"):     # tolerate a moved/renamed directory
        return cand
    return None


def sun_of(tif_path):
    """Geographic sun axis, incidence and north azimuth from the product's .LBL."""
    lbl = tif_path.with_suffix(".LBL")
    if not lbl.exists():
        return np.nan, np.nan, np.nan
    t = lbl.read_text(errors="replace")

    def g(k):
        m = re.search(rf"{k}\s*=\s*([-\d.]+)", t)
        return float(m.group(1)) if m else np.nan
    ss, na, inc = g("SUB_SOLAR_AZIMUTH"), g("NORTH_AZIMUTH"), g("INCIDENCE_ANGLE")
    return (ss - na) % 180.0, inc, na


def is_valid(tile):
    """Does this tile contain real ground, rather than a product nodata margin?

    **Do not test "fraction of nonzero pixels".** These products declare no nodata value and encode
    it as 0, but *deep shadow is also 0* — and on a high-incidence scene shadow can be a fifth of the
    image. An earlier version of this script required 99.9% nonzero and consequently accepted only
    **2 of 131** tiles on ESP_030222_1220 (23% zeros, median DN 114 — perfectly good imagery), while
    also selecting mildly against shadowed ground, which is the last thing an illumination study
    should do.

    Nodata margins are *large contiguous* zero regions; shadow is scattered and small. So the test is
    on structure, not on count: tile down into ``DEAD_BLOCK``-px blocks and reject if too many blocks
    are **entirely** zero. Measured against the old test on three scenes, this accepts the same tiles
    wherever the old one worked (ESP_017355 76 vs 72 of 83, mean DN 153.7 vs 154.4; TRA_000828 61 vs
    48 of 69, DN 172.0 vs 174.0) and fixes the scene where it did not (109 vs 2 of 131).
    """
    if tile.max() == tile.min():
        return False
    h = (tile.shape[0] // DEAD_BLOCK) * DEAD_BLOCK
    w = (tile.shape[1] // DEAD_BLOCK) * DEAD_BLOCK
    if not h or not w:
        return False
    b = tile[:h, :w].reshape(h // DEAD_BLOCK, DEAD_BLOCK, w // DEAD_BLOCK, DEAD_BLOCK)
    return float((b.max(axis=(1, 3)) == 0).mean()) <= MAX_DEAD_BLOCKS


def tile_order(W, H, seed):
    """All tile origins covering the raster, in a reproducible random order."""
    xs = np.arange(0, max(W - TILE, 1), TILE)
    ys = np.arange(0, max(H - TILE, 1), TILE)
    grid = np.array([(int(x), int(y)) for y in ys for x in xs])
    rng = np.random.default_rng(seed)
    return grid[rng.permutation(len(grid))]


def scan_scene(obsid, tif_path, model, sam, target_km2):
    import rasterio
    from rasterio.windows import Window
    from YOLOv8BeyondEarth.orientation_validation import (angles_from_moments,
                                                          detect_sam2_mask_sets)
    sun, inc, north = sun_of(tif_path)
    rows, t0 = [], time.time()
    # `n_failed` was used by the per-tile crash guard below but never initialised, so the guard
    # raised UnboundLocalError *from inside its own handler* and a single bad tile still lost the
    # whole scene — the protection its comment promises never worked.
    n_seen = n_valid = realigned = n_failed = 0
    trunc = {"tiles_at_max_det": 0, "tiles_near_max_det": 0, "max_det_returned": 0}
    with rasterio.open(tif_path) as src:
        res = float(src.res[0])
        km2_per_tile = (TILE * res) ** 2 / 1e6
        need = int(np.ceil(target_km2 / km2_per_tile))
        order = tile_order(src.width, src.height, SEED)
        print(f"[{obsid}] res {res:.2f} m/px | {src.width}x{src.height} | "
              f"{len(order):,} tiles available | need {need:,} valid tiles for "
              f"{target_km2:g} km²", flush=True)
        for x0, y0 in order:
            if n_valid >= need or n_seen >= MAX_TILES:
                break
            n_seen += 1
            tile = src.read(1, window=Window(int(x0), int(y0), TILE, TILE))
            if tile.shape != (TILE, TILE) or not is_valid(tile):
                continue
            n_valid += 1
            # One bad tile must not lose a whole scene on an unattended multi-hour run. A very dense
            # tile can exhaust GPU memory in the SAM re-mask; drop it, free the cache, keep going, and
            # count it so the loss is visible rather than silent.
            try:
                df, masks = detect_sam2_mask_sets(model, sam, tile, slice_size=TILE, imgsz=1024,
                                                  conf=0.10, device=0, configs=("yolo", "sam256"),
                                                  sam_batch=64, empty_cache_every=64,
                                                  max_det=MAX_DET)
            except Exception as exc:                                  # noqa: BLE001
                n_failed += 1
                print(f"  [{obsid}] tile ({x0},{y0}) FAILED ({type(exc).__name__}: "
                      f"{str(exc)[:120]}) — skipped, {n_failed} so far", flush=True)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:                                     # noqa: BLE001,S110
                    pass
                continue
            realigned += int(df.attrs.get("realigned_chunks", 0))
            tt = df.attrs.get("truncation") or {}
            trunc["tiles_at_max_det"] += int(tt.get("tiles_at_max_det", 0))
            trunc["tiles_near_max_det"] += int(tt.get("tiles_near_max_det", 0))
            trunc["max_det_returned"] = max(trunc["max_det_returned"],
                                            int(tt.get("max_det_returned", 0)))
            if len(df) == 0:
                continue
            ay, asm = angles_from_moments(masks["yolo"]), angles_from_moments(masks["sam256"])
            keep = ((df.cx >= BORDER) & (df.cx < TILE - BORDER)
                    & (df.cy >= BORDER) & (df.cy < TILE - BORDER)).to_numpy()
            for di in np.where(keep)[0]:
                rows.append(dict(
                    scene=obsid, res=res, tile_x=int(x0), tile_y=int(y0),
                    angle_yolo=float(ay.angle180.iloc[di]), aspect_yolo=float(ay.aspect.iloc[di]),
                    dpx_yolo=float(ay.diameter_px.iloc[di]),
                    angle_sam=float(asm.angle180.iloc[di]), aspect_sam=float(asm.aspect.iloc[di]),
                    dpx_sam=float(asm.diameter_px.iloc[di]),
                    dm_sam=float(asm.diameter_px.iloc[di]) * res,   # metres, for a physical cut
                    shadow_axis=sun, incidence=inc))
            if n_valid % 100 == 0:
                el = sum(1 for r in rows if r["aspect_sam"] > 1.35 and r["dpx_sam"] >= 8)
                print(f"  [{obsid}] {n_valid}/{need} valid tiles "
                      f"({n_valid * km2_per_tile:.1f} km²)  dets={len(rows):,}  "
                      f"elong≥8px={el:,}  {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    km2 = n_valid * km2_per_tile
    e = df[df.aspect_sam > 1.35] if len(df) else df
    e8 = e[e.dpx_sam >= 8] if len(e) else e
    meta = dict(scene=obsid, res_m=res, shadow_axis=None if not np.isfinite(sun) else round(sun, 1),
                incidence=None if not np.isfinite(inc) else round(inc, 3),
                north_azimuth=None if not np.isfinite(north) else north,
                target_km2=target_km2, km2_valid=round(km2, 2), tiles_valid=n_valid,
                tiles_examined=n_seen, valid_fraction=round(n_valid / max(n_seen, 1), 3),
                reached_target=bool(n_valid >= need), hit_tile_guard=bool(n_seen >= MAX_TILES),
                n_det=len(df), n_elong=len(e), n_elong_ge8px=len(e8),
                per_km2_det=round(len(df) / km2, 1) if km2 else None,
                per_km2_elong_ge8px=round(len(e8) / km2, 2) if km2 else None,
                median_dpx_sam=None if not len(df) else round(float(df.dpx_sam.median()), 2),
                realigned_chunks=realigned, tiles_failed=n_failed,
                # X-06 provenance. `max_det` absent from a meta means the 2026-07-27 capped run;
                # `tiles_at_max_det > 0` means this scene's per_km2_* figures are lower bounds.
                max_det=MAX_DET, tiles_at_max_det=trunc["tiles_at_max_det"],
                tiles_near_max_det=trunc["tiles_near_max_det"],
                max_det_returned=trunc["max_det_returned"],
                wall_s=round(time.time() - t0, 1),
                # Absolute start/end so a wall clock can always be audited against the machine's power
                # log afterwards. This run's own figures turned out to be inflated ~1.27x by Modern
                # Standby (the job keeps computing, at ~68% throughput) and nothing recorded here could
                # have detected it. Also run `powercfg /change standby-timeout-ac 0` before timing.
                started_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                ended_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                seed=SEED, tile_px=TILE, max_dead_blocks=MAX_DEAD_BLOCKS,
                dead_block_px=DEAD_BLOCK, order="shuffled")
    return df, meta


def dry_run(groups, target_km2):
    """Cost estimate from the previous pass's measured seconds-per-tile. No GPU."""
    import rasterio
    prev = {}
    for p in (ROOT / "data/sam2_validation/sam_chirality_meta.json",
              ROOT / "data/sam2_validation/t1_controls_meta.json"):
        if p.exists():
            for k, v in json.loads(p.read_text()).items():
                if v.get("tiles"):
                    prev[k] = v["wall_s"] / v["tiles"]
    total, rows = 0.0, []
    for grp in groups:
        for obsid, rel in GROUPS[grp]:
            tif_path = resolve(rel, obsid)
            if tif_path is None:
                rows.append((grp, obsid, None, None, None, "MISSING TIF"))
                continue
            with rasterio.open(tif_path) as src:
                res = float(src.res[0])
                avail = (max(src.width - TILE, 1) // TILE + 1) * (max(src.height - TILE, 1) // TILE + 1)
            need = int(np.ceil(target_km2 / ((TILE * res) ** 2 / 1e6)))
            spt = prev.get(obsid, 1.0)
            hrs = need * spt / 3600
            total += hrs
            rows.append((grp, obsid, res, need, avail, f"{hrs * 60:.0f} min @ {spt:.2f} s/tile"
                         + ("  ** NEED > AVAILABLE **" if need > avail else "")))
    print(f"{'group':9s} {'scene':17s} {'m/px':>5s} {'tiles needed':>13s} {'available':>10s}  est")
    for g, o, r, n, a, note in rows:
        print(f"{g:9s} {o:17s} {'' if r is None else f'{r:5.2f}'} "
              f"{'' if n is None else f'{n:13,}'} {'' if a is None else f'{a:10,}'}  {note}")
    print(f"\nTOTAL estimated {total:.1f} h for {target_km2:g} km² per scene "
          f"(previous pass's s/tile; dense scenes cost more per tile)")


def archive_scene_outputs(csv, mj):
    """Move an existing per-scene CSV+JSON aside before a re-scan overwrites them (the A-2 pattern).

    The five control-site outputs are the **only** record of the sample behind the published
    control-site null (R 0.196-0.443, f_lower 0.573-0.756, grid 0.255-0.344) and of every
    ``per_km2_*`` figure quoted from it. A re-scan writes to the same paths, so without this the
    re-run destroys the thing a reader would want to compare it against -- and the numbers it
    replaces are cited in CLAUDE.md and in the Goal-4 gate.

    Idempotent by timestamp, and never overwrites an existing archive.
    """
    import shutil
    moved = []
    for path in (csv, mj):
        if not path.exists():
            continue
        dst = path.parent / "_archive" / f"{path.stem}__{time.strftime('%Y%m%dT%H%M%S')}{path.suffix}"
        if dst.exists():
            raise FileExistsError(f"{dst} exists; refusing to overwrite an archive")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        moved.append(dst.name)
    if moved:
        print(f"[archive] copied {', '.join(moved)} to _archive/ before re-scanning (A-2)",
              flush=True)
    return moved


def main():
    args = sys.argv[1:]
    target = next((float(a.split("=")[1]) for a in args if a.startswith("--target-km2=")),
                  TARGET_KM2)
    grp_arg = next((a.split("=")[1] for a in args if a.startswith("--group=")), "hirise,controls")
    groups = [g.strip() for g in grp_arg.split(",")]
    only = next((set(a.split("=")[1].split(",")) for a in args if a.startswith("--scenes=")), None)
    # `--max-det` because the module constant may not clear this arm: TILE=512 whole-tile slices are
    # what X-06 measured returning up to 1013 on dense HiRISE terrain, so 600 is a hope, not a bound.
    # Whatever is used lands in the meta together with `max_det_returned`, so one run answers both
    # "is this measurement truncated?" and "would 600 have been enough?".
    global MAX_DET
    MAX_DET = next((int(a.split("=")[1]) for a in args if a.startswith("--max-det=")), MAX_DET)
    # A re-scan at the same target is skipped by default (the resume path). `--force` re-measures,
    # archiving whatever is there first.
    force = "--force" in args
    if "--dry-run" in args:
        dry_run(groups, target)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    from ultralytics import SAM, YOLO
    model, sam = YOLO(str(WEIGHTS)), SAM(str(SAM_WEIGHTS))
    t0 = time.time()
    for grp in groups:
        for obsid, rel in GROUPS[grp]:
            if only and obsid not in only:
                continue
            csv = OUT / f"{grp}__{obsid}.csv"
            mj = OUT / f"{grp}__{obsid}.json"
            if csv.exists() and mj.exists() \
                    and json.loads(mj.read_text()).get("target_km2") == target:
                if not force:
                    prior = json.loads(mj.read_text())
                    # Say when the cached result is the capped run, rather than reporting "done" for
                    # a measurement whose densities are lower bounds (X-06).
                    note = ("" if "max_det" in prior else
                            "  (no max_det recorded => the capped 2026-07-27 run, so its per_km2_* "
                            "are LOWER BOUNDS; pass --force to re-measure)")
                    print(f"[{obsid}] already done at {target:g} km² — skipping{note}", flush=True)
                    continue
                archive_scene_outputs(csv, mj)
            tif_path = resolve(rel, obsid)
            if tif_path is None:
                print(f"[{obsid}] MISSING TIF ({rel}) — skipped, recorded", flush=True)
                mj.write_text(json.dumps(dict(scene=obsid, error="missing tif", path=rel), indent=1))
                continue
            df, meta = scan_scene(obsid, tif_path, model, sam, target)
            meta["group"] = grp
            df.to_csv(csv, index=False)
            mj.write_text(json.dumps(meta, indent=1))
            print(f"[{obsid}] DONE {meta['km2_valid']:.1f} km² "
                  f"({meta['tiles_valid']:,}/{meta['tiles_examined']:,} tiles valid) | "
                  f"dets {meta['n_det']:,} | elong≥8px {meta['n_elong_ge8px']:,} "
                  f"({meta['per_km2_elong_ge8px']}/km²) | {meta['wall_s'] / 60:.1f} min "
                  f"| elapsed {(time.time() - t0) / 3600:.2f} h", flush=True)
            if not meta["reached_target"]:
                print(f"  WARNING {obsid} did not reach {target:g} km² of valid ground "
                      f"(hit_tile_guard={meta['hit_tile_guard']}, "
                      f"valid_fraction={meta['valid_fraction']}) — its area is "
                      f"{meta['km2_valid']:.1f} km² and must be reported as such", flush=True)

    # combine
    frames, metas = [], []
    for p in sorted(OUT.glob("*.csv")):
        frames.append(pd.read_csv(p))
    for p in sorted(OUT.glob("*.json")):
        metas.append(json.loads(p.read_text()))
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        all_df.to_csv(ROOT / "data" / "summary" / "fixed_area_scenes.csv", index=False)
        (ROOT / "data" / "summary" / "fixed_area_scenes_meta.json").write_text(
            json.dumps(dict(target_km2=target, tile_px=TILE, seed=SEED, order="shuffled",
                            size_cut_caveat="figures cut at >=8 px for angle reliability; 8 px is "
                                            "2 m at 0.25 m/px and 4 m at 0.5 m/px, so the physical "
                                            "population differs across scenes. dm_sam holds the "
                                            "metric diameter for a common physical cut.",
                            scenes=metas), indent=1))
        print(f"\ncombined -> data/summary/fixed_area_scenes.csv ({len(all_df):,} rows, "
              f"{len(metas)} scenes) in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
