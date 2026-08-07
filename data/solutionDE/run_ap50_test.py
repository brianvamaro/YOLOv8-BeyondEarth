# Goal-3 phase 4: test-set AP50, contextual not gating (plan §Validation 9).
# Runs the fused ladder {512, 256, SR x2} + the 256-only production baseline on the 280
# boulder2024 test tiles and scores both with the SAME polygon-catalog evaluator
# (YOLOv8BeyondEarth.ap_eval) — the arm-to-arm comparison is the number that matters; the
# absolute value sits NEXT TO the published anchors (Amaro 2026 test AP50 0.58, Prieur 2023
# 0.52) but is not computed by the papers' evaluator (SAHI-sliced inference + polygon AP here
# vs whole-tile model.val there). Expectation (pre-registered): fused >= baseline on the
# labeled size range; a drop is a red flag, a flat number is not a failure.
#
# Tiles are 500x500 non-georeferenced PNGs; the pipeline wants GeoTIFFs and exact slice
# geometry, so each tile is materialized once as a 512x512 zero-padded uint8 GeoTIFF
# (identity-scale transform, 1 m/px dummy CRS -> native px == map units; labels get the same
# transform). Padding keeps every arm's slices exactly {512, 256, 128-native} — a 500-px
# "512 slice" would otherwise come back from ultralytics in a 512-frame with a 2.4% size error.
# Resume is per tile and **identity-checked**: each tile's marker records what produced it, and a
# marker from a different configuration is refused rather than silently reused. Catalogs + AP -> ap50/.
#
# Three defects fixed 2026-08-06 (re-run blocker A-1); all three would have corrupted the re-run this
# script exists to feed:
#
#  1. **Config-blind resume.** The marker was an empty `{stem}.done` file. Every catalog on disk
#     predates X-01 (conf reached ultralytics as 0.25, not the requested 0.10), X-02 (`score` was
#     overwritten with a literal 0.10 for truncated rows — the executed fingerprint is no `nms_prio`
#     column and 28 of 507 rows at score exactly 0.10), X-06 (`max_det`), G3-01, G3-10 and G1-06. A
#     re-run would have found the markers, read those catalogs and published them as today's ladder.
#  2. **`missing file == empty catalog` on resume.** pyogrio cannot write an empty layer, so a real
#     empty result and a failed/absent write are indistinguishable on disk. Substituting an empty
#     frame for a *missing* one moves the pre-registered `fused >= baseline` gate in the **passing**
#     direction whenever it is the baseline that went missing. Emptiness is now recorded per tile at
#     write time, so resume can tell the two apart and raises on the ambiguous case.
#  3. **The size-gated AP was the wrong quantity (G3-04).** `size_gated` filtered *predictions* only
#     and left every truth in the denominator — including the 32.3% below 5 px — so the detector was
#     charged for missing boulders the evaluation had declared out of scope. `catalog_ap` has taken a
#     `size_range` with COCO ignore semantics since `c17f704`; this runner never migrated. On the
#     frozen catalogs the correction moves the headline pair 0.4803/0.4848 -> 0.5659/0.5695: the
#     direction survives, the margin narrows from +0.0045 to +0.0036. Both are computed now, with the
#     prediction-only variant kept and labelled so the published numbers stay reproducible.
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")

DATA = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\data_prieur\boulder2024\test")
BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data")
WEIGHTS = BASE / "weights" / "yolov8-m-boulder-detection-tmp.pt"
OUT = BASE / "solutionDE" / "ap50"
TILE_DIR = OUT / "tiles"
PRED_DIR = OUT / "preds"
PAD = 512          # tile canvas: 500-px tiles zero-padded so slice grids are exact
RES = 1.0          # dummy m/px: native px == map units everywhere
LADDER = (512, 256, (256, 2))
WINNER = dict(rule="bands", bands={256: (0, np.inf)}, engulf_thr=0.7)  # 256pref_engulf


#: Size floors scored. Each is evaluated two ways: `size_range` (truth AND predictions, COCO ignore
#: semantics — the correct quantity, G3-04) and prediction-only gating (what the published table
#: measured, kept so those numbers stay reproducible and the delta stays visible).
SIZE_FLOORS = (4, 5)


def run_identity(conf, ladder, winner, weights):
    """Everything that determines a tile's cached prediction, so resume can be *verified*.

    Mirrors `scripts/review/measure_operating_point.py`'s B2 sidecar. The point is not to be clever:
    it is that an empty `.done` file cannot answer "which contract produced this?", and the honest
    answer when it cannot is to recompute.
    """
    from YOLOv8BeyondEarth import presets
    from YOLOv8BeyondEarth.predict import INFERENCE_CONTRACT_VERSION
    arm = {**presets.resolve("goal1_production", confidence_threshold=float(conf)),
           "ladder": [list(s) if isinstance(s, tuple) else s for s in ladder],
           "fusion": {k: (str(v) if k == "bands" else v) for k, v in winner.items()}}
    code = presets.code_identity()
    return {
        "conf": round(float(conf), 4),
        "weights": presets.file_identity(weights),
        "preset": "goal1_production+ladder",
        "preset_version": presets.PRESET_VERSION,
        "inference_contract_version": INFERENCE_CONTRACT_VERSION,
        "arm_config_hash": presets.config_hash(arm),
        "pad": PAD, "res": RES,
        "code": {"git_sha": code.get("git_sha"), "git_dirty": code.get("git_dirty")},
    }


#: Identity fields whose mismatch means a reused catalog is a *different measurement*. `code` is
#: excluded for the same reason as in `measure_operating_point.py` — a docs commit must not force a
#: 280-tile GPU re-run — and reported as a warning instead.
_FATAL_IDENTITY_FIELDS = ("conf", "weights", "preset", "preset_version",
                          "inference_contract_version", "arm_config_hash", "pad", "res")


def read_marker(path, want):
    """``None`` if this tile must be recomputed; the marker dict if its cache is verified.

    Raises when reuse would silently produce the wrong number — the pre-X-01 catalogs on disk are
    exactly that case.
    """
    if not path.exists():
        return None
    try:
        have = json.loads(path.read_text())
    except (ValueError, UnicodeDecodeError):
        # The pre-2026-08-06 marker was a zero-byte `.done` file. It cannot say what produced it,
        # and everything it marks predates X-01/X-02/X-06, so it is refused rather than trusted.
        raise FileExistsError(
            f"{path.name} is a legacy empty marker with no identity. Every catalog it marks was "
            "produced before the X-01/X-02/X-06 contract fixes (effective conf 0.25, `score` "
            "overwritten with 0.10 on truncated rows, `max_det` unpassed), so reusing it would "
            "score pre-fix geometry as today's ladder. Delete the ap50/preds directory and re-run.")
    differing = [k for k in _FATAL_IDENTITY_FIELDS
                 if have.get("identity", {}).get(k) != want.get(k)]
    if differing:
        detail = "\n".join(f"    {k}: cached {have.get('identity', {}).get(k)!r} != "
                           f"this run {want.get(k)!r}" for k in differing)
        raise FileExistsError(
            f"{path.name} was produced by a different configuration, so reusing it would score the "
            f"wrong catalog. Differing:\n{detail}\nDelete the ap50/preds directory and re-run.")
    return have


def materialize_tile(png_path, tif_path):
    """500x500 PNG -> 512x512 zero-padded GeoTIFF with a north-up identity-scale transform."""
    import rasterio
    from rasterio.transform import from_origin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # NotGeoreferencedWarning on the source PNG
        with rasterio.open(png_path) as src:
            img = src.read(1)
    h, w = img.shape
    assert h <= PAD and w <= PAD, f"{png_path.name}: {w}x{h} exceeds the {PAD} canvas"
    canvas = np.zeros((PAD, PAD), dtype=np.uint8)
    canvas[:h, :w] = img
    transform = from_origin(0, PAD, RES, RES)
    with rasterio.open(tif_path, "w", driver="GTiff", height=PAD, width=PAD, count=1,
                       dtype="uint8", crs="EPSG:32633", transform=transform) as dst:
        dst.write(canvas, 1)
    return w, h


if __name__ == "__main__":
    import geopandas as gpd
    import pandas as pd
    from sahi import AutoDetectionModel

    from YOLOv8BeyondEarth.predict import get_multiscale_prediction
    from YOLOv8BeyondEarth.ap_eval import catalog_ap, load_yolo_seg_labels, size_gated

    for d in (TILE_DIR, PRED_DIR):
        d.mkdir(parents=True, exist_ok=True)

    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=str(WEIGHTS), device="cuda:0",
        image_size=1024, confidence_threshold=0.1)

    pngs = sorted(DATA.glob("images/*.png"))
    print(f"{len(pngs)} test tiles", flush=True)

    scales_str = "-".join(str(s) if isinstance(s, int) else f"{s[0]}sr{s[1]}" for s in LADDER)
    rule_str = f"{WINNER['rule']}-engulf{int(round(WINNER['engulf_thr'] * 100)):02d}"

    identity = run_identity(0.1, LADDER, WINNER, WEIGHTS)
    tile_meta, failures, wall = {}, [], {"total_s": 0.0}
    fused_frames, base_frames = [], []
    t_all = time.time()
    for k, png in enumerate(pngs):
        stem = png.stem
        tif = TILE_DIR / f"{stem}.tif"
        if not tif.exists():
            materialize_tile(png, tif)
        fused_gpkg = PRED_DIR / f"{stem}-predictions-fused-{rule_str}-{scales_str}-mask.gpkg"
        base_gpkg = PRED_DIR / (f"{stem}-predictions-ct-010-ss-256-is-1024-ov-020"
                                "-downscaled-mask-nms.gpkg")
        # An identity-carrying marker, not a touch-file. pyogrio cannot write an empty layer, so
        # "no file" is ambiguous between a genuinely empty result and a write that never happened —
        # the marker resolves it by recording emptiness at the moment it is known.
        done_marker = PRED_DIR / f"{stem}.done.json"
        legacy_marker = PRED_DIR / f"{stem}.done"
        if legacy_marker.exists() and not done_marker.exists():
            raise FileExistsError(
                f"{legacy_marker.name} is a legacy empty marker (pre-2026-08-06) with no record of "
                "what produced it, and everything it marks predates the X-01/X-02/X-06 contract "
                "fixes. Delete the ap50/preds directory and re-run; see this file's header.")
        try:
            marker = read_marker(done_marker, identity)
            if marker is None:
                t0 = time.time()
                fused, audit, per_scale = get_multiscale_prediction(
                    tif, model, scales=LADDER, output_dir=PRED_DIR, output_format="gpkg",
                    confidence_threshold=0.1, inference_size=1024,
                    overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                    downscale_pred=True, **WINNER)
                wall[stem] = round(time.time() - t0, 2)
                base = per_scale[256]
                # Written after the layers, so a crash mid-write leaves no verified-looking marker.
                done_marker.write_text(json.dumps(
                    {"identity": identity, "n_fused": int(len(fused)), "n_base": int(len(base)),
                     "wall_s": wall[stem]}, indent=1))
            else:
                def _resume(path, n_expected, what):
                    """A missing file is only an empty catalog if the marker says it was empty."""
                    if path.exists():
                        return gpd.read_file(path)
                    if n_expected:
                        raise FileNotFoundError(
                            f"{path.name} is missing but its marker records {n_expected} "
                            f"{what} detections, so this is a lost write, not an empty tile. "
                            "Substituting an empty catalog here would move the pre-registered "
                            "'fused >= baseline' gate in the passing direction. Delete "
                            f"{done_marker.name} to recompute this tile.")
                    return gpd.GeoDataFrame({"score": []}, geometry=[], crs="EPSG:32633")
                fused = _resume(fused_gpkg, marker.get("n_fused"), "fused")
                base = _resume(base_gpkg, marker.get("n_base"), "baseline")
        except Exception as e:                                    # keep going; report at the end
            failures.append((stem, repr(e)))
            print(f"[FAIL {len(failures)}] {stem}: {e!r}", flush=True)
            continue
        # keep only the eval columns (in-memory frames carry array-valued polygon/bbox
        # columns that gpkg cannot serialize; read-back frames carry renamed extras)
        for cat, frames in ((fused, fused_frames), (base, base_frames)):
            slim = gpd.GeoDataFrame({"image_id": stem,
                                     "score": cat["score"].to_numpy() if len(cat) else []},
                                    geometry=cat.geometry.values, crs="EPSG:32633")
            frames.append(slim)
        if (k + 1) % 20 == 0:
            print(f"[{k + 1}/{len(pngs)}] elapsed {time.time() - t_all:.0f}s", flush=True)
    wall["total_s"] = round(time.time() - t_all, 2)

    # A tile that failed contributed no predictions but its TRUTHS are still built into `gts` below,
    # so it scores as pure recall loss and silently depresses both arms' AP. Both arms lose the same
    # tiles, so the arm-to-arm comparison (the number that matters) survives — but the absolute value
    # does not, and the pre-registered gate is a comparison of absolutes. Drop the failed tiles from
    # the truth side and say so, rather than quietly charging the detector for them.
    if failures:
        failed_stems = {stem for stem, _ in failures}
        print(f"WARNING: {len(failed_stems)} of {len(pngs)} tiles failed; excluding them from the "
              "truth set so they are not scored as recall loss. The gate below is computed on "
              f"{len(pngs) - len(failed_stems)} tiles.", flush=True)
    else:
        failed_stems = set()
    if not fused_frames or not base_frames:
        raise RuntimeError(
            f"every tile failed ({len(failures)} failures); nothing to score. First: "
            f"{failures[0] if failures else None}")
    fused_cat = pd.concat(fused_frames, ignore_index=True)
    base_cat = pd.concat(base_frames, ignore_index=True)
    gpd.GeoDataFrame(fused_cat).to_file(OUT / "fused_catalog.gpkg")
    gpd.GeoDataFrame(base_cat).to_file(OUT / "baseline_catalog.gpkg")

    # truths in the same padded map frame as the predictions
    from rasterio.transform import from_origin
    transform = from_origin(0, PAD, RES, RES)
    gts = {}
    for png in pngs:
        if png.stem in failed_stems:
            continue                       # see the WARNING above: not scored, not charged
        txt = DATA / "labels" / f"{png.stem}.txt"
        gts[png.stem] = load_yolo_seg_labels(txt, 500, 500, transform=transform) \
            if txt.exists() else []

    results = {"n_tiles": len(pngs), "n_failures": len(failures), "failures": failures,
               "wall": wall,
               "anchors": {"amaro2026_test_ap50": 0.58, "prieur2023_test_ap50": 0.52,
                           "note": "published numbers, different evaluator — context only"}}
    for name, cat in (("baseline_256", base_cat), ("fused_256pref_engulf", fused_cat)):
        entry = {}
        for geom in ("mask", "box"):
            entry[f"ap50_{geom}"] = catalog_ap(cat, gts, iou_thr=0.5, geom=geom)
        # Labels follow the Prieur 5x5-px floor, so the ">= floor" question is asked two ways.
        for floor in SIZE_FLOORS:
            # The correct quantity (G3-04): the range applies to truth AND predictions, with
            # COCO ignore semantics, so out-of-scope truths leave the recall denominator instead
            # of being charged as misses.
            entry[f"ap50_mask_ge{floor}px"] = catalog_ap(
                cat, gts, iou_thr=0.5, geom="mask", size_range=(floor, np.inf), res=RES)
            # What the published table measured: predictions gated, all 8,990 truths kept
            # (2,904 of them below 5 px). Retained so the shipped numbers stay reproducible and
            # the size of the correction stays on the record, NOT as an alternative reading.
            entry[f"ap50_mask_ge{floor}px_predfiltered_LEGACY"] = catalog_ap(
                size_gated(cat, min_diam_px=floor, res=RES), gts, iou_thr=0.5, geom="mask")
        results[name] = entry
        print(name, json.dumps(entry, indent=2), flush=True)

    # The pre-registered gate, evaluated explicitly rather than left to a reader comparing two
    # printouts — and on the G3-04-correct quantity.
    results["pre_registered_gate"] = {
        "statement": "fused >= baseline on the labeled size range (>= 5 px)",
        "metric": "ap50_mask_ge5px (size_range applied to truth and predictions)",
        "baseline": results["baseline_256"]["ap50_mask_ge5px"],
        "fused": results["fused_256pref_engulf"]["ap50_mask_ge5px"],
        "margin": (results["fused_256pref_engulf"]["ap50_mask_ge5px"]
                   - results["baseline_256"]["ap50_mask_ge5px"]),
        "passes": (results["fused_256pref_engulf"]["ap50_mask_ge5px"]
                   >= results["baseline_256"]["ap50_mask_ge5px"]),
        "note": "Development numbers (G3-11): all six candidate rules were built and scored on "
                "this population. n_failures must be 0 for this gate to mean anything.",
        "n_failures": len(failures),
    }
    print("gate", json.dumps(results["pre_registered_gate"], indent=2), flush=True)

    (OUT / "ap50_results.json").write_text(json.dumps(results, indent=2))
    print("written", OUT / "ap50_results.json", flush=True)
