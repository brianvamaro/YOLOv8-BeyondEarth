# Goal-3 phase 4: full-scene benchmark (plan §Phases 4) — fused ladder vs 256-only production
# on the full NAC scene (M139694087LE, 5118x59183, 0.484 m/px) and the full HiRISE scene
# (ESP_017355_2260_RED, 22490x85922, 0.5 m/px).
#
# SUBPROCESS-PER-ARM (2026-07-11): the first cut ran all three arms + fusion in ONE long-lived
# process; by the HiRISE 256 arm it had ratcheted to ~24 GB committed (mostly cold/paged-out
# leftovers from the 512 arm — CPython pymalloc arenas + fragmentation don't return millions of
# small polygon/DataFrame objects to the OS between arms), driving the system to 78 GB commit
# and pagefile thrash. Fix = the old Goal-1 fullbench pattern: each arm (and the fusion) runs in
# its OWN subprocess that EXITS, fully reclaiming memory. So no process ever holds more than one
# arm's working set (~10-15 GB), well inside 31 GB. Dual-mode:
#   (no args)            driver: orchestrate arms + fuse via subprocess.run([sys.executable,...])
#   arm <scene> <ss> <sr>  worker: run ONE detection arm, write gpkg + JSON keys, exit
#   fuse <scene>           worker: load the 3 arms, fuse (256pref_engulf), gates, exit
# Idempotent (skip if the arm gpkg / fused layer exists); resumable; each worker read-modify-
# writes only its own keys in fullscene_results.json, so completed scenes/arms are preserved.
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data")
WEIGHTS = BASE / "weights" / "yolov8-m-boulder-detection-tmp.pt"
OUT = BASE / "solutionDE" / "fullscene"
RESULTS = OUT / "fullscene_results.json"

SCENES = {
    "nac": BASE / "raster" / "M139694087LE.tif",
    "hirise": BASE / "mars_hirise" / "raster" / "ESP_017355_2260_RED.tif",
}
ARMS = ((512, 1), (256, 1), (256, 2))          # the production ladder
FUSED_NAME = "-predictions-fused-bands-engulf70-512-256-256sr2-mask.gpkg"


def nms_gpkg(out_dir, raster, ss, sr):
    tag = f"{ss}sr{sr}" if sr > 1 else f"{ss}"
    return out_dir / (raster.stem + f"-predictions-ct-010-ss-{tag}-is-1024-ov-020"
                      "-downscaled-mask-nms.gpkg")


def load_results():
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def update_results(mutate):
    """Read-modify-write only this worker's keys (sequential workers, no race)."""
    res = load_results()
    mutate(res)
    RESULTS.write_text(json.dumps(res, indent=2))


# --------------------------------------------------------------------------- arm worker
def run_arm(scene, ss, sr):
    import geopandas as gpd  # noqa: F401  (import cost paid once per worker)
    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth.predict import get_sliced_prediction

    raster = SCENES[scene]
    out_dir = OUT / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    key = f"ss{ss}" + (f"sr{sr}" if sr > 1 else "")
    if nms_gpkg(out_dir, raster, ss, sr).exists():
        print(f"[skip] {scene}/{key}", flush=True)
        return
    print(f"[run ] {scene}/{key} (pid {__import__('os').getpid()})", flush=True)

    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=str(WEIGHTS), device="cuda:0",
        image_size=1024, confidence_threshold=0.1)
    t0 = time.time()
    gdf, gdf_nms = get_sliced_prediction(
        raster, detection_model=model, confidence_threshold=0.1,
        output_dir=out_dir, slice_size=ss, sr_factor=sr, inference_size=1024,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        downscale_pred=True, postprocess=True,
        save_prenms=False, save_bbox=False, output_format="gpkg")
    wall = round(time.time() - t0, 2)
    n_raw, n_nms = int(len(gdf)), int(len(gdf_nms))

    def mut(res):
        s = res.setdefault(scene, {})
        s.setdefault("arm_wall_s", {})[key] = wall
        s[f"n_raw_{key}"] = n_raw
        s[f"n_nms_{key}"] = n_nms
    update_results(mut)
    print(f"[done] {scene}/{key}: {wall}s  n_raw={n_raw} n_nms={n_nms}", flush=True)


# --------------------------------------------------------------------------- fuse worker
def run_fuse(scene):
    import geopandas as gpd
    from rastertools_BOULDERING import metadata as raster_metadata
    from YOLOv8BeyondEarth.fusion import (fuse_predictions, matched_recall, duplicate_rate)
    from YOLOv8BeyondEarth.polygon import outlines_to_shp

    raster = SCENES[scene]
    out_dir = OUT / scene
    fused_path = out_dir / (raster.stem + FUSED_NAME)
    print(f"[fuse] {scene} (pid {__import__('os').getpid()})", flush=True)

    per_scale = {}
    for ss, sr in ARMS:
        key = ss if sr == 1 else (ss, sr)
        per_scale[key] = gpd.read_file(nms_gpkg(out_dir, raster, ss, sr))
        print(f"  loaded {key}: {len(per_scale[key]):,}", flush=True)
    native_res = raster_metadata.get_resolution(raster)[0]

    t0 = time.time()
    fused, audit = fuse_predictions(per_scale, rule="bands", bands={256: (0, np.inf)},
                                    engulf_thr=0.7, native_res=native_res)
    fuse_wall = round(time.time() - t0, 2)
    outlines_to_shp(fused, fused_path)
    audit.to_csv(fused_path.with_suffix("").as_posix() + "-audit.csv.gz", index=False)

    rec, unmatched = matched_recall(per_scale[256], fused, engulf_thr=0.7)
    dup = duplicate_rate(fused, native_res=native_res)
    comp = fused["scale_key"].value_counts().to_dict()

    def mut(res):
        s = res.setdefault(scene, {})
        stimes = s.get("arm_wall_s", {})
        s.update({
            "fuse_wall_s": fuse_wall,
            "n_fused": int(len(fused)),
            "parity_matched_recall": rec,
            "parity_unmatched": int(len(unmatched)),
            "composition": {k: int(v) for k, v in comp.items()},
            "engulfed": int(audit.engulfed.sum()),
            "dupe_audit": dup.to_dict(orient="records"),
            "ladder_factor_vs_256": (round(sum(stimes.values()) / stimes["ss256"], 2)
                                     if "ss256" in stimes else None),
        })
    update_results(mut)
    print(f"[done] {scene}/fuse: {fuse_wall}s  n_fused={len(fused):,}  parity={rec:.4f} "
          f"unmatched={len(unmatched)}  engulfed={int(audit.engulfed.sum()):,}", flush=True)


# --------------------------------------------------------------------------- driver
def spawn(*args):
    """Run one worker in a fresh subprocess (inherits the activated conda env → CUDA).
    Memory is fully reclaimed on exit. Raises if the worker fails (don't fuse a missing arm)."""
    cmd = [sys.executable, __file__, *map(str, args)]
    print(f">>> spawn {' '.join(map(str, args))}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"worker {args} failed rc={r.returncode} — stopping")


def driver():
    for scene, raster in SCENES.items():
        out_dir = OUT / scene
        out_dir.mkdir(parents=True, exist_ok=True)
        for ss, sr in ARMS:
            if nms_gpkg(out_dir, raster, ss, sr).exists():
                key = f"ss{ss}" + (f"sr{sr}" if sr > 1 else "")
                print(f"[skip] {scene}/{key} (exists)", flush=True)
                continue
            spawn("arm", scene, ss, sr)
        fused_path = out_dir / (raster.stem + FUSED_NAME)
        res = load_results()
        if fused_path.exists() and "fuse_wall_s" in res.get(scene, {}):
            print(f"[skip] {scene}/fuse (exists)", flush=True)
            continue
        spawn("fuse", scene)
    print("\n=== full-scene benchmark complete ===", flush=True)
    print(json.dumps(load_results(), indent=2, default=str), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        driver()
    elif sys.argv[1] == "arm":
        run_arm(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif sys.argv[1] == "fuse":
        run_fuse(sys.argv[2])
    else:
        raise SystemExit(f"unknown mode {sys.argv[1]!r} (use: arm <scene> <ss> <sr> | fuse <scene>)")
