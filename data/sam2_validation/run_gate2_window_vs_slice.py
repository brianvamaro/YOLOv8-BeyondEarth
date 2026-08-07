"""Goal-3 phase 5a, entry gate 2 — window-vs-slice SAM equivalence.

The gate (plan §Phase-5 / 5a): *"SAM's context in a window crop differs from the in-loop slice
crop; re-mask one dev crop both ways on identical boxes and compare angle/area on matched pairs
(tolerance from the sam256/sam1024 equivalence precedent). A context sensitivity here would also be
a finding for the validation plan."*

Why the two paths can differ at all — SAM's encoder resizes whatever it is handed to a fixed input,
so what matters is the crop *around* the boulder:
  * slice path  : SAHI's **overlapping** 256 grid; the boulder sits wherever the slice grid puts it.
  * window path : ``remask.remask_windows``' **non-overlapping** 256 grid, grown only to contain its
                  own members; the same boulder sits somewhere else in its 256 box, with different
                  neighbours near the edges.
Same nominal 256 px of context, different translation. This measures whether SAM cares.

Precedent for the tolerance: the sam256-vs-sam1024 equivalence check in the SAM2 validation gave
median |Δangle| **1.29°** (p90 5.9°) for a *much* bigger perturbation (a 4× resolution change), and
that was accepted as "config-invariant". A translation of the crop should be no worse, so:

    PASS      median |Δangle| <= 1.3 deg  AND  median |Δarea|/area <= 5%
    MARGINAL  median |Δangle| <= 3.0 deg
    FAIL      anything worse -> 5a does not ship on windows without a fix, and the context
              sensitivity is itself a finding for the validation plan.

Usage (needs GPU + weights; ~minutes on the 4k crop):
    conda run -n bouldernet --no-capture-output python data/sam2_validation/run_gate2_window_vs_slice.py
Writes gate2_window_vs_slice.csv (per matched pair) + gate2_summary.json next to this script.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from YOLOv8BeyondEarth.predict import get_sliced_prediction
from YOLOv8BeyondEarth.orientation_validation import sam2_remasker
from YOLOv8BeyondEarth.remask import remask_windows
from YOLOv8BeyondEarth.orientation import attach_orientations, axial_distance
from YOLOv8BeyondEarth.benchmark import match_detections

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent

WEIGHTS = ROOT / "data/weights/yolov8-m-boulder-detection-tmp.pt"
SAM_WEIGHTS = REPO / "sam2.1_b.pt"
CROPS = {
    "hirise_crop4k": ROOT / "data/solutionDE/hirise_crop4k.tif",
    "nac_crop2k": ROOT / "data/raster/crop.tif",
}
OUT_DIR = HERE / "gate2_out"

SLICE = 256
CONF = 0.10
IMGSZ = 1024
ASPECT_MIN = 1.35     # angles are only meaningful on elongated boulders (standing gate)
MATCH_FACTOR = 2.0    # centroid match radius = 2 x native res, the standing pairing tolerance


def _load_models():
    from sahi import AutoDetectionModel
    from ultralytics import SAM
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=str(WEIGHTS), device="cuda:0",
        image_size=IMGSZ, confidence_threshold=CONF)
    sam = SAM(SAM_WEIGHTS.as_posix())
    return model, sam


def run_crop(name, path, model, sam):
    import rasterio as rio
    with rio.open(path) as src:
        res = abs(src.transform.a)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    common = dict(detection_model=model, confidence_threshold=CONF, inference_size=IMGSZ,
                  slice_size=SLICE, overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                  downscale_pred=True, postprocess=True, output_dir=OUT_DIR,
                  output_format="gpkg", save_prenms=False, save_bbox=False)

    # --- path B: the production in-slice re-masker (the validated reference) ---
    _, gdf_slice = get_sliced_prediction(path, remask_fn=sam2_remasker(sam),
                                         mask_tag="sam2-slice", **common)

    # --- path A: plain YOLO detections, then the phase-5a window re-masker ---
    _, gdf_yolo = get_sliced_prediction(path, **common)
    gdf_win = remask_windows(path, gdf_yolo, sam, window_size=SLICE, progress=True)
    gdf_win_sam = gdf_win[gdf_win.mask_src == "sam2"].reset_index(drop=True)

    # --- compare on spatially matched pairs (standing gate: never raw populations) ---
    pairs = match_detections(gdf_win_sam, gdf_slice, max_dist=MATCH_FACTOR * res)
    a = attach_orientations(gdf_win_sam.iloc[pairs.idx_a.values].reset_index(drop=True), res)
    b = attach_orientations(gdf_slice.iloc[pairs.idx_b.values].reset_index(drop=True), res)

    df = pd.DataFrame({
        "crop": name,
        "dist": pairs.dist.values,
        "angle_window": a.angle180.values, "angle_slice": b.angle180.values,
        "aspect_window": a.aspect_ra.values, "aspect_slice": b.aspect_ra.values,
        "dpx_window": a.diameter_m.values / res, "dpx_slice": b.diameter_m.values / res,
        "area_window": a.geometry.area.values, "area_slice": b.geometry.area.values,
    })
    df["dangle"] = axial_distance(df.angle_window.values, df.angle_slice.values)
    df["darea_frac"] = (df.area_window - df.area_slice).abs() / df.area_slice.clip(lower=1e-9)

    elong = df[(df.aspect_window > ASPECT_MIN) & (df.aspect_slice > ASPECT_MIN)]
    summary = {
        "crop": name, "res": res,
        "n_slice": int(len(gdf_slice)), "n_yolo": int(len(gdf_yolo)),
        "n_window_sam": int(len(gdf_win_sam)),
        "window_notes": gdf_win.remask_note.value_counts().to_dict(),
        "n_matched": int(len(df)),
        "match_rate_vs_window": float(len(df) / max(len(gdf_win_sam), 1)),
        "median_dangle_elong": float(elong.dangle.median()) if len(elong) else None,
        "p90_dangle_elong": float(elong.dangle.quantile(0.9)) if len(elong) else None,
        "n_elong": int(len(elong)),
        "median_darea_frac": float(df.darea_frac.median()),
        "p90_darea_frac": float(df.darea_frac.quantile(0.9)),
        "frac_angle_within_10deg": float((elong.dangle <= 10).mean()) if len(elong) else None,
    }
    return df, summary


def verdict(s):
    if s["median_dangle_elong"] is None:
        return "NO DATA"
    if s["median_dangle_elong"] <= 1.3 and s["median_darea_frac"] <= 0.05:
        return "PASS"
    if s["median_dangle_elong"] <= 3.0:
        return "MARGINAL"
    return "FAIL"


def main():
    model, sam = _load_models()
    frames, summaries = [], []
    for name, path in CROPS.items():
        if not path.exists():
            print(f"[skip] {name}: {path} missing")
            continue
        print(f"\n=== {name} ===", flush=True)
        df, s = run_crop(name, path, model, sam)
        s["verdict"] = verdict(s)
        frames.append(df); summaries.append(s)
        print(json.dumps(s, indent=2, default=str), flush=True)

    if frames:
        pd.concat(frames, ignore_index=True).to_csv(HERE / "gate2_window_vs_slice.csv", index=False)
        (HERE / "gate2_summary.json").write_text(json.dumps(summaries, indent=2, default=str))
        print("\nVERDICTS:", {s["crop"]: s["verdict"] for s in summaries})


if __name__ == "__main__":   # Windows spawn guard (conda-run gotcha)
    main()
