"""Audit the campaign-scale and SR-arm cost arithmetic (verifies cost_model_optimization_plan claims).

The 2026-08-02/03 optimisation research returned several arithmetic claims single-source, and its
verifiers never ran. These are the ones settleable from slice geometry plus saved catalog counts, with
no timing run — which matters, because every wall clock in that pass is Modern-Standby contaminated
and the GPU is power-capped.

Claims audited:

* **"131 model-pixels per scene-pixel under the full ladder, 25 at the 256 arm alone."** The GPU term
  is `imgsz**2 / step**2` summed over arms, where `step = native_slice - int(overlap*native_slice)`.
* **"The SR arm is (256, 2) -> 128 native px, i.e. 3.96x the slices of the 256 arm."**
* **"Shipping the SR arm is 64% of the full ladder."**
* **"The published NAC 6.64x [ledger: 6.63x] is arithmetically impossible (capped at ~4x by slice
  count) and is therefore a Modern-Standby artifact."** This is the interesting one. A slice-count
  ratio only bounds the *GPU* term; total wall is GPU + per-detection CPU + the SR Lanczos step. So
  the bound holds only if the SR arm's per-detection CPU work does not *exceed* the 256 arm's — which
  is a question about detection counts, and it goes opposite ways on the two scenes.

    conda run -n bouldernet python scripts/review/check_ladder_arithmetic.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_max_det_binding import slice_starts   # noqa: E402

#: The GOAL3_PRODUCTION ladder, as (slice_size, sr_factor).
LADDER = [(512, 1), (256, 1), (256, 2)]

SCENES = {
    "hirise": {
        "raster": REPO / "data" / "mars_hirise" / "raster" / "ESP_017355_2260_RED.tif",
        "dir": REPO / "data" / "solutionDE" / "fullscene" / "hirise",
        "stem": "ESP_017355_2260_RED",
        "ledger_sr_arm_wall_ratio": 2.73,
    },
    "nac": {
        "raster": REPO / "data" / "raster" / "M139694087LE.tif",
        "dir": REPO / "data" / "solutionDE" / "fullscene" / "nac",
        "stem": "M139694087LE",
        "ledger_sr_arm_wall_ratio": 6.63,
    },
}


def arm_layer(scene, slice_size, sr_factor):
    ss = str(slice_size) if sr_factor == 1 else f"{slice_size}sr{sr_factor}"
    return scene["dir"] / (f"{scene['stem']}-predictions-ct-010-ss-{ss}"
                           f"-is-1024-ov-020-downscaled-mask-nms.gpkg")


def n_features(path):
    from pyogrio import read_info
    try:
        return int(read_info(path)["features"])
    except Exception as exc:                      # missing layer: report rather than guess
        return f"unavailable ({type(exc).__name__})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import rasterio

    record = {"imgsz": args.imgsz, "overlap": args.overlap, "ladder": [list(a) for a in LADDER],
              "scenes": {}}

    for name, scene in SCENES.items():
        if not scene["raster"].exists():
            record["scenes"][name] = {"error": f"raster missing: {scene['raster']}"}
            print(f"[skip] {name}: raster missing")
            continue
        with rasterio.open(scene["raster"]) as src:
            width, height = src.width, src.height
        scene_px = width * height

        arms = {}
        for slice_size, sr in LADDER:
            native = slice_size // sr
            nx = len(slice_starts(width, native, args.overlap))
            ny = len(slice_starts(height, native, args.overlap))
            n_slices = nx * ny
            model_px = n_slices * args.imgsz ** 2
            # closed form, for cross-checking the empirical grid
            step = native - int(args.overlap * native)
            arms[f"{slice_size}sr{sr}" if sr > 1 else str(slice_size)] = {
                "native_slice": native, "step_px": step,
                "n_slices": n_slices, "grid": [nx, ny],
                "model_px_per_scene_px": round(model_px / scene_px, 2),
                "closed_form_imgsz2_over_step2": round(args.imgsz ** 2 / step ** 2, 2),
                "n_detections": n_features(arm_layer(scene, slice_size, sr)),
            }

        total_slices = sum(a["n_slices"] for a in arms.values())
        total_ratio = sum(a["model_px_per_scene_px"] for a in arms.values())
        sr_key, base_key = "256sr2", "256"
        sr, base = arms[sr_key], arms[base_key]

        det_ratio = None
        if isinstance(sr["n_detections"], int) and isinstance(base["n_detections"], int):
            det_ratio = round(sr["n_detections"] / base["n_detections"], 3)

        entry = {
            "raster_px": [width, height], "scene_px": scene_px,
            "arms": arms,
            "total_slices": total_slices,
            "model_px_per_scene_px_full_ladder": round(total_ratio, 2),
            "sr_arm_share_of_model_px": round(sr["model_px_per_scene_px"] / total_ratio, 3),
            "sr_over_256_slice_ratio": round(sr["n_slices"] / base["n_slices"], 3),
            "sr_over_256_detection_ratio": det_ratio,
            "ledger_sr_arm_wall_ratio": scene["ledger_sr_arm_wall_ratio"],
        }

        # The bound. Total wall = GPU(n_slices) + CPU(n_detections) + Lanczos(SR only). Expressing
        # both arms as multiples of the 256 arm, and writing g for the 256 arm's GPU fraction of its
        # own wall, the SR arm's ratio is  g*S + (1-g)*D + L  with S the slice ratio and D the
        # detection ratio. That is maximised over g in [0, 1] at max(S, D) -- so max(S, D) (plus the
        # small Lanczos term) is a HARD upper bound on the SR arm's wall ratio, whatever the CPU/GPU
        # split turns out to be.
        if det_ratio is not None:
            bound = max(entry["sr_over_256_slice_ratio"], det_ratio)
            entry["wall_ratio_upper_bound"] = bound
            entry["ledger_exceeds_bound"] = bool(scene["ledger_sr_arm_wall_ratio"] > bound)
            entry["excess_over_bound"] = round(
                scene["ledger_sr_arm_wall_ratio"] / bound, 2)

        record["scenes"][name] = entry

        print(f"\n=== {name}  ({width} x {height} = {scene_px/1e9:.3f} Gpx) ===")
        for key, a in arms.items():
            print(f"  arm {key:<8} native={a['native_slice']:<4} step={a['step_px']:<4} "
                  f"slices={a['n_slices']:>8,}  model_px/scene_px={a['model_px_per_scene_px']:>6.2f}"
                  f"  detections={a['n_detections']}")
        print(f"  full ladder model_px/scene_px = {total_ratio:.2f}"
              f"   (SR arm is {entry['sr_arm_share_of_model_px']*100:.1f}% of it)")
        print(f"  SR/256: slices x{entry['sr_over_256_slice_ratio']}, "
              f"detections x{det_ratio}")
        if "wall_ratio_upper_bound" in entry:
            verdict = ("EXCEEDS the bound -> not explainable by slice count or detections"
                       if entry["ledger_exceeds_bound"] else "within the bound")
            print(f"  ledger wall ratio {scene['ledger_sr_arm_wall_ratio']}x vs upper bound "
                  f"{entry['wall_ratio_upper_bound']}x  ->  {verdict}")

    out = Path(args.out) if args.out else REPO / "data" / "review" / "ladder_arithmetic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
