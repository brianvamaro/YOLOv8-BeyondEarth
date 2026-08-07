"""Cache "what the segmenter actually sees" for a handful of real boulders (mechanism figure).

The artifact claim in the summary notebook is that YOLOv8-seg's small-boulder orientations are set by
the **resolution of the segmentation head's prototype basis**, not by the imagery and not by the angle
fit. That is a claim about an array nobody normally looks at, so this caches it.

`orientation_validation.detect_segmenter_view` returns, per detection, the same boulder at each stage:
native slice pixels -> the bilinear 4x upscale the network is fed -> **the prototype-resolution
probability field** -> the final production mask. At the production config
(`slice_size=256, imgsz=1024`) the prototype basis runs at `imgsz/4 = 256`, i.e. **one prototype cell
per native raster pixel**, which is the whole mechanism in one number: a 7-px boulder gets ~7x7 cells
to describe its shape, so its long axis can only land where such a coarse grid permits.

Selection. Boulders are picked to *show the mechanism honestly*, not to flatter it: from the small
well-detected population (`dpx` in `DPX_RANGE`), prefer those whose final-mask angle sits closest to a
pixel diagonal AND whose proto field has few cells — i.e. the regime the archive's dose-response study
implicated — while spanning a range of sizes so a reader can see the effect fade. The score and the
alternative (a random sample) are both cached so the figure can show either; the notebook states which
it draws and why.

GPU, small (a few tiles). __main__ guard (Windows spawn). Run from inside YOLOv8-BeyondEarth/:
  C:\\Users\\brian\\anaconda3\\Scripts\\conda.exe run -n bouldernet --no-capture-output \\
      python data/summary/run_segmenter_view.py
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet")
ROOT = BASE / "YOLOv8-BeyondEarth"
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data" / "summary"
DE = ROOT / "data" / "solutionDE"
WEIGHTS = ROOT / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"

SLICE, IMGSZ, CONF = 256, 1024, 0.10
DPX_RANGE = (5.0, 14.0)      # the band the archive's dose-response found carries the lock
ASPECT_MIN = 1.35
N_KEEP = 6
N_TILES = 4                  # HiRISE ROM crops to scan (dense; a few is plenty)
MAX_PER_CROP = 600           # a mechanism figure needs a handful of boulders, not thousands


def diag_distance(a):
    """Axial distance from an azimuth to the nearer pixel diagonal (45 / 135), in [0, 45]."""
    from YOLOv8BeyondEarth.orientation import axial_distance
    return float(min(axial_distance(a, 45.0), axial_distance(a, 135.0)))


def main():
    from ultralytics import YOLO
    from YOLOv8BeyondEarth.orientation_validation import detect_segmenter_view

    OUT.mkdir(parents=True, exist_ok=True)
    import rasterio
    crops = sorted(DE.glob("hirise_rom*.tif"))[:N_TILES]
    if not crops:
        print("no HiRISE ROM crops found (data/solutionDE/hirise_rom*.tif)")
        return

    model = YOLO(str(WEIGHTS))
    recs, meta, t0 = [], None, time.time()
    for p in crops:
        with rasterio.open(p) as src:
            arr = src.read(1)
        r, m = detect_segmenter_view(model, arr, slice_size=SLICE, imgsz=IMGSZ, conf=CONF, device=0,
                                     max_records=MAX_PER_CROP)
        for x in r:
            x["crop"] = p.stem
        recs += r
        meta = m
        print(f"  {p.stem}: {len(r)} detections from {m['n_slices']} slices"
              + (f" (stopped at the {MAX_PER_CROP} cap)" if m["truncated"] else "")
              + f" ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{len(recs)} detections total | proto cells per native px = "
          f"{meta['proto_px_per_native_px']:.2f} (slice {meta['slice_size']} -> imgsz "
          f"{meta['imgsz']}, proto {meta['proto_res']:.0f})")

    cand = [r for r in recs
            if DPX_RANGE[0] <= r["dpx"] <= DPX_RANGE[1] and r["aspect"] > ASPECT_MIN
            and np.isfinite(r["angle"])]
    print(f"{len(cand)} candidates (dpx {DPX_RANGE[0]}-{DPX_RANGE[1]} px, aspect > {ASPECT_MIN})")
    if not cand:
        print("no candidates — widen DPX_RANGE")
        return

    for r in cand:
        r["diag_dist"] = diag_distance(r["angle"])
    # Spread the picks over size so the figure shows the effect fading, and within each size bucket
    # take the most diagonal-locked example (the failure mode being explained).
    edges = np.linspace(DPX_RANGE[0], DPX_RANGE[1], N_KEEP + 1)
    picked = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bucket = [r for r in cand if lo <= r["dpx"] < hi]
        if bucket:
            picked.append(min(bucket, key=lambda r: r["diag_dist"]))
    rng = np.random.default_rng(0)
    random_sample = [cand[i] for i in rng.choice(len(cand), min(N_KEEP, len(cand)), replace=False)]

    with open(OUT / "segmenter_view.pkl", "wb") as f:
        pickle.dump(dict(picked=picked, random=random_sample, meta=meta), f)
    stats = dict(
        meta=meta, n_detections=len(recs), n_candidates=len(cand),
        dpx_range=DPX_RANGE, aspect_min=ASPECT_MIN, crops=[p.stem for p in crops],
        picked=[dict(crop=r["crop"], dpx=round(r["dpx"], 2), angle=round(r["angle"], 1),
                     proto_angle=round(r["proto_angle"], 1), aspect=round(r["aspect"], 2),
                     proto_cells=r["proto_cells"], diag_dist=round(r["diag_dist"], 1))
                for r in picked],
        # The population-level number the figure's claim rests on: how tightly the proto-resolution
        # angle already agrees with the final reported angle. If they agree, everything downstream of
        # the prototype basis (upsample, threshold, contour, ellipse fit) is not the cause.
        proto_vs_final_median_abs_deg=round(float(np.median([
            abs(((r["angle"] - r["proto_angle"] + 90) % 180) - 90) for r in cand])), 2),
        proto_cells_median=int(np.median([r["proto_cells"] for r in cand])))
    (OUT / "segmenter_view_meta.json").write_text(json.dumps(stats, indent=2))
    print(f"\npicked {len(picked)} examples spanning "
          f"{min(r['dpx'] for r in picked):.1f}-{max(r['dpx'] for r in picked):.1f} px")
    print(f"median |final angle - proto angle| over all candidates: "
          f"{stats['proto_vs_final_median_abs_deg']:.2f} deg  "
          f"(small => the angle is already set at prototype resolution)")
    print(f"median proto cells above 0.5 per boulder: {stats['proto_cells_median']}")
    print(f"wrote {OUT / 'segmenter_view.pkl'}")


if __name__ == "__main__":
    main()
