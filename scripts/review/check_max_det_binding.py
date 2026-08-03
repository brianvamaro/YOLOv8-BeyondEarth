"""Does ultralytics' default ``max_det=300`` truncate any slice? (found in the 2026-08-02 review)

``predict.py`` never passes ``max_det`` to the model, so ultralytics' ``cfg/default.yaml`` value of
**300** silently caps the detections returned for every slice. If any slice hits it, the catalog is
missing boulders with no error, no warning and no record — a completeness bug, not a performance
one, and it would bite hardest exactly where the terrain is densest.

Rather than re-run a scene to find out, this reconstructs the per-slice load from a **saved**
catalog: every detection's centroid is placed on the arm's own slice grid and counted per slice.

Read the number it prints as a **lower bound**, for two compounding reasons:

* the saved layer is post-NMS, so every cross-slice duplicate the model emitted has already been
  removed — the model returned more per slice than this counts;
* a slice also emits detections for objects only partly inside it, which a centroid test misses.

So "max observed 250" does not clear the cap, but "max observed 300+" would prove it binds. The
headroom this reports is the honest quantity: how close the densest slice gets, knowing the true
count is higher.

    conda run -n bouldernet python scripts/review/check_max_det_binding.py \
        --catalog data/solutionDE/fullscene/hirise/ESP_017355_2260_RED-predictions-ct-010-ss-512-is-1024-ov-020-downscaled-mask-nms.gpkg \
        --raster data/mars_hirise/raster/ESP_017355_2260_RED.tif --slice-size 512
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

#: ultralytics `cfg/default.yaml`. Not overridden anywhere in `predict.py`.
ULTRALYTICS_DEFAULT_MAX_DET = 300


def slice_starts(extent, slice_size, overlap_ratio):
    """SAHI's 1-D slice origins: step by ``slice_size - int(overlap*slice_size)``, clamped at the end."""
    overlap = int(overlap_ratio * slice_size)
    step = slice_size - overlap
    starts = list(range(0, max(extent - slice_size, 0) + 1, step))
    if not starts or starts[-1] + slice_size < extent:
        starts.append(max(extent - slice_size, 0))
    return np.array(sorted(set(starts)), dtype=np.int64)


def per_slice_counts(px, py, width, height, slice_size, overlap_ratio):
    """Detections per slice, exactly, by banding on y then searching x within each band."""
    xs = slice_starts(width, slice_size, overlap_ratio)
    ys = slice_starts(height, slice_size, overlap_ratio)

    order = np.argsort(py, kind="stable")
    py_sorted, px_by_y = py[order], px[order]

    counts = np.zeros((len(ys), len(xs)), dtype=np.int32)
    for i, y0 in enumerate(ys):
        lo = np.searchsorted(py_sorted, y0, side="left")
        hi = np.searchsorted(py_sorted, y0 + slice_size, side="left")
        if hi <= lo:
            continue
        band = np.sort(px_by_y[lo:hi], kind="stable")
        left = np.searchsorted(band, xs, side="left")
        right = np.searchsorted(band, xs + slice_size, side="left")
        counts[i] = right - left
    return counts, len(xs), len(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--raster", required=True)
    ap.add_argument("--slice-size", type=int, required=True)
    ap.add_argument("--sr-factor", type=int, default=1,
                    help="the arm's SR factor; the native slice is slice_size // sr_factor")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import rasterio
    from pyogrio import read_bounds

    # bounds only: no geometry is materialised, so a 1.6 GB layer costs a few hundred MB
    _, bounds = read_bounds(args.catalog)
    cx = 0.5 * (bounds[0] + bounds[2])
    cy = 0.5 * (bounds[1] + bounds[3])

    with rasterio.open(args.raster) as src:
        inv = ~src.transform
        width, height = src.width, src.height
    px, py = inv * (cx, cy)
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)

    native_slice = args.slice_size // args.sr_factor
    counts, nx, ny = per_slice_counts(px, py, width, height, native_slice, args.overlap)
    flat = counts.ravel()
    occupied = flat[flat > 0]

    record = {
        "catalog": Path(args.catalog).name,
        "raster": Path(args.raster).name,
        "raster_px": [width, height],
        "slice_size": args.slice_size,
        "sr_factor": args.sr_factor,
        "native_slice": native_slice,
        "n_detections": int(len(px)),
        "n_slices": int(nx * ny),
        "n_slices_with_detections": int((flat > 0).sum()),
        "mean_per_occupied_slice": round(float(occupied.mean()), 1) if len(occupied) else 0.0,
        "p50": int(np.percentile(occupied, 50)) if len(occupied) else 0,
        "p99": int(np.percentile(occupied, 99)) if len(occupied) else 0,
        "p999": int(np.percentile(occupied, 99.9)) if len(occupied) else 0,
        "max_observed": int(flat.max()),
        "ultralytics_default_max_det": ULTRALYTICS_DEFAULT_MAX_DET,
        "headroom_at_max": ULTRALYTICS_DEFAULT_MAX_DET - int(flat.max()),
        "n_slices_over_200": int((flat > 200).sum()),
        "n_slices_over_250": int((flat > 250).sum()),
        "n_slices_at_or_over_300": int((flat >= ULTRALYTICS_DEFAULT_MAX_DET).sum()),
        "note": ("post-NMS centroid counts: a LOWER BOUND on what the model emitted per slice, "
                 "since cross-slice duplicates are already removed and partly-overlapping objects "
                 "are not counted"),
    }
    print(json.dumps(record, indent=2))
    out = Path(args.out) if args.out else REPO / "data" / "review" / (
        f"max_det_binding-{Path(args.catalog).stem[:40]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
