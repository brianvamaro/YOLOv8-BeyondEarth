"""How many SAM2 image encodes does a scene actually cost? (verifies a cost-model plan claim)

The 2026-08-02/03 optimisation research concluded:

    "Encoder reuse is already achieved. `sam_masks_in_order` encodes ceil(n_boxes/sam_batch) times
     per slice; at production density (~35-37 detections per 256 px slice, `sam_batch=96`) that is
     already exactly 1 encode. A native encode-once/decode-many path saves zero on real scenes.
     Strike this from the candidate list."

That was single-source and its verifier never ran. The premise is a *mean* density, but the cost is a
**sum over slices**, and encodes are a ceiling function — so the mean cannot settle it. Dense slices
each pay several encodes and the mean hides them.

This computes the real multiplier from a saved per-arm catalog: place every detection's centroid on
the arm's own slice grid, then sum `ceil(n / sam_batch)` over occupied slices and compare against
"one encode per occupied slice".

Two corrections are applied, because the saved catalogs understate the load twice:

* **X-01** — every saved catalog ran at an *effective* conf 0.25 while being labelled 0.10. Measured
  on the 16 densest 256-arm slices (`probe_max_det.py`, 2026-08-04): mean 317.4 detections at conf
  0.10 against 217.8 at 0.25, i.e. **1.457x**.
* the counts are **post-NMS**, so cross-slice duplicates are already gone and partly-overlapping
  objects are uncounted.

Both push the same way, so every number here is a **lower bound** on the true encode count.

    conda run -n bouldernet python scripts/review/check_sam_encode_multiplier.py \
        --catalog data/solutionDE/fullscene/hirise/ESP_017355_2260_RED-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg \
        --raster data/mars_hirise/raster/ESP_017355_2260_RED.tif --slice-size 256
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

from check_max_det_binding import per_slice_counts   # noqa: E402

#: Measured conf-0.25 -> conf-0.10 detection ratio on the densest 256-arm slices (probe_max_det.py).
CONF_010_OVER_025 = 317.4 / 217.8


def multiplier(counts, sam_batch):
    """``(encodes, occupied_slices, multiplier)`` for one per-slice count array."""
    occupied = counts[counts > 0]
    if not len(occupied):
        return 0, 0, 0.0
    encodes = int(np.ceil(occupied / sam_batch).sum())
    return encodes, int(len(occupied)), encodes / len(occupied)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--raster", required=True)
    ap.add_argument("--slice-size", type=int, required=True)
    ap.add_argument("--sr-factor", type=int, default=1)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--sam-batch", type=int, default=96,
                    help="sam2_remasker's chunk size; the plan's claim is specific to 96")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import rasterio
    from pyogrio import read_bounds

    _, bounds = read_bounds(args.catalog)
    cx = 0.5 * (bounds[0] + bounds[2])
    cy = 0.5 * (bounds[1] + bounds[3])
    with rasterio.open(args.raster) as src:
        inv = ~src.transform
        width, height = src.width, src.height
    px, py = inv * (cx, cy)

    native_slice = args.slice_size // args.sr_factor
    counts, _, _ = per_slice_counts(np.asarray(px, float), np.asarray(py, float),
                                    width, height, native_slice, args.overlap)
    flat = counts.ravel().astype(np.float64)

    record = {"catalog": Path(args.catalog).name, "slice_size": args.slice_size,
              "sr_factor": args.sr_factor, "sam_batch": args.sam_batch,
              "n_detections": int(len(px)), "n_slices": int(flat.size),
              "claim": "at production density, ceil(n_boxes/sam_batch) is already exactly 1 encode",
              "caveat": ("post-NMS centroid counts are a lower bound on what the model returns per "
                         "slice, so every multiplier below is a lower bound"),
              "cases": {}}

    for name, scaled in (("as_saved_effective_conf_0.25", flat),
                         ("corrected_to_conf_0.10", flat * CONF_010_OVER_025)):
        encodes, occupied, mult = multiplier(scaled, args.sam_batch)
        over = scaled[scaled > args.sam_batch]
        record["cases"][name] = {
            "encodes": encodes, "occupied_slices": occupied,
            "encodes_per_occupied_slice": round(mult, 3),
            "slices_over_sam_batch": int(len(over)),
            "fraction_of_occupied_over_sam_batch": round(len(over) / occupied, 4) if occupied else 0,
            "mean_per_occupied": round(float(scaled[scaled > 0].mean()), 1) if occupied else 0,
            "max_per_slice": round(float(scaled.max()), 1),
            "encodes_above_one_each": encodes - occupied,
        }
        print(f"{name:<32} encodes={encodes:>9,}  occupied={occupied:>7,}  "
              f"x{mult:.3f} per slice   over-batch slices={len(over):>6,} "
              f"({len(over)/occupied*100 if occupied else 0:5.1f}%)  mean={scaled[scaled>0].mean():.1f}")

    a = record["cases"]["as_saved_effective_conf_0.25"]
    b = record["cases"]["corrected_to_conf_0.10"]
    record["verdict"] = {
        "claim_holds_at_effective_conf_0.25": a["encodes_per_occupied_slice"] < 1.05,
        "claim_holds_at_declared_conf_0.10": b["encodes_per_occupied_slice"] < 1.05,
        "wasted_encodes_at_conf_0.10": b["encodes_above_one_each"],
        "summary": (
            f"{b['encodes_per_occupied_slice']:.2f} encodes per occupied slice at the declared conf "
            f"0.10 ({b['fraction_of_occupied_over_sam_batch']*100:.1f}% of occupied slices exceed "
            f"sam_batch={args.sam_batch}), so an encode-once/decode-many path would remove "
            f"{b['encodes_above_one_each']:,} redundant image encodes on this scene — not zero."),
    }
    print("\n" + json.dumps(record["verdict"], indent=2))

    out = Path(args.out) if args.out else (
        REPO / "data" / "review" /
        f"sam_encode_multiplier-ss{args.slice_size}sr{args.sr_factor}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
