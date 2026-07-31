"""How much do review fixes X-01 and X-03 actually move real output? (2026-07-30)

Both fixes invalidate cached catalogs in principle. This measures by how much, on the tracked
NAC dev crop, so the re-run ledger can be prioritised instead of guessed.

X-01 — the requested confidence never reached ultralytics, so every catalog ran at 0.25.
       Measured: how many detections a true 0.10 run adds, and where they sit in size/score.
X-03 — the skimage tracer picked the contour with the most *vertices*, not the greatest area.
       Measured: how many masks have more than one contour at all, and in how many of those the
       two selection rules disagree (and by how much in area / equivalent diameter).
G1-06 — the area gate counted pixels *exactly* equal to 1 while the tracer works at the 0.5 level.
       This matters because ultralytics returns masks at ``imgsz`` (1024), not at ``slice_size``,
       so ``downscale_pred=True`` is a genuine 4x (slice 256) or 2x (slice 512) INTER_AREA
       *downscale* and the resulting masks really are fractional. Measured: how many detections
       the two gate conventions disagree about at the production threshold.

Sizes are reported in **native px** (mask px / (mask_size / slice_size)), not mask px.

Writes a JSON summary plus a per-detection CSV next to the crop.

    conda run -n bouldernet python scripts/review/measure_contract_impact.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from YOLOv8BeyondEarth.polygon import (  # noqa: E402
    MASK_SUPPORT_LEVEL, _enclosed_area, _mask_bounds, mask_support_area,
)


def _slice_masks(model, raster_png, slice_size, inference_size, conf, batch_size=8, limit=None,
                 downscale_pred=True):
    """Run the production slicing + batched model call, measuring each detection's mask."""
    import cv2
    import skimage
    from sahi.slicing import slice_image
    from YOLOv8BeyondEarth.predict import _to_bgr

    sliced = slice_image(image=str(raster_png), slice_height=slice_size, slice_width=slice_size,
                         overlap_height_ratio=0.2, overlap_width_ratio=0.2, out_ext=".png")
    images = list(sliced.images)
    if limit:
        images = images[:limit]

    rows = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        results = model.model([_to_bgr(im) for im in batch], imgsz=inference_size, conf=conf,
                              verbose=False, device=model.device)
        for res in results:
            if res.boxes.data.shape[0] == 0:
                continue
            scores = res.boxes.data[:, 4].cpu().numpy()
            masks = (res.masks.data >= 0.5).to(torch.float32).cpu().numpy()
            for score, raw in zip(scores, masks):
                # the production path for downscale_pred=True: INTER_AREA the imgsz-sized mask
                # down onto the slice grid, which is what makes it fractional
                mask = (cv2.resize(raw, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
                        if downscale_pred and raw.shape[0] != slice_size else raw)
                px_per_native = mask.shape[0] / slice_size
                bounds = _mask_bounds(mask)
                if bounds is None:
                    continue
                y0, y1, x0, x1 = bounds
                padded = np.pad(mask[y0:y1, x0:x1], 1, constant_values=0)
                contours = skimage.measure.find_contours(padded, MASK_SUPPORT_LEVEL)
                if not contours:
                    continue
                by_len = max(contours, key=len)
                by_area = max(contours, key=_enclosed_area)
                a_len, a_area = _enclosed_area(by_len), _enclosed_area(by_area)
                scale2 = px_per_native ** 2
                rows.append(dict(
                    score=float(score),
                    # G1-06: the two gate conventions, in native px
                    support_ge05=mask_support_area(mask) / scale2,
                    support_eq1=float(np.count_nonzero(mask == 1)) / scale2,
                    n_fractional=int(((mask > 0) & (mask < 1)).sum()),
                    n_contours=len(contours),
                    differs=bool(by_len is not by_area),
                    dpx_by_len=float(np.sqrt(4 * a_len / np.pi)) / px_per_native if a_len > 0 else 0.0,
                    dpx_by_area=float(np.sqrt(4 * a_area / np.pi)) / px_per_native if a_area > 0 else 0.0,
                ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", default=str(REPO / "data" / "raster" / "crop.tif"))
    ap.add_argument("--weights",
                    default=str(REPO / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"))
    ap.add_argument("--out", default=str(REPO / "data" / "review"))
    ap.add_argument("--slice-size", type=int, default=256)
    ap.add_argument("--inference-size", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="cap the number of slices (smoke run)")
    ap.add_argument("--area-gate", type=int, default=6, help="min_area_threshold under test")
    args = ap.parse_args()

    from sahi import AutoDetectionModel
    from rastertools_BOULDERING import convert as raster_convert

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raster = Path(args.raster)
    png = raster.with_suffix(".png")
    if not png.exists():
        raster_convert.tiff_to_png(raster, png)

    summary = {"raster": raster.name, "slice_size": args.slice_size,
               "inference_size": args.inference_size}

    # --- X-01: what a true 0.10 run adds over the 0.25 that actually ran ---------------------
    per_conf = {}
    for conf in (0.10, 0.25):
        model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=args.weights, device="cuda:0",
            image_size=args.inference_size, confidence_threshold=conf)
        df = _slice_masks(model, png, args.slice_size, args.inference_size, conf,
                          limit=args.limit)
        per_conf[conf] = df
        df.to_csv(out_dir / f"contract_impact_conf{int(conf * 100):03d}.csv", index=False)

    d10, d25 = per_conf[0.10], per_conf[0.25]
    added = d10[d10.score < 0.25]
    summary["x01"] = {
        "n_at_conf_010": int(len(d10)),
        "n_at_conf_025": int(len(d25)),
        "extra_detections": int(len(d10) - len(d25)),
        "extra_fraction": float((len(d10) - len(d25)) / max(len(d25), 1)),
        "added_median_dpx_native": float(added.dpx_by_area.median()) if len(added) else None,
        "kept_median_dpx_native": float(d25.dpx_by_area.median()) if len(d25) else None,
    }

    # --- G1-06: does the area-gate convention change what survives? --------------------------
    gate = int(args.area_gate)
    ref = d10
    keep_old = ref.support_eq1 > gate      # pixels exactly == 1 (the old gate)
    keep_new = ref.support_ge05 > gate     # pixels at or above the 0.5 level the tracer uses
    summary["g1_06"] = {
        "area_gate_px": gate,
        "downscale_pred": True,
        "fraction_of_masks_with_fractional_px": float((ref.n_fractional > 0).mean()),
        "n_kept_old_gate": int(keep_old.sum()),
        "n_kept_new_gate": int(keep_new.sum()),
        "n_rescued_by_new_gate": int((keep_new & ~keep_old).sum()),
        "n_dropped_by_new_gate": int((keep_old & ~keep_new).sum()),
        "rescued_fraction": float((keep_new & ~keep_old).sum() / max(int(keep_old.sum()), 1)),
        "rescued_median_dpx_native": (float(ref.loc[keep_new & ~keep_old, "dpx_by_area"].median())
                                      if int((keep_new & ~keep_old).sum()) else None),
    }

    # --- X-03: how often does contour selection change the emitted polygon? ------------------
    ref = d10
    multi = ref[ref.n_contours > 1]
    diff = ref[ref.differs]
    with np.errstate(divide="ignore", invalid="ignore"):
        dpx_ratio = (diff.dpx_by_len / diff.dpx_by_area).replace([np.inf, -np.inf], np.nan)
    summary["x03"] = {
        "n_detections": int(len(ref)),
        "n_multi_contour": int(len(multi)),
        "multi_contour_fraction": float(len(multi) / max(len(ref), 1)),
        "n_selection_differs": int(len(diff)),
        "selection_differs_fraction": float(len(diff) / max(len(ref), 1)),
        "median_dpx_ratio_old_over_new": float(dpx_ratio.median()) if len(diff) else None,
        "min_dpx_ratio_old_over_new": float(dpx_ratio.min()) if len(diff) else None,
        "n_worse_than_10pct": int((dpx_ratio < 0.9).sum()) if len(diff) else 0,
    }

    (out_dir / "contract_impact.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
