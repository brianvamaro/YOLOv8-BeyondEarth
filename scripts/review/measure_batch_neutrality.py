"""Is batched inference geometry-neutral? (review item G1-02)

Goal 1 demonstrated exact equality only at ``batch_size=1`` on one 1,616-object crop, then shipped
``batch_size=8`` and validated full scenes on near-equal *final counts*. Count equality is not
geometry equality: two catalogs can agree on how many boulders there are and disagree about where
they are, and the review's estimate was that batching moves roughly 1% of borderline detections.

This measures it properly on the current code: same raster, same preset, only ``batch_size``
differing, compared by **one-to-one spatial matching** rather than by counting. Reports unmatched
rows on both sides, score deltas, and how many matched pairs are byte-identical in geometry.

    conda run -n bouldernet python scripts/review/measure_batch_neutrality.py \\
        --raster data/raster/crop.tif
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from YOLOv8BeyondEarth.matching import unique_centroid_match     # noqa: E402


def compare_catalogs(a, b, tol_m):
    """One-to-one compare two catalogs of the same scene. Counting is not comparing."""
    ca = np.c_[a.geometry.centroid.x.to_numpy(), a.geometry.centroid.y.to_numpy()]
    cb = np.c_[b.geometry.centroid.x.to_numpy(), b.geometry.centroid.y.to_numpy()]
    ia, ib = unique_centroid_match(ca, cb, tol_m)

    ga = a.geometry.to_numpy()[ia]
    gb = b.geometry.to_numpy()[ib]
    identical = sum(1 for x, y in zip(ga, gb) if x.equals_exact(y, 0.0))
    area_a = np.array([g.area for g in ga])
    area_b = np.array([g.area for g in gb])
    with np.errstate(divide="ignore", invalid="ignore"):
        area_ratio = np.where(area_a > 0, area_b / area_a, np.nan)
    d_score = (b["score"].to_numpy()[ib] - a["score"].to_numpy()[ia]
               if "score" in a and "score" in b else np.zeros(len(ia)))
    return {
        "n_a": int(len(a)), "n_b": int(len(b)), "n_matched": int(len(ia)),
        "n_unmatched_a": int(len(a) - len(ia)), "n_unmatched_b": int(len(b) - len(ib)),
        "frac_unmatched": float((len(a) - len(ia)) / max(len(a), 1)),
        "n_geometry_identical": int(identical),
        "frac_geometry_identical": float(identical / max(len(ia), 1)),
        "area_ratio_median": float(np.nanmedian(area_ratio)) if len(ia) else None,
        "area_ratio_p99": float(np.nanpercentile(area_ratio, 99)) if len(ia) else None,
        "max_abs_score_delta": float(np.abs(d_score).max()) if len(ia) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", default=str(REPO / "data" / "raster" / "crop.tif"))
    ap.add_argument("--weights",
                    default=str(REPO / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"))
    ap.add_argument("--out", default=str(REPO / "data" / "review"))
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth import presets
    from YOLOv8BeyondEarth.predict import predict_production

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = abs(__import__("rasterio").open(args.raster).transform.a)
    tol_m = 2.0 * res                       # a 2 px centroid tolerance: same boulder, small shift

    catalogs = {}
    for bs in args.batches:
        model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=args.weights, device=args.device,
            image_size=presets.GOAL1_PRODUCTION["inference_size"],
            confidence_threshold=presets.GOAL1_PRODUCTION["confidence_threshold"])
        _, gdf_nms = predict_production(args.raster, model, out_dir / f"batch{bs}",
                                        batch_size=bs, on_conflict="overwrite")
        catalogs[bs] = gdf_nms
        print(f"batch_size={bs}: {len(gdf_nms)} boulders")

    ref = args.batches[0]
    summary = {"raster": Path(args.raster).name, "reference_batch": ref,
               "tolerance_m": tol_m, "comparisons": {}}
    for bs in args.batches[1:]:
        cmp = compare_catalogs(catalogs[ref], catalogs[bs], tol_m)
        summary["comparisons"][f"{ref}_vs_{bs}"] = cmp
        print(f"\nbatch {ref} vs {bs}:")
        print(json.dumps(cmp, indent=2))

    (out_dir / "batch_neutrality.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir / 'batch_neutrality.json'}")


if __name__ == "__main__":
    main()
