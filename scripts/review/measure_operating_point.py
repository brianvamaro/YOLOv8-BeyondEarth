"""conf 0.10 or conf 0.25? The decision that gates the re-run (runbook step 0).

X-01 established that the requested confidence never reached ultralytics, so **every published
number was measured at 0.25 while being labelled 0.10**, and that a true 0.10 run yields +58.6% raw
detections whose median size is not just the sub-floor tail. That leaves two options and no way to
choose between them from the code: re-run everything at the declared 0.10 and accept ~1.6x the
detections, or re-declare the operating point as the 0.25 the science actually used.

The maintainer asked to measure first, through the **full** production path rather than on raw
detections — area gate, slice-edge filter, NMS and fusion all interact with the threshold — and to
report precision against human labels plus where the extra detections land in the SFD. That is what
this does:

``--mode tiles``
    The labelled half, and the one that decides it. A sample of **boulder2024 validation** tiles
    (not ``test``, which stays clean for confirmation — G3-11) through ``get_multiscale_prediction``
    at the ``goal3_production`` preset, at each confidence. Reports AP50, and precision/recall at
    the operating point, per size band, with COCO ignore semantics on both sides (G3-04), plus a
    paired bootstrap CI on the AP difference that resamples *images*.

``--mode crop``
    The unlabelled half: the tracked NAC dev crop at ``goal1_production``, compared with the
    one-to-one spatial matching G1-02 requires. Gives the "+X% detections" figure through the whole
    path, where ``measure_contract_impact.py`` gave it on raw model output.

**A caveat that belongs on the labelled numbers.** G5-04's filename audit found the boulder2024
splits share source prefixes, so validation tiles are not independent of training and the absolute
precision/recall here are optimistic. The *comparison* is unaffected — both operating points see
the same tiles — which is the only thing this measurement is asked to settle.

    conda run -n bouldernet python scripts/review/measure_operating_point.py
    conda run -n bouldernet python scripts/review/measure_operating_point.py --mode crop
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BOULDER2024 = REPO.parent / "data_prieur" / "boulder2024"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_batch_neutrality import compare_catalogs                      # noqa: E402
from YOLOv8BeyondEarth import presets                                      # noqa: E402
from YOLOv8BeyondEarth.ap_eval import (catalog_ap, equivalent_diameter,    # noqa: E402
                                       load_yolo_seg_labels, paired_ap_ci)
from YOLOv8BeyondEarth.benchmark import DEFAULT_SIZE_BANDS, _band_label    # noqa: E402
from YOLOv8BeyondEarth.matching import unique_centroid_match               # noqa: E402

#: 500 px tiles on a 512 canvas with a 1 m/px identity transform, so native px == map units and the
#: arms' slice grids stay exact. Same convention as the phase-4 AP runner it must be comparable to.
PAD, RES = 512, 1.0
CONFIDENCES = (0.10, 0.25)


def materialize_tile(png_path, tif_path):
    """500x500 PNG -> zero-padded 512x512 GeoTIFF. A 500 px "512 slice" would come back from
    ultralytics in a 512 frame with a 2.4% size error, so the canvas is not cosmetic."""
    import rasterio
    from rasterio.transform import from_origin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # NotGeoreferencedWarning on the source
        with rasterio.open(png_path) as src:
            img = src.read(1)
    h, w = img.shape
    assert h <= PAD and w <= PAD, f"{png_path.name}: {w}x{h} exceeds the {PAD} canvas"
    canvas = np.zeros((PAD, PAD), dtype=np.uint8)
    canvas[:h, :w] = img
    with rasterio.open(tif_path, "w", driver="GTiff", height=PAD, width=PAD, count=1,
                       dtype="uint8", crs="EPSG:32633",
                       transform=from_origin(0, PAD, RES, RES)) as dst:
        dst.write(canvas, 1)


def sample_tiles(split_dir, n, seed):
    """A seeded sample of tiles, drawn once and stable: the same tiles must feed both confidences."""
    pngs = sorted(split_dir.glob("images/*.png"))
    if n and n < len(pngs):
        idx = np.random.default_rng(seed).choice(len(pngs), size=n, replace=False)
        pngs = [pngs[i] for i in sorted(idx)]
    return pngs


def run_tiles(pngs, tile_dir, pred_dir, weights, conf, device):
    """Every tile through the fused production ladder at one confidence. Returns one catalog."""
    import geopandas as gpd
    import pandas as pd
    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth.predict import get_multiscale_prediction

    cached = pred_dir / f"catalog-conf{int(conf * 100):03d}.gpkg"
    if cached.exists():
        print(f"[skip] conf {conf}: reusing {cached.name}")
        return gpd.read_file(cached)

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(weights), device=device,
        image_size=presets.GOAL1_PRODUCTION["inference_size"], confidence_threshold=conf)

    frames, failures = [], []
    for k, png in enumerate(pngs):
        tif = tile_dir / f"{png.stem}.tif"
        if not tif.exists():
            materialize_tile(png, tif)
        try:
            fused, _, _ = get_multiscale_prediction(
                tif, model, output_dir=pred_dir / f"conf{int(conf * 100):03d}",
                output_format="gpkg", on_conflict="overwrite", confidence_threshold=conf)
        except Exception as exc:                       # keep going, but never silently
            failures.append((png.stem, repr(exc)))
            print(f"  [FAIL {len(failures)}] {png.stem}: {exc!r}", flush=True)
            continue
        # in-memory frames carry array-valued polygon/bbox columns that gpkg cannot serialize
        frames.append(gpd.GeoDataFrame(
            {"image_id": png.stem,
             "score": fused["score"].to_numpy() if len(fused) else []},
            geometry=fused.geometry.values, crs="EPSG:32633"))
        if (k + 1) % 25 == 0:
            print(f"  conf {conf}: [{k + 1}/{len(pngs)}]", flush=True)

    if failures:
        (pred_dir / f"failures-conf{int(conf * 100):03d}.json").write_text(json.dumps(failures))
        print(f"  conf {conf}: {len(failures)} tile failures recorded")
    if not frames:
        raise RuntimeError(f"every tile failed at conf {conf}; first: {failures[0] if failures else None}")
    catalog = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:32633")
    catalog.to_file(cached)
    return catalog


def per_image_diff(a, b, tol):
    """One-to-one match the two catalogs *within each image*, pooled. Counting is not comparing."""
    added = removed = matched = 0
    for image_id in sorted(set(a.image_id) | set(b.image_id)):
        sa, sb = a[a.image_id == image_id], b[b.image_id == image_id]
        if len(sa) == 0 or len(sb) == 0:
            added += len(sb)
            removed += len(sa)
            continue
        ca = np.c_[sa.geometry.centroid.x.to_numpy(), sa.geometry.centroid.y.to_numpy()]
        cb = np.c_[sb.geometry.centroid.x.to_numpy(), sb.geometry.centroid.y.to_numpy()]
        ia, _ = unique_centroid_match(ca, cb, tol)
        matched += len(ia)
        removed += len(sa) - len(ia)
        added += len(sb) - len(ia)
    return {"n_matched": int(matched), "n_added": int(added), "n_removed": int(removed)}


def band_table(catalog, gts):
    """AP50 and the operating-point precision/recall, overall and per size band."""
    rows = [{"band": "all", **{k: v for k, v in catalog_ap(catalog, gts, res=RES).items()
                               if k != "size_range"}}]
    for lo, hi in DEFAULT_SIZE_BANDS:
        entry = catalog_ap(catalog, gts, size_range=(lo, hi), res=RES)
        rows.append({"band": _band_label(lo, hi),
                     **{k: v for k, v in entry.items() if k != "size_range"}})
    return rows


def mode_tiles(args):
    split_dir = BOULDER2024 / args.split
    out_dir = Path(args.out) / f"operating_point_{args.split}"
    tile_dir, pred_dir = out_dir / "tiles", out_dir / "preds"
    for d in (tile_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)

    pngs = sample_tiles(split_dir, args.n_tiles, args.seed)
    print(f"{len(pngs)} {args.split} tiles (seed {args.seed})")

    catalogs = {c: run_tiles(pngs, tile_dir, pred_dir, args.weights, c, args.device)
                for c in CONFIDENCES}

    gts = {}
    for png in pngs:
        txt = split_dir / "labels" / f"{png.stem}.txt"
        gts[png.stem] = (load_yolo_seg_labels(txt, 500, 500,
                                              transform=__import__("rasterio").transform
                                              .from_origin(0, PAD, RES, RES))
                         if txt.exists() else [])

    result = {"split": args.split, "n_tiles": len(pngs), "seed": args.seed,
              "preset": "goal3_production", "res_m_per_px": RES,
              "n_truths": int(sum(len(v) for v in gts.values())),
              "caveat": ("boulder2024 splits share source prefixes (G5-04), so absolute "
                         "precision/recall are optimistic; the 0.10-vs-0.25 contrast is not"),
              "by_conf": {}}

    for conf, cat in catalogs.items():
        dpx = equivalent_diameter(cat.geometry.to_numpy(), RES)
        result["by_conf"][f"{conf:.2f}"] = {
            "n_detections": int(len(cat)),
            "median_dpx": float(np.median(dpx)) if len(cat) else None,
            "bands": band_table(cat, gts),
        }

    lo, hi = catalogs[0.10], catalogs[0.25]
    result["diff_010_vs_025"] = per_image_diff(hi, lo, tol=2.0 * RES)
    result["diff_010_vs_025"]["extra_fraction"] = round(
        result["diff_010_vs_025"]["n_added"] / max(len(hi), 1), 4)

    # where the extras land in the SFD: `score` is the detector's real confidence now (X-02), so
    # the sub-0.25 rows of the 0.10 catalog *are* the population 0.25 never saw
    sub = lo[lo.score < 0.25]
    d_sub = equivalent_diameter(sub.geometry.to_numpy(), RES)
    d_kept = equivalent_diameter(lo[lo.score >= 0.25].geometry.to_numpy(), RES)
    result["sfd_placement"] = {
        "n_below_025": int(len(sub)),
        "median_dpx_below_025": float(np.median(d_sub)) if len(sub) else None,
        "median_dpx_at_or_above_025": float(np.median(d_kept)) if len(d_kept) else None,
        "frac_below_025_under_5px": float(np.mean(d_sub < 5)) if len(sub) else None,
        "dpx_deciles_below_025": [round(float(x), 2)
                                  for x in np.percentile(d_sub, np.arange(10, 100, 10))]
        if len(sub) else None,
    }

    # is the AP difference bigger than image-to-image noise?
    ci = paired_ap_ci(hi, lo, gts, n_boot=args.n_boot, seed=args.seed, res=RES)
    result["paired_ap_ci_025_to_010"] = {k: (float(v) if np.isscalar(v) else v)
                                         for k, v in ci.items()}
    return result, out_dir


def mode_crop(args):
    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth.predict import predict_production

    out_dir = Path(args.out) / "operating_point_crop"
    out_dir.mkdir(parents=True, exist_ok=True)
    import rasterio
    res = abs(rasterio.open(args.raster).transform.a)

    catalogs = {}
    for conf in CONFIDENCES:
        model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(args.weights), device=args.device,
            image_size=presets.GOAL1_PRODUCTION["inference_size"], confidence_threshold=conf)
        _, gdf_nms = predict_production(args.raster, model,
                                        out_dir / f"conf{int(conf * 100):03d}",
                                        confidence_threshold=conf, on_conflict="overwrite")
        catalogs[conf] = gdf_nms
        print(f"conf {conf}: {len(gdf_nms)} boulders")

    cmp = compare_catalogs(catalogs[0.25], catalogs[0.10], 2.0 * res)
    lo = catalogs[0.10]
    d_sub = equivalent_diameter(lo[lo.score < 0.25].geometry.to_numpy(), res)
    d_kept = equivalent_diameter(lo[lo.score >= 0.25].geometry.to_numpy(), res)
    return {
        "raster": Path(args.raster).name, "preset": "goal1_production",
        "res_m_per_px": res,
        "n_at_010": int(len(catalogs[0.10])), "n_at_025": int(len(catalogs[0.25])),
        "extra_fraction": round((len(catalogs[0.10]) - len(catalogs[0.25]))
                                / max(len(catalogs[0.25]), 1), 4),
        "one_to_one": cmp,
        "median_dpx_below_025": float(np.median(d_sub)) if len(d_sub) else None,
        "median_dpx_at_or_above_025": float(np.median(d_kept)) if len(d_kept) else None,
    }, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=("tiles", "crop", "both"))
    ap.add_argument("--split", default="validation", choices=("validation", "test"))
    ap.add_argument("--n-tiles", type=int, default=250, help="0 for the whole split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--raster", default=str(REPO / "data" / "raster" / "crop.tif"))
    ap.add_argument("--weights",
                    default=str(REPO / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"))
    ap.add_argument("--out", default=str(REPO / "data" / "review"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    summary = {}
    if args.mode in ("crop", "both"):
        summary["crop"], _ = mode_crop(args)
        print(json.dumps(summary["crop"], indent=2), flush=True)
    if args.mode in ("tiles", "both"):
        summary["tiles"], _ = mode_tiles(args)
        print(json.dumps(summary["tiles"], indent=2), flush=True)

    path = Path(args.out) / "operating_point.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
