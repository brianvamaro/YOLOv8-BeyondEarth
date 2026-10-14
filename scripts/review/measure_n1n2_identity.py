"""Do levers N1 and N2 change any detection? Old code against new, on real ground with fill.

N2 (windowed reads; streamed footprint hull) claims identity by construction and N1 (skip slices that
are entirely fill) claims it by measurement -- the weights return nothing on an all-zero slice. The
unit tests check both claims piece by piece against the old code; this checks the whole pipeline with
the real model on a real crop, so nothing between the pieces is assumed.

Three variants per arm, each in its own interpreter so the two package versions cannot mix:

- ``old``      -- the pre-change package (``--old-src``, e.g. ``git archive <rev> src``): PNG hop,
                  whole-scene sahi slicing, ``true_footprint``;
- ``new_all``  -- this package with ``skip_fill_slices=False``: N2 alone. Batches are composed exactly
                  as before, so this must be **byte-identical** to ``old``;
- ``new_skip`` -- this package as shipped: N2 + N1. Kept slices are re-batched, so the final partial
                  batch can change size, which G1-02 measured moving ~0.7% of polygons sub-perceptibly
                  at batch 1 vs 8. Reported, not assumed.

Comparison is G1-02's standard (``compare_catalogs``: one-to-one matching, unmatched rows both sides,
exact polygon equality, score deltas) plus a whole-catalog byte check over every column.

    conda run -n bouldernet python scripts/review/measure_n1n2_identity.py \\
        --raster <crop.tif> --old-src <dir>/src
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO = Path(__file__).resolve().parents[2]
VARIANTS = ("old", "new_all", "new_skip")


def _peak_rss_bytes():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", 0) or 0) or None
    except Exception:
        pass
    try:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return None


def worker(args):
    src = Path(args.src).resolve()
    sys.path.insert(0, str(src))
    import torch
    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth import predict as P, presets
    assert Path(P.__file__).resolve().is_relative_to(src), f"imported {P.__file__}, not {src}"
    slice_size, sr = (int(x) for x in (args.arm.split(":") + ["1"])[:2])
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=args.weights, device=args.device,
        image_size=1024, confidence_threshold=0.10)
    cfg = {k: v for k, v in presets.GOAL1_PRODUCTION.items()
           if k not in ("slice_size", "sr_factor", "confidence_threshold", "inference_size",
                        "save_bbox", "save_prenms", "output_format", "max_det", "batch_size")}
    cfg.update(max_det=args.max_det, batch_size=args.batch_size)
    if args.variant != "old":
        cfg["skip_fill_slices"] = args.variant == "new_skip"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    gdf, nms = P.get_sliced_prediction(
        args.raster, model, confidence_threshold=0.10, output_dir=args.out_dir,
        slice_size=slice_size, sr_factor=sr, inference_size=1024, save_bbox=False,
        save_prenms=False, save_nms=True, output_format="gpkg", on_conflict="overwrite",
        on_max_det="warn", **cfg)
    rec = {"variant": args.variant, "arm": args.arm, "src": str(src),
           "wall_seconds": round(time.perf_counter() - t0, 2), "n_nms": int(len(nms)),
           "slices": gdf.attrs.get("slices"), "truncation": gdf.attrs.get("truncation", {}),
           "peak_rss_bytes": _peak_rss_bytes(),
           "peak_cuda_bytes": int(torch.cuda.max_memory_allocated())}
    print("RESULT " + json.dumps(rec), flush=True)


def _compare_catalogs():
    spec = importlib.util.spec_from_file_location(
        "measure_batch_neutrality", Path(__file__).with_name("measure_batch_neutrality.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compare_catalogs


def _rows(gdf):
    import shapely
    cols = [c for c in gdf.columns if c not in ("geometry", "id")]
    return sorted(zip(shapely.to_wkb(gdf.geometry.values), *(gdf[c].tolist() for c in cols)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raster", required=True)
    ap.add_argument("--old-src", required=True, help="the pre-change package's src directory")
    ap.add_argument("--out", default=str(REPO / "data" / "review" / "n1n2_identity"),
                    help="the per-variant layers go here (gitignored)")
    ap.add_argument("--report", default=str(REPO / "data" / "review" / "n1n2_identity.json"),
                    help="the summary (tracked)")
    ap.add_argument("--arms", nargs="+", default=["256", "512", "256:2"])
    ap.add_argument("--weights",
                    default=str(REPO / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"))
    ap.add_argument("--max-det", type=int, default=1100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--batch-size-512", type=int, default=2,
                    help="the 512 arm spills an 8 GiB card at batch 8 on dense ground")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    ap.add_argument("--arm", help=argparse.SUPPRESS)
    ap.add_argument("--src", help=argparse.SUPPRESS)
    ap.add_argument("--out-dir", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.variant:
        return worker(args)

    import geopandas as gpd
    import rasterio
    compare = _compare_catalogs()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.raster) as src:
        tol_m = 2.0 * abs(src.transform.a)
    srcs = {"old": Path(args.old_src), "new_all": REPO / "src", "new_skip": REPO / "src"}
    report = {"raster": str(args.raster), "old_src": str(args.old_src), "max_det": args.max_det,
              "runs": [], "comparisons": {}}
    for arm in args.arms:
        bs = args.batch_size_512 if arm.split(":")[0] == "512" else args.batch_size
        layers = {}
        for v in VARIANTS:
            d = out / arm.replace(":", "sr") / v
            cmd = [sys.executable, __file__, "--raster", args.raster, "--old-src", args.old_src,
                   "--variant", v, "--arm", arm, "--src", str(srcs[v]), "--out-dir", str(d),
                   "--weights", args.weights, "--max-det", str(args.max_det),
                   "--batch-size", str(bs), "--device", args.device]
            res = subprocess.run(cmd, capture_output=True, text=True)
            line = [x for x in res.stdout.splitlines() if x.startswith("RESULT ")]
            if res.returncode or not line:
                raise SystemExit(f"{v} {arm} failed:\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
            rec = json.loads(line[-1][len("RESULT "):])
            rec["batch_size"] = bs
            report["runs"].append(rec)
            layers[v] = gpd.read_file(next(d.glob("*-nms.gpkg")))
            print(f"{arm:6s} {v:9s} {rec['n_nms']:6d} det  {rec['wall_seconds']:7.1f} s  "
                  f"slices {rec['slices']}  peak RSS {(rec['peak_rss_bytes'] or 0) / 2**30:.2f} GiB",
                  flush=True)
        for a, b in (("old", "new_all"), ("new_all", "new_skip"), ("old", "new_skip")):
            c = compare(layers[a], layers[b], tol_m)
            c["byte_identical_catalog"] = _rows(layers[a]) == _rows(layers[b])
            report["comparisons"][f"{arm}:{a}_vs_{b}"] = c
            print(f"   {a} vs {b}: byte-identical {c['byte_identical_catalog']}, "
                  f"unmatched {c['n_unmatched_a']}/{c['n_unmatched_b']}, geometry identical "
                  f"{c['n_geometry_identical']}/{c['n_matched']}, max |dscore| "
                  f"{c['max_abs_score_delta']:.2e}", flush=True)
    Path(args.report).write_text(json.dumps(report, indent=2, default=str))
    print(f"\n-> {args.report}")


if __name__ == "__main__":
    main()
