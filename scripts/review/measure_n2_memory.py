"""Lever N2's RAM claim, measured: bytes per scene pixel for each whole-raster step, old and new.

The 2026-10-14 review estimated, from reading the code, that the old prediction path held the scene
three times over: the TIFF->PNG copy (``tiff_to_png``), sahi's decoded RGB scene (``slice_image``;
every slice is a view into it, so it stays resident for the whole run), and
``rastertools.raster.true_footprint`` (an int64 copy of the validity mask). This measures each step,
and its replacement, in a fresh interpreter so one step's arena cannot inflate the next, and reports
peak resident memory above an import-only baseline, per scene pixel. Peak RSS of a whole-raster step
is linear in pixel count, so the coefficient carries to a 7 Gpx scene; the RAM a run needs is the
largest resident set at any moment, which for the old path was the RGB scene plus the footprint step.

    conda run -n bouldernet python scripts/review/measure_n2_memory.py --raster <8-bit tif>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STEPS = ("baseline", "old_png", "old_slices", "old_footprint", "new_slices", "new_footprint")


def _peak():
    try:
        import psutil
        info = psutil.Process().memory_info()
        if getattr(info, "peak_wset", 0):
            return int(info.peak_wset)
    except Exception:
        pass
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def step(name, raster, tmp):
    sys.path.insert(0, str(REPO / "src"))
    from rastertools_BOULDERING import convert, raster as rt_raster
    from sahi.slicing import slice_image
    from YOLOv8BeyondEarth import coverage, predict
    tif, png = Path(raster), Path(tmp) / "scene.png"
    t0 = time.perf_counter()
    keep = None
    if name == "old_png":
        convert.tiff_to_png(tif, png)                       # writes beside the raster's name...
        produced = tif.with_name(tif.name.split(".tif")[0] + ".png")
        if produced.exists():
            produced.replace(png)
    elif name == "old_slices":
        keep = slice_image(image=str(png), slice_height=256, slice_width=256,
                           overlap_height_ratio=0.2, overlap_width_ratio=0.2)
    elif name == "old_footprint":
        keep = rt_raster.true_footprint(tif, Path(tmp) / "fp.shp")
    elif name == "new_slices":
        s = predict._WindowedSlices(tif, 256, 0.2, 0.2)
        for i in range(len(s)):
            keep = s.images[i]
        s.close()
    elif name == "new_footprint":
        keep = coverage.valid_convex_hull(tif)
    print("RESULT " + json.dumps({"step": name, "peak_bytes": _peak(),
                                  "seconds": round(time.perf_counter() - t0, 1)}), flush=True)
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raster", required=True)
    ap.add_argument("--report", default=str(REPO / "data" / "review" / "n2_memory.json"))
    ap.add_argument("--step", choices=STEPS, help=argparse.SUPPRESS)
    ap.add_argument("--tmp", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.step:
        step(args.step, args.raster, args.tmp)
        return

    import rasterio
    with rasterio.open(args.raster) as src:
        n_px = src.width * src.height
    out = {"raster": str(args.raster), "pixels": n_px, "steps": {}}
    with tempfile.TemporaryDirectory(prefix="n2mem-") as tmp:
        for name in STEPS:
            res = subprocess.run([sys.executable, __file__, "--raster", args.raster, "--step", name,
                                  "--tmp", tmp], capture_output=True, text=True)
            line = [x for x in res.stdout.splitlines() if x.startswith("RESULT ")]
            if res.returncode or not line:
                raise SystemExit(f"{name} failed:\n{res.stderr[-3000:]}")
            out["steps"][name] = json.loads(line[-1][7:])
    base = out["steps"]["baseline"]["peak_bytes"]
    for name, rec in out["steps"].items():
        rec["above_baseline_bytes"] = rec["peak_bytes"] - base
        rec["bytes_per_pixel"] = round(rec["above_baseline_bytes"] / n_px, 3)
        print(f"{name:14s} {rec['above_baseline_bytes'] / 2**30:7.2f} GiB above baseline  "
              f"{rec['bytes_per_pixel']:6.2f} B/px  {rec['seconds']:6.1f} s")
    Path(args.report).write_text(json.dumps(out, indent=2))
    print(f"-> {args.report}")


if __name__ == "__main__":
    main()
