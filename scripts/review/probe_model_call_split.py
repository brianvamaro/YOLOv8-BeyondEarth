"""Split the ultralytics model call into preprocess / inference / postprocess (2026-08-04 profile).

The AC-power per-stage profile put **63.8%** of the production 256 arm's wall inside
``detection_model.model(...)``. That timer wraps the call in its entirety, so it is *not* a GPU
measurement — it bundles ultralytics' CPU preprocessing (letterbox, normalise, host->device), the GPU
forward, and postprocessing (NMS, ``process_mask``, ``Results`` construction). Calling it "GPU-bound"
was an overclaim and is withdrawn; this settles it instead.

The split decides which lever is next, three different ways:

* **inference dominates** -> genuine GPU compute. Levers are FLOP/resolution reductions (a smaller
  encoder, a smaller ``imgsz``), and the ~88%-not-arithmetic finding says expect little.
* **postprocess dominates** -> that is where ``ops.process_mask`` materialises masks at ``imgsz``,
  so **retina_masks=True** (masks at ``slice_size``) is the direct fix and returns to the top of the
  list on throughput grounds, not only for the VRAM wall.
* **preprocess dominates** -> CPU framework overhead. The lever is batching or bypassing it, and
  neither a faster card nor ``retina_masks`` helps.

No instrumentation needed: ultralytics already records all three per call in ``predictor.profilers``
and attaches them to every ``Results`` as ``.speed`` (ms **per image**, i.e. ``dt / batch_size``, so
summing over all results recovers the true totals). Those profilers call
``torch.cuda.synchronize()``, so the numbers are properly synchronised rather than async-launch
artifacts.

    conda run -n bouldernet python scripts/review/probe_model_call_split.py --slice-size 256
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", default=str(REPO / "data" / "raster" / "crop.tif"))
    ap.add_argument("--weights", default=str(REPO / "data" / "weights"
                                             / "yolov8-m-boulder-detection-tmp.pt"))
    ap.add_argument("--slice-sizes", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--inference-size", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--max-det", type=int, default=None, help="default: the production preset")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3, help="repetitions after a discarded warm-up")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from sahi.slicing import slice_image
    from ultralytics import YOLO

    from YOLOv8BeyondEarth import presets
    from YOLOv8BeyondEarth.predict import _ensure_png, _to_bgr

    max_det = args.max_det if args.max_det is not None else presets.GOAL1_PRODUCTION["max_det"]
    raster = Path(args.raster)
    png = _ensure_png(raster, presets.file_identity(raster))
    model = YOLO(args.weights, task="segment")

    record = {"raster": raster.name, "inference_size": args.inference_size, "conf": args.conf,
              "max_det": max_det, "batch_size": args.batch_size, "reps": args.reps,
              "note": ("speed values are ms per image (dt/batch), so summing over all results "
                       "recovers per-run totals; ultralytics' profilers synchronise CUDA"),
              "arms": {}}

    for ss in args.slice_sizes:
        sl = slice_image(image=png.as_posix(), slice_height=ss, slice_width=ss,
                         overlap_height_ratio=0.2, overlap_width_ratio=0.2, out_ext=".png")
        images = list(sl.images)
        batches = [images[i:i + args.batch_size]
                   for i in range(0, len(images), args.batch_size)]

        def one_pass():
            totals = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
            n_det = 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for batch in batches:
                res = model([_to_bgr(im) for im in batch], imgsz=args.inference_size,
                            conf=args.conf, max_det=max_det, verbose=False, device=args.device)
                for r in res:
                    for k in totals:
                        totals[k] += float(r.speed[k]) / 1000.0      # ms -> s
                    n_det += 0 if r.boxes is None else int(r.boxes.data.shape[0])
                del res
            torch.cuda.synchronize()
            return totals, time.perf_counter() - t0, n_det

        one_pass()                                                   # warm-up, discarded
        runs = [one_pass() for _ in range(args.reps)]
        walls = sorted(r[1] for r in runs)
        median_i = len(runs) // 2
        totals, wall, n_det = sorted(runs, key=lambda r: r[1])[median_i]
        accounted = sum(totals.values())

        arm = {
            "n_slices": len(images), "n_batches": len(batches), "n_detections": n_det,
            "model_call_wall_s": round(wall, 3),
            "wall_range_s": [round(walls[0], 3), round(walls[-1], 3)],
            "wall_spread_pct": round(100 * (walls[-1] - walls[0]) / walls[0], 1),
            "seconds": {k: round(v, 3) for k, v in totals.items()},
            "share_of_model_call": {k: round(v / wall, 3) for k, v in totals.items()},
            "accounted_s": round(accounted, 3),
            "unaccounted_s": round(wall - accounted, 3),
            "unaccounted_share": round((wall - accounted) / wall, 3),
            "cpu_framework_share": round((totals["preprocess"] + totals["postprocess"]) / wall, 3),
        }
        record["arms"][str(ss)] = arm
        print(f"\n=== slice {ss}  ({len(images)} slices, {n_det} detections) ===")
        print(f"  model-call wall (median of {args.reps}) {wall:.3f} s   "
              f"range {walls[0]:.3f}-{walls[-1]:.3f} ({arm['wall_spread_pct']}% spread)")
        for k in ("preprocess", "inference", "postprocess"):
            print(f"    {k:<12} {totals[k]:>7.3f} s  {100*totals[k]/wall:>5.1f}%")
        print(f"    {'unaccounted':<12} {arm['unaccounted_s']:>7.3f} s  "
              f"{100*arm['unaccounted_share']:>5.1f}%")
        print(f"  -> CPU framework (pre+post) {100*arm['cpu_framework_share']:.1f}% of the model call")

    out = Path(args.out) if args.out else REPO / "data" / "review" / "model_call_split.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
