"""Where does a production run actually spend its time? (re-run runbook, pre-run optimisation pass)

Goal 1's profile predates three things that changed the per-detection path, so the runbook asks
for a fresh one before the long re-run starts:

* **G3-10's mask-size discovery.** Ultralytics returns ``results.masks.data`` at ``imgsz``
  (1024), *not* at the prototype or the slice size. Under ``downscale_pred=True`` the production
  path therefore moves a 1024x1024 float mask per detection off the GPU and then does a genuine
  4x (slice 256) or 2x (slice 512) ``INTER_AREA`` downscale on it. Nobody has timed that.
* **X-03** added a shoelace area per traced contour — cheap per call, ~1e6 calls per scene.
* **G1-06** moved the area gate onto ``mask_support_area``, a full-array scan per detection.

This measures the real breakdown instead of arguing about it. Every stage is wrapped in place, so
the numbers come from the production entry point (``predict_production``) rather than a
re-implementation that might drift from it.

Threading note: the per-detection work runs on a ``ThreadPoolExecutor``. Summing wall time across
workers over-counts against the run's wall clock, so the default is ``--workers 1`` — a clean
serial attribution — and ``--workers 8`` is run separately for the wall-clock number.

    conda run -n bouldernet python scripts/review/profile_production_path.py
    conda run -n bouldernet python scripts/review/profile_production_path.py --slice-size 512
    conda run -n bouldernet python scripts/review/profile_production_path.py \
        --sha256 data/mars_hirise/raster/ESP_017355_2260_RED.tif
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


# --- instrumentation ------------------------------------------------------------------------------

class Stopwatch:
    """Thread-safe accumulation of (calls, seconds) per named stage.

    Each thread appends into its own list — ``list.append`` is atomic under CPython and needs no
    lock, whereas a shared ``total += dt`` would silently lose samples across eight workers.
    """

    def __init__(self):
        self._per_thread = defaultdict(list)
        self.shapes = Counter()

    def _bucket(self):
        return self._per_thread[threading.get_ident()]

    def wrap(self, name, fn, note=None):
        bucket_of = self._bucket

        @functools.wraps(fn)
        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                bucket_of().append((name, time.perf_counter() - t0))
                if note is not None:
                    note(self.shapes, args, kwargs)
        return timed

    def reset(self):
        """Discard everything recorded so far — used to drop the warm-up repetition."""
        self._per_thread.clear()
        self.shapes.clear()

    def totals(self):
        calls, seconds = Counter(), defaultdict(float)
        for samples in self._per_thread.values():
            for name, dt in samples:
                calls[name] += 1
                seconds[name] += dt
        return {name: {"calls": calls[name], "seconds": round(seconds[name], 4),
                       "us_per_call": round(1e6 * seconds[name] / calls[name], 2)}
                for name in sorted(seconds, key=lambda k: -seconds[k])}


def _note_resize(shapes, args, kwargs):
    """Record what ``cv2.resize`` is actually being handed — settles the mask-size question."""
    src = args[0] if args else kwargs.get("src")
    dsize = args[1] if len(args) > 1 else kwargs.get("dsize")
    if src is not None and getattr(src, "ndim", 0) == 2:
        shapes[f"resize {src.shape[0]}->{(dsize or (None,))[0]}"] += 1


def instrument(watch):
    """Wrap the hot path in place. Returns nothing; the patches stay for the process lifetime."""
    import cv2
    import skimage.measure
    import torch

    from YOLOv8BeyondEarth import polygon, predict, presets

    # The masks come off the GPU as `imgsz`-sized float32 — 4 MB *per detection* at imgsz 1024 —
    # inside a closure that cannot be patched by name, so time the transfer itself. `.cpu()`
    # synchronizes, so this is the real device-to-host cost and not an async illusion.
    _cpu = torch.Tensor.cpu

    def timed_cpu(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return _cpu(self, *a, **kw)
        finally:
            watch._bucket().append(("Tensor.cpu (mask GPU->CPU)", time.perf_counter() - t0))
            watch.shapes["gpu_to_cpu_MiB"] += self.numel() * self.element_size() // (1024 ** 2)
    torch.Tensor.cpu = timed_cpu

    # per-detection, inside the worker threads
    cv2.resize = watch.wrap("cv2.resize (mask downscale)", cv2.resize, note=_note_resize)
    predict.cv2 = cv2
    predict.mask_support_area = watch.wrap(
        "mask_support_area (G1-06 gate)", polygon.mask_support_area)
    predict.binary_mask_to_polygon = watch.wrap(
        "binary_mask_to_polygon (total)", polygon.binary_mask_to_polygon)
    predict.binary_mask_to_polygon_cv = watch.wrap(
        "binary_mask_to_polygon_cv (total)", polygon.binary_mask_to_polygon_cv)
    predict.shift_polygon = watch.wrap("shift_polygon", polygon.shift_polygon)

    # nested inside the tracer — reported separately, NOT additive with it
    polygon._mask_bounds = watch.wrap("  ..._mask_bounds", polygon._mask_bounds)
    polygon._enclosed_area = watch.wrap("  ..._enclosed_area (X-03 shoelace)", polygon._enclosed_area)
    skimage.measure.find_contours = watch.wrap(
        "  ...find_contours", skimage.measure.find_contours)
    polygon.skimage = skimage

    # once per slice / once per run, on the main thread
    predict._result_to_df = watch.wrap("_result_to_df (per slice, total)", predict._result_to_df)
    predict.add_geometries = watch.wrap("add_geometries", polygon.add_geometries)
    polygon.write_vector = watch.wrap("write_vector (+readback)", polygon.write_vector)
    predict._ensure_png = watch.wrap("_ensure_png (TIFF->PNG)", predict._ensure_png)
    presets.file_identity = watch.wrap("file_identity (sha256)", presets.file_identity)
    predict.presets = presets


class TimedModel:
    """Delegating proxy that times the batched ultralytics call and nothing else."""

    def __init__(self, inner, watch):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_call", watch.wrap("model call (GPU inference)", inner))

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


# --- the two standalone costs the runbook names ---------------------------------------------------

def time_sha256(path):
    from YOLOv8BeyondEarth import presets
    path = Path(path)
    t0 = time.perf_counter()
    ident = presets.file_identity(path)
    dt = time.perf_counter() - t0
    gb = ident.get("bytes", 0) / 1024 ** 3
    return {"path": path.name, "gib": round(gb, 3), "seconds": round(dt, 2),
            "gib_per_s": round(gb / dt, 2) if dt else None,
            "hashed": ident.get("sha256") is not None}


# --- driver ---------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", default=str(REPO / "data" / "raster" / "crop.tif"))
    ap.add_argument("--weights",
                    default=str(REPO / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"))
    ap.add_argument("--out", default=str(REPO / "data" / "review"))
    ap.add_argument("--slice-size", type=int, default=None, help="override the preset slice size")
    ap.add_argument("--contour-method", default=None, choices=("skimage", "cv2"))
    ap.add_argument("--workers", type=int, default=1,
                    help="1 = clean serial attribution; 8 = the production wall clock")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-warmup", action="store_true",
                    help="profile the first repetition instead (measures the cold path)")
    ap.add_argument("--sha256", default=None, help="time file_identity on this file and exit")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sha256:
        record = time_sha256(args.sha256)
        print(json.dumps(record, indent=2))
        (out_dir / "profile_sha256.json").write_text(json.dumps(record, indent=2))
        return

    watch = Stopwatch()
    instrument(watch)

    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth import presets
    from YOLOv8BeyondEarth.predict import predict_production

    overrides = {"batch_size": 8}
    if args.slice_size is not None:
        overrides["slice_size"] = args.slice_size
    if args.contour_method is not None:
        overrides["contour_method"] = args.contour_method
    cfg = presets.resolve("goal1_production", **overrides)

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=args.weights, device=args.device,
        image_size=cfg["inference_size"], confidence_threshold=cfg["confidence_threshold"])
    model.model = TimedModel(model.model, watch)

    tag = args.tag or (f"ss{cfg['slice_size']}-{cfg['contour_method']}-w{args.workers}")
    run_dir = out_dir / f"profile-{tag}"

    # The first repetition on this machine costs 2-3x the steady state (CUDA context, cuDNN
    # autotune, page cache) — exactly the warm-up that produced G1-05's 94.5% spread. Profile the
    # second one, or the model call swamps every CPU stage and the split is a lie about warm-up.
    if not args.no_warmup:
        predict_production(args.raster, model, run_dir, n_workers=args.workers,
                           on_conflict="overwrite", **overrides)
        watch.reset()

    t0 = time.perf_counter()
    gdf, gdf_nms = predict_production(args.raster, model, run_dir, n_workers=args.workers,
                                      on_conflict="overwrite", **overrides)
    wall = time.perf_counter() - t0

    n_kept = int(len(gdf_nms if gdf_nms is not None else gdf))
    record = {
        "tag": tag,
        "raster": Path(args.raster).name,
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "n_workers": args.workers,
        "wall_seconds": round(wall, 2),
        "n_detections": int(len(gdf)),
        "n_after_nms": n_kept,
        "mask_resize_shapes": dict(watch.shapes),
        "stages": watch.totals(),
    }
    # what fraction of the run each stage explains (serial runs only: with >1 worker the summed
    # thread time legitimately exceeds the wall clock and a percentage would be nonsense)
    if args.workers == 1:
        for name, entry in record["stages"].items():
            entry["pct_of_wall"] = round(100.0 * entry["seconds"] / wall, 2)

    path = out_dir / f"profile_production_path-{tag}.json"
    path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
