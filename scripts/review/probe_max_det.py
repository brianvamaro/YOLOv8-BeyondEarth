"""Size ``max_det`` from evidence, and settle whether the cap actually engages (review item X-06).

``check_max_det_binding.py`` reconstructed the per-slice load from *saved* catalogs and reported a
**lower bound** (post-NMS centroids: cross-slice duplicates already removed, partly-overlapping
objects uncounted). Three questions it could not answer, and this can:

1. **Does the cap engage?** A footprint holding >300 detections implies truncation only if the model
   returned >300 *for that slice*. Run the densest real slices with the cap lifted far out of the
   way and read the true count.
2. **Does it engage at the confidence the shipped catalogs actually ran at?** X-01 means every saved
   catalog ran at an effective 0.25 while being labelled 0.10. If the cap binds at 0.25 too, the
   existing catalogs are truncated, not just the planned 0.10 re-run.
3. **What can VRAM afford?** Ultralytics' segmentation path materialises masks at ``imgsz``, not at
   ``slice_size``: ``ops.process_mask`` builds ``N x imgsz x imgsz`` float32, interpolates, then
   casts to byte — so **every** detection in a batch retains ``imgsz**2`` bytes inside its
   ``Results`` object for the whole batch, and the current image transiently costs ~5x that. The cap
   and ``batch_size`` therefore multiply. The review item assumed a 12 GB card; this machine has
   **8 GiB**, so the ceiling has to be measured.

**One cell per subprocess.** Ultralytics keeps the previous call's ``Results`` alive on
``model.predictor``, and the caching allocator carries state, so a single-process sweep contaminates
later cells with earlier ones — a first version of this probe measured batch 4 as costing *more*
than batch 8 for that reason. Isolation also mirrors the production rule
(``fullscene-runs-subprocess-per-arm``). Within a cell several consecutive batches run so the peak
includes the retained previous batch, which is production's real steady state.

**There is no OOM to catch.** Measured on this machine: a single 14 GiB tensor allocates on the
8.15 GiB card without raising, because Windows' WDDM driver spills into shared host memory. So an
oversized ``max_det`` does not fail — it silently collapses throughput over PCIe. Read the
``spills_beyond_device`` column, not the absence of an exception.

    conda run -n bouldernet python scripts/review/probe_max_det.py \
        --raster data/mars_hirise/raster/ESP_017355_2260_RED.tif \
        --catalog data/solutionDE/fullscene/hirise/ESP_017355_2260_RED-predictions-ct-010-ss-512-is-1024-ov-020-downscaled-mask-nms.gpkg \
        --slice-size 512

Fidelity: slices are read from the **GeoTIFF** by window, while production slices the 8-bit PNG
conversion of the same raster through SAHI. For an already-8-bit source that conversion is a
straight copy, so detection counts are comparable — production-faithful to within that assumption
rather than bit-identical. Peak memory and detection counts are power-state independent, so unlike
every wall clock in this pass they are not Modern-Standby contaminated.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_max_det_binding import per_slice_counts, slice_starts   # noqa: E402


def power_state():
    """AC/battery and the live clock line. Peak *memory* does not depend on either, but G1-05's rule
    is that the power state is part of the record, and a reader must be able to tell at a glance that
    these are not throughput numbers."""
    state = {}
    try:
        import psutil
        battery = psutil.sensors_battery()
        state["on_ac_power"] = True if battery is None else bool(battery.power_plugged)
        if battery is not None:
            state["battery_percent"] = round(battery.percent, 1)
    except Exception as exc:
        state["power_error"] = str(exc)
    try:
        state["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,power.limit",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:
        state["nvidia_smi_error"] = str(exc)
    return state


def densest_slice_origins(catalog, raster, native_slice, overlap, n_batches, batch_size):
    """Origins of the fullest runs of ``batch_size`` consecutive slices, from the saved centroids.

    Consecutive in **slice order** (row-major, the order SAHI emits and production batches), because
    peak memory is set by the whole batch and dense terrain is spatially clustered — a realistic
    worst case is a whole batch dense at once, not one dense slice among seven empty ones.
    """
    import rasterio
    from pyogrio import read_bounds

    _, bounds = read_bounds(catalog)
    cx = 0.5 * (bounds[0] + bounds[2])
    cy = 0.5 * (bounds[1] + bounds[3])
    with rasterio.open(raster) as src:
        inv = ~src.transform
        width, height = src.width, src.height
    px, py = inv * (cx, cy)
    counts, _, _ = per_slice_counts(np.asarray(px, float), np.asarray(py, float),
                                    width, height, native_slice, overlap)
    xs = slice_starts(width, native_slice, overlap)
    ys = slice_starts(height, native_slice, overlap)

    flat = counts.ravel()
    nx = len(xs)
    if len(flat) < batch_size:
        raise SystemExit(f"only {len(flat)} slices on this grid")
    window_sums = np.convolve(flat, np.ones(batch_size, dtype=np.int64), mode="valid")

    picked, used = [], np.zeros(len(flat), bool)
    for start in np.argsort(window_sums)[::-1]:
        if len(picked) == n_batches:
            break
        if used[start:start + batch_size].any():
            continue
        used[start:start + batch_size] = True
        picked.append({
            "first_slice": int(start),
            "expected_total": int(window_sums[start]),
            "expected_per_slice": [int(v) for v in flat[start:start + batch_size]],
            "origins": [(int(xs[(start + k) % nx]), int(ys[(start + k) // nx]))
                        for k in range(batch_size)],
        })
    return picked, int(flat.max()), int((flat > 0).sum())


def read_slices(raster, origins, native_slice, slice_size):
    """Each slice as an RGB uint8 array, Lanczos-upscaled to ``slice_size`` when SR is in play."""
    import cv2
    import rasterio
    from rasterio.windows import Window

    out = []
    with rasterio.open(raster) as src:
        for (x0, y0) in origins:
            band = src.read(1, window=Window(x0, y0, native_slice, native_slice),
                            boundless=True, fill_value=0)
            if band.shape != (native_slice, native_slice):       # clamped at the raster edge
                pad = np.zeros((native_slice, native_slice), band.dtype)
                pad[:band.shape[0], :band.shape[1]] = band
                band = pad
            img = np.stack([band] * 3, axis=-1)
            if slice_size != native_slice:
                img = cv2.resize(img, (slice_size, slice_size), interpolation=cv2.INTER_LANCZOS4)
            out.append(np.ascontiguousarray(img))
    return out


# --- the child: exactly one (batch_size, max_det, conf) cell -------------------------------------

def fraction_discarded_at_cap(counts, cap):
    """Fraction of the model's detections that a ``max_det=cap`` discards, over ``counts`` slices.

    ``counts`` must be the **uncapped** per-slice detection counts. Computed per slice and then
    summed — ``sum(max(0, c - cap)) / sum(c)`` — which is what makes it exact for any cap.

    This replaces ``max(0, 1 - cap / mean(counts))`` (review item B1), which pushed the mean through
    the truncation nonlinearity. Truncation is ``min(c, cap)``, and ``mean(min(c, cap)) !=
    min(mean(c), cap)`` unless *every* slice is at the cap, so the old form was right only in the
    saturated regime and the ``max(0, ...)`` clamp then rendered any *partial* truncation as exactly
    ``0.0``. That is how ``max_det_probe-hirise-ss256.json`` came to publish
    ``"cap_300_engages": true`` beside ``"fraction_discarded_at_300": 0.0`` — the truth is 1.6%, from
    3 of 16 slices at the cap. The saturated cells are unaffected, which is the tell: the 512 cells
    with 16/16 at the cap give the same 0.604 / 0.397 both ways.

    It is the fraction of detections the *model returns* — ultralytics' empty-mask filter in
    ``SegmentationPredictor.construct_result`` runs afterwards on both sides.
    """
    total = sum(counts)
    if not total:
        return 0.0
    return sum(max(0, c - cap) for c in counts) / total


def run_cell(args):
    import torch
    from ultralytics import YOLO

    origins = json.loads(Path(args.origins_json).read_text())
    native_slice = args.slice_size // args.sr_factor
    dev = torch.device(args.device)
    total_mib = torch.cuda.get_device_properties(dev).total_memory / 1024 ** 2

    images = [read_slices(args.raster, [tuple(o) for o in batch], native_slice, args.slice_size)
              for batch in origins]
    model = YOLO(args.weights, task="segment")

    # Warm up on one slice: the first call builds the CUDA context and cuDNN workspaces, which would
    # otherwise be charged to this cell.
    model(images[0][:1], imgsz=args.inference_size, conf=args.conf, max_det=args.max_det,
          verbose=False, device=args.device)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated(dev) / 1024 ** 2

    peak_alloc = peak_reserved = 0.0
    counts = []
    torch.cuda.reset_peak_memory_stats(dev)
    for batch in images:
        for start in range(0, len(batch), args.batch_size):
            chunk = batch[start:start + args.batch_size]
            if len(chunk) < args.batch_size:
                continue
            results = model(chunk, imgsz=args.inference_size, conf=args.conf,
                            max_det=args.max_det, verbose=False, device=args.device)
            torch.cuda.synchronize()
            peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated(dev) / 1024 ** 2)
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(dev) / 1024 ** 2)
            counts.extend(int(r.boxes.data.shape[0]) for r in results)
            del results

    cell = {
        "conf": args.conf, "batch_size": args.batch_size, "max_det": args.max_det,
        "resident_after_warmup_mib": round(resident, 1),
        "peak_alloc_mib": round(peak_alloc, 1),
        "peak_reserved_mib": round(peak_reserved, 1),
        "peak_frac_of_device": round(peak_reserved / total_mib, 3),
        "spills_beyond_device": bool(peak_reserved > total_mib),
        "slices_measured": len(counts),
        # Retained per slice, not just summarised: every question this probe answers about truncation
        # is a per-slice sum, and reducing to a mean first is exactly the error B1 records. It is 16
        # integers.
        "detections_per_slice": [int(c) for c in counts],
        "detections_per_slice_max": max(counts) if counts else 0,
        "detections_per_slice_mean": round(float(np.mean(counts)), 1) if counts else 0.0,
        "slices_at_cap": int(sum(1 for c in counts if c >= args.max_det)),
        "cap_bound": bool(any(c >= args.max_det for c in counts)),
    }
    print("CELL " + json.dumps(cell))
    return cell


# --- the parent: discovery once, then one subprocess per cell ------------------------------------

def orchestrate(args):
    import tempfile

    native_slice = args.slice_size // args.sr_factor
    biggest_batch = max(args.batch_sizes)
    batches, saved_max, n_occupied = densest_slice_origins(
        args.catalog, args.raster, native_slice, args.overlap, args.n_batches, biggest_batch)

    tmp = Path(tempfile.mkdtemp(prefix="maxdet-probe-")) / "origins.json"
    tmp.write_text(json.dumps([b["origins"] for b in batches]))

    record = {
        "raster": Path(args.raster).name, "catalog": Path(args.catalog).name,
        "slice_size": args.slice_size, "sr_factor": args.sr_factor,
        "native_slice": native_slice, "inference_size": args.inference_size,
        "weights": Path(args.weights).name,
        "power": power_state(),
        "analytic_bytes_per_detection": {
            "retained_uint8_at_imgsz": args.inference_size ** 2,
            "transient_float32_at_imgsz": 4 * args.inference_size ** 2,
            "note": ("ops.process_mask builds N x imgsz^2 float32 in the interpolate and casts to "
                     "N x imgsz^2 uint8, which the Results object keeps for the whole batch"),
        },
        "saved_catalog_max_per_slice": saved_max,
        "saved_catalog_occupied_slices": n_occupied,
        "dense_batches": [{k: v for k, v in b.items() if k != "origins"} for b in batches],
        "isolation": "one subprocess per cell",
        "cells": [],
    }

    for conf in args.confs:
        for batch_size in args.batch_sizes:
            for max_det in args.max_dets:
                cmd = [sys.executable, str(Path(__file__).resolve()),
                       "--cell", "--raster", str(args.raster), "--catalog", str(args.catalog),
                       "--origins-json", str(tmp), "--slice-size", str(args.slice_size),
                       "--sr-factor", str(args.sr_factor),
                       "--inference-size", str(args.inference_size), "--weights", str(args.weights),
                       "--device", args.device, "--conf", str(conf),
                       "--batch-size", str(batch_size), "--max-det", str(max_det)]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("CELL ")), None)
                if line is None:
                    print(f"[fail] conf={conf} bs={batch_size} max_det={max_det}\n"
                          f"{proc.stdout[-800:]}\n{proc.stderr[-1500:]}")
                    record["cells"].append({"conf": conf, "batch_size": batch_size,
                                            "max_det": max_det, "failed": True,
                                            "stderr_tail": proc.stderr[-500:]})
                    continue
                cell = json.loads(line[5:])
                record["cells"].append(cell)
                print(f"conf={cell['conf']:<5} bs={cell['batch_size']:<2} "
                      f"max_det={cell['max_det']:<5} peak={cell['peak_reserved_mib']:>8.1f} MiB "
                      f"({cell['peak_frac_of_device']:.2f}x device"
                      f"{', SPILLS' if cell['spills_beyond_device'] else ''}) "
                      f"det/slice max={cell['detections_per_slice_max']:<5} "
                      f"mean={cell['detections_per_slice_mean']:<6} "
                      f"at_cap={cell['slices_at_cap']}/{cell['slices_measured']}")

    # The decisive read on questions 1 and 2: with the cap lifted past any plausible load, what is
    # the true per-slice count on the densest terrain — at each confidence?
    record["true_load"] = {}
    for conf in args.confs:
        free = [c for c in record["cells"]
                if not c.get("failed") and c["conf"] == conf and not c["cap_bound"]]
        if free:
            true_max = max(c["detections_per_slice_max"] for c in free)
            true_mean = max(c["detections_per_slice_mean"] for c in free)
            # Prefer the uncapped per-slice counts, which make this exact for *any* cap. Cells
            # written before this fix have no such list, so fall back to the two-cell mean ratio.
            best = max(free, key=lambda c: len(c.get("detections_per_slice") or []))
            counts = best.get("detections_per_slice") or []
            capped = [c for c in record["cells"] if not c.get("failed")
                      and c["conf"] == conf and c["max_det"] == 300]
            entry = {"max_per_slice": true_max, "mean_per_slice": true_mean,
                     "cap_300_engages": bool(true_max > 300)}
            if counts:
                entry["fraction_discarded_at_300"] = round(fraction_discarded_at_cap(counts, 300), 4)
                entry["fraction_discarded_basis"] = "exact, per-slice uncapped counts"
            elif capped and true_mean:
                mean_at_cap = max(c["detections_per_slice_mean"] for c in capped)
                entry["fraction_discarded_at_300"] = round(1 - mean_at_cap / true_mean, 4)
                entry["fraction_discarded_basis"] = (
                    f"mean_at_cap {mean_at_cap} / mean_uncapped {true_mean}, same slices")
            else:
                # No cap-300 cell to compare against. Say so; do not emit 0.0, which reads as
                # "nothing was truncated" and is what made this defect publishable.
                entry["fraction_discarded_at_300"] = None
                entry["fraction_discarded_basis"] = "not measured: no max_det=300 cell at this conf"
            record["true_load"][str(conf)] = entry

    out = Path(args.out) if args.out else (
        REPO / "data" / "review" /
        f"max_det_probe-{Path(args.raster).stem[:24]}-ss{args.slice_size}sr{args.sr_factor}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out}")
    print(json.dumps({k: v for k, v in record.items() if k != "cells"}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", required=True)
    ap.add_argument("--catalog", required=True, help="a saved per-arm layer for the SAME arm")
    ap.add_argument("--slice-size", type=int, required=True)
    ap.add_argument("--sr-factor", type=int, default=1)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--inference-size", type=int, default=1024)
    ap.add_argument("--weights",
                    default=str(REPO / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--confs", type=float, nargs="+", default=[0.10, 0.25])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 4, 2, 1])
    ap.add_argument("--max-dets", type=int, nargs="+", default=[300, 600, 800, 1200, 3000])
    ap.add_argument("--n-batches", type=int, default=2,
                    help="distinct dense batches per cell")
    ap.add_argument("--out", default=None)
    # child-only
    ap.add_argument("--cell", action="store_true", help="internal: run one cell and exit")
    ap.add_argument("--origins-json", default=None)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-det", type=int, default=300)
    args = ap.parse_args()
    run_cell(args) if args.cell else orchestrate(args)


if __name__ == "__main__":
    main()
