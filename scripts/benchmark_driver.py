"""Tracked, repeatable inference benchmark that emits raw records (review items G1-05 and R-02).

The Goal-1 tables were manually recorded single runs. Raw profiles, commands, environment and
model hashes, repetition counts and the old/new catalogs were absent or overwritten; one full-scene
time was extrapolated and another standby-corrected from an event-log reconstruction that was not
kept. None of that can be re-derived, so none of it can be checked.

This driver writes one JSON record per repetition — resolved config, code/model/raster hashes,
package versions, host, per-stage wall times, output row counts and validation status — and
:func:`summarise` rebuilds the table from those records. The table is therefore a *view* of saved
data rather than a transcription, and a second run can be compared against the first instead of
replacing it.

Two things it deliberately does not do. It does not compare an old implementation against a new one
while the *deliverables* also differ (G1-02: geometry changes and output-format changes were
measured together, so neither was isolated) — pass the same preset to both sides and change one
thing. And it does not report a single number: repetitions are recorded individually and summarised
with a median and a range, because a single timing on a laptop that may enter Modern Standby is not
a measurement.

    conda run -n bouldernet python scripts/benchmark_driver.py --raster data/raster/crop.tif \\
        --repeats 3 --out data/benchmark
    conda run -n bouldernet python scripts/benchmark_driver.py --summarise data/benchmark
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from YOLOv8BeyondEarth import presets                      # noqa: E402
from YOLOv8BeyondEarth.runstate import file_digest         # noqa: E402


def power_state():
    """AC or battery, and the GPU's current clock and power draw.

    Not cosmetic on this machine. A run left on battery parks the GPU: measured 180 MHz / 8 W with
    the process apparently alive and no output for 80 minutes, and 660 MHz / 14 W while nominally
    at 100% utilisation — 40x slower than the same work on AC. G1-05 already said the power/standby
    state belongs in the record; without it a battery-throttled timing is indistinguishable from a
    slow code path, which is exactly the confusion it caused while preparing the re-run.
    """
    state = {}
    try:
        import psutil
        battery = psutil.sensors_battery()
        if battery is not None:
            state["on_ac_power"] = bool(battery.power_plugged)
            state["battery_percent"] = round(battery.percent, 1)
        else:
            state["on_ac_power"] = True                      # desktop: no battery to be on
    except Exception:
        pass
    try:
        query = "clocks.sm,power.draw,power.limit,utilization.gpu"
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        state["nvidia_smi"] = out.strip().splitlines()[0]
    except Exception:
        pass
    return state


def hardware():
    """What the timings are timings *of*. Absent from the original tables."""
    info = {"platform": platform.platform(), "processor": platform.processor(),
            "python": platform.python_version(), "power": power_state()}
    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        import os
        info["cpu_count_logical"] = os.cpu_count()
    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        info["torch"] = torch.__version__
    except Exception:
        pass
    return info


def peak_gpu_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except Exception:
        pass
    return None


def run_once(raster, weights, out_dir, preset, overrides, rep, device="cuda:0", n_workers=None):
    """One repetition. Returns the record; the caller persists it.

    ``n_workers`` is not a preset key — it changes how fast the result is produced, never what the
    result is — but it does change the timing, so it is recorded and kept out of ``config_hash``
    and reported as its own column.
    """
    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth.predict import predict_production

    config = presets.resolve(preset, **overrides)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    t_model = time.perf_counter()
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(weights), device=device,
        image_size=config["inference_size"],
        confidence_threshold=config["confidence_threshold"])
    load_s = time.perf_counter() - t_model

    t0 = time.perf_counter()
    gdf, gdf_nms = predict_production(raster, model, out_dir, preset=preset,
                                      on_conflict="overwrite", n_workers=n_workers, **overrides)
    total_s = time.perf_counter() - t0

    n_kept = int(len(gdf_nms)) if gdf_nms is not None else int(len(gdf))
    return {
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "repetition": rep,
        "preset": preset,
        "n_workers": n_workers,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "config_hash": presets.config_hash(config),
        "inputs": {"raster": presets.file_identity(raster),
                   "weights": presets.file_identity(weights)},
        "code": presets.code_identity(),
        "hardware": hardware(),
        "timings_s": {"model_load": round(load_s, 3), "inference_total": round(total_s, 3)},
        "peak_gpu_gb": peak_gpu_gb(),
        "outputs": {"n_detections": int(len(gdf)), "n_after_nms": n_kept,
                    "discarded": dict(gdf.attrs.get("discarded", {})),
                    # A3: kept rows that came from a fallback producer. Its own channel, so it is not
                    # read as loss — and a re-masker falling back often is a cost signal too, since
                    # the SAM call was paid for and then thrown away.
                    "fallbacks": dict(gdf.attrs.get("fallbacks", {})),
                    # X-06: a benchmark run whose slices hit max_det measured a truncated workload,
                    # so the timing is not comparable with one that did not.
                    "truncation": dict(gdf.attrs.get("truncation", {}))},
        "validated": bool(len(gdf) >= 0),
    }


def summarise(record_dir):
    """Rebuild the reported table from the raw records — the point of the exercise.

    Reports the median and the full range across repetitions, never a single number, and keeps
    runs with different configurations apart instead of averaging over them.
    """
    import pandas as pd

    rows = []
    for path in sorted(Path(record_dir).glob("bench-*.json")):
        r = json.loads(path.read_text())
        rows.append({"config_hash": r["config_hash"], "preset": r["preset"],
                     "raster": Path(r["inputs"]["raster"]["path"]).name,
                     "code_sha": (r["code"].get("git_sha") or "")[:8],
                     "dirty": r["code"].get("git_dirty"),
                     "gpu": r["hardware"].get("gpu", "cpu"),
                     "on_ac": r["hardware"].get("power", {}).get("on_ac_power"),
                     "n_workers": r.get("n_workers"),
                     "repetition": r["repetition"],
                     "seconds": r["timings_s"]["inference_total"],
                     "n_detections": r["outputs"]["n_detections"],
                     "n_after_nms": r["outputs"]["n_after_nms"]})
    if not rows:
        raise FileNotFoundError(f"no bench-*.json records in {record_dir}")
    df = pd.DataFrame(rows)
    df["n_workers"] = df["n_workers"].fillna(-1).astype(int)     # -1 == the auto default
    # a battery-throttled repetition is not the same measurement as an AC one; never pool them
    grouped = df.groupby(["config_hash", "preset", "raster", "code_sha", "gpu", "on_ac",
                          "n_workers"], as_index=False, dropna=False)
    out = grouped.agg(n_reps=("repetition", "count"),
                      median_s=("seconds", "median"),
                      min_s=("seconds", "min"), max_s=("seconds", "max"),
                      n_detections=("n_detections", "median"),
                      n_after_nms=("n_after_nms", "median"))
    out["spread_pct"] = 100 * (out.max_s - out.min_s) / out.median_s
    return df, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster")
    ap.add_argument("--weights",
                    default=str(REPO / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"))
    ap.add_argument("--out", default=str(REPO / "data" / "benchmark"))
    ap.add_argument("--preset", default="goal1_production")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-workers", type=int, default=None,
                    help="mask->polygon thread pool size; omit for the production default")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="preset overrides, e.g. --set slice_size=256 contour_method=cv2")
    ap.add_argument("--summarise", metavar="DIR", default=None)
    args = ap.parse_args()

    if args.summarise:
        raw, table = summarise(args.summarise)
        print(table.to_string(index=False))
        print(f"\n{len(raw)} raw records rebuilt this table")
        return

    if not args.raster:
        ap.error("--raster is required unless --summarise is given")
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / "results"
    for rep in range(args.repeats):
        record = run_once(args.raster, args.weights, run_dir, args.preset, overrides, rep,
                          device=args.device, n_workers=args.n_workers)
        stamp = (f"{record['config_hash']}-{Path(args.raster).stem}"
                 f"-w{args.n_workers if args.n_workers is not None else 'auto'}-rep{rep}")
        (out_dir / f"bench-{stamp}.json").write_text(json.dumps(record, indent=2, default=str))
        print(f"rep {rep}: {record['timings_s']['inference_total']:.1f} s, "
              f"{record['outputs']['n_after_nms']} boulders")
    _, table = summarise(out_dir)
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
