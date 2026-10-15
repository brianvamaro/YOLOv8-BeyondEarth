"""Cap escalation end to end, on real inference: is an escalated arm the uncapped catalog? (2026-10-15)

The unit tests (``tests/test_arm_escalation.py``) fake the arm's child process. This runs the real
runner on one ROM tile's 256 arm, four times:

1. ``--on-max-det escalate`` from a cap low enough to bind (25), so the real child stops, exits with
   its code, and the runner climbs the ladder until an attempt completes;
2. a direct run at a cap that cannot bind (1100) at the batch the escalation finished at. The claim
   under test is that 1 and 2 are **the same catalog**;
3. the same at the preset's batch, which measures the one thing escalation changes: the halved batch;
4. command 1 again, which must skip the finished escalated arm rather than start over at 25.

It owns ``--work`` and clears it first, since a leftover ledger would make run 1 a resume.

    conda run -n bouldernet python scripts/review/check_escalation_e2e.py --out data/review/escalation_e2e.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "review"))

import geopandas as gpd                                                        # noqa: E402
from measure_batch_neutrality import compare_catalogs                          # noqa: E402
from YOLOv8BeyondEarth import presets                                          # noqa: E402
from YOLOv8BeyondEarth.runstate import StageLedger                            # noqa: E402

RUNNER = REPO / "scripts" / "run_multiscale_arms.py"
TILE = REPO / "data" / "rom_label_error" / "ESP_017355_2260" / "ESP_017355_2260_rom00_t1242.tif"


def run(raster, out_dir, *flags):
    cmd = [sys.executable, str(RUNNER), "--raster", str(raster), "--out-dir", str(out_dir),
           "--arms", "256", "--no-fuse", *map(str, flags)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if res.returncode:
        raise SystemExit(f"runner failed ({res.returncode}): {' '.join(cmd)}\n{res.stdout}")
    return res.stdout


def layer(out_dir):
    (path,) = Path(out_dir).glob("*-ss-256-*-downscaled-mask-nms.gpkg")
    return gpd.read_file(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", default=str(TILE))
    ap.add_argument("--work", default=str(REPO / "data" / "review" / "escalation_e2e"))
    ap.add_argument("--start-cap", type=int, default=25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    work = Path(args.work)
    shutil.rmtree(work, ignore_errors=True)

    esc = work / "escalate"
    run(args.raster, esc, "--max-det", args.start_cap, "--on-max-det", "escalate")
    attempts = [{k: r.get(k) for k in ("state", "max_det", "batch_size", "stopped", "bound",
                                       "next_attempt", "n", "truncation", "wall_seconds")}
                for r in StageLedger(esc / "run_ledger.jsonl").entries("arm:256")
                if r.get("state") != "running"]
    final = attempts[-1]
    assert final["state"] == "complete", attempts

    direct = work / f"direct_cap1100_b{final['batch_size']}"
    run(args.raster, direct, "--max-det", 1100, "--batch-size", final["batch_size"],
        "--on-max-det", "error")
    preset_b = presets.GOAL1_PRODUCTION["batch_size"]
    direct_preset = work / f"direct_cap1100_b{preset_b}"
    run(args.raster, direct_preset, "--max-det", 1100, "--on-max-det", "error")

    resume_out = run(args.raster, esc, "--max-det", args.start_cap, "--on-max-det", "escalate")
    resumed_skip = "[skip] arm 256" in resume_out and "[run ]" not in resume_out

    tol = 0.5          # one native px at 0.5 m/px
    a, b, c = layer(esc), layer(direct), layer(direct_preset)
    out = {
        "what": "real-inference check that an escalated arm equals the uncapped catalog at its batch",
        "raster": Path(args.raster).name, "arm": "256", "start_cap": args.start_cap,
        "attempts": attempts,
        "escalated_vs_uncapped_same_batch": compare_catalogs(a, b, tol),
        "escalated_vs_uncapped_preset_batch": compare_catalogs(a, c, tol),
        "uncapped_same_batch_vs_preset_batch": compare_catalogs(b, c, tol),
        "resubmit_skipped_without_running": resumed_skip,
        "code": presets.code_identity(),
    }
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    same = out["escalated_vs_uncapped_same_batch"]
    print(f"attempts: {[(x['max_det'], x['batch_size'], x['state']) for x in attempts]}")
    print(f"escalated vs uncapped, same batch: n {same['n_a']} vs {same['n_b']}, unmatched "
          f"{same['n_unmatched_a']}/{same['n_unmatched_b']}, geometry identical "
          f"{same['frac_geometry_identical']:.4f}, max |dscore| {same['max_abs_score_delta']:.2e}")
    pre = out["escalated_vs_uncapped_preset_batch"]
    print(f"escalated vs uncapped, batch {preset_b}: n {pre['n_a']} vs {pre['n_b']}, unmatched "
          f"{pre['n_unmatched_a']}/{pre['n_unmatched_b']}, geometry identical "
          f"{pre['frac_geometry_identical']:.4f}, max |dscore| {pre['max_abs_score_delta']:.2e}")
    print(f"resubmit skipped without running: {resumed_skip}")


if __name__ == "__main__":
    main()
