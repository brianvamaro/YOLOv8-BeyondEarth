"""What each fusion rule costs to compute, on a real full-scene ladder (2026-10-15).

The rules differ only in the priority order a cluster's keeper is chosen by (`fusion._priority_order`,
one `np.lexsort`); candidate search, DIoU and greedy suppression are shared. `engulf_thr` adds a second
pass with polygon intersections. This times each variant on the NAC full-scene arms (the HiRISE arm
layers are not on this machine), median of `--reps`, layer reading excluded.

    conda run -n bouldernet python scripts/review/time_fusion_rules.py --out data/review/fusion_rule_timing.json
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import geopandas as gpd                                                    # noqa: E402
from rastertools_BOULDERING import metadata as raster_metadata             # noqa: E402
from YOLOv8BeyondEarth import presets                                      # noqa: E402
from YOLOv8BeyondEarth.fusion import fuse_predictions                      # noqa: E402

ARMS = {(512, 1): "ss-512", (256, 1): "ss-256", (256, 2): "ss-256sr2"}
P = presets.GOAL3_PRODUCTION
VARIANTS = {
    "256pref_engulf (production)": dict(rule="bands", bands=P["bands"], engulf_thr=P["engulf_thr"]),
    "256pref, no engulf": dict(rule="bands", bands=P["bands"], engulf_thr=None),
    "scale_aware": dict(rule="scale_aware", engulf_thr=None),
    "scale_aware_engulf": dict(rule="scale_aware", engulf_thr=P["engulf_thr"]),
    "score (plain cross-arm NMS)": dict(rule="score", engulf_thr=None),
}
COMBOS = {"512+256+SR": [(512, 1), (256, 1), (256, 2)], "512+256": [(512, 1), (256, 1)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", default=str(REPO / "data" / "rerun_2026-08-17" / "nac"))
    ap.add_argument("--raster", default=str(REPO / "data" / "raster" / "M139694087LE.tif"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    arm_dir = Path(args.arm_dir)
    layers = {k: next(arm_dir.glob(f"*-ct-010-{tag}-is-1024-ov-020-downscaled-mask-nms.gpkg"))
              for k, tag in ARMS.items()}
    gdfs = {k: gpd.read_file(p) for k, p in layers.items()}
    res = raster_metadata.get_resolution(args.raster)[0]
    rows = []
    for cname, keys in COMBOS.items():
        sub = {k: gdfs[k] for k in keys}
        n_pool = sum(len(g) for g in sub.values())
        for vname, kw in VARIANTS.items():
            walls, n_fused = [], None
            for _ in range(args.reps):
                t0 = time.perf_counter()
                fused, audit = fuse_predictions(sub, metric=P["metric"],
                                                cross_arm_only=P["cross_arm_only"],
                                                native_res=res, **kw)
                walls.append(time.perf_counter() - t0)
                n_fused = len(fused)
            rows.append(dict(combo=cname, variant=vname, n_pool=n_pool, n_fused=int(n_fused),
                             wall_s_median=round(statistics.median(walls), 3),
                             wall_s_all=[round(w, 3) for w in walls]))
            print(f"{cname:11s} {vname:30s} pool {n_pool:6d} fused {n_fused:6d}  "
                  f"{statistics.median(walls):6.2f} s  (reps {[round(w, 2) for w in walls]})",
                  flush=True)
    out = {"what": "fuse_predictions wall time per rule variant, NAC full-scene arms "
                   "(layer reads excluded), median of reps",
           "layers": {str(k): p.name for k, p in layers.items()}, "native_res": res,
           "reps": args.reps, "code": presets.code_identity(), "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
