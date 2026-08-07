# Goal-3 phase 1: rule comparison on the real treatment pools (plan §Validation 1, second half —
# "fuse the treatment scene per-scale outputs and compare rules on one pool"). CPU-only, runs on
# the three-arm outputs. NOTE the pool here is {256, 128-native, SR x2} — the three-arm arms —
# not the production ladder (512 doesn't exist yet; phase 2/3 adds it). The first half of the
# gate (the SS4c content-free-field per-scale null) cannot run from cache: the null-field
# artifacts store angles/centroids only, no polygons, and only at ss=256 — it needs a small GPU
# session and is deferred to phase 2/3 alongside the 512 runs.
# Per rule: fused size, per-scale composition, parity gate vs 256 baseline, duplicate-rate
# audit, wall clock; plus a quick orientation look (diag-frac on the elongated subset) — the
# citable orientation readout stays phase 3's matched-pair analysis.
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
from YOLOv8BeyondEarth.fusion import fuse_predictions, matched_recall, duplicate_rate
from YOLOv8BeyondEarth.orientation import attach_orientations, diagonal_fraction, grid_fraction

DE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionDE")
FIT_RES = 0.25
ASPECT_MIN = 1.35

SCENES = {
    "hirise": dict(native_res=0.5, axis="diagonal", arms={
        256:        ("hirise_baseline", "hirise_crop4k-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg"),
        128:        ("hirise_dctl", "hirise_crop4k-predictions-ct-010-ss-128-is-1024-ov-020-downscaled-mask-nms.gpkg"),
        (256, 2):   ("hirise_e2", "hirise_crop4k_sr2-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg"),
    }),
    "nac": dict(native_res=0.48357597846618, axis="cardinal", arms={
        256:        ("nac_baseline", "nac_crop2k-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg"),
        128:        ("nac_dctl", "nac_crop2k-predictions-ct-010-ss-128-is-1024-ov-020-downscaled-mask-nms.gpkg"),
        (256, 2):   ("nac_e2", "nac_crop2k_sr2-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg"),
    }),
}

# Provisional bands boundary (native px): SR-arm matched recall of baseline falls below ~0.5
# above ~10 px in the three-arm run (three_arm_pairs.csv) — phase 3's rule ablation re-derives
# this properly; here it only exercises the rule on real data.
PROVISIONAL_BANDS = {(256, 2): (0.0, 10.0), 128: (10.0, 10.0), 256: (10.0, np.inf)}


def frac_fn(axis):
    return diagonal_fraction if axis == "diagonal" else grid_fraction


def rose_frac(gdf, axis):
    g = attach_orientations(gdf, FIT_RES)
    el = g[(g.aspect_ra > ASPECT_MIN) & g.angle180.notna()]
    return frac_fn(axis)(el.angle180.values), len(el)


if __name__ == "__main__":
    rows = []
    for scene, cfg in SCENES.items():
        res, axis = cfg["native_res"], cfg["axis"]
        gdfs = {k: gpd.read_file(DE / "preds" / d / f) for k, (d, f) in cfg["arms"].items()}
        base = gdfs[256]
        print(f"\n=== {scene}: pool sizes {{k: len}} = { {str(k): len(g) for k, g in gdfs.items()} } ===")

        for rule in ("scale_aware", "score", "bands"):
            kw = dict(rule=rule)
            if rule == "bands":
                kw.update(bands=PROVISIONAL_BANDS, native_res=res)
            t0 = time.perf_counter()
            fused, audit = fuse_predictions(gdfs, **kw)
            dt = time.perf_counter() - t0
            rec, unmatched = matched_recall(base, fused)
            dup = duplicate_rate(fused, native_res=res)
            comp = fused["scale_key"].value_counts().to_dict()
            fr, n_el = rose_frac(fused, axis)
            n_rescued = int((~fused["in_band"]).sum()) if "in_band" in fused.columns else 0
            row = dict(scene=scene, rule=rule, n_fused=len(fused), n_audit=len(audit),
                       n_rescued_out_of_band=n_rescued,
                       parity=rec, n_unmatched=len(unmatched),
                       near_pairs=int(dup.n_near_pairs.sum()), bug_pairs=int(dup.n_bug_pairs.sum()),
                       fuse_seconds=round(dt, 2), frac=round(fr, 4), n_elongated=n_el,
                       **{f"from_{k}": v for k, v in comp.items()})
            rows.append(row)
            print(f"  rule={rule:12s} fused={len(fused):6d} comp={comp} parity={rec:.4f} "
                  f"near_pairs={int(dup.n_near_pairs.sum())} bugs={int(dup.n_bug_pairs.sum())} "
                  f"t={dt:.2f}s  {'diag' if axis == 'diagonal' else 'grid'}-frac={fr:.3f} (n_el={n_el})")

        # input roses for reference (fused must sit in this range, not outside it)
        for k, g in gdfs.items():
            fr, n_el = rose_frac(g, axis)
            print(f"  input {str(k):9s} {'diag' if axis == 'diagonal' else 'grid'}-frac={fr:.3f} (n_el={n_el})")
            rows.append(dict(scene=scene, rule=f"input_{k}", n_fused=len(g), frac=round(fr, 4),
                             n_elongated=n_el))

    pd.DataFrame(rows).to_csv(DE / "fusion_treatment_pool.csv", index=False)
    print(f"\nwrote {DE / 'fusion_treatment_pool.csv'}")
