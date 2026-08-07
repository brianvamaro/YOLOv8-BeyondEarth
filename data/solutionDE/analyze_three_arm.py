# Quick-look analysis of the D/E three-arm crop run (first pass; the citable version becomes a
# jupytext notebook). Per arm: elongated-subset diag/grid fraction + chirality, by native-px size
# band. Matched pairs (benchmark.match_detections, baseline<->arm): same-population fraction
# comparison, |delta angle| between pairs, matched recall by band, and unmatched-new-detection
# counts (the recall-below-floor readout). CSVs land next to the run outputs in data/solutionDE/.
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
from YOLOv8BeyondEarth.orientation import (attach_orientations, grid_fraction,
                                           diagonal_fraction, asymmetry_ratio, axial_distance)
from YOLOv8BeyondEarth.benchmark import match_detections

DE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionDE")
FIT_RES = 0.25          # uniform segmentize spacing for the ellipse fit, all arms/scenes
ASPECT_MIN = 1.35
BANDS = [(3, 4), (4, 6), (6, 8), (8, 10), (10, 14), (14, np.inf)]

SCENES = {
    "hirise": dict(native_res=0.5, axis="diagonal", arms={
        "baseline": ("hirise_baseline", "hirise_crop4k-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg"),
        "e2":       ("hirise_e2", "hirise_crop4k_sr2-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg"),
        "dctl":     ("hirise_dctl", "hirise_crop4k-predictions-ct-010-ss-128-is-1024-ov-020-downscaled-mask-nms.gpkg"),
    }),
    "nac": dict(native_res=0.48357597846618, axis="cardinal", arms={
        "baseline": ("nac_baseline", "nac_crop2k-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg"),
        "e2":       ("nac_e2", "nac_crop2k_sr2-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg"),
        "dctl":     ("nac_dctl", "nac_crop2k-predictions-ct-010-ss-128-is-1024-ov-020-downscaled-mask-nms.gpkg"),
    }),
}


def frac_fn(axis):
    return diagonal_fraction if axis == "diagonal" else grid_fraction


def band_label(dpx):
    for lo, hi in BANDS:
        if lo <= dpx < hi:
            return f"{lo}-{hi if np.isfinite(hi) else 'inf'}"
    return "<3"


def arm_summary(df, axis, native_res):
    el = df[(df.aspect_ra > ASPECT_MIN) & df.angle180.notna()].copy()
    el["dpx"] = el.diameter_m / native_res
    fr = frac_fn(axis)
    rows = [dict(band="all", n=len(el), frac=fr(el.angle180.values),
                 chirality=asymmetry_ratio(el.angle180.values))]
    for lo, hi in BANDS:
        b = el[(el.dpx >= lo) & (el.dpx < hi)]
        if len(b) >= 30:
            rows.append(dict(band=f"{lo}-{hi if np.isfinite(hi) else 'inf'}", n=len(b),
                             frac=fr(b.angle180.values),
                             chirality=asymmetry_ratio(b.angle180.values)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    per_arm_rows, pair_rows, newdet_rows = [], [], []
    for scene, cfg in SCENES.items():
        res, axis = cfg["native_res"], cfg["axis"]
        gdfs = {}
        for arm, (d, fname) in cfg["arms"].items():
            g = gpd.read_file(DE / "preds" / d / fname)
            g = attach_orientations(g, FIT_RES)
            g["dpx"] = g.diameter_m / res
            gdfs[arm] = g
            s = arm_summary(g, axis, res)
            s.insert(0, "arm", arm); s.insert(0, "scene", scene)
            per_arm_rows.append(s)

        base = gdfs["baseline"]
        for arm in ("e2", "dctl"):
            other = gdfs[arm]
            m = match_detections(base, other, max_dist=2.0 * res)
            a = base.iloc[m.idx_a.values]
            b = other.iloc[m.idx_b.values]
            # matched recall by baseline size band
            base_band = base.dpx.map(band_label)
            matched_band = a.dpx.map(band_label)
            for band in base_band.unique():
                nb = int((base_band == band).sum()); nm = int((matched_band == band).sum())
                pair_rows.append(dict(scene=scene, arm=arm, kind="matched_recall", band=band,
                                      n_base=nb, n_matched=nm, value=nm / nb if nb else np.nan))
            # same-population orientation comparison + pair |dA| (elongated-in-both pairs)
            el = (a.aspect_ra.values > ASPECT_MIN) & (b.aspect_ra.values > ASPECT_MIN) \
                 & np.isfinite(a.angle180.values) & np.isfinite(b.angle180.values)
            fr = frac_fn(axis)
            dpx_a = a.dpx.values
            for lo, hi in [(3, np.inf)] + BANDS:
                sel = el & (dpx_a >= lo) & (dpx_a < hi)
                if sel.sum() < 30:
                    continue
                da = axial_distance(a.angle180.values[sel], b.angle180.values[sel])
                pair_rows.append(dict(
                    scene=scene, arm=arm, kind="pairs",
                    band=("all" if not np.isfinite(hi) and lo == 3 else f"{lo}-{hi if np.isfinite(hi) else 'inf'}"),
                    n_base=int(sel.sum()), n_matched=int(sel.sum()),
                    value=float(np.median(da)),
                    frac_base=fr(a.angle180.values[sel]), frac_arm=fr(b.angle180.values[sel]),
                    chir_base=asymmetry_ratio(a.angle180.values[sel]),
                    chir_arm=asymmetry_ratio(b.angle180.values[sel])))
            # arm detections with no baseline partner (candidate below-floor recall)
            new = other.iloc[[i for i in range(len(other)) if i not in set(m.idx_b.values)]]
            nb_band = new.dpx.map(band_label)
            for band, n in nb_band.value_counts().items():
                newdet_rows.append(dict(scene=scene, arm=arm, band=band, n_new=int(n)))

    pd.concat(per_arm_rows).to_csv(DE / "three_arm_per_arm.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(DE / "three_arm_pairs.csv", index=False)
    pd.DataFrame(newdet_rows).to_csv(DE / "three_arm_newdet.csv", index=False)

    pa = pd.concat(per_arm_rows)
    print("=== per-arm elongated-subset fraction (uniform=0.222) ===")
    print(pa.to_string(index=False))
    pr = pd.DataFrame(pair_rows)
    print("\n=== matched pairs ===")
    print(pr[pr.kind == "pairs"].to_string(index=False))
    print("\n=== matched recall (of baseline) ===")
    print(pr[pr.kind == "matched_recall"].sort_values(["scene", "arm", "band"]).to_string(index=False))
    print("\n=== unmatched new detections by size band ===")
    print(pd.DataFrame(newdet_rows).sort_values(["scene", "arm", "band"]).to_string(index=False))
