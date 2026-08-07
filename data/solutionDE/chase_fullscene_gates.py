# Goal-3 phase 4: chase the two full-scene gate loose ends (run AFTER analyze_fullscene):
#  1. PARITY: HiRISE parity = 0.9999991 with exactly 1 unmatched 256-baseline detection of
#     1,112,849 (pre-registered gate is exact 1.0). Recompute matched_recall, pull the offender,
#     and characterize it (geometry validity, size, score, best DIoU + containment vs its fused
#     neighbours) — expected: a degenerate/sliver geometry below every threshold, but verify.
#  2. LARGE-BAND DUPES: near-pair rate at 14+ px is 4.9% (HiRISE) / 6.6% (NAC) vs the ~2%
#     target the small bands pass. duplicate_rate is an upper bound: adjacent DISTINCT boulders
#     land in the audit window too (big boxes -> big windows). Classify every surviving 14+
#     near-pair by polygon containment (fragment-inside-whole median was 0.99 in the phase-3
#     deep dive) and mask-IoU: containment >= 0.5 = fragment-mode (true dupe escaping engulf),
#     else neighbour-mode (benign). Writes goal3_fullscene_dupe_classify.csv + prints verdicts.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
from YOLOv8BeyondEarth.fusion import (matched_recall, equivalent_diameter_px,
                                      pairwise_box_metric, _candidate_pairs,
                                      _poly_containment, DEFAULT_CLUSTER_THR)

DE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionDE")
FS = DE / "fullscene"

SCENES = {
    "nac": ("M139694087LE", 0.48357597846618, FS / "nac"),
    "hirise": ("ESP_017355_2260_RED", 0.5, FS / "hirise"),
}
FUSED_NAME = "-predictions-fused-bands-engulf70-512-256-256sr2-mask.gpkg"


def base_gpkg(scene_dir, raster):
    return scene_dir / (raster + "-predictions-ct-010-ss-256-is-1024-ov-020"
                        "-downscaled-mask-nms.gpkg")


if __name__ == "__main__":
    rows = []
    for scene, (raster, res, scene_dir) in SCENES.items():
        base = gpd.read_file(base_gpkg(scene_dir, raster))
        fused = gpd.read_file(scene_dir / (raster + FUSED_NAME))
        print(f"[{scene}] base={len(base):,} fused={len(fused):,}", flush=True)

        # ---- 1. parity offender(s) ---------------------------------------------------
        rec, unmatched = matched_recall(base, fused, engulf_thr=0.7)
        print(f"[{scene}] parity={rec:.7f} unmatched={len(unmatched)}", flush=True)
        for u in unmatched[:10]:
            g = base.geometry.iloc[u]
            dpx = float(equivalent_diameter_px(base.iloc[[u]], res)[0])
            # nearest fused neighbours by envelope intersection of a small buffer
            hood = fused.iloc[list(fused.sindex.query(g.buffer(5 * res)))]
            best_diou = best_cont = np.nan
            if len(hood):
                bb = np.repeat(np.asarray(g.bounds, float)[None, :], len(hood), axis=0)
                fb = hood.geometry.bounds.to_numpy(float)
                best_diou = float(pairwise_box_metric(bb, fb).max())
                best_cont = float(_poly_containment(
                    np.repeat(np.array([g], dtype=object), len(hood)),
                    hood.geometry.values).max())
            print(f"    offender idx={u}: valid={g.is_valid} empty={g.is_empty} "
                  f"area={g.area:.4f} m2 dpx={dpx:.2f} score={base.score.iloc[u]:.3f} "
                  f"n_hood={len(hood)} best_DIoU={best_diou:.3f} best_contain={best_cont:.3f}",
                  flush=True)

        # ---- 2. classify surviving 14+ px near-pairs -----------------------------------
        dpx = equivalent_diameter_px(fused, res)
        boxes = fused.geometry.bounds.to_numpy(float)
        ii, jj, vals = _candidate_pairs(boxes, 1e-9, "diou")
        big = (dpx[ii] >= 14) | (dpx[jj] >= 14)
        near = big & (vals < DEFAULT_CLUSTER_THR)
        i2, j2, v2 = ii[near], jj[near], vals[near]
        print(f"[{scene}] 14+ near-pairs: {len(i2):,}", flush=True)
        if len(i2):
            gi, gj = fused.geometry.values[i2], fused.geometry.values[j2]
            ai, aj = shapely.area(gi), shapely.area(gj)
            small_first = ai <= aj
            gs = np.where(small_first, gi, gj)
            gb = np.where(small_first, gj, gi)
            contain = _poly_containment(gs, gb)
            inter = shapely.area(shapely.intersection(gi, gj))
            miou = inter / (ai + aj - inter)
            frag = contain >= 0.5
            df = pd.DataFrame(dict(scene=scene, i=i2, j=j2, diou=v2, containment=contain,
                                   mask_iou=miou, fragment_mode=frag))
            rows.append(df)
            n14 = int((dpx >= 14).sum())
            print(f"[{scene}] fragment-mode (contain>=0.5): {int(frag.sum()):,} "
                  f"({frag.mean():.1%} of near-pairs) -> TRUE-dupe rate at 14+ = "
                  f"{frag.sum() / n14:.4f} (was {len(i2) / n14:.4f} unclassified); "
                  f"neighbour-mode contain median={np.median(contain[~frag]) if (~frag).any() else np.nan:.3f}",
                  flush=True)
        del base, fused

    if rows:
        pd.concat(rows, ignore_index=True).to_csv(DE / "goal3_fullscene_dupe_classify.csv",
                                                  index=False)
        print("written", DE / "goal3_fullscene_dupe_classify.csv", flush=True)
