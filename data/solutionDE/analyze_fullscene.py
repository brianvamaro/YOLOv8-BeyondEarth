# Goal-3 phase 4: full-scene analysis (plan §Validation 5, 6, 8 at scene scale). Consumes the
# fullscene run's on-disk layers (run_fullscene_goal3.py) + fullscene_results.json and produces
# the numbers the D-card run-3 column and the plan's benchmark card need at FULL-SCENE scale,
# where the crop estimates may not hold:
#   goal3_fullscene_sfd.csv        - cumulative N(>D) fused vs 256-only + exp fit + roll-off px
#   goal3_fullscene_orientation.csv- diag/grid frac, well-resolved elongated, fused vs 256
#   goal3_fullscene_cost.csv       - per-arm wall clock + n_nms + ladder factor breakdown
# The SFD estimator is IDENTICAL to analyze_goal3_rom.sfd_rows (shared FIT_RANGE_PX/ROLLOFF_TOL)
# so full-scene completeness limits are directly comparable to the crop/ROM ones. No human truth
# over the whole scene, so SFD here is a fused-vs-256 self-comparison of completeness (the
# roll-off caveat from the plan applies: a completeness indicator only under the fitted-law
# assumption; the ROM spot-check breaks the degeneracy). CPU-only; run AFTER the GPU job frees
# RAM. Idempotent; analyzes whatever scenes are complete in the results JSON.
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
from YOLOv8BeyondEarth.fusion import equivalent_diameter_px
from YOLOv8BeyondEarth.orientation import (attach_orientations, grid_fraction,
                                           diagonal_fraction, asymmetry_ratio)
# reuse the crop/ROM estimator constants verbatim (identity == comparability)
sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionDE")
from analyze_goal3_rom import FIT_RANGE_PX, ROLLOFF_TOL

DE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionDE")
FS = DE / "fullscene"
RESULTS = FS / "fullscene_results.json"

SCENES = {
    "nac": dict(res=0.48357597846618,
                raster="M139694087LE", area_px=5118 * 59183),
    "hirise": dict(res=0.5,
                   raster="ESP_017355_2260_RED", area_px=22490 * 85922),
}
GRID = np.arange(2.0, 60.5, 0.5)   # px, same grid as analyze_goal3_rom.sfd_rows


def nms_gpkg(scene_dir, raster, ss, sr):
    tag = f"{ss}sr{sr}" if sr > 1 else f"{ss}"
    return scene_dir / (raster + f"-predictions-ct-010-ss-{tag}-is-1024-ov-020"
                        "-downscaled-mask-nms.gpkg")


def sfd_and_rolloff(scene, cats, res, area_m2):
    """Cumulative N(>D)/km^2 + exp fit N(>D)=A exp(-lam D) over FIT_RANGE_PX + roll-off px.
    Identical math to analyze_goal3_rom.sfd_rows, catalog labels ours (no scene-wide human)."""
    rows, fits = [], []
    for name, cat in cats.items():
        dpx = np.sort(equivalent_diameter_px(cat, res))
        n_gt = len(dpx) - np.searchsorted(dpx, GRID, side="right")
        dens = n_gt / (area_m2 / 1e6)
        for d, n, v in zip(GRID, n_gt, dens):
            rows.append(dict(scene=scene, catalog=name, d_px=float(d), n_gt=int(n),
                             per_km2=float(v)))
        sel = (GRID >= FIT_RANGE_PX[0]) & (GRID <= FIT_RANGE_PX[1]) & (n_gt > 5)
        if sel.sum() >= 5:
            lam, loga = np.polyfit(GRID[sel], np.log(n_gt[sel]), 1)
            fit = np.exp(loga + lam * GRID)
            below = GRID < FIT_RANGE_PX[0]
            ratio = np.where(fit > 0, n_gt / fit, np.nan)
            rolled = below & (ratio < ROLLOFF_TOL)
            rolloff = float(GRID[rolled].max()) if rolled.any() else np.nan
            fits.append(dict(scene=scene, catalog=name, lam=float(lam),
                             rolloff_px=rolloff, n_total=len(dpx)))
        else:
            fits.append(dict(scene=scene, catalog=name, lam=np.nan,
                             rolloff_px=np.nan, n_total=len(dpx)))
    return rows, fits


def orientation_nonreg(scene, source, cat, res):
    """diag/grid frac on the well-resolved elongated subset (dpx>=8, aspect>1.35). Pre-filter by
    dpx (area-only, no orientation needed) to skip orientation on the sub-floor mass."""
    dpx_all = equivalent_diameter_px(cat, res)
    sub = cat.loc[dpx_all >= 6.0].copy()                 # superset of the >=8 well-resolved gate
    if len(sub) == 0:
        return []
    ori = attach_orientations(sub, res=res, n_workers=8)
    ori["dpx"] = ori.diameter_m / res
    wr = ori[(ori.aspect_ra > 1.35) & (ori.dpx >= 8.0)]
    a = wr.angle180.values
    return [dict(scene=scene, source=source, n=len(a),
                 grid_frac=grid_fraction(a) if len(a) else np.nan,
                 diag_frac=diagonal_fraction(a) if len(a) else np.nan,
                 r135=asymmetry_ratio(a) if len(a) else np.nan)]


if __name__ == "__main__":
    results = json.loads(RESULTS.read_text())
    sfd_rows, fit_rows, ori_rows, cost_rows = [], [], [], []

    for scene, meta in SCENES.items():
        if scene not in results or "n_fused" not in results[scene]:
            print(f"[skip] {scene}: not complete in results JSON", flush=True)
            continue
        res = meta["res"]
        scene_dir = FS / scene
        raster = meta["raster"]
        area_m2 = meta["area_px"] * res * res

        base = gpd.read_file(nms_gpkg(scene_dir, raster, 256, 1))
        fused_path = scene_dir / (raster + "-predictions-fused-bands-engulf70"
                                  "-512-256-256sr2-mask.gpkg")
        fused = gpd.read_file(fused_path)
        print(f"[{scene}] base256={len(base):,} fused={len(fused):,}", flush=True)

        # --- SFD roll-off completeness (validation 5) -----------------------------------
        sr, fr = sfd_and_rolloff(scene, {"256_only": base, "fused": fused}, res, area_m2)
        sfd_rows += sr; fit_rows += fr

        # --- orientation non-regression (validation 6) ----------------------------------
        ori_rows += orientation_nonreg(scene, "256_only", base, res)
        ori_rows += orientation_nonreg(scene, "fused", fused, res)

        # --- cost reconciliation (validation 8): per-arm wall clock + yield -------------
        arm_wall = results[scene]["arm_wall_s"]
        base_s = arm_wall.get("ss256")
        for key in ("ss512", "ss256", "ss256sr2"):
            n_nms = results[scene].get(f"n_nms_{key}")
            cost_rows.append(dict(scene=scene, arm=key, wall_s=arm_wall.get(key),
                                  x_vs_256=(arm_wall[key] / base_s if base_s else np.nan),
                                  n_nms=n_nms))
        cost_rows.append(dict(scene=scene, arm="LADDER", wall_s=sum(arm_wall.values()),
                              x_vs_256=results[scene].get("ladder_factor_vs_256"),
                              n_nms=results[scene].get("n_fused")))

    pd.DataFrame(sfd_rows).to_csv(DE / "goal3_fullscene_sfd.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(DE / "goal3_fullscene_sfd_fit.csv", index=False)
    pd.DataFrame(ori_rows).to_csv(DE / "goal3_fullscene_orientation.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(DE / "goal3_fullscene_cost.csv", index=False)

    print("\n=== SFD roll-off (completeness limit px) ===")
    print(pd.DataFrame(fit_rows).to_string(index=False))
    print("\n=== orientation non-regression (well-resolved elongated) ===")
    print(pd.DataFrame(ori_rows).to_string(index=False))
    print("\n=== cost reconciliation ===")
    print(pd.DataFrame(cost_rows).to_string(index=False))
    print("\nCSVs written to", DE)
