# Solution G card + gating analysis (CPU). Reads G_detections.csv (aligned yolo/sam256/sam1024
# angles + native size per detection) written by run_G.py. Produces: per-side diag-fraction &
# chirality (elongated subset, overall + by native band); the gating sweep (SAM for dpx<tau, YOLO
# else -> smallest tau reaching uniform); area ratios; and the implied per-scene cost.
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
from YOLOv8BeyondEarth.orientation import diagonal_fraction, asymmetry_ratio

G = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionG")
UNIFORM = 2 * 10.0 / 90.0  # diagonal_fraction uniform at +/-10 deg = 0.222
ASPECT_MIN = 1.35
BANDS = [(3, 4), (4, 6), (6, 8), (8, 10), (10, 14), (14, np.inf)]
# full HiRISE scene / the 4096 crop -> per-scene extrapolation factor (area ratio)
CROP_PX = 4096 * 4096
SCENE_PX = 22490 * 85922


def frac_chir(ang):
    ang = ang[np.isfinite(ang)]
    return diagonal_fraction(ang), asymmetry_ratio(ang)


def per_side_table(df):
    rows = []
    for side in ("yolo", "sam256", "sam1024"):
        a = df[f"angle_{side}"].values
        asp = df[f"aspect_{side}"].values
        el = (asp > ASPECT_MIN) & np.isfinite(a)
        dpx = df["dpx_native"].values
        f, c = frac_chir(a[el])
        rows.append(dict(side=side, band="all", n=int(el.sum()), diag_frac=f, chirality=c))
        for lo, hi in BANDS:
            m = el & (dpx >= lo) & (dpx < hi)
            if m.sum() >= 30:
                f, c = frac_chir(a[m])
                rows.append(dict(side=side, band=f"{lo}-{hi if np.isfinite(hi) else 'inf'}",
                                 n=int(m.sum()), diag_frac=f, chirality=c))
    return pd.DataFrame(rows)


def gating_sweep(df, side="sam256"):
    """Combined rose: SAM angle where dpx_native < tau, YOLO angle otherwise. Elongated subset by
    the YOLO aspect (the production gate is applied before re-masking, so selection uses YOLO)."""
    a_yolo = df["angle_yolo"].values
    a_sam = df[f"angle_{side}"].values
    asp_yolo = df["aspect_yolo"].values
    dpx = df["dpx_native"].values
    rows = []
    for tau in [0, 4, 5, 6, 7, 8, 10, 12, np.inf]:
        use_sam = dpx < tau
        ang = np.where(use_sam, a_sam, a_yolo)
        el = (asp_yolo > ASPECT_MIN) & np.isfinite(ang)
        f, c = frac_chir(ang[el])
        rows.append(dict(tau=("inf" if not np.isfinite(tau) else tau),
                         remask_frac=float(use_sam.mean()),
                         remask_frac_elong=float((use_sam & (asp_yolo > ASPECT_MIN)).mean()),
                         diag_frac=f, chirality=c, n_elong=int(el.sum())))
    return pd.DataFrame(rows)


def area_ratio_table(df):
    rows = []
    dpx = df["dpx_native"].values
    for side in ("sam256", "sam1024"):
        r = df[f"area_{side}"].values / np.maximum(df["area_yolo"].values, 1)
        # sam1024 areas are in imgsz px (4x native linear -> 16x area); normalize to native
        scale = 16.0 if side == "sam1024" else 1.0
        r = r / scale
        for lo, hi in [(0, np.inf)] + BANDS:
            m = (dpx >= lo) & (dpx < hi) & np.isfinite(r)
            if m.sum() >= 30:
                rows.append(dict(side=side,
                                 band=("all" if lo == 0 and not np.isfinite(hi)
                                       else f"{lo}-{hi if np.isfinite(hi) else 'inf'}"),
                                 n=int(m.sum()), median_ratio=float(np.median(r[m]))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(G / "G_detections.csv")
    meta = json.loads((G / "G_run_meta.json").read_text())
    ms_per = meta["ms_per_boulder"]

    print(f"n={len(df)}  wall={meta['wall_s']:.0f}s  {ms_per:.1f} ms/boulder  "
          f"failed={meta['failed']}")
    print(f"\n=== per-side diag-fraction (uniform={UNIFORM:.3f}) ===")
    ps = per_side_table(df); print(ps.to_string(index=False)); ps.to_csv(G / "G_per_side.csv", index=False)

    print("\n=== gating sweep (sam256 for dpx<tau, YOLO else; elongated) ===")
    gs = gating_sweep(df, "sam256"); print(gs.to_string(index=False)); gs.to_csv(G / "G_gating.csv", index=False)

    print("\n=== area ratio (SAM vs YOLO, native-normalized) ===")
    ar = area_ratio_table(df); print(ar.to_string(index=False)); ar.to_csv(G / "G_area.csv", index=False)

    # per-scene cost extrapolation
    scene_factor = SCENE_PX / CROP_PX
    ungated_scene_s = meta["wall_s"] * scene_factor
    print(f"\n=== cost ===")
    print(f"crop re-mask wall: {meta['wall_s']:.0f}s for {len(df)} boulders "
          f"({ms_per:.1f} ms/boulder)")
    print(f"ungated full-scene extrapolation (x{scene_factor:.1f} area): "
          f"{ungated_scene_s/3600:.1f} h")
    for tau in (5, 6, 8):
        frac = float((df['dpx_native'].values < tau).mean())
        print(f"  gated tau={tau}: re-mask {frac*100:.0f}% -> "
              f"{ungated_scene_s*frac/3600:.1f} h/scene")
