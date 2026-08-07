"""Two more rose figures for the SAM orientation-artifact thread (CPU only, cached data).

FIG 1 `t2_pair_roses.png` — T2 repeat pairs: SAME GROUND, different sun. Per pair, both members for
  SAM and YOLO, with each member's own sun axis (orange) and the fitted population peak (dashed).
  This is the picture behind the T2 table: if the rose were illumination-driven the peak would swing
  WITH the sun between members; if ground-fixed it stays put.
FIG 2 `val_body_roses.png` — the T1-T5 validation set (`t35_val_detections.csv`) by BODY:
  Mars / Moon / Earth x SAM / YOLO / human. The human column is the rock-tracing reference.

Both at the standing well-resolved cut: aspect>1.35, dpx>=8.
  conda run -n bouldernet python data/sam2_validation/make_more_roses.py
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from YOLOv8BeyondEarth.orientation import half_plane_fraction as fl, asymmetry_ratio as asym

OUT = Path(r"c:/Users/brian/Documents/PhD/BoulderNet/YOLOv8-BeyondEarth/data/sam2_validation")
FIG = Path(r"c:/Users/brian/Documents/PhD/BoulderNet/docs/figures/sam_artifact")
MIN_DPX, ASP = 8.0, 1.35
BINS = np.arange(0, 181, 10); CTR = np.deg2rad(np.r_[BINS[:-1], BINS[:-1] + 180] + 5); W = np.deg2rad(10)
COL = {"sam": "tab:green", "yolo": "tab:blue", "human": "tab:gray"}


def axial_peak(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 8:
        return np.nan, np.nan
    z = np.exp(2j * np.deg2rad(a)).mean()
    return (np.rad2deg(np.angle(z)) / 2.0) % 180.0, float(abs(z))


def rose(ax, a, color, title, sun=None, mark_grid=True):
    a = np.asarray(a, float); a = a[np.isfinite(a)] % 180.0
    c, _ = np.histogram(a, bins=BINS); r = np.r_[c, c] / max(1, c.sum())
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.bar(CTR, r, width=W, color=color, alpha=0.6, edgecolor="k", lw=0.3)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(th, np.full_like(th, 1 / 18), "k--", lw=0.6, alpha=0.6)          # uniform
    top = r.max() * 1.05
    if mark_grid:
        for d in (45, 135):
            ax.plot([np.deg2rad(d)] * 2, [0, top], color="purple", ls=":", lw=0.9, alpha=0.8)
    pk, R = axial_peak(a)
    if np.isfinite(pk):                                                       # fitted peak
        pr = np.deg2rad(pk)
        ax.plot([pr, pr + np.pi], [top, top], color=color, ls="--", lw=2.0, zorder=7)
    if sun is not None:                                                       # sun axis
        sr = np.deg2rad(sun)
        ax.plot([sr, sr + np.pi], [top * 1.02] * 2, color="orange", lw=2.4, zorder=6)
    ax.set_yticklabels([]); ax.set_xticklabels([]); ax.set_title(title, fontsize=8.5)
    return pk, R


# ---------------- FIG 1: T2 repeat pairs ----------------
d2 = pd.read_csv(OUT / "t2_hirise_sam.csv")
pairs = list(d2.pair.unique())
fig, axes = plt.subplots(len(pairs), 4, figsize=(14, 3.6 * len(pairs)), subplot_kw={"projection": "polar"})
axes = np.atleast_2d(axes)
for i, pid in enumerate(pairs):
    p = d2[d2.pair == pid]; scenes = list(p.scene.unique())
    col = 0
    for side in ("sam", "yolo"):
        for sc in scenes:
            s = p[(p.scene == sc) & (p[f"aspect_{side}"] > ASP) & (p.dpx_sam >= MIN_DPX)]
            sun = float(s.sun.iloc[0]) if len(s) else np.nan
            inc = float(s.inc.iloc[0]) if len(s) else np.nan
            pk0, R0 = axial_peak(s[f"angle_{side}"])
            rose(axes[i, col], s[f"angle_{side}"], COL[side],
                 f"{pid} {side}\n{sc[4:]}  sun={sun:.0f}° inc={inc:.0f}°\n"
                 f"n={len(s)} pk={pk0:.0f}° R={R0:.2f}", sun=sun)
            col += 1
fig.suptitle("T2 — repeat pairs: SAME GROUND, different sun (orange = that member's sun axis, "
             "dashed = fitted population peak, purple dotted = pixel diagonals).\nIllumination would swing "
             "the peak WITH the sun between the two members of a pair; a ground-fixed signal keeps it put. "
             "Columns: SAM member A, SAM member B, YOLO A, YOLO B.", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "t2_pair_roses.png", dpi=115, bbox_inches="tight")
print("saved", FIG / "t2_pair_roses.png")

# ---------------- FIG 2: validation set by body ----------------
t = pd.read_csv(OUT / "t35_val_detections.csv")
bodies = ["mars", "moon", "earth"]
fig2, ax2 = plt.subplots(3, 3, figsize=(10.5, 11), subplot_kw={"projection": "polar"})
for i, body in enumerate(bodies):
    b = t[t.body == body]
    sets = [("sam", b.loc[(b.aspect_sam > ASP) & (b.dpx_sam >= MIN_DPX), "angle_sam"]),
            ("yolo", b.loc[(b.aspect_yolo > ASP) & (b.dpx_sam >= MIN_DPX), "angle_yolo"]),
            ("human", b.loc[(b.h_aspect > ASP) & (b.h_dpx >= MIN_DPX), "h_angle"])]
    for j, (tag, a) in enumerate(sets):
        a = a.dropna()
        pk0, R0 = axial_peak(a)
        rose(ax2[i, j], a, COL[tag],
             f"{body} — {tag}\nn={len(a)} pk={pk0:.0f}° R={R0:.2f}\n"
             f"f_low={fl(a):.2f} asym={asym(a):.2f}")
fig2.suptitle("Validation set (boulder2024 val tiles, aspect>1.35 & dpx≥8) by body: SAM vs YOLO vs HUMAN.\n"
              "Pooled across many scenes per body, so real per-scene fabric partly cancels — read SAM-vs-human "
              "SHAPE agreement, not absolute peaks.\n(dashed = fitted peak, purple dotted = pixel diagonals, "
              "black dashed circle = uniform)", fontsize=9.5)
fig2.tight_layout(); fig2.savefig(FIG / "val_body_roses.png", dpi=115, bbox_inches="tight")
print("saved", FIG / "val_body_roses.png")
