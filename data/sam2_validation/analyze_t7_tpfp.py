"""T7 (SAM) — detection-selection / TP-FP orientation bias, from the t35 val run (no GPU).
Split SAM's elongated detections into TP (matched to a human label) vs FP (unmatched) and compare
their roses. If the sun-ward lean lives in the FP set, it's a detection-selection artifact (SAM
finds spurious sun-aligned blobs); if TP leans too, it's in the real boulders.
Filters: aspect_sam>1.35, dpx_sam>=8 (the well-resolved regime)."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from YOLOv8BeyondEarth.orientation import (half_plane_fraction as fl, asymmetry_ratio as asym,
                                           axis_alignment_fraction as align, axial_distance)
OUT = Path(r"c:/Users/brian/Documents/PhD/BoulderNet/YOLOv8-BeyondEarth/data/sam2_validation")
FIG = Path(r"c:/Users/brian/Documents/PhD/BoulderNet/docs/figures/sam_artifact")
t = pd.read_csv(OUT / "t35_val_detections.csv")
SUN = {"ESP_017355_2260_RED": 75.3, "ESP_025650_2165_RED": 77.5, "ESP_028537_2270_RED": 65.2}
BINS = np.arange(0, 181, 10); CTR = np.deg2rad(np.r_[BINS[:-1], BINS[:-1] + 180] + 5); W = np.deg2rad(10)


def stats(a, sun):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    al = align(a, sun, 45) if sun is not None else np.nan
    return len(a), fl(a), asym(a), al


def rose(ax, a, sun, color):
    a = np.asarray(a, float); a = a[np.isfinite(a)] % 180.0
    c, _ = np.histogram(a, bins=BINS); r = np.r_[c, c] / max(1, c.sum())
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.bar(CTR, r, width=W, color=color, alpha=0.6, edgecolor="k", lw=0.3)
    th = np.linspace(0, 2 * np.pi, 200); ax.plot(th, np.full_like(th, 1 / 18), "k--", lw=0.6, alpha=0.6)
    sr = np.deg2rad(sun); ax.plot([sr, sr + np.pi], [r.max() * 1.05] * 2, color="orange", lw=2.2, zorder=6)
    ax.set_yticklabels([]); ax.set_xticklabels([])


print("T7 SAM TP/FP split (aspect>1.35, dpx>=8). along-sun null 0.5, asym null 1.0, f_lower null 0.5\n")
print(f"{'scene':22s} {'sun':>5s} | {'set':4s} {'n':>5s} {'f_low':>6s} {'asym':>6s} {'alongSun':>8s}")
for scene, sun in SUN.items():
    b = t[t.scene == scene]
    s = b[(b.aspect_sam > 1.35) & (b.dpx_sam >= 8)]
    tp = s[s.h_angle.notna()]; fp = s[s.h_angle.isna()]
    for tag, d in [("TP", tp), ("FP", fp), ("hum", s[s.h_angle.notna()])]:
        col = d.h_angle if tag == "hum" else d.angle_sam
        n, f, a, al = stats(col, sun)
        print(f"{scene[:22]:22s} {sun:5.1f} | {tag:4s} {n:5d} {f:6.3f} {a:6.2f} {al:8.3f}")
    print(f"{'':22s}       | FP fraction of SAM-elong = {len(fp)/max(1,len(s)):.2f}\n")

# figure for 2260 (the strong-lean scene): TP-SAM, FP-SAM, human
b = t[t.scene == "ESP_017355_2260_RED"]; s = b[(b.aspect_sam > 1.35) & (b.dpx_sam >= 8)]
fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), subplot_kw={"projection": "polar"})
rose(ax[0], s.loc[s.h_angle.notna(), "angle_sam"], 75.3, "tab:green")
ax[0].set_title(f"2260 SAM — TP (matched)\nn={s.h_angle.notna().sum()}", fontsize=9)
rose(ax[1], s.loc[s.h_angle.isna(), "angle_sam"], 75.3, "tab:red")
ax[1].set_title(f"2260 SAM — FP (unmatched)\nn={s.h_angle.isna().sum()}", fontsize=9)
rose(ax[2], b.loc[(b.h_aspect > 1.35) & (b.h_dpx >= 8), "h_angle"], 75.3, "tab:gray")
ax[2].set_title("2260 HUMAN", fontsize=9)
fig.suptitle("T7 — 2260 SAM detections split by match to human (orange = sun 75°). "
             "Is the sun-ward lean in the spurious (FP) detections?", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "t7_tpfp_2260.png", dpi=120, bbox_inches="tight")
print("saved", FIG / "t7_tpfp_2260.png")
