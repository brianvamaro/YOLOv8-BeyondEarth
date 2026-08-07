"""Per-scene boulder-orientation rose plots for the SAM-chirality characterization.
Fig 1: 11 HiRISE scenes (generality run), SAM roses sorted by sun axis, sun axis marked.
Fig 2: the two human-labeled HiRISE val scenes (2260, 2270), SAM vs human roses.
Roses: elongated (aspect>1.35) long-axis azimuths, North up, CW, doubled to [0,360); per-scene
fraction (so shapes compare across n)."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from YOLOv8BeyondEarth.orientation import half_plane_fraction as fl, asymmetry_ratio as asym

OUT = Path(r"c:/Users/brian/Documents/PhD/BoulderNet/YOLOv8-BeyondEarth/data/sam2_validation")
FIG = Path(r"c:/Users/brian/Documents/PhD/BoulderNet/docs/figures/sam_artifact")
MIN_DPX = 8.0          # size floor: below the ~8 px pixelation-floor knee angles are noise (Component A)
BINS = np.arange(0, 181, 10)
CTR = np.deg2rad(np.r_[BINS[:-1], BINS[:-1] + 180] + 5)
W = np.deg2rad(10)


def rose(ax, a, sun=None, color="tab:green", label=None):
    a = np.asarray(a, float); a = a[np.isfinite(a)] % 180.0
    c, _ = np.histogram(a, bins=BINS)
    frac = c / max(1, c.sum())
    r = np.r_[frac, frac]
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.bar(CTR, r, width=W, color=color, alpha=0.55, edgecolor="k", lw=0.3)
    uni = 1.0 / len(BINS[:-1])                    # uniform per-bin fraction
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(th, np.full_like(th, uni), "k--", lw=0.6, alpha=0.6)
    if sun is not None:
        sr = np.deg2rad(sun)
        rm = r.max() * 1.05
        ax.plot([sr, sr + np.pi], [rm, rm], color="orange", lw=2.2, zorder=6)
    ax.set_yticklabels([]); ax.set_xticklabels([])
    return a


# ---- Fig 1: 11 generality scenes ----
df = pd.read_csv(OUT / "sam_chirality_hirise.csv")
order = df.groupby("scene").shadow_axis.first().sort_values()
scenes = order.index.tolist()
n = len(scenes); ncol = 4; nrow = int(np.ceil(n / ncol))
fig = plt.figure(figsize=(4 * ncol, 4 * nrow))
for i, sc in enumerate(scenes):
    ax = fig.add_subplot(nrow, ncol, i + 1, projection="polar")
    s = df[df.scene == sc]; a = s.angle_sam[(s.aspect_sam > 1.35) & (s.dpx_sam >= MIN_DPX)]
    sun = float(s.shadow_axis.iloc[0])
    aa = rose(ax, a, sun=sun)
    ax.set_title(f"{sc}\nsun {sun:.0f}\u00b0 | n={len(aa)} | asym {asym(aa):.2f} | fl {fl(aa):.2f}",
                 fontsize=8.5)
fig.suptitle(f"SAM boulder-orientation roses across HiRISE scenes \u2014 aspect>1.35, dpx\u2265{MIN_DPX:.0f} "
             "(orange = sun/shadow axis; dashed = uniform).\nWell-resolved subset: does removing the "
             "small-boulder pixelation noise sharpen a rotation with the sun?",
             fontsize=10.5, y=1.008)
fig.tight_layout(); fig.savefig(FIG / "hirise_roses_sam.png", dpi=115, bbox_inches="tight")
print("saved", FIG / "hirise_roses_sam.png")

# ---- Fig 2: the two human-labeled val scenes, SAM vs human ----
t = pd.read_csv(OUT / "t35_val_detections.csv")
SUN = {"ESP_017355_2260_RED": 75.3, "ESP_028537_2270_RED": (155.20 - 270) % 180}
fig2, axes = plt.subplots(2, 2, figsize=(9, 9), subplot_kw={"projection": "polar"})
for row, (scene, sun) in enumerate(SUN.items()):
    b = t[t.scene == scene]
    sam = b.angle_sam[(b.aspect_sam > 1.35) & (b.dpx_sam >= MIN_DPX)]
    hum = b.h_angle[(b.h_aspect > 1.35) & (b.h_dpx >= MIN_DPX)]
    aa = rose(axes[row, 0], sam, sun=sun, color="tab:green")
    axes[row, 0].set_title(f"{scene}  SAM\nn={len(aa)} asym {asym(aa):.2f} fl {fl(aa):.2f}", fontsize=9)
    hh = rose(axes[row, 1], hum, sun=sun, color="tab:red")
    axes[row, 1].set_title(f"{scene}  HUMAN\nn={np.isfinite(hh).sum()} asym {asym(hh):.2f} fl {fl(hh):.2f}",
                           fontsize=9)
fig2.suptitle(f"Human-labeled HiRISE val scenes: SAM (green) vs human (red). aspect>1.35, dpx≥{MIN_DPX:.0f}. "
              "Orange = sun axis.\n2260: SAM leans sun-ward, humans ~uniform (the divergence). 2270: both milder.",
              fontsize=10)
fig2.tight_layout(); fig2.savefig(FIG / "hirise_roses_sam_vs_human.png", dpi=115, bbox_inches="tight")
print("saved", FIG / "hirise_roses_sam_vs_human.png")
