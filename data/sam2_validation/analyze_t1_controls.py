"""T1 (SAM) analysis — on geologically-null control ground, does SAM prefer ANY axis, and which?

Per site: is the rose non-uniform (Rayleigh on doubled angles), and if so does the peak sit at
  the pixel grid (0/90) / diagonal (45/135) -> decoder-grid artifact
  that site's SUN axis                      -> illumination
  neither                                   -> unexplained (local fabric / noise)
Sites span 53-64°S + 69°N so sun and grid are de-confounded ACROSS sites: the aggregate
"mean |peak - X|" over sites is the discriminator (smallest wins).
Compares SAM vs YOLO on the SAME detections (the original Test 1 found YOLO peaked ~44-55°, i.e. the
45° diagonal, on this same null ground)."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from YOLOv8BeyondEarth.orientation import (half_plane_fraction as fl, asymmetry_ratio as asym,
                                           diagonal_fraction as diag, grid_fraction as grid,
                                           grid_fraction_uniform, axial_distance,
                                           axis_alignment_fraction as align)
OUT = Path(r"c:/Users/brian/Documents/PhD/BoulderNet/YOLOv8-BeyondEarth/data/sam2_validation")
FIG = Path(r"c:/Users/brian/Documents/PhD/BoulderNet/docs/figures/sam_artifact")
MIN_DPX, ASP = 8.0, 1.35
BINS = np.arange(0, 181, 10); CTR = np.deg2rad(np.r_[BINS[:-1], BINS[:-1] + 180] + 5); W = np.deg2rad(10)


def axial(a):
    """(peak_deg, R, rayleigh_p) for axial angles; p from z=nR^2, p~exp(-z) (approx, large n)."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 8:
        return np.nan, np.nan, np.nan
    z = np.exp(2j * np.deg2rad(a)).mean()
    R = float(abs(z)); pk = (np.rad2deg(np.angle(z)) / 2.0) % 180.0
    return pk, R, float(np.exp(-len(a) * R ** 2))


def nearest(pk, axes):
    return float(np.min([axial_distance(np.array([pk]), x)[0] for x in axes]))


def rose(ax, a, sun, color, title):
    a = np.asarray(a, float); a = a[np.isfinite(a)] % 180.0
    c, _ = np.histogram(a, bins=BINS); r = np.r_[c, c] / max(1, c.sum())
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.bar(CTR, r, width=W, color=color, alpha=0.6, edgecolor="k", lw=0.3)
    th = np.linspace(0, 2 * np.pi, 200); ax.plot(th, np.full_like(th, 1 / 18), "k--", lw=0.6, alpha=0.6)
    for d, col in [(45, "purple"), (135, "purple")]:
        ax.plot([np.deg2rad(d)] * 2, [0, r.max() * 1.05], color=col, ls=":", lw=1)
    sr = np.deg2rad(sun); ax.plot([sr, sr + np.pi], [r.max() * 1.02] * 2, color="orange", lw=2.2, zorder=6)
    ax.set_yticklabels([]); ax.set_xticklabels([]); ax.set_title(title, fontsize=8.5)


d = pd.read_csv(OUT / "t1_controls_sam.csv")
print(f"T1 SAM controls (aspect>{ASP}, dpx>={MIN_DPX:.0f}). uniform: diag=grid={grid_fraction_uniform(10):.3f}, "
      f"f_lower=0.5, asym=1.0\n")
hdr = (f"{'site':18s} {'sun':>5s} {'inc':>5s} {'n':>5s} | {'peak':>5s} {'R':>5s} {'p_uni':>7s} "
       f"{'|pk-sun|':>8s} {'|pk-card|':>9s} {'|pk-diag|':>9s} | {'diag':>5s} {'grid':>5s} {'f_low':>5s}")
rows = []
for side in ("sam", "yolo"):
    print(f"--- {side.upper()} ---"); print(hdr)
    for sc in d.scene.unique():
        s = d[(d.scene == sc) & (d[f"aspect_{side}"] > ASP) & (d.dpx_sam >= MIN_DPX)]
        a = s[f"angle_{side}"].to_numpy(); sun = float(d[d.scene == sc].sun.iloc[0])
        inc = float(d[d.scene == sc].incidence.iloc[0])
        pk, R, p = axial(a)
        if not np.isfinite(pk):
            print(f"{sc:18s} too few (n={len(a)})"); continue
        dsun, dcard, ddiag = axial_distance(np.array([pk]), sun)[0], nearest(pk, [0, 90]), nearest(pk, [45, 135])
        rows.append(dict(side=side, scene=sc, sun=sun, inc=inc, n=len(a), peak=pk, R=R, p=p,
                         d_sun=dsun, d_card=dcard, d_diag=ddiag, diag=diag(a), grid=grid(a),
                         fl=fl(a), asym=asym(a), along_sun=align(a, sun, 45)))
        print(f"{sc:18s} {sun:5.0f} {inc:5.0f} {len(a):5d} | {pk:5.0f} {R:5.2f} {p:7.1e} "
              f"{dsun:8.0f} {dcard:9.0f} {ddiag:9.0f} | {diag(a):5.3f} {grid(a):5.3f} {fl(a):5.2f}")
    print()
Rd = pd.DataFrame(rows)
print("=== AGGREGATE over control sites: which axis explains the peaks? (smallest mean wins; 45=random) ===")
for side in ("sam", "yolo"):
    r = Rd[Rd.side == side]
    print(f"  {side:4s}: mean |peak-SUN|={r.d_sun.mean():5.1f}  mean |peak-CARDINAL|={r.d_card.mean():5.1f}  "
          f"mean |peak-DIAGONAL|={r.d_diag.mean():5.1f}  | mean R={r.R.mean():.2f}  "
          f"n_sites_nonuniform(p<0.01)={int((r.p < 0.01).sum())}/{len(r)}  mean along-sun={r.along_sun.mean():.3f}")
Rd.to_csv(OUT / "t1_controls_perscene.csv", index=False)

sites = list(d.scene.unique())
fig, axes = plt.subplots(2, len(sites), figsize=(3.5 * len(sites), 7.4), subplot_kw={"projection": "polar"})
for j, sc in enumerate(sites):
    sun = float(d[d.scene == sc].sun.iloc[0])
    for i, side in enumerate(("sam", "yolo")):
        s = d[(d.scene == sc) & (d[f"aspect_{side}"] > ASP) & (d.dpx_sam >= MIN_DPX)]
        pk, R, p = axial(s[f"angle_{side}"])
        rose(axes[i, j], s[f"angle_{side}"], sun, "tab:green" if side == "sam" else "tab:blue",
             f"{sc}\n{side} n={len(s)} pk={pk:.0f}° R={R:.2f}\nsun={sun:.0f}°")
fig.suptitle("T1 — geologically-null control sites (aspect>1.35, dpx≥8). Orange = sun axis, "
             "purple dotted = pixel diagonals 45/135, dashed = uniform.\nTop SAM, bottom YOLO. On "
             "fabric-free ground any preferred axis is an artifact; WHERE it sits names the cause.",
             fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "t1_controls_roses.png", dpi=115, bbox_inches="tight")
print("\nsaved", FIG / "t1_controls_roses.png")
