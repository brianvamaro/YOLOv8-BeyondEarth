"""Build the figures for docs/orientation_results_summary.md from cached run outputs.

No GPU / no inference — everything is read from the CSV/NPZ caches written by the SolutionG and
SolutionH run scripts (data/solutionG, data/solutionH):
  1. summary/orientation_threeway.png  — YOLO-seg vs G/SAM vs H-obb R-A on the real HiRISE crop
     (rose + linear histogram per method, elongated subset).
  2. summary/G_hist_linear.png         — YOLO vs SAM (sam256) long-axis linear histogram.
  3. summary/G_axis_fractions.png      — per-axis (0/45/90/135) azimuth fractions, YOLO vs SAM:
     the quantification behind the linear histogram.
  4. summary/G_roses.png               — YOLO vs SAM axial roses.
  5. summary/G_synth_rose.png          — SAM on synthetic uniform-truth boulders (shadowed arm
     only, matching the OBB-null render convention): the "SAM adds no artifact" control.
  6. summary/G_diagfrac_by_band.png    — diagonal fraction by native-px size band, YOLO vs SAM.
  7. solutionH/ra_null.png             — the section 4c-null: uniform truth vs d0 vs R-A.
  8. solutionH/ra_synth_overlay.png    — one synthetic uniform-truth tile, d0 and R-A boxes
     overlaid (truth orange, d0 red dashed, R-A cyan).
The G figures are the summary-doc's sam256-focused versions of the SolutionG notebook figures (the
notebook keeps the full YOLO/sam256/sam1024 comparison and both synthetic arms as the record; the
summary uses the chosen config/arm only). Everything reproduces cells of the SolutionG/SolutionH
notebooks.
One palette across the document: YOLO-seg blue, SAM green, OBB-d0 red, OBB-R-A cyan,
truth/uniform gray — matching the SolutionG notebook figures.
Run: python scripts/solutionH/make_orientation_summary_figs.py
"""
import os

import matplotlib
import numpy as np
import pandas as pd

from YOLOv8BeyondEarth.benchmark import orientation_by_band, DEFAULT_SIZE_BANDS
from YOLOv8BeyondEarth.orientation import (plot_rose, diagonal_fraction, grid_fraction,
                                           asymmetry_ratio, grid_fraction_uniform, axial_distance)

INNER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO = os.path.abspath(os.path.join(INNER, ".."))
GDIR = os.path.join(INNER, "data", "solutionG")
HDIR = os.path.join(INNER, "data", "solutionH")
FIG_SUMMARY = os.path.join(REPO, "docs", "figures", "summary")
FIG_H = os.path.join(REPO, "docs", "figures", "solutionH")
UNI = grid_fraction_uniform()   # 0.222 — the isotropic reference, not a method

ASPECT_MIN = 1.35               # elongated = meaningful long axis (matches well_resolved_subset)

# one palette across the whole document (matches the SolutionG notebook figures)
C_YOLO, C_SAM = "tab:blue", "tab:green"
C_D0, C_RA, C_TRUTH = "tab:red", "tab:cyan", "0.4"


def elongated(ang, asp):
    d = pd.DataFrame({"a": np.asarray(ang, dtype=float), "r": np.asarray(asp, dtype=float)}).dropna()
    return d.loc[d.r > ASPECT_MIN, "a"].values % 180.0


def rose_and_hist(fig, ncols, i, name, a, col):
    import matplotlib.pyplot as plt  # noqa: F401  (registered backend)
    axr = fig.add_subplot(2, ncols, i + 1, projection="polar")
    plot_rose(axr, a, bins=36, color=col,
              title=f"{name}\ndiag {diagonal_fraction(a):.2f} · grid {grid_fraction(a):.2f}")
    axh = fig.add_subplot(2, ncols, ncols + i + 1)
    axh.hist(a, bins=np.arange(0, 181, 5), color=col, alpha=0.85, density=True)
    axh.axhline(1.0 / 180.0, color="k", lw=0.8, ls="--", alpha=0.6)
    for x in (0, 45, 90, 135, 180):
        axh.axvline(x, color="k", lw=0.6, ls=":" if x in (45, 135) else "-", alpha=0.5)
    axh.set_xlim(0, 180); axh.set_xticks([0, 45, 90, 135, 180])
    axh.set_xlabel("azimuth from North (°)")
    return axh


def draw_obb(ax, corners, color="cyan", lw=1.2, ls="-"):
    for c in corners:               # c: (4, 2) col,row
        p = np.vstack([c, c[0]])
        ax.plot(p[:, 0], p[:, 1], ls, color=color, lw=lw)


def axis_fraction(a, center, tol_deg=10.0):
    """Share of azimuths within tol of ONE axis (uniform reference = 2*tol/180 = 0.111)."""
    return float((axial_distance(a, center) <= tol_deg).mean())


def main():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_SUMMARY, exist_ok=True)
    gdet = pd.read_csv(os.path.join(GDIR, "G_detections.csv"))
    crop_rot = pd.read_csv(os.path.join(HDIR, "obb_crop_angles_rot.csv"))
    null_d0 = pd.read_csv(os.path.join(HDIR, "obb_null_pred.csv"))
    null_rot = pd.read_csv(os.path.join(HDIR, "obb_null_pred_rot.csv"))
    null_truth = pd.read_csv(os.path.join(HDIR, "obb_null_truth.csv"))

    # 1 — the headline three-way on the real crop
    series = [("YOLOv8-seg mask (before)", elongated(gdet.angle_yolo, gdet.aspect_yolo), C_YOLO),
              ("G — SAM2 re-mask (fix)", elongated(gdet.angle_sam256, gdet.aspect_sam256), C_SAM),
              ("H-obb R-A — OBB head, degrees=180", elongated(crop_rot.angle180, crop_rot.aspect), C_RA)]
    fig = plt.figure(figsize=(13.5, 6.4))
    for i, (name, a, col) in enumerate(series):
        axh = rose_and_hist(fig, 3, i, f"{name}\n(n={len(a):,})", a, col)
        if i == 0:
            axh.set_ylabel("density")
        print(f"real crop  {name:38} n={len(a):>6}  diag {diagonal_fraction(a):.3f}  "
              f"grid {grid_fraction(a):.3f}  chir {asymmetry_ratio(a):.2f}")
    fig.suptitle("Same HiRISE crop, elongated subset — YOLO locks to the 135° diagonal; "
                 "SAM2 re-masking dissolves it; the rotation-aug OBB head keeps a mild cardinal lean",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_SUMMARY, "orientation_threeway.png"), dpi=130, bbox_inches="tight")

    # G figures (summary-doc versions, sam256-only): linear hist, roses, diag-frac by band.
    # The SolutionG notebook keeps YOLO/sam256/sam1024; here we show only the production config.
    yolo_a = elongated(gdet.angle_yolo, gdet.aspect_yolo)
    sam_a = elongated(gdet.angle_sam256, gdet.aspect_sam256)
    gseries = [("YOLO-seg", yolo_a, C_YOLO), ("SAM2 (sam256)", sam_a, C_SAM)]

    # 2 — linear histogram over [0, 180°), diagonals shaded
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for name, a, col in gseries:
        ax.hist(a, bins=36, range=(0, 180), density=True, histtype="step", lw=2,
                color=col, label=f"{name} (diag-frac {diagonal_fraction(a):.3f})")
    ax.axhline(1 / 180, ls="--", color="gray", lw=1, label="uniform")
    for d in (45, 135):
        ax.axvspan(d - 10, d + 10, color="red", alpha=0.07)
    ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("long-axis azimuth from North (deg)"); ax.set_ylabel("density")
    ax.set_title("YOLO's 135° diagonal lock → SAM2 flattens it (elongated subset, HiRISE)")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(FIG_SUMMARY, "G_hist_linear.png"), dpi=130, bbox_inches="tight")

    # 3 — per-axis fractions (the metric behind the linear histogram)
    axes_deg = (0, 45, 90, 135)
    uni_axis = 2 * 10.0 / 180.0                       # 0.111 — one axis's share under uniformity
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.34
    for j, (name, a, col) in enumerate(gseries):
        fr = [axis_fraction(a, c) for c in axes_deg]
        ax.bar(np.arange(4) + (j - 0.5) * w, fr, width=w, color=col, label=name)
        print(f"axis-frac  {name:14}  " + "  ".join(f"{c}°={f:.3f}" for c, f in zip(axes_deg, fr)))
    for j in (1, 3):                                  # highlight the diagonal (artifact) axes
        ax.axvspan(j - 0.5, j + 0.5, color="red", alpha=0.06)
    ax.axhline(uni_axis, color="k", ls="--", lw=1.0, label=f"uniform ({uni_axis:.3f}/axis)")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels([f"{c}°" for c in axes_deg])
    ax.set_xlabel("pixel axis (±10°); diagonals shaded"); ax.set_ylabel("fraction of azimuths")
    ax.set_title("Per-axis share of long-axis azimuths, matched detections (elongated subset)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_SUMMARY, "G_axis_fractions.png"), dpi=130, bbox_inches="tight")

    # 4 — axial roses, YOLO vs SAM
    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(9, 4.4))
    for ax, (name, a, col) in zip(axes, gseries):
        plot_rose(ax, a, bins=36, color=col, title=f"{name}  (diag-frac {diagonal_fraction(a):.3f})")
    fig.suptitle("Long-axis rose (North up, clockwise; elongated aspect>1.35): "
                 "YOLO diagonal lock → SAM2 dissolved")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_SUMMARY, "G_roses.png"), dpi=130, bbox_inches="tight")

    # 5 — synthetic uniform-truth control, shadowed arm only (same render convention as the
    # OBB null): does SAM ADD a directional artifact? (uniform in -> uniform out)
    synth = pd.read_csv(os.path.join(GDIR, "synth_rose", "synth_rose.csv"))
    d = synth[synth.cond == "shadow"]
    t = d.truth_angle.values % 180.0
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.hist(t, bins=36, range=(0, 180), density=True, histtype="stepfilled",
            color="gray", alpha=0.25, label=f"truth (uniform, n={len(t)})")
    for s, name, col in (("yolo", "YOLO-seg", C_YOLO), ("sam256", "SAM2 (sam256)", C_SAM)):
        a = d[s].values
        a = a[np.isfinite(a)] % 180.0
        ax.hist(a, bins=36, range=(0, 180), density=True, histtype="step", lw=2,
                color=col, label=f"{name} (diag {diagonal_fraction(a):.3f})")
        print(f"synth-ctl  {name:14}  n={len(a):>5}  diag {diagonal_fraction(a):.3f}  "
              f"chir {asymmetry_ratio(a):.2f}")
    ax.axhline(1 / 180, ls="--", color="k", lw=0.8)
    for dg in (45, 135):
        ax.axvspan(dg - 10, dg + 10, color="red", alpha=0.06)
    ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("long-axis azimuth (deg)"); ax.set_ylabel("density")
    ax.set_title("Synthetic boulders (shadowed, elongated), UNIFORM truth: "
                 "YOLO manufactures a diagonal excess; SAM2 stays ~uniform")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_SUMMARY, "G_synth_rose.png"), dpi=130, bbox_inches="tight")

    # 6 — diagonal fraction by native-px size band
    bt = {s: orientation_by_band(gdet[f"angle_{s}"], gdet[f"aspect_{s}"], gdet["dpx_native"],
                                 grid_axis="diagonal") for s in ("yolo", "sam256")}
    common = [b for b in bt["yolo"].band if b != "all" and b in set(bt["sam256"].band)]
    x = np.arange(len(common)); w = 0.34
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, (s, name, col) in enumerate([("yolo", "YOLO-seg", C_YOLO), ("sam256", "SAM2 (sam256)", C_SAM)]):
        vals = [bt[s].set_index("band").loc[b, "frac"] for b in common]
        ax.bar(x + (j - 0.5) * w, vals, w, color=col, label=name)
    ax.axhline(UNI, ls="--", color="gray", label=f"uniform {UNI:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(common); ax.set_xlabel("native-px size band")
    ax.set_ylabel("diagonal fraction")
    ax.set_title("YOLO's diagonal excess is in the 6–14 px bulk; SAM2 removes it everywhere")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(FIG_SUMMARY, "G_diagfrac_by_band.png"), dpi=130, bbox_inches="tight")

    # 7 — the section 4c-null: uniform truth in (left) -> d0 out vs R-A out
    truth_a = null_truth.angle180.values % 180.0
    d0_a = elongated(null_d0.angle180, null_d0.aspect)
    panels = [("TRUTH (rendered, uniform)", truth_a, C_TRUTH),
              ("H-obb d0 (degrees=0) OUTPUT", d0_a, C_D0),
              ("H-obb R-A (degrees=180) OUTPUT", elongated(null_rot.angle180, null_rot.aspect), C_RA)]
    fig = plt.figure(figsize=(13, 6))
    for i, (name, a, col) in enumerate(panels):
        rose_and_hist(fig, 3, i, name, a, col)
        print(f"4c-null    {name:38} n={len(a):>6}  diag {diagonal_fraction(a):.3f}  "
              f"grid {grid_fraction(a):.3f}")
    fig.suptitle("§4c-null (uniform-truth in → what comes out): as trained the head snaps to a hard "
                 "0/90 cross (grid 0.58); rotation aug cuts its prior ~70% (0.32) but stays short of "
                 f"uniform ({UNI:.2f})", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_H, "ra_null.png"), dpi=130, bbox_inches="tight")

    # 8 — the synthetic overlay: d0 and R-A boxes on the SAME field (same render seed)
    from matplotlib.lines import Line2D
    ex_d0 = np.load(os.path.join(HDIR, "obb_examples.npz"))
    ex_rot = np.load(os.path.join(HDIR, "obb_examples_rot.npz"))
    assert np.array_equal(ex_d0["synth_img"], ex_rot["synth_img"]), "d0/R-A synth fields differ"
    fig, ax = plt.subplots(figsize=(8.6, 8.6))
    ax.imshow(ex_d0["synth_img"], cmap="gray")
    for cx, cy, a_px, b_px, ang in ex_d0["synth_truth"]:
        A = np.deg2rad(ang)
        dx, dy = np.sin(A) * a_px, -np.cos(A) * a_px
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], "-", color="orange", lw=1.8)
    draw_obb(ax, ex_rot["synth_corners"], color="cyan", lw=1.3, ls="-")
    draw_obb(ax, ex_d0["synth_corners"], color="red", lw=1.7, ls="--")  # dashed on top: gaps keep cyan visible
    ax.legend(handles=[Line2D([], [], color="orange", lw=1.8, label="true long axis"),
                       Line2D([], [], color="red", lw=1.5, ls="--", label="d0 box (degrees=0)"),
                       Line2D([], [], color="cyan", lw=1.3, label="R-A box (degrees=180)")],
              loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_title("Same synthetic uniform-truth field: d0 boxes snap axis-aligned, "
                 "R-A boxes track the tilt", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_H, "ra_synth_overlay.png"), dpi=130, bbox_inches="tight")
    print("wrote summary/:", "orientation_threeway, G_hist_linear, G_axis_fractions, G_roses, "
          "G_synth_rose, G_diagfrac_by_band")
    print("wrote solutionH/:", "ra_null, ra_synth_overlay")


if __name__ == "__main__":
    main()
