"""Build the three figures for docs/orientation_results_summary.md from cached run outputs.

No GPU / no inference — everything is read from the CSV/NPZ caches written by the SolutionG and
SolutionH run scripts (data/solutionG, data/solutionH):
  1. summary/orientation_threeway.png  — YOLO-seg vs G/SAM vs H-obb R-A on the real HiRISE crop
     (rose + linear histogram per method, elongated subset).
  2. summary/G_axis_fractions.png      — per-axis (0/45/90/135) azimuth fractions for the three
     mask configs: the quantification behind the G linear histograms.
  3. solutionH/ra_null.png             — the section 4c-null, d0 vs R-A vs uniform truth.
  4. solutionH/ra_synth_overlay.png    — one synthetic uniform-truth tile, d0 and R-A boxes
     overlaid (truth yellow, d0 red dashed, R-A cyan).
These reproduce cells of notebooks/SolutionH_obb_retrain (the citable record) as standalone PNGs.
Run: python scripts/solutionH/make_orientation_summary_figs.py
"""
import os

import matplotlib
import numpy as np
import pandas as pd

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
    series = [("YOLOv8-seg mask (before)", elongated(gdet.angle_yolo, gdet.aspect_yolo), "tab:red"),
              ("G — SAM2 re-mask (fix)", elongated(gdet.angle_sam256, gdet.aspect_sam256), "tab:green"),
              ("H-obb R-A — OBB head, degrees=180", elongated(crop_rot.angle180, crop_rot.aspect), "tab:purple")]
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

    # 2 — per-axis fractions for the three mask configs (the metric behind G's linear hists)
    axes_deg = (0, 45, 90, 135)
    configs = [("yolo", elongated(gdet.angle_yolo, gdet.aspect_yolo), "tab:blue"),
               ("sam256", elongated(gdet.angle_sam256, gdet.aspect_sam256), "tab:green"),
               ("sam1024", elongated(gdet.angle_sam1024, gdet.aspect_sam1024), "tab:orange")]
    uni_axis = 2 * 10.0 / 180.0                       # 0.111 — one axis's share under uniformity
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.26
    for j, (name, a, col) in enumerate(configs):
        fr = [axis_fraction(a, c) for c in axes_deg]
        ax.bar(np.arange(4) + (j - 1) * w, fr, width=w, color=col, label=name)
        print(f"axis-frac  {name:8}  " + "  ".join(f"{c}°={f:.3f}" for c, f in zip(axes_deg, fr)))
    for j in (1, 3):                                  # highlight the diagonal (artifact) axes
        ax.axvspan(j - 0.5, j + 0.5, color="red", alpha=0.06)
    ax.axhline(uni_axis, color="k", ls="--", lw=1.0, label=f"uniform ({uni_axis:.3f}/axis)")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels([f"{c}°" for c in axes_deg])
    ax.set_xlabel("pixel axis (±10°); diagonals shaded"); ax.set_ylabel("fraction of azimuths")
    ax.set_title("Per-axis share of long-axis azimuths, matched detections (elongated subset)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_SUMMARY, "G_axis_fractions.png"), dpi=130, bbox_inches="tight")

    # 3 — the section 4c-null: d0 vs R-A vs uniform truth
    panels = [("H-obb d0 (degrees=0) OUTPUT", elongated(null_d0.angle180, null_d0.aspect), "tab:red"),
              ("H-obb R-A (degrees=180) OUTPUT", elongated(null_rot.angle180, null_rot.aspect), "tab:orange"),
              ("TRUTH (rendered, uniform)", null_truth.angle180.values % 180.0, "0.4")]
    fig = plt.figure(figsize=(13, 6))
    for i, (name, a, col) in enumerate(panels):
        rose_and_hist(fig, 3, i, name, a, col)
        print(f"4c-null    {name:38} n={len(a):>6}  diag {diagonal_fraction(a):.3f}  "
              f"grid {grid_fraction(a):.3f}")
    fig.suptitle("§4c-null (uniform-truth in → what comes out): rotation aug cuts the head's own "
                 f"cardinal prior ~70% (grid 0.58 → 0.32) but stays short of uniform ({UNI:.2f})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_H, "ra_null.png"), dpi=130, bbox_inches="tight")

    # 4 — the synthetic overlay: d0 and R-A boxes on the SAME field (same render seed)
    from matplotlib.lines import Line2D
    ex_d0 = np.load(os.path.join(HDIR, "obb_examples.npz"))
    ex_rot = np.load(os.path.join(HDIR, "obb_examples_rot.npz"))
    assert np.array_equal(ex_d0["synth_img"], ex_rot["synth_img"]), "d0/R-A synth fields differ"
    fig, ax = plt.subplots(figsize=(8.6, 8.6))
    ax.imshow(ex_d0["synth_img"], cmap="gray")
    for cx, cy, a_px, b_px, ang in ex_d0["synth_truth"]:
        A = np.deg2rad(ang)
        dx, dy = np.sin(A) * a_px, -np.cos(A) * a_px
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], "-", color="yellow", lw=1.8)
    draw_obb(ax, ex_rot["synth_corners"], color="cyan", lw=1.3, ls="-")
    draw_obb(ax, ex_d0["synth_corners"], color="red", lw=1.7, ls="--")  # dashed on top: gaps keep cyan visible
    ax.legend(handles=[Line2D([], [], color="yellow", lw=1.8, label="true long axis"),
                       Line2D([], [], color="red", lw=1.5, ls="--", label="d0 box (degrees=0)"),
                       Line2D([], [], color="cyan", lw=1.3, label="R-A box (degrees=180)")],
              loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_title("Same synthetic uniform-truth field: d0 boxes snap axis-aligned, "
                 "R-A boxes track the tilt", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_H, "ra_synth_overlay.png"), dpi=130, bbox_inches="tight")
    print("wrote:", os.path.join(FIG_SUMMARY, "orientation_threeway.png"), "and G_axis_fractions.png")
    print("wrote:", os.path.join(FIG_H, "ra_null.png"), "and ra_synth_overlay.png")


if __name__ == "__main__":
    main()
