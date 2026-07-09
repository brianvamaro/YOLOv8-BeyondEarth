"""Solution H pre-gate label audits (H plan, "Label audits"):
(a) angle rose of the *human* labels themselves — if the labels are grid-locked, that bounds
    everything downstream (and is a finding in its own right);
(b) aspect distribution — fraction of near-circular instances whose angle is ill-defined;
(c) size distribution vs the 4-14 px band the fix targets.
Plus an OBB-on-image overlay for a visual sanity check.

Reads the shipped seg polygons (ground truth), fits the unbiased EllipseModel, aggregates over a
split. Saves figures to docs/figures/solutionH/. Run: python scripts/audit_obb_labels.py [split]
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import Polygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from YOLOv8BeyondEarth.orientation import (ellipse_obb, plot_rose, grid_fraction, diagonal_fraction,
                                           grid_fraction_uniform)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "data_prieur", "boulder2024")
FIGDIR = os.path.join(REPO, "docs", "figures", "solutionH")
TILE_PX = 500
DENSIFY = 1.0 / TILE_PX
ASPECT_MIN = 1.35     # elongated cut (matches orientation.well_resolved_subset)
DPX_MIN = 8           # well-resolved cut (px)


def iter_polys(split):
    lbl = os.path.join(SRC, split, "labels")
    for fn in os.listdir(lbl):
        if not fn.endswith(".txt"):
            continue
        with open(os.path.join(lbl, fn)) as f:
            for row in f:
                v = row.split()
                if len(v) < 7:
                    continue
                xy = np.array(v[1:], dtype=float)
                if xy.size % 2:
                    continue
                yield xy.reshape(-1, 2)


def measure(poly_norm):
    """(angle180, aspect, diameter_px) for one normalized polygon; nan on failed fit."""
    _, ang, asp = ellipse_obb(Polygon(poly_norm), DENSIFY)
    if not np.isfinite(ang):
        return np.nan, np.nan, np.nan
    area_px = Polygon(poly_norm).area * TILE_PX * TILE_PX
    dpx = 2.0 * np.sqrt(area_px / np.pi)
    return ang, asp, dpx


def overlay(split, n_examples=6):
    """Render OBB corners over a few tiles that actually contain boulders."""
    from PIL import Image
    lbl = os.path.join(SRC, split, "labels")
    img = os.path.join(SRC, split, "images")
    picks, files = [], sorted(os.listdir(lbl))
    for fn in files:
        with open(os.path.join(lbl, fn)) as f:
            rows = [r for r in f if len(r.split()) >= 7]
        if len(rows) >= 8:
            picks.append((fn, rows))
        if len(picks) == n_examples:
            break
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, (fn, rows) in zip(axes.ravel(), picks):
        stem = fn[:-4]
        ipath = os.path.join(img, stem + ".png")
        ax.imshow(np.asarray(Image.open(ipath)), cmap="gray")
        for r in rows:
            xy = np.array(r.split()[1:], dtype=float).reshape(-1, 2) * TILE_PX
            corners, _, _ = ellipse_obb(Polygon(np.array(r.split()[1:], dtype=float).reshape(-1, 2)), DENSIFY)
            if corners is not None:
                c = np.vstack([corners, corners[0]]) * TILE_PX
                ax.plot(c[:, 0], c[:, 1], "-", color="lime", lw=0.8)
        ax.set_title(stem[:28], fontsize=7); ax.axis("off")
    fig.suptitle(f"OBB labels (green) from EllipseModel fits — {split}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "obb_overlay.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")                  # headless save, standalone only (not on import -> lets
                                           # the notebook keep the inline backend for figure capture)
    os.makedirs(FIGDIR, exist_ok=True)
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    print(f"auditing split={split} ...")
    polys = list(iter_polys(split))
    with ThreadPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(measure, polys))
    ang = np.array([o[0] for o in out]); asp = np.array([o[1] for o in out]); dpx = np.array([o[2] for o in out])
    ok = np.isfinite(ang)
    ang, asp, dpx = ang[ok], asp[ok], dpx[ok]
    n = len(ang)

    # --- (b) aspect: near-circular fraction (ill-defined angle) ---
    near_circ = float((asp <= ASPECT_MIN).mean())
    # --- (c) size vs 4-14 px band ---
    frac_sub4 = float((dpx < 4).mean()); frac_4_14 = float(((dpx >= 4) & (dpx < 14)).mean())
    frac_ge14 = float((dpx >= 14).mean())
    # --- (a) angle bias on the meaningful (elongated + well-resolved) subset ---
    sel = (asp > ASPECT_MIN) & (dpx >= DPX_MIN)
    gf = grid_fraction(ang[sel]); df = diagonal_fraction(ang[sel]); uni = grid_fraction_uniform()

    print(f"\n=== {split}: {n} instances (fitted) ===")
    print(f"(b) aspect: median {np.median(asp):.2f}, near-circular (<= {ASPECT_MIN}) = {100*near_circ:.1f}%")
    print(f"(c) size px-diam: median {np.median(dpx):.1f}px | <4px {100*frac_sub4:.1f}% | "
          f"4-14px {100*frac_4_14:.1f}% | >=14px {100*frac_ge14:.1f}%")
    print(f"(a) elongated+well-resolved subset n={int(sel.sum())} ({100*sel.mean():.1f}%): "
          f"grid-frac(0/90)={gf:.3f}  diag-frac(45/135)={df:.3f}  uniform={uni:.3f}")

    # figures
    fig = plt.figure(figsize=(15, 4.5))
    axr = fig.add_subplot(1, 3, 1, projection="polar")
    plot_rose(axr, ang[sel], bins=36, title=f"(a) label angle rose\nelongated+resolved (n={int(sel.sum())})")
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.hist(asp[np.isfinite(asp)], bins=np.linspace(1, 4, 61), color="tab:orange")
    ax2.axvline(ASPECT_MIN, color="k", ls="--", lw=1); ax2.set_xlabel("aspect ratio (a/b)")
    ax2.set_title(f"(b) aspect — near-circ <= {ASPECT_MIN}: {100*near_circ:.0f}%"); ax2.set_ylabel("count")
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(dpx[dpx < 40], bins=np.linspace(0, 40, 81), color="tab:green")
    ax3.axvspan(4, 14, color="red", alpha=0.12); ax3.axvline(DPX_MIN, color="k", ls="--", lw=1)
    ax3.set_xlabel("equiv. diameter (px)"); ax3.set_title(f"(c) size — 4-14px band: {100*frac_4_14:.0f}%")
    ax3.set_ylabel("count")
    fig.suptitle(f"boulder2024 OBB label audit — {split} ({n} instances)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"label_audit_{split}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures to {FIGDIR}")
    overlay(split)
    print("wrote obb_overlay.png")


if __name__ == "__main__":
    main()
