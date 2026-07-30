"""Figures for Goal-3 phase-5a entry gate 2 — SAM crop-position sensitivity.

CPU only; reads the cached pair table written by
``data/sam2_validation/run_gate2b_identical_boxes.py``. Three figures, each answering one question:

  gate2_control.png   Is the window grid at fault?      (treatment vs control ECDF + medians)
  gate2_size.png      How does it scale with size?      (median |dAngle| and |dArea| by band)
  gate2_roses.png     Does the aggregate rose move?     (three crop grids, same boulders)

Run:  conda run -n bouldernet python scripts/summary/make_gate2_figs.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from YOLOv8BeyondEarth import figstyle as F
from YOLOv8BeyondEarth.orientation import grid_fraction, diagonal_fraction, half_plane_fraction

HERE = Path(__file__).resolve().parents[2]
PAIRS = HERE / "data/sam2_validation/gate2b_pairs.csv"
FIG = HERE.parent / "docs/figures/goal3"
ASPECT_MIN = 1.35
BAR = 1.3          # the pre-registered literal bar, in degrees

CROPS = [("hirise_crop4k", "Mars HiRISE (0.5 m/px)"), ("nac_crop2k", "Moon NAC (0.48 m/px)")]
# treatment vs control must be visually distinguishable but NOT read as two producers:
C_TREAT, C_CTRL = F.PALETTE["sam"], F.PALETTE["obb_d0"]


def _elong(d):
    return d[(d.aspect_win > ASPECT_MIN) & (d.aspect_sli > ASPECT_MIN)].dropna(
        subset=["angle_win", "angle_sli", "angle_shx"])


def fig_control(df):
    """The gate's actual evidence: treatment and control land on top of each other."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (crop, title) in zip(axes, CROPS):
        el = _elong(df[df.crop == crop])
        for vals, color, label in (
                (el.dangle_win_sli, C_TREAT, "window vs slice  (treatment)"),
                (el.dangle_win_shx, C_CTRL, "window vs shifted window  (control)")):
            v = np.sort(vals.dropna().values)
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, lw=2.0, label=label)
            ax.axvline(np.median(v), color=color, lw=1.0, ls=":", alpha=0.8)
        ax.axvline(BAR, color=F.NULL_GREY, lw=1.2, ls="--")
        ax.annotate(f"pre-registered bar {BAR}°", xy=(BAR, 0.06), xytext=(BAR + 1.2, 0.06),
                    color=F.NULL_GREY, fontsize=8, va="center")
        ax.set_xlim(0, 25); ax.set_ylim(0, 1)
        ax.set_xlabel("|Δ angle| for the SAME boulder (deg)")
        ax.set_ylabel("cumulative fraction")
        ax.set_title(f"{title}   n={len(el):,}", fontsize=10)
        ax.grid(alpha=0.25, color=F.GRID_GREY)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        m1, m2 = el.dangle_win_sli.median(), el.dangle_win_shx.median()
        ax.text(0.03, 0.94, f"median  treatment {m1:.2f}°\nmedian  control    {m2:.2f}°",
                transform=ax.transAxes, fontsize=8.5, va="top", color=F.INK_SOFT, family="monospace")
    fig.suptitle("Gate 2 — re-masking the same boulder from a different crop position", fontsize=12)
    fig.text(0.5, -0.02,
             "Same YOLO boxes, same SAM, same cleanup; only the crop around the boulder moves. "
             "The control merely shifts the window grid half a cell — it is not a different scheme. "
             "Treatment and control agree, so the disagreement is SAM's sensitivity to crop position, "
             "not a property of the phase-5a window grid.",
             ha="center", fontsize=8.5, color=F.INK_SOFT, wrap=True)
    fig.tight_layout()
    fig.savefig(FIG / "gate2_control.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_size(df):
    bands = [(0, 6), (6, 8), (8, 12), (12, 1e9)]
    labels = ["<6", "6–8", "8–12", "≥12"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, metric, ylab, scale in ((axes[0], "dangle_win_sli", "median |Δ angle|  (deg)", 1.0),
                                    (axes[1], "darea_win_sli", "median |Δ area| / area  (%)", 100.0)):
        for (crop, title), mk in zip(CROPS, ("o-", "s--")):
            d = df[df.crop == crop]
            d = d[(d.aspect_win > ASPECT_MIN) & (d.aspect_sli > ASPECT_MIN)]
            ys, ns = [], []
            for lo, hi in bands:
                sub = d[(d.dpx_win >= lo) & (d.dpx_win < hi)]
                ys.append(sub[metric].median() * scale if len(sub) else np.nan)
                ns.append(len(sub))
            color = C_TREAT if crop.startswith("hirise") else C_CTRL
            ax.plot(labels, ys, mk, color=color, lw=1.8, ms=5, label=f"{title}")
        if metric == "dangle_win_sli":
            ax.axhline(BAR, color=F.NULL_GREY, lw=1.2, ls="--")
            ax.text(0.02, BAR + 0.15, f"bar {BAR}°", color=F.NULL_GREY, fontsize=8,
                    transform=ax.get_yaxis_transform())
        ax.set_xlabel("boulder size (native px, equivalent diameter)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, color=F.GRID_GREY)
        ax.legend(fontsize=8, frameon=False)
        ax.set_ylim(bottom=0)
    fig.suptitle("Gate 2 — the jitter shrinks with size but never reaches the bar", fontsize=12)
    fig.text(0.5, -0.04,
             "Elongated subset (aspect > 1.35). Even well-resolved boulders (≥12 px) disagree by "
             "3.4° (Mars) / 1.6° (Moon) between crop positions. Measured at effective conf 0.25 "
             "(review item X-01), so the small-size bins are under-populated relative to production.",
             ha="center", fontsize=8.5, color=F.INK_SOFT, wrap=True)
    fig.tight_layout()
    fig.savefig(FIG / "gate2_size.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_roses(df):
    """The question that matters for Goal 2: does the AGGREGATE rose move?"""
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.6),
                             subplot_kw={"projection": "polar"})
    arms = [("angle_win", "window grid"), ("angle_sli", "production slice grid"),
            ("angle_shx", "shifted window (control)")]
    for r, (crop, title) in enumerate(CROPS):
        el = _elong(df[df.crop == crop])
        rmax = F.shared_rmax([el[c].dropna().values for c, _ in arms])
        for c, (col, arm) in enumerate(arms):
            a = el[col].dropna().values
            ax = axes[r][c]
            F.rose(ax, a, color=C_TREAT if c != 2 else C_CTRL, rmax=rmax,
                   show_rticks=(r == 0 and c == 0))
            ax.set_title(f"{arm}\ngrid {grid_fraction(a,10):.3f} · diag {diagonal_fraction(a,10):.3f}"
                         f" · f_lo {half_plane_fraction(a):.3f}", fontsize=8.5, pad=12)
        axes[r][0].text(-0.32, 0.5, title, transform=axes[r][0].transAxes, rotation=90,
                        va="center", ha="center", fontsize=10)
    fig.suptitle("Gate 2 — the same boulders, re-masked from three crop positions", fontsize=12)
    fig.text(0.5, 0.005,
             "Per-boulder angles jitter by 4.65° (Mars) / 2.69° (Moon), but the jitter is "
             "overwhelmingly random: every aggregate fraction statistic moves ≤0.018 across the "
             "three grids — far below the effects Goal 2's conclusions rest on (G's diagonal "
             "0.428→0.200). Peak azimuth is the exception, wandering up to 3.8°/8.5°.",
             ha="center", fontsize=8.5, color=F.INK_SOFT, wrap=True)
    fig.tight_layout(rect=(0.02, 0.045, 1, 1))
    fig.savefig(FIG / "gate2_roses.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    F.apply_rcparams()
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PAIRS)
    fig_control(df); fig_size(df); fig_roses(df)
    print("wrote:", *[p.name for p in sorted(FIG.glob("gate2_*.png"))])


if __name__ == "__main__":
    main()
