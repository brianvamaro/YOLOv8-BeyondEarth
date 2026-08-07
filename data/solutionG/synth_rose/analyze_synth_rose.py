# Analyze the synthetic-boulder SAM rose experiment (run_synth_rose.py). Does SAM introduce a
# directional artifact? For each condition (shadowfree / shadow) and side (yolo/sam256/sam1024):
# output diag-frac + chirality + [0,90)/[90,180) split (truth is uniform -> a faithful side stays
# ~uniform); median |dAngle| vs truth; and the directional bias = mean signed axial deviation binned
# by TRUTH angle (a side that pulls angles toward cardinal shows a systematic pattern here).
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
from YOLOv8BeyondEarth.orientation import diagonal_fraction, asymmetry_ratio, axial_distance

G = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionG\synth_rose")
UNIFORM = 0.222
SIDES = ("yolo", "sam256", "sam1024")


def signed_axial(a, b):
    """Signed axial deviation a-b in (-90, 90] (a,b in [0,180))."""
    d = (np.asarray(a) - np.asarray(b) + 90.0) % 180.0 - 90.0
    return d


if __name__ == "__main__":
    df = pd.read_csv(G / "synth_rose.csv")
    print(f"n={len(df)}  by cond:\n{df.groupby('cond').size()}\n")

    for cond in ("shadowfree", "shadow"):
        d = df[df.cond == cond]
        t = d.truth_angle.values
        print(f"===== {cond}  (n={len(d)}) =====")
        print(f"  TRUTH: diag {diagonal_fraction(t):.3f}  chir {asymmetry_ratio(t):.2f}  "
              f"[0,90) {np.mean(t<90):.3f} / [90,180) {np.mean(t>=90):.3f}   (uniform ref {UNIFORM})")
        for s in SIDES:
            a = d[s].values
            ok = np.isfinite(a)
            a, tt = a[ok], t[ok]
            err = np.abs(axial_distance(a, tt))
            print(f"  {s:8s}: diag {diagonal_fraction(a):.3f}  chir {asymmetry_ratio(a):.2f}  "
                  f"[0,90) {np.mean(a<90):.3f} / [90,180) {np.mean(a>=90):.3f}  "
                  f"med|dA| {np.median(err):.2f}  p90 {np.percentile(err,90):.1f}")
        # directional bias: mean signed dev binned by truth angle (sam256)
        print("  sam256 signed dev (out-truth) by truth-angle bin (deg): "
              "a faithful side ~0 everywhere")
        edges = np.arange(0, 181, 30)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (t >= lo) & (t < hi) & np.isfinite(d.sam256.values)
            if m.sum() >= 20:
                sd = np.median(signed_axial(d.sam256.values[m], t[m]))
                print(f"    truth [{lo:3d},{hi:3d}): n={int(m.sum()):4d}  median signed dev {sd:+.1f}")
        print()

    # ---- figures ----
    import matplotlib.pyplot as plt
    FIG = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\docs\figures\solutionG")
    FIG.mkdir(parents=True, exist_ok=True)
    colors = {"truth": "gray", "yolo": "tab:blue", "sam256": "tab:green", "sam1024": "tab:orange"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
    for ax, cond in zip(axes, ("shadowfree", "shadow")):
        d = df[df.cond == cond]
        ax.hist(d.truth_angle.values, bins=36, range=(0, 180), density=True, histtype="stepfilled",
                color="gray", alpha=0.25, label="truth (uniform)")
        for s in ("yolo", "sam256"):
            a = d[s].values; a = a[np.isfinite(a)]
            ax.hist(a, bins=36, range=(0, 180), density=True, histtype="step", lw=2,
                    color=colors[s], label=f"{s} (diag {diagonal_fraction(a):.3f})")
        ax.axhline(1 / 180, ls="--", color="k", lw=0.8)
        for dg in (45, 135):
            ax.axvspan(dg - 10, dg + 10, color="red", alpha=0.06)
        ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlabel("long-axis azimuth (deg)"); ax.set_title(f"{cond}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("Synthetic boulders, UNIFORM truth: YOLO manufactures a diagonal excess; "
                 "SAM2 stays ~uniform (no directional artifact)")
    fig.tight_layout(); fig.savefig(FIG / "G_synth_rose.png", dpi=130, bbox_inches="tight")
    print(f"wrote {FIG/'G_synth_rose.png'}")
