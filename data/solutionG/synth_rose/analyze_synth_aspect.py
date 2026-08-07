# Full-aspect synthetic control: is SAM faithful across the WHOLE aspect range, including the
# near-circular (Test 4 §4b) regime where YOLO's grid-snap is worst? Bin by TRUE aspect and report
# each side's output diag-frac + chirality (uniform truth -> a faithful side stays ~uniform in EVERY
# band). Near-circular truth angle is ill-defined, so the test is "output rose uniform?", not accuracy.
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
from YOLOv8BeyondEarth.orientation import diagonal_fraction, asymmetry_ratio

G = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionG\synth_rose")
UNIFORM = 0.222
ABANDS = [(1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 3.1)]

if __name__ == "__main__":
    df = pd.read_csv(G / "synth_rose_fullaspect.csv")
    print(f"n={len(df)}  aspect range [{df.aspect_true.min():.2f},{df.aspect_true.max():.2f}]\n")

    for lo, hi in ABANDS:
        d = df[(df.aspect_true >= lo) & (df.aspect_true < hi)]
        if len(d) < 30:
            continue
        t = d.truth_angle.values
        print(f"=== aspect [{lo},{hi})  n={len(d)}  TRUTH diag {diagonal_fraction(t):.3f} "
              f"chir {asymmetry_ratio(t):.2f} ===")
        for s in ("yolo", "sam256", "sam1024"):
            a = d[s].values; a = a[np.isfinite(a)]
            print(f"   {s:8s} diag {diagonal_fraction(a):.3f}  chir {asymmetry_ratio(a):.2f}  "
                  f"[90,180) {np.mean(a>=90):.3f}")
        print()

    # figure: diag-frac vs aspect band, YOLO vs SAM (uniform line) — both shadow conds pooled
    import matplotlib.pyplot as plt
    FIG = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\docs\figures\solutionG")
    labels, yolo_d, sam_d = [], [], []
    for lo, hi in ABANDS:
        d = df[(df.aspect_true >= lo) & (df.aspect_true < hi)]
        if len(d) < 30:
            continue
        labels.append(f"{lo}-{hi if hi < 3 else '3.0'}")
        yolo_d.append(diagonal_fraction(d.yolo.values[np.isfinite(d.yolo.values)]))
        sam_d.append(diagonal_fraction(d.sam256.values[np.isfinite(d.sam256.values)]))
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - w / 2, yolo_d, w, color="tab:blue", label="YOLO")
    ax.bar(x + w / 2, sam_d, w, color="tab:green", label="SAM256")
    ax.axhline(UNIFORM, ls="--", color="gray", label=f"uniform {UNIFORM}")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_xlabel("true aspect ratio band")
    ax.set_ylabel("output diagonal fraction")
    ax.set_title("SAM ≤ uniform at EVERY aspect (no diagonal artifact);\nYOLO's diagonal excess grows with elongation")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG / "G_synth_aspect.png", dpi=130, bbox_inches="tight")
    print(f"wrote {FIG/'G_synth_aspect.png'}")
