"""Analyze the HiRISE SAM-chirality generality run. Key question: across scenes with DIFFERENT
shadow axes (36-77 deg), does SAM's dominant boulder axis MOVE WITH the shadow (SAM tracks the sun)
while YOLO stays grid-locked? Per-scene chirality table + a peak-vs-shadow-axis figure."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from YOLOv8BeyondEarth.orientation import (half_plane_fraction as fl, asymmetry_ratio as asym,
                                           diagonal_fraction as diag, grid_fraction as grid,
                                           axial_distance, axis_alignment_fraction)

OUT = Path(r"c:/Users/brian/Documents/PhD/BoulderNet/YOLOv8-BeyondEarth/data/sam2_validation")
FIG = Path(r"c:/Users/brian/Documents/PhD/BoulderNet/docs/figures/sam_artifact"); FIG.mkdir(parents=True, exist_ok=True)
ASPECT = 1.35
MIN_DPX = 8.0          # size floor (0 = none). ≥8 px = the pixelation-floor knee (Component A).


def axial_mean(a):
    """Dominant axis (deg, [0,180)) via the doubled-angle circular mean; + resultant length R."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, 0.0
    z = np.exp(2j * np.deg2rad(a)); m = z.mean()
    return (np.rad2deg(np.angle(m)) / 2.0) % 180.0, float(abs(m))


def main():
    df = pd.read_csv(OUT / "sam_chirality_hirise.csv")
    scenes = df.groupby("scene").shadow_axis.first().sort_values().index.tolist()
    print(f"{'scene':20s} {'shadow':>6s} {'inc':>5s} {'nEl':>5s} | "
          f"{'SAMpeak':>7s} {'d(pk,sh)':>8s} {'SAMasym':>7s} {'SAMfl':>6s} | "
          f"{'YOLOpk':>6s} {'d(pk,sh)':>8s} {'YOLOasym':>8s}")
    rows = []
    for sc in scenes:
        s = df[df.scene == sc]
        sa = float(s.shadow_axis.iloc[0]); inc = float(s.incidence.iloc[0])
        sam = s.angle_sam[(s.aspect_sam > ASPECT) & (s.dpx_sam >= MIN_DPX)].to_numpy()
        yolo = s.angle_yolo[(s.aspect_yolo > ASPECT) & (s.dpx_sam >= MIN_DPX)].to_numpy()
        spk, sR = axial_mean(sam); ypk, yR = axial_mean(yolo)
        d_sam = axial_distance(np.array([spk]), sa)[0]
        d_yolo = axial_distance(np.array([ypk]), sa)[0]
        rows.append(dict(scene=sc, shadow=sa, inc=inc, n=len(sam), sam_peak=spk, sam_R=sR,
                         d_sam=d_sam, sam_asym=asym(sam), sam_fl=fl(sam),
                         sam_along=axis_alignment_fraction(sam, sa, 45),
                         yolo_peak=ypk, d_yolo=d_yolo, yolo_asym=asym(yolo),
                         yolo_along=axis_alignment_fraction(yolo, sa, 45)))
        print(f"{sc:20s} {sa:6.1f} {inc:5.1f} {len(sam):5d} | {spk:7.1f} {d_sam:8.1f} "
              f"{asym(sam):7.2f} {fl(sam):6.3f} | {ypk:6.1f} {d_yolo:8.1f} {asym(yolo):8.2f}")
    R = pd.DataFrame(rows)

    # Do the peaks track the shadow axis across scenes? (mean |peak - shadow|; null ~45 for random)
    print(f"\nmean |SAM peak - shadow axis|  = {R.d_sam.mean():.1f} deg  (0=tracks sun, 45=unrelated)")
    print(f"mean |YOLO peak - shadow axis| = {R.d_yolo.mean():.1f} deg")
    print(f"mean SAM along-shadow fraction  = {R.sam_along.mean():.3f} (null 0.5)")
    print(f"mean YOLO along-shadow fraction = {R.yolo_along.mean():.3f}")

    # figure: peak vs shadow axis
    fig, ax = plt.subplots(figsize=(6.4, 6))
    lim = [0, 180]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.5, label="peak = shadow axis (tracks sun)")
    for d in (45, 135):
        ax.axhline(d, color="purple", ls=":", lw=1, alpha=0.6)
    for c in (0, 90, 180):
        ax.axhline(c, color="gray", ls=":", lw=1, alpha=0.5)
    ax.scatter(R.shadow, R.sam_peak, s=60, c="tab:green", label="SAM peak", zorder=5)
    ax.scatter(R.shadow, R.yolo_peak, s=60, c="tab:blue", marker="s", label="YOLO peak", zorder=5)
    ax.set_xlabel("scene shadow (=sun) axis, deg"); ax.set_ylabel("population dominant axis, deg")
    ax.set_xlim(0, 90); ax.set_ylim(0, 180); ax.legend(fontsize=8)
    ax.set_title("Does the boulder rose track the sun across HiRISE scenes?\n"
                 "(SAM on sun-diagonal => tracks; YOLO flat at grid => locked)")
    fig.tight_layout(); fig.savefig(FIG / "hirise_peak_vs_shadow.png", dpi=120, bbox_inches="tight")
    R.to_csv(OUT / "sam_chirality_perscene.csv", index=False)
    print(f"\nsaved figure -> {FIG/'hirise_peak_vs_shadow.png'}  + perscene csv")


if __name__ == "__main__":
    main()
