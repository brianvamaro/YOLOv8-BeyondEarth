"""T2 (SAM) analysis — on fixed ground, does the boulder rose rotate with the sun?
Discriminator (per repeat pair, same ground, sun differs by Δsun):
  illumination  -> the rose follows the sun: each member's rose aligns to ITS OWN sun (both along-sun
                   high), and the peak rotates ~Δsun between members.
  fabric/fixed  -> the rose is fixed: along-sun high for the member whose sun ≈ the fixed axis, low for
                   the other; peak rotation ≈ 0.
Compare SAM vs YOLO (YOLO grid-locked -> should not rotate with the sun). Weak arm: Δsun 13-29°."""
import numpy as np, pandas as pd
from pathlib import Path
from YOLOv8BeyondEarth.orientation import (signed_axial_deviation, axis_alignment_fraction as align,
                                           half_plane_fraction as fl)
OUT = Path(r"c:/Users/brian/Documents/PhD/BoulderNet/YOLOv8-BeyondEarth/data/sam2_validation")


def axial_mean(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 5:
        return np.nan
    return (np.rad2deg(np.angle(np.exp(2j * np.deg2rad(a)).mean())) / 2.0) % 180.0


def sub(df, side):
    ac, pc = f"angle_{side}", f"aspect_{side}"
    m = (df[pc] > 1.35) & (df.dpx_sam >= 8)
    return df.loc[m, ac].to_numpy()


d = pd.read_csv(OUT / "t2_hirise_sam.csv")
print("T2 HiRISE repeat pairs (same ground; aspect>1.35, dpx>=8). along-sun null 0.5.\n")
for pid in d.pair.unique():
    p = d[d.pair == pid]
    scenes = list(p.scene.unique())
    if len(scenes) != 2:
        print(f"{pid}: {len(scenes)} member(s) — skip"); continue
    (sA, sB) = scenes
    A, B = p[p.scene == sA], p[p.scene == sB]
    sunA, sunB = A.sun.iloc[0], B.sun.iloc[0]
    dsun = signed_axial_deviation(np.array([sunB]), sunA)[0]
    print(f"=== pair {pid}: {sA}(sun {sunA:.0f},inc {A.inc.iloc[0]:.0f}) ↔ "
          f"{sB}(sun {sunB:.0f},inc {B.inc.iloc[0]:.0f})  Δsun={dsun:+.0f}° ===")
    for side in ("sam", "yolo"):
        aA, aB = sub(A, side), sub(B, side)
        pkA, pkB = axial_mean(aA), axial_mean(aB)
        rot = signed_axial_deviation(np.array([pkB]), pkA)[0]
        alA = align(aA, sunA, 45); alB = align(aB, sunB, 45)
        tag = "illum-like" if (alA > 0.5 and alB > 0.5 and abs(rot) > 0.3 * abs(dsun)) else "fixed-like"
        print(f"  {side:4s}: nA={len(aA):4d} nB={len(aB):4d} | peakA={pkA:5.0f} peakB={pkB:5.0f} "
              f"peak-rot={rot:+5.0f}° (Δsun {dsun:+.0f}) | along-sun A={alA:.2f} B={alB:.2f} | "
              f"f_lower A={fl(aA):.2f} B={fl(aB):.2f}  -> {tag}")
    print()
