# Does SAM2 re-masking introduce its OWN directional orientation artifact? The decisive test:
# render synthetic boulders at UNIFORM-random known angles, SAM-re-mask them, and check whether the
# OUTPUT rose stays uniform. Two arms: shadow-free (isolates the pure decoder+shape) and with-shadow
# (tests whether SAM keys to illumination). If SAM output stays ~uniform -> it does NOT manufacture a
# directional artifact (we are not trading YOLO's 135 lock for a SAM one). Saves per-boulder CSV.
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")

OUT = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionG\synth_rose")
OUT.mkdir(parents=True, exist_ok=True)
WEIGHTS = r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\weights\yolov8-m-boulder-detection-tmp.pt"

N_FIELDS = 12          # 1024px fields per condition
N_PER = 300            # boulders per field (uniform-random angles)
DIAM = (5, 15)         # native px equivalent diameter (spans artifact 6-8 and well-resolved bands)
ASPECT = (1.0, 3.0)    # FULL aspect range incl. near-circular (the Test-4 §4b grid-snap regime)
MATCH_PX = 4.0
CONDITIONS = [("shadowfree", 0.0), ("shadow", 0.6)]

if __name__ == "__main__":
    from ultralytics import YOLO, SAM
    from YOLOv8BeyondEarth.transforms_test import render_boulder_field
    from YOLOv8BeyondEarth.orientation_validation import detect_sam2_mask_sets, angles_from_moments

    model = YOLO(WEIGHTS)
    sam = SAM("sam2.1_b.pt")

    rows = []
    t0 = time.time()
    for cond, sf in CONDITIONS:
        for k in range(N_FIELDS):
            img, truth = render_boulder_field(size=1024, n=N_PER, angles=None, diam_px=DIAM,
                                              aspect=ASPECT, shadow_frac=sf, seed=1000 + k)
            truth["aspect_true"] = truth.a_px.values / truth.b_px.values
            df, ms = detect_sam2_mask_sets(model, sam, img, slice_size=256, imgsz=1024,
                                           conf=0.10, device=0)
            if len(df) == 0:
                continue
            ang = {s: angles_from_moments(ms[s])["angle180"].values
                   for s in ("yolo", "sam256", "sam1024")}
            tree = cKDTree(np.c_[truth.cx.values, truth.cy.values])
            dist, idx = tree.query(np.c_[df.cx.values, df.cy.values], k=1)
            for i in np.where(dist <= MATCH_PX)[0]:
                j = idx[i]
                rows.append(dict(cond=cond, truth_angle=float(truth.angle180.values[j]),
                                 dpx=float(truth.diam_px.values[j]),
                                 aspect_true=float(truth.aspect_true.values[j]),
                                 yolo=float(ang["yolo"][i]), sam256=float(ang["sam256"][i]),
                                 sam1024=float(ang["sam1024"][i])))
            print(f"[{cond}] field {k}: {len(df)} det, "
                  f"{int((dist <= MATCH_PX).sum())} matched (cum {len(rows)})", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "synth_rose_fullaspect.csv", index=False)
    print(f"\nwrote {OUT/'synth_rose_fullaspect.csv'}  n={len(out)}  wall={time.time()-t0:.0f}s")
    print(out.groupby("cond").size())
