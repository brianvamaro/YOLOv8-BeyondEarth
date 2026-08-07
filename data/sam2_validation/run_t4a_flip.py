# T4a — mirror-flip half-plane cause test (docs/sam2_orientation_validation_plan.md).
# The clean pixel-vs-scene separator for the [0,90) vs [90,180) imbalance (T2 showed rotation alone
# can't attribute it; T3 found the imbalance is body-specific = scene content; T4a proves it on the
# imbalanced Mars HiRISE scene). For each dense HiRISE crop, detect + SAM2 re-mask the ORIGINAL and
# its LEFT-RIGHT MIRROR, map the flipped angles back to scene coords (theta -> 180-theta; a horizontal
# mirror reflects East<->West), and match the SAME boulders by scene position.
#
#   scene-locked (real illumination/fabric): scene-referenced f_lower INVARIANT under the flip, and
#     per-boulder theta_orig ~ theta_flip_sceneref (SAM re-derives the same physical angle).
#   pixel/decoder handedness: scene-referenced f_lower FLIPS toward 1-f_lower (~0.38), and per-boulder
#     angles disagree systematically.
# The per-boulder median |dAngle(orig, flip->scene)| is the decisive, self-validating metric (a wrong
# mapping or a real handedness both blow it up; a small value can only mean faithful + correct map).
#
# Same crops as T2 (HiRISE, crop=768, seed=2) for continuity (T2 measured f_lower=0.611 at alpha=0).
# Both operations are LOSSLESS (fliplr, no resampling) so the boulder populations match tightly.
# __main__ guard required (Windows spawn). Run from inside YOLOv8-BeyondEarth/.
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth")
sys.path.insert(0, str(BASE / "src"))

OUT = BASE / "data" / "sam2_validation"
WEIGHTS = BASE / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"
SAM_WEIGHTS = BASE / "sam2.1_b.pt"
HIRISE = BASE / "data" / "mars_hirise" / "raster" / "ESP_017355_2260_RED.tif"
CROP, N_CROPS, SEED = 768, 6, 2
MATCH_R = 6.0


def _angles(masks_side):
    from YOLOv8BeyondEarth.orientation_validation import angles_from_moments
    a = angles_from_moments(masks_side)
    return a["angle180"].values, a["aspect"].values, a["diameter_px"].values


if __name__ == "__main__":
    from ultralytics import YOLO, SAM
    from YOLOv8BeyondEarth.orientation_validation import detect_sam2_mask_sets
    from YOLOv8BeyondEarth.transforms_test import read_dense_crops

    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    sam = SAM(str(SAM_WEIGHTS))
    crops = read_dense_crops(str(HIRISE), crop=CROP, n=N_CROPS, min_valid=0.99, seed=SEED)
    print(f"{len(crops)} HiRISE crops", flush=True)

    t0 = time.time()
    rows = []
    realigned = 0
    for ci, crop in enumerate(crops):
        S = crop.shape[1]
        fc = np.ascontiguousarray(crop[:, ::-1])           # left-right mirror (lossless)
        dfo, mo = detect_sam2_mask_sets(model, sam, crop, slice_size=256, imgsz=1024, conf=0.10,
                                        device=0, configs=("yolo", "sam256"))
        dff, mf = detect_sam2_mask_sets(model, sam, fc, slice_size=256, imgsz=1024, conf=0.10,
                                        device=0, configs=("yolo", "sam256"))
        realigned += int(dfo.attrs.get("realigned_chunks", 0)) + int(dff.attrs.get("realigned_chunks", 0))
        for side in ("yolo", "sam256"):
            to, aspo, dpo = _angles(mo[side])
            tf, aspf, dpf = _angles(mf[side])
            # original scene coords = image coords; flipped -> scene
            oc = dfo[["cx", "cy"]].to_numpy()
            fc_scene = np.c_[(S - 1) - dff.cx.to_numpy(), dff.cy.to_numpy()]
            tf_scene = (180.0 - tf) % 180.0
            if len(oc) == 0 or len(fc_scene) == 0:
                continue
            pairs = cKDTree(oc).query_ball_tree(cKDTree(fc_scene), r=MATCH_R)
            cand = sorted((float(np.hypot(*(oc[i] - fc_scene[j]))), i, j)
                          for i, js in enumerate(pairs) for j in js
                          if np.hypot(*(oc[i] - fc_scene[j])) <= MATCH_R)
            uo, uf = set(), set()
            for d, i, j in cand:
                if i in uo or j in uf:
                    continue
                uo.add(i); uf.add(j)
                rows.append(dict(crop=ci, side=side, theta_orig=float(to[i]),
                                 theta_flip_sceneref=float(tf_scene[j]),
                                 aspect=float(aspo[i]), dpx=float(dpo[i]), dist=d))
        print(f"  crop {ci}: orig {len(dfo)} / flip {len(dff)} dets, matched so far {len(rows)}",
              flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "t4a_flip.csv", index=False)
    wall = time.time() - t0
    meta = dict(wall_s=wall, crop=CROP, n_crops=len(crops), scene="hirise_mars",
                match_r=MATCH_R, realigned_chunks=realigned,
                n_matched={s: int((out.side == s).sum()) for s in ("yolo", "sam256")})
    (OUT / "t4a_run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print(f"wall {wall:.0f}s", flush=True)
