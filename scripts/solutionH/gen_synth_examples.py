"""Generate the per-boulder SYNTHETIC example cache for Fig 5 of the orientation results summary
(the synthetic analogue of Fig 2: individual boulders with each producer's outline/box + angle,
but here against a KNOWN true shape instead of a human tracing).

GPU. Renders several 256-px synthetic fields (single tiles → the same 256→1024 upscale production
uses per slice, so the YOLO masks show the real discretisation artifact), then on each field runs:
  - YOLO-seg  (yolov8-m-boulder)      → per-boulder binary mask (blue)
  - SAM2      (sam2.1_b, box-prompted) → per-boulder binary mask (green)   [option G]
  - OBB R-A   (yolo26m-h-obb-rot)      → per-boulder oriented box (cyan)
Detections are matched to the rendered truth by centroid. Like Fig 2 (which deliberately shows real
boulders whose YOLO angle snapped to a pixel diagonal), we render the small artifact-regime size
band and SELECT the boulders where YOLO's mask most mis-measures the long axis (|YOLO−true| large)
— the honest showcase of the failure mode and SAM's recovery, now against a KNOWN true angle. Each
kept boulder is stored self-contained (its own image patch + mask patches + local-frame OBB corners
+ all four angles) so the plotter is field-agnostic. Cached to data/solutionH/synth_examples.pkl;
the GPU-free `make_orientation_summary_figs.py` reads it and draws Fig 5.

Run from repo root: python YOLOv8-BeyondEarth/scripts/solutionH/gen_synth_examples.py
"""
import os
import pickle
import sys

import numpy as np

INNER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(INNER, "src"))

WEIGHTS = os.path.join(INNER, "data", "weights", "yolov8-m-boulder-detection-tmp.pt")
OBB_WEIGHTS = os.path.join(INNER, "data", "weights", "yolo26m-h-obb-rot.pt")
SAM_WEIGHTS = "sam2.1_b.pt"
OUT = os.path.join(INNER, "data", "solutionH", "synth_examples.pkl")

SIZE = 256                 # one production slice; upscaled 4x to imgsz=1024 (= production per-tile)
IMGSZ = 1024
N_RENDER = 40              # requested boulders per field (fewer placed after separation rejection)
DIAM_PX = (7, 20)          # the artifact regime + a few larger, clearer boulders (cf. Fig 2's range)
ASPECT = (1.7, 3.0)        # clearly elongated so the true long axis is well defined
SUN_AZ = 255.0             # matches the Fig 4 / Fig 7 shadowed null render convention
SEEDS = [5, 11, 23, 42, 67, 88]     # pool several fields to build a selection pool
MATCH_PX = 5.0
N_PANELS = 12
MIN_YOLO_ERR = 10.0        # a boulder is an "artifact case" if |YOLO angle − true| >= this


def _axial(a, t):
    return abs(((a - t + 90.0) % 180.0) - 90.0)


def _clean(mb):
    from scipy.ndimage import label, binary_fill_holes
    lab, n = label(mb)
    if n == 0:
        return None
    if n > 1:
        big = 1 + int(np.argmax([(lab == j).sum() for j in range(1, n + 1)]))
        mb = (lab == big).astype(np.uint8)
    return binary_fill_holes(mb).astype(np.uint8)


def _field_records(field_seed, model, sam, obb, cv2, cKDTree, render_boulder_field,
                   angles_from_moments, obb_angle180):
    """Run all three producers on one rendered field; return self-contained per-boulder records."""
    img, truth = render_boulder_field(size=SIZE, n=N_RENDER, angles=None, diam_px=DIAM_PX,
                                      aspect=ASPECT, shadow_frac=0.6, sun_az_deg=SUN_AZ, seed=field_seed)
    truth = truth.reset_index(drop=True)
    rgb = np.stack([img] * 3, axis=-1)

    r0 = model(rgb, imgsz=IMGSZ, conf=0.10, verbose=False, device=0)[0]
    if r0.masks is None or len(r0.boxes) == 0:
        return []
    boxes = r0.boxes.xyxy.cpu().numpy()
    md = r0.masks.data.cpu().numpy()
    ymasks = []
    for k in range(len(boxes)):
        m = (md[k] >= 0.5).astype(np.uint8)
        if m.shape[0] != SIZE:
            m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        ymasks.append(_clean(m))
    ycx = (boxes[:, 0] + boxes[:, 2]) / 2
    ycy = (boxes[:, 1] + boxes[:, 3]) / 2

    sres = sam(rgb, bboxes=boxes, verbose=False, device=0)[0].masks.data.cpu().numpy()
    smasks = [_clean(sres[k].astype(np.uint8)) for k in range(len(boxes))]

    ro = obb.predict(rgb, imgsz=IMGSZ, conf=0.10, verbose=False, device=0)[0]
    ocorners = ro.obb.xyxyxyxy.cpu().numpy() if (ro.obb is not None and len(ro.obb)) else np.empty((0, 4, 2))
    ocx = ocorners[:, :, 0].mean(1) if len(ocorners) else np.empty(0)
    ocy = ocorners[:, :, 1].mean(1) if len(ocorners) else np.empty(0)

    tt = cKDTree(np.c_[truth.cx.values, truth.cy.values])
    dist, jrow = tt.query(np.c_[ycx, ycy], k=1)
    recs = {}
    for k in range(len(boxes)):
        if dist[k] > MATCH_PX or ymasks[k] is None or smasks[k] is None:
            continue
        j = int(jrow[k])
        if truth.a_px[j] / truth.b_px[j] <= 1.7 or j in recs:      # elongated; first match per boulder
            continue
        ya = float(angles_from_moments([ymasks[k]])["angle180"].iloc[0])
        sa = float(angles_from_moments([smasks[k]])["angle180"].iloc[0])
        cx, cy, dpx = float(truth.cx[j]), float(truth.cy[j]), float(truth.diam_px[j])
        pad = max(int(round(dpx)) + 2, 12)                         # tight crop (~2x diam), matches Fig 2
        c0, r0c = max(int(cx) - pad, 0), max(int(cy) - pad, 0)
        sl = (slice(r0c, r0c + 2 * pad), slice(c0, c0 + 2 * pad))
        recs[j] = dict(patch=img[sl].copy(), cx=cx - c0, cy=cy - r0c,
                       a_px=float(truth.a_px[j]), b_px=float(truth.b_px[j]),
                       true_angle=float(truth.angle180[j]), true_dpx=dpx,
                       yolo_mask=ymasks[k][sl].copy(), yolo_angle=ya, yolo_err=_axial(ya, truth.angle180[j]),
                       sam_mask=smasks[k][sl].copy(), sam_angle=sa, sam_err=_axial(sa, truth.angle180[j]),
                       obb_corners=None, obb_angle=np.nan, _c0=c0, _r0=r0c)
    if len(ocx):
        od, oj = tt.query(np.c_[ocx, ocy], k=1)
        for k in range(len(ocorners)):
            j = int(oj[k])
            if od[k] <= MATCH_PX and j in recs and recs[j]["obb_corners"] is None:
                rec = recs[j]
                rec["obb_corners"] = ocorners[k] - np.array([rec["_c0"], rec["_r0"]])   # local frame
                rec["obb_angle"] = float(obb_angle180(ocorners[k])[0])
    for r in recs.values():
        r.pop("_c0"); r.pop("_r0")
    return list(recs.values())


def main():
    import cv2
    from scipy.spatial import cKDTree
    from ultralytics import YOLO, SAM
    from YOLOv8BeyondEarth.transforms_test import render_boulder_field
    from YOLOv8BeyondEarth.orientation_validation import angles_from_moments
    from YOLOv8BeyondEarth.orientation import obb_angle180

    model = YOLO(WEIGHTS)
    sam = SAM(SAM_WEIGHTS)
    obb = YOLO(OBB_WEIGHTS, task="obb")

    pool = []
    for s in SEEDS:
        recs = _field_records(s, model, sam, obb, cv2, cKDTree, render_boulder_field,
                              angles_from_moments, obb_angle180)
        pool.extend(recs)
        print(f"seed {s}: +{len(recs)} matched elongated (pool {len(pool)})")

    # SELECT the artifact cases (parallels Fig 2's diagonal-locked selection): boulders where YOLO's
    # mask most mis-measures the long axis, ranked by how much SAM recovers it (yolo_err - sam_err).
    artifact = [r for r in pool if r["yolo_err"] >= MIN_YOLO_ERR]
    artifact.sort(key=lambda r: r["yolo_err"] - r["sam_err"], reverse=True)
    recs = artifact[:N_PANELS] if len(artifact) >= N_PANELS else \
        sorted(pool, key=lambda r: r["yolo_err"] - r["sam_err"], reverse=True)[:N_PANELS]
    recs.sort(key=lambda r: r["true_dpx"])                # display order: small -> large
    n_obb = sum(r["obb_corners"] is not None for r in recs)
    my = float(np.median([r["yolo_err"] for r in recs]))
    ms = float(np.median([r["sam_err"] for r in recs]))
    print(f"\npool {len(pool)}; artifact-cases (|Y-T|>={MIN_YOLO_ERR:.0f}) {len(artifact)}; "
          f"kept {len(recs)} panels; OBB matched {n_obb}/{len(recs)}")
    print(f"selected-panel median error: YOLO {my:.1f}°  SAM {ms:.1f}°")
    if len(recs) < 8:
        raise RuntimeError("too few example boulders — add SEEDS or relax MIN_YOLO_ERR")

    with open(OUT, "wb") as f:
        pickle.dump({"boulders": recs, "sun_az_deg": SUN_AZ,
                     "median_yolo_err": my, "median_sam_err": ms}, f)
    print("wrote:", OUT)


if __name__ == "__main__":
    main()
