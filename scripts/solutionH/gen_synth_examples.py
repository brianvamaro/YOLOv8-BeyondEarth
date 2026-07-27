"""Generate the per-boulder SYNTHETIC example cache for Fig 5 of the orientation results summary
(the synthetic analogue of Fig 2: individual boulders with each producer's outline/box + angle,
but here against a KNOWN true shape instead of a human tracing).

GPU. Renders several 256-px synthetic fields (single tiles → the same 256→1024 upscale production
uses per slice, so the YOLO masks show the real discretisation artifact), then on each field runs:
  - YOLO-seg  (yolov8-m-boulder)      → per-boulder binary mask (blue)
  - SAM2      (sam2.1_b, box-prompted) → per-boulder binary mask (green)   [option G]
  - OBB R-A   (yolo26m-h-obb-rot)      → per-boulder oriented box (cyan)
Detections are matched to the rendered truth by centroid, then the pool is kept **stratified by size**:
within each equivalent-diameter band, the boulders where YOLO's mask most mis-measures the long axis
and SAM recovers it most. Ranking the whole pool by that error instead — as an earlier version did —
can only ever return small boulders, because the artifact *is* size-dependent: measured over 330
rendered boulders, the share with |YOLO − true| >= 10 deg runs 8/32 at 4-6 px but **0/113 above 15 px**.
Stratifying lets the figure show the failure mode *and* its size dependence, and that per-band rate is
saved alongside the panels so the figure can quote it.

Each kept boulder is stored self-contained (its own image patch + mask patches + local-frame OBB
corners + all four angles) so the plotter is field-agnostic. Cached to
data/solutionH/synth_examples.pkl, read by `scripts/summary/make_summary_figs.py` (fig03) and the
older `make_orientation_summary_figs.py`.

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
# `diam_px` is the MAJOR-axis length, so the equivalent diameter that ends up in `true_dpx` is
# 2*sqrt(a*b), i.e. noticeably smaller for an elongated shape: (7, 34) here yields ~5-24 px equivalent.
# The old (7, 20) topped out near 14 px equivalent, and after the artifact selection below nothing
# above 10 px survived — the artifact is size-dependent, so ranking purely by YOLO error can only ever
# return small boulders. The pool is therefore stratified by size instead.
DIAM_PX = (7, 34)
ASPECT = (1.7, 3.0)        # clearly elongated so the true long axis is well defined
SUN_AZ = 255.0             # matches the Fig 4 / Fig 7 shadowed null render convention
SEEDS = [5, 11, 23, 42, 67, 88, 101, 137, 199]   # more fields: large boulders are rarer per field
MATCH_PX = 5.0
# Size-stratified pool: within each equivalent-diameter band keep the boulders where YOLO's mask most
# mis-measures the axis AND SAM recovers it most (the same "honest showcase" ranking as before), so the
# plotter can draw a row spanning size rather than four near-identical small boulders.
BANDS = [(4, 6), (6, 8), (8, 11), (11, 15), (15, 99)]
PER_BAND = 4
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

    # SELECT per size band. Within a band, prefer the artifact cases (|YOLO - true| >= MIN_YOLO_ERR)
    # ranked by how much SAM recovers (yolo_err - sam_err); if a band has none, fall back to its
    # best-ranked boulders so the band is still represented. Stratifying is what lets the figure show
    # the failure mode AND the size dependence: ranking the whole pool by error returns only small
    # boulders, because that is where the artifact lives.
    recs = []
    for lo, hi in BANDS:
        band = [r for r in pool if lo <= r["true_dpx"] < hi]
        band.sort(key=lambda r: r["yolo_err"] - r["sam_err"], reverse=True)
        # Artifact cases first, then top up from the band's next-best so every band offers the
        # plotter a real choice. Above ~15 px there are typically NO artifact cases at all, which is
        # the size dependence stated as a count rather than asserted.
        hits = [r for r in band if r["yolo_err"] >= MIN_YOLO_ERR]
        # dedupe by identity: these records hold numpy arrays, so `r not in hits` would compare them
        # elementwise and raise on the ambiguous truth value.
        seen = {id(r) for r in hits}
        hits += [r for r in band if id(r) not in seen]
        recs.extend(hits[:PER_BAND])
        print(f"  band {lo}-{hi} px: {len(band):3d} in pool, "
              f"{sum(1 for r in band if r['yolo_err'] >= MIN_YOLO_ERR):3d} artifact cases, "
              f"kept {len(hits[:PER_BAND])}")
    recs.sort(key=lambda r: r["true_dpx"])                # display order: small -> large
    n_obb = sum(r["obb_corners"] is not None for r in recs)
    my = float(np.median([r["yolo_err"] for r in recs]))
    ms = float(np.median([r["sam_err"] for r in recs]))
    n_art = sum(1 for r in pool if r["yolo_err"] >= MIN_YOLO_ERR)
    print(f"\npool {len(pool)}; artifact-cases (|Y-T|>={MIN_YOLO_ERR:.0f}) {n_art}; "
          f"kept {len(recs)} panels over {len(BANDS)} size bands; OBB matched {n_obb}/{len(recs)}")
    print(f"selected-panel median error: YOLO {my:.1f}°  SAM {ms:.1f}°")
    # The artifact-case rate per band is the size dependence, measured; kept for the figure to quote.
    band_rate = {(f"{lo}-{hi}" if hi < 99 else f"{lo}+"): dict(
        n=sum(1 for r in pool if lo <= r["true_dpx"] < hi),
        artifact_cases=sum(1 for r in pool if lo <= r["true_dpx"] < hi
                           and r["yolo_err"] >= MIN_YOLO_ERR)) for lo, hi in BANDS}
    print("artifact-case rate by size band:",
          {k: f"{v['artifact_cases']}/{v['n']}" for k, v in band_rate.items()})
    if len(recs) < 8:
        raise RuntimeError("too few example boulders — add SEEDS or relax MIN_YOLO_ERR")

    with open(OUT, "wb") as f:
        pickle.dump({"boulders": recs, "sun_az_deg": SUN_AZ,
                     "median_yolo_err": my, "median_sam_err": ms,
                     "bands": BANDS, "min_yolo_err": MIN_YOLO_ERR,
                     "artifact_rate_by_band": band_rate, "pool_size": len(pool),
                     "diam_px_rendered": DIAM_PX}, f)
    print("wrote:", OUT)


if __name__ == "__main__":
    main()
