"""Per-boulder example panels: the same rock as human, YOLO and SAM2 see it.

Two figures in the summary notebook need per-boulder panels, and they need *different* boulders:

* the **mechanism** figure wants rocks where YOLO's angle has snapped onto a pixel diagonal while the
  human tracing and the SAM2 re-mask of the same box have not — the failure mode being explained;
* the **area** figure wants rocks where YOLO's mask is visibly inflated relative to the human outline —
  the claim that bar chart makes.

Both come from one pass (YOLO + SAM2 + human labels on dense boulder2024 validation tiles, one per
body), so this script caches several *named selections* over a common candidate pool rather than
hard-coding one figure's taste:

  ``diag_locked``  YOLO angle within DIAG_TOL of 45/135 and far from the human angle  -> mechanism
  ``area_bloat``   largest YOLO/human area ratio                                      -> area, tail
  ``area_spread``  quantiles of the YOLO/human ratio, per body                        -> area, typical

Selecting the most illustrative cases is legitimate for a *mechanism* panel and misleading for a
*magnitude* claim, so both the extreme (``area_bloat``) and the representative (``area_spread``)
selections are cached, the pool's own median ratios are written to the meta file, and the notebook
states which selection each panel draws.

Two deliberate choices:

* the angle selection requires ``aspect > ASPECT_MIN`` (a round mask has no meaningful long axis) while
  the area selection does **not** — area inflation is not an elongation phenomenon, and filtering on
  aspect would quietly change the population the figure speaks for;
* human angles use the y-flip convention `run_t35_val.load_labels` applies to tile-pixel label
  polygons (row index increases downward, so North is -y), matching the mask-frame angles.

Supersedes the ad-hoc `data/sam2_validation/run_examples.py` for summary figures. GPU, small.
__main__ guard (Windows spawn). Run from inside YOLOv8-BeyondEarth/:
  C:\\Users\\brian\\anaconda3\\Scripts\\conda.exe run -n bouldernet --no-capture-output \\
      python data/summary/run_boulder_examples.py
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet")
ROOT = BASE / "YOLOv8-BeyondEarth"
sys.path.insert(0, str(ROOT / "src"))
VAL = BASE / "data_prieur" / "boulder2024" / "validation"
OUT = ROOT / "data" / "summary"
WEIGHTS = ROOT / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"
SAM_WEIGHTS = ROOT / "sam2.1_b.pt"

# Dense, human-labelled tiles — one per body, the same ones the validation area/angle statistics are
# computed on, so the panels are drawn from the population the bar charts describe.
TILES = [("mars", "ESP_017355_2260_RED_2327_image"),
         ("moon", "M104827900_0405_image"),
         ("earth", "courtright-reservoir-sept2022-morning_0075_image")]
DPX_RANGE = (6.0, 30.0)
ASPECT_MIN = 1.35
DIAG_TOL = 12.0        # "on a pixel diagonal": within this many degrees of 45 or 135
N_PER_SEL = 6
N_SPREAD_PER_BODY = 2
EDGE = 20


def axial(a, b):
    return abs(((a - b + 90.0) % 180.0) - 90.0)


def collect(model, sam, body, stem):
    """All matched (human, YOLO, SAM) triples on one tile, with masks and areas."""
    import cv2
    import torch
    from PIL import Image
    from shapely.geometry import Polygon
    from YOLOv8BeyondEarth.orientation import ellipse_angle180
    from YOLOv8BeyondEarth.orientation_validation import _largest_filled, angles_from_moments

    ip = VAL / "images" / f"{stem}.png"
    if not ip.exists():
        print(f"[{body}] missing tile {stem}")
        return []
    arr = np.array(Image.open(ip).convert("L"))
    H, W = arr.shape
    rgb = np.stack([arr] * 3, -1)
    r = model(rgb, imgsz=1024, conf=0.10, verbose=False, device=0)[0]
    if r.masks is None:
        return []
    boxes = r.boxes.xyxy.cpu().numpy()
    ymd = r.masks.data.cpu().numpy()
    parts = []
    for i in range(0, len(boxes), 64):                # dense tiles OOM on one giant prompt batch
        parts.append(sam(rgb, bboxes=boxes[i:i + 64], verbose=False,
                         device=0)[0].masks.data.cpu().numpy())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    sm = np.concatenate(parts, 0)

    humans, hcent = [], []
    lp = VAL / "labels" / f"{stem}.txt"
    if lp.exists():
        for line in lp.read_text().strip().splitlines():
            v = line.split()
            if len(v) < 7:
                continue
            xy = np.array(v[1:], float).reshape(-1, 2) * [W, H]
            if len(xy) >= 3:
                humans.append(xy)
                hcent.append(xy.mean(0))
    hcent = np.asarray(hcent) if hcent else np.empty((0, 2))
    if not len(hcent):
        return []

    out = []
    for k in range(len(boxes)):
        ym = _largest_filled((cv2.resize(ymd[k], (W, H)) >= 0.5).astype(np.uint8))
        smk = _largest_filled((sm[k] >= 0.5).astype(np.uint8))
        if ym is None or smk is None:
            continue
        bx = boxes[k]
        cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
        if min(cx, cy) < EDGE or cx > W - EDGE or cy > H - EDGE:
            continue
        a_y, a_s = float(ym.sum()), float(smk.sum())
        dpx = 2 * np.sqrt(a_s / np.pi)
        if not DPX_RANGE[0] <= dpx <= DPX_RANGE[1]:
            continue
        j = int(np.argmin(np.hypot(hcent[:, 0] - cx, hcent[:, 1] - cy)))
        if np.hypot(hcent[j, 0] - cx, hcent[j, 1] - cy) > max(dpx, 8):
            continue
        hp = humans[j]
        poly = Polygon(hp)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            continue
        gy, gs = angles_from_moments([ym]).iloc[0], angles_from_moments([smk]).iloc[0]
        ah, asph = ellipse_angle180(Polygon(np.c_[hp[:, 0], -hp[:, 1]]), 1.0)
        if not np.isfinite(ah):
            continue
        out.append(dict(body=body, tile=stem, cx=float(cx), cy=float(cy), dpx=float(dpx),
                        area_h=float(poly.area), area_y=a_y, area_s=a_s,
                        r_yolo=a_y / poly.area, r_sam=a_s / poly.area,
                        angle_y=float(gy.angle180), angle_s=float(gs.angle180), angle_h=float(ah),
                        aspect_y=float(gy.aspect), aspect_s=float(gs.aspect), aspect_h=float(asph),
                        _ym=ym, _sm=smk, _hp=hp, _arr=arr))
    return out


def crop(rec):
    """Cut a square patch around the boulder and re-base masks/polygon into patch coordinates."""
    half = int(max(rec["dpx"] * 1.7, 20))
    c0, r0 = int(rec["cx"] - half), int(rec["cy"] - half)
    c1, r1 = c0 + 2 * half, r0 + 2 * half
    arr = rec["_arr"]
    if c0 < 0 or r0 < 0 or c1 > arr.shape[1] or r1 > arr.shape[0]:
        return None
    sl = (slice(r0, r1), slice(c0, c1))
    keep = {k: v for k, v in rec.items() if not k.startswith("_")}
    keep.update(patch=arr[sl].copy(), ymask=rec["_ym"][sl].copy(), smask=rec["_sm"][sl].copy(),
                hpoly=rec["_hp"] - [c0, r0])
    return keep


def main():
    from ultralytics import SAM, YOLO
    OUT.mkdir(parents=True, exist_ok=True)
    model, sam = YOLO(str(WEIGHTS)), SAM(str(SAM_WEIGHTS))
    pool, t0 = [], time.time()
    for body, stem in TILES:
        got = collect(model, sam, body, stem)
        pool += got
        print(f"[{body}] {stem}: {len(got)} matched candidates ({time.time()-t0:.0f}s)", flush=True)
    if not pool:
        print("no candidates found")
        return

    for r in pool:
        r["diag_dist"] = min(axial(r["angle_y"], 45.0), axial(r["angle_y"], 135.0))
        r["gap_y_h"] = axial(r["angle_y"], r["angle_h"])
        r["gap_s_h"] = axial(r["angle_s"], r["angle_h"])

    sel = {}
    # --- mechanism: YOLO on a diagonal, human elsewhere. Spread over size so the reader can see the
    # effect fade with resolution rather than seeing six cherry-picked tiny rocks.
    elong = [r for r in pool if min(r["aspect_y"], r["aspect_h"]) > ASPECT_MIN
             and r["diag_dist"] <= DIAG_TOL]
    edges = np.linspace(DPX_RANGE[0], min(18.0, DPX_RANGE[1]), N_PER_SEL + 1)
    diag = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bucket = [r for r in elong if lo <= r["dpx"] < hi]
        if bucket:
            diag.append(max(bucket, key=lambda r: r["gap_y_h"]))
    sel["diag_locked"] = [c for c in (crop(r) for r in diag) if c]

    # --- area: the extreme tail, and a representative spread, kept separately and labelled as such
    bloat = sorted(pool, key=lambda r: -r["r_yolo"])[:N_PER_SEL]
    sel["area_bloat"] = [c for c in (crop(r) for r in bloat) if c]
    spread = []
    for body, _ in TILES:
        b = sorted([r for r in pool if r["body"] == body], key=lambda r: r["r_yolo"])
        if not b:
            continue
        for q in np.linspace(0.45, 0.85, N_SPREAD_PER_BODY):
            spread.append(b[min(len(b) - 1, int(q * (len(b) - 1)))])
    sel["area_spread"] = [c for c in (crop(r) for r in spread) if c]

    with open(OUT / "boulder_examples.pkl", "wb") as f:
        pickle.dump(sel, f)

    def med(vals):
        return round(float(np.median(vals)), 3) if len(vals) else None

    meta = dict(
        n_pool=len(pool), dpx_range=DPX_RANGE, aspect_min=ASPECT_MIN, diag_tol=DIAG_TOL,
        tiles=[t[1] for t in TILES],
        selections={k: len(v) for k, v in sel.items()},
        # Pool-level truth, so the notebook can say how typical the chosen panels are.
        # Two angle-gap columns on purpose. The pool carries NO aspect filter (area inflation is not
        # an elongation phenomenon), and |dAngle| on a near-round mask is meaningless — so the
        # all-shapes gap runs large and is NOT comparable to the published 3.8-8.5 deg figures. The
        # `_elong` columns restrict to boulders both the human and the producer call elongated and
        # well-resolved, which is the comparable subset.
        pool_by_body={b: dict(
            n=int(sum(r["body"] == b for r in pool)),
            median_r_yolo=med([r["r_yolo"] for r in pool if r["body"] == b]),
            median_r_sam=med([r["r_sam"] for r in pool if r["body"] == b]),
            median_gap_y_h_all_shapes=med([r["gap_y_h"] for r in pool if r["body"] == b]),
            median_gap_s_h_all_shapes=med([r["gap_s_h"] for r in pool if r["body"] == b]),
            n_elong=int(sum(r["body"] == b and min(r["aspect_h"], r["aspect_s"]) > ASPECT_MIN
                            and r["dpx"] >= 8 for r in pool)),
            median_gap_y_h_elong=med([r["gap_y_h"] for r in pool if r["body"] == b
                                      and min(r["aspect_h"], r["aspect_y"]) > ASPECT_MIN
                                      and r["dpx"] >= 8]),
            median_gap_s_h_elong=med([r["gap_s_h"] for r in pool if r["body"] == b
                                      and min(r["aspect_h"], r["aspect_s"]) > ASPECT_MIN
                                      and r["dpx"] >= 8]))
            for b, _ in TILES if any(r["body"] == b for r in pool)},
        panels={k: [dict(body=r["body"], dpx=round(r["dpx"], 1), r_yolo=round(r["r_yolo"], 2),
                         r_sam=round(r["r_sam"], 2), angle_h=round(r["angle_h"], 1),
                         angle_y=round(r["angle_y"], 1), angle_s=round(r["angle_s"], 1))
                    for r in v] for k, v in sel.items()})
    (OUT / "boulder_examples_meta.json").write_text(json.dumps(meta, indent=2))

    print("\npool medians. Cross-check against the published validation numbers: area SAM/human "
          "1.06-1.27x, YOLO/human 1.58-2.13x; matched |dAngle| SAM 3.8-8.5deg (ELONGATED subset).")
    for b, s in meta["pool_by_body"].items():
        print(f"  {b:6s} n={s['n']:4d}  YOLO/hum {s['median_r_yolo']}x  SAM/hum {s['median_r_sam']}x"
              f"  |  elongated n={s['n_elong']:3d}: |dAng| YOLO {s['median_gap_y_h_elong']}deg  "
              f"SAM {s['median_gap_s_h_elong']}deg"
              f"  (all shapes, not comparable: {s['median_gap_s_h_all_shapes']}deg)")
    print("\nselections: " + ", ".join(f"{k}={len(v)}" for k, v in sel.items()))
    print(f"wrote {OUT / 'boulder_examples.pkl'}")


if __name__ == "__main__":
    main()
