"""Incidence-scaling sanity check (extends validate_calibration.py): does SAM's half-plane shift
grow with incidence toward the real ~0.05, as the shadow-length lever (L ~ tan(incidence)) predicts?

One shadow-free baseline + shadowed arms at several incidences (sun geometry fixed at the treatment
scene's, sun-axis 75.34 so f_lower rises with a sun-ward pull). Reports f_lower(SAM), f_lower(proxy),
SAM-minus-proxy and the shift vs baseline per incidence. GPU; run from inside YOLOv8-BeyondEarth/.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))

from YOLOv8BeyondEarth.transforms_test import render_boulder_field                       # noqa: E402
from YOLOv8BeyondEarth.orientation_validation import (                                   # noqa: E402
    detect_sam2_mask_sets, angles_from_moments, rock_trace_orientations)
from YOLOv8BeyondEarth.orientation import half_plane_fraction, axial_distance            # noqa: E402

WEIGHTS = BASE / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"
SAM_WEIGHTS = BASE / "sam2.1_b.pt"
CAL = json.loads((Path(__file__).resolve().parent / "shadow_calibration.json").read_text())
OUT = Path(__file__).resolve().parent / "calib_incidence.json"

SUN = CAL["sun_from_deg"]; SUN_AXIS = SUN % 180.0
CAP = CAL["cap"]["median"]; SHADOW = CAL["shadow"]["median"]; HF = CAL["height_frac"]["median"]
INCIDENCES = [30.0, 42.0, 55.0, 68.0]

TARGET = 3000
SIZE, NPF, BG, NOISE = 1024, 400, 110.0, 12.0
DIAM_PX, ASPECT = (6.0, 20.0), (1.2, 3.0)
DEVICE, CONF, SLICE, IMGSZ, SAM_BATCH = 0, 0.10, 256, 1024, 64


def _match(df_det, ang_det, truth):
    from scipy.spatial import cKDTree
    if len(df_det) == 0 or len(truth) == 0:
        return pd.DataFrame()
    tree = cKDTree(truth[["cx", "cy"]].to_numpy())
    dist, ti = tree.query(df_det[["cx", "cy"]].to_numpy(), k=1)
    tol = np.maximum(0.75 * truth["diam_px"].to_numpy()[ti], 6.0)
    used_t, used_d, rows = set(), set(), []
    for di in np.argsort(dist):
        if dist[di] > tol[di] or ti[di] in used_t or di in used_d:
            continue
        used_t.add(ti[di]); used_d.add(di); rows.append((di, ti[di]))
    if not rows:
        return pd.DataFrame()
    da = np.array([r[0] for r in rows]); ta = np.array([r[1] for r in rows])
    return pd.DataFrame({
        "angle": ang_det["angle180"].to_numpy()[da],
        "truth_angle": truth["angle180"].to_numpy()[ta],
        "aspect_true": truth["a_px"].to_numpy()[ta] / truth["b_px"].to_numpy()[ta],
        "diam_true": truth["diam_px"].to_numpy()[ta],
        "truth_idx": ta,
    })


def run_arm(model, sam, inc_deg, seed0):
    rows = []
    seed = seed0
    while sum(len(r) for r in rows) < TARGET:
        kw = dict(size=SIZE, n=NPF, diam_px=DIAM_PX, aspect=ASPECT, bg=BG, noise=NOISE,
                  cap=CAP, sun_az_deg=SUN, seed=seed)
        if inc_deg is None:
            img, truth = render_boulder_field(inc_deg=None, shadow_frac=0.0, **kw)
        else:
            img, truth = render_boulder_field(inc_deg=inc_deg, shadow=SHADOW, height_frac=HF, **kw)
        seed += 1
        df, masks = detect_sam2_mask_sets(model, sam, img, slice_size=SLICE, imgsz=IMGSZ, conf=CONF,
                                          device=DEVICE, configs=("yolo", "sam256"),
                                          sam_batch=SAM_BATCH, empty_cache_every=8)
        if len(df) == 0:
            continue
        m = _match(df, angles_from_moments(masks["sam256"]), truth)
        if len(m):
            proxy = rock_trace_orientations(img, truth, bg=BG, cap=CAP)
            m = m[m["aspect_true"] > 1.35].copy()
            m["proxy_angle"] = proxy["angle180"].to_numpy()[m["truth_idx"].to_numpy()]
            rows.append(m)
    return pd.concat(rows, ignore_index=True)


def summ(df):
    return {
        "n": int(len(df)),
        "f_lower_sam": round(float(half_plane_fraction(df["angle"])), 4),
        "f_lower_proxy": round(float(half_plane_fraction(df["proxy_angle"])), 4),
        "conc_sam_to_sun": round(float(np.nanmean(axial_distance(df["angle"].to_numpy(), SUN_AXIS))), 2),
        "median_abs_err_sam": round(float(np.nanmedian(axial_distance(
            df["angle"].to_numpy(), df["truth_angle"].to_numpy()))), 2),
    }


def main():
    from ultralytics import YOLO, SAM
    t0 = time.time()
    model = YOLO(str(WEIGHTS)); sam = SAM(str(SAM_WEIGHTS))
    base = summ(run_arm(model, sam, None, 1000))
    print("baseline (shadow-free):", base)
    res = {"sun_axis_deg": SUN_AXIS, "cap": CAP, "shadow": SHADOW, "height_frac": HF,
           "shadow_free": base, "by_incidence": {}}
    for i, inc in enumerate(INCIDENCES):
        s = summ(run_arm(model, sam, inc, 5000 + 1000 * i))
        s["f_lower_shift_vs_free"] = round(s["f_lower_sam"] - base["f_lower_sam"], 4)
        s["sam_minus_proxy"] = round(s["f_lower_sam"] - s["f_lower_proxy"], 4)
        res["by_incidence"][f"{inc:.0f}"] = s
        print(f"inc {inc:.0f}: shift={s['f_lower_shift_vs_free']:+.3f} "
              f"sam-proxy={s['sam_minus_proxy']:+.3f} conc={s['conc_sam_to_sun']:.1f}")
    res["wall_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
