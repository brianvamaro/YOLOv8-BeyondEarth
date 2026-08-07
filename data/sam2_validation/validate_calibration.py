"""Pre-sweep validation: does the CALIBRATED synthetic shadow reproduce the real half-plane shift?

The checkpoint gate for Component A. Renders synthetic uniform-truth boulder fields under the
shadow-calibration.json numbers (treatment scene: inc 42.1, sun-from 255.34, cap 32, shadow -86,
height_frac 0.23) two ways -- SHADOW-FREE and SHADOWED -- pushes both through the production SAM2
re-mask path (sam256) plus YOLO and the rock-tracing proxy, matches each mask to known truth, and
measures the half-plane fraction f_lower (fixed cardinal frame) per masker per arm.

Expectation if the shadow is realistic: SAM's f_lower moves from ~0.5 (shadow-free) toward the sun
axis (75.34 deg is in [0,90), so f_lower rises) by ~the real ~0.05, while the rock-tracing proxy
(shadow excluded) stays ~0.5 -- i.e. SAM-minus-proxy ~ the SAM-specific illumination excess. Writes
calib_validation.json. GPU; run from inside YOLOv8-BeyondEarth/ in the bouldernet env.
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
OUT = Path(__file__).resolve().parent / "calib_validation.json"

INC = CAL["incidence_deg"]
SUN = CAL["sun_from_deg"]
CAP = CAL["cap"]["median"]
SHADOW = CAL["shadow"]["median"]
HF = CAL["height_frac"]["median"]
SUN_AXIS = SUN % 180.0

TARGET = 5000            # matched elongated boulders per arm
SIZE, NPF = 1024, 400
BG, NOISE = 110.0, 12.0
DIAM_PX, ASPECT = (6.0, 20.0), (1.2, 3.0)
DEVICE, CONF, SLICE, IMGSZ = 0, 0.10, 256, 1024
SAM_BATCH = 64


def _match(df_det, ang_det, truth):
    """Greedy one-to-one nearest match of detections to known truth centroids (within
    max(0.75*diam_true, 6) px). Returns a DataFrame with the matched det angle + truth columns."""
    from scipy.spatial import cKDTree
    if len(df_det) == 0 or len(truth) == 0:
        return pd.DataFrame()
    tx = truth[["cx", "cy"]].to_numpy()
    tree = cKDTree(tx)
    dx = df_det[["cx", "cy"]].to_numpy()
    dist, ti = tree.query(dx, k=1)
    tol = np.maximum(0.75 * truth["diam_px"].to_numpy()[ti], 6.0)
    order = np.argsort(dist)
    used_t, used_d, rows = set(), set(), []
    for di in order:
        if dist[di] > tol[di] or ti[di] in used_t or di in used_d:
            continue
        used_t.add(ti[di]); used_d.add(di)
        rows.append((di, ti[di]))
    if not rows:
        return pd.DataFrame()
    di_arr = np.array([r[0] for r in rows]); ti_arr = np.array([r[1] for r in rows])
    return pd.DataFrame({
        "angle": ang_det["angle180"].to_numpy()[di_arr],
        "aspect_det": ang_det["aspect"].to_numpy()[di_arr],
        "truth_angle": truth["angle180"].to_numpy()[ti_arr],
        "aspect_true": (truth["a_px"].to_numpy()[ti_arr] / truth["b_px"].to_numpy()[ti_arr]),
        "diam_true": truth["diam_px"].to_numpy()[ti_arr],
        "truth_idx": ti_arr,
    })


def run_arm(model, sam, shadowed, seed0):
    sam_rows, yolo_rows, proxy_rows = [], [], []
    seed = seed0
    while sum(len(r) for r in sam_rows) < TARGET:
        kw = dict(size=SIZE, n=NPF, diam_px=DIAM_PX, aspect=ASPECT, bg=BG, noise=NOISE,
                  cap=CAP, sun_az_deg=SUN, seed=seed)
        if shadowed:
            img, truth = render_boulder_field(inc_deg=INC, shadow=SHADOW, height_frac=HF, **kw)
        else:
            img, truth = render_boulder_field(inc_deg=None, shadow_frac=0.0, **kw)
        seed += 1
        df, masks = detect_sam2_mask_sets(model, sam, img, slice_size=SLICE, imgsz=IMGSZ,
                                          conf=CONF, device=DEVICE, configs=("yolo", "sam256"),
                                          sam_batch=SAM_BATCH, empty_cache_every=8)
        if len(df) == 0:
            continue
        a_sam = angles_from_moments(masks["sam256"])
        a_yolo = angles_from_moments(masks["yolo"])
        m_sam = _match(df, a_sam, truth)
        m_yolo = _match(df, a_yolo, truth)
        proxy = rock_trace_orientations(img, truth, bg=BG, cap=CAP)
        if len(m_sam):
            m_sam = m_sam[m_sam["aspect_true"] > 1.35]
            m_sam["proxy_angle"] = proxy["angle180"].to_numpy()[m_sam["truth_idx"].to_numpy()]
            sam_rows.append(m_sam)
        if len(m_yolo):
            yolo_rows.append(m_yolo[m_yolo["aspect_true"] > 1.35])
    S = pd.concat(sam_rows, ignore_index=True)
    Y = pd.concat(yolo_rows, ignore_index=True) if yolo_rows else pd.DataFrame()
    return S, Y


def summarise(S, Y):
    def fl(a):
        return round(float(half_plane_fraction(a)), 4)

    def conc(a):                      # mean axial distance to the sun axis (lower = pulled to sun)
        d = axial_distance(a.to_numpy() if hasattr(a, "to_numpy") else a, SUN_AXIS)
        return round(float(np.nanmean(d)), 2)
    return {
        "n_sam": int(len(S)), "n_yolo": int(len(Y)),
        "f_lower_sam": fl(S["angle"]), "f_lower_yolo": fl(Y["angle"]) if len(Y) else None,
        "f_lower_proxy": fl(S["proxy_angle"]),
        "conc_sam_to_sun": conc(S["angle"]), "conc_proxy_to_sun": conc(S["proxy_angle"]),
        "median_abs_err_sam": round(float(np.nanmedian(axial_distance(
            S["angle"].to_numpy(), S["truth_angle"].to_numpy()))), 2),
        "median_abs_err_proxy": round(float(np.nanmedian(axial_distance(
            S["proxy_angle"].to_numpy(), S["truth_angle"].to_numpy()))), 2),
    }


def main():
    from ultralytics import YOLO, SAM
    t0 = time.time()
    model = YOLO(str(WEIGHTS))
    sam = SAM(str(SAM_WEIGHTS))
    print(f"calib: inc={INC:.1f} sun={SUN:.1f} (axis {SUN_AXIS:.1f}) cap={CAP} shadow={SHADOW} hf={HF}")

    Sf, Yf = run_arm(model, sam, shadowed=False, seed0=1000)
    print("shadow-free arm done", len(Sf))
    Ss, Ys = run_arm(model, sam, shadowed=True, seed0=5000)
    print("shadowed arm done", len(Ss))

    res = {
        "calibration": {"incidence_deg": INC, "sun_from_deg": SUN, "sun_axis_deg": SUN_AXIS,
                        "cap": CAP, "shadow": SHADOW, "height_frac": HF},
        "shadow_free": summarise(Sf, Yf),
        "shadowed": summarise(Ss, Ys),
        "wall_s": round(time.time() - t0, 1),
    }
    res["deltas"] = {
        "sam_f_lower_shift": round(res["shadowed"]["f_lower_sam"] - res["shadow_free"]["f_lower_sam"], 4),
        "proxy_f_lower_shift": round(res["shadowed"]["f_lower_proxy"] - res["shadow_free"]["f_lower_proxy"], 4),
        "sam_minus_proxy_shadowed": round(res["shadowed"]["f_lower_sam"] - res["shadowed"]["f_lower_proxy"], 4),
        "sam_conc_gain": round(res["shadow_free"]["conc_sam_to_sun"] - res["shadowed"]["conc_sam_to_sun"], 2),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
