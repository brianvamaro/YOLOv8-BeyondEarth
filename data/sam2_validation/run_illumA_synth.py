"""Component A of the illumination-artifact exploration (docs/illumination_artifact_exploration_plan.md):
the realistically-shadowed synthetic sweep that measures illumination-shaping MAGNITUDE against known
truth, separated from real elongation.

Grid: sun_az (8 values over 180 deg) x incidence (30..68 deg) on uniform-random-truth boulder fields
rendered with the shadow-calibration.json numbers (cap 32, shadow -86, height_frac 0.23) at a wide
size/aspect range. Three maskers read each field the SAME way (angles_from_moments): SAM2 (sam256,
the production re-mask), YOLO, and the rock-tracing proxy (cap-only threshold = the human/rock prior).
Also records, per cell, SAM's box-blowup FAILURE count (detections dropped because the SAM mask grabbed
> 1.5x the YOLO box -- i.e. the shadow) vs kept, to characterise the self-limiting of shaping at high
incidence.

Writes (gitignored, read by notebooks/Goal2_illumination_artifact):
  illumA_boulders.csv   per matched boulder: inc, sun_az, sun_axis, truth_angle, aspect_true,
                        diam_true_px, angle_sam, angle_yolo, angle_proxy
  illumA_cellmeta.csv   per cell: inc, sun_az, shadowed, n_kept, n_fail_sam, n_fields, wall_s
  illumA_run_meta.json  grid + calibration + totals

GPU; run from inside YOLOv8-BeyondEarth/ in the bouldernet env (one process; never two SAM jobs at once).
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

HERE = Path(__file__).resolve().parent
WEIGHTS = BASE / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"
SAM_WEIGHTS = BASE / "sam2.1_b.pt"
CAL = json.loads((HERE / "shadow_calibration.json").read_text())
OUT_B = HERE / "illumA_boulders.csv"
OUT_C = HERE / "illumA_cellmeta.csv"
OUT_M = HERE / "illumA_run_meta.json"

CAP = CAL["cap"]["median"]; SHADOW = CAL["shadow"]["median"]; HF = CAL["height_frac"]["median"]
SUN_AZ = list(np.arange(0.0, 180.0, 22.5))          # 8 values; sun AXIS = sun_az % 180
INCIDENCES = [30.0, 42.0, 50.0, 58.0, 68.0]
TARGET, MAX_FIELDS = 2500, 80
SIZE, NPF, BG, NOISE = 1024, 400, 110.0, 12.0
DIAM_PX, ASPECT = (5.0, 22.0), (1.2, 3.0)
DEVICE, CONF, SLICE, IMGSZ, SAM_BATCH = 0, 0.10, 256, 1024, 64
ASPECT_MIN = 1.35


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
    return pd.DataFrame({"di": da, "ti": ta})


def run_cell(model, sam, inc_deg, sun_az, seed0):
    """Accumulate >=TARGET matched elongated boulders for one (inc, sun) cell (inc_deg=None = free)."""
    from scipy.spatial import cKDTree  # noqa: F401  (used inside _match)
    boulders, seed, n_fail, n_kept, nf = [], seed0, 0, 0, 0
    while sum(len(b) for b in boulders) < TARGET and nf < MAX_FIELDS:
        kw = dict(size=SIZE, n=NPF, diam_px=DIAM_PX, aspect=ASPECT, bg=BG, noise=NOISE,
                  cap=CAP, sun_az_deg=sun_az, seed=seed)
        if inc_deg is None:
            img, truth = render_boulder_field(inc_deg=None, shadow_frac=0.0, **kw)
        else:
            img, truth = render_boulder_field(inc_deg=inc_deg, shadow=SHADOW, height_frac=HF, **kw)
        seed += 1; nf += 1
        df, masks = detect_sam2_mask_sets(model, sam, img, slice_size=SLICE, imgsz=IMGSZ, conf=CONF,
                                          device=DEVICE, configs=("yolo", "sam256"),
                                          sam_batch=SAM_BATCH, empty_cache_every=8)
        n_fail += int(df.attrs.get("failed", {}).get("sam256", 0))
        n_kept += int(len(df))
        if len(df) == 0:
            continue
        a_sam = angles_from_moments(masks["sam256"])
        a_yolo = angles_from_moments(masks["yolo"])
        proxy = rock_trace_orientations(img, truth, bg=BG, cap=CAP)
        m = _match(df, a_sam, truth)
        if not len(m):
            continue
        di, ti = m["di"].to_numpy(), m["ti"].to_numpy()
        asp_true = truth["a_px"].to_numpy()[ti] / truth["b_px"].to_numpy()[ti]
        keep = asp_true > ASPECT_MIN
        if not keep.any():
            continue
        di, ti, asp_true = di[keep], ti[keep], asp_true[keep]
        boulders.append(pd.DataFrame({
            "inc": inc_deg if inc_deg is not None else -1.0,
            "sun_az": sun_az, "sun_axis": sun_az % 180.0,
            "truth_angle": truth["angle180"].to_numpy()[ti],
            "aspect_true": asp_true, "diam_true_px": truth["diam_px"].to_numpy()[ti],
            "angle_sam": a_sam["angle180"].to_numpy()[di],
            "angle_yolo": a_yolo["angle180"].to_numpy()[di],
            "angle_proxy": proxy["angle180"].to_numpy()[ti],
        }))
    B = pd.concat(boulders, ignore_index=True) if boulders else pd.DataFrame()
    meta = {"inc": inc_deg if inc_deg is not None else -1.0, "sun_az": sun_az,
            "shadowed": inc_deg is not None, "n_matched": int(len(B)), "n_kept": n_kept,
            "n_fail_sam": n_fail, "n_fields": nf}
    return B, meta


def main():
    from ultralytics import YOLO, SAM
    t0 = time.time()
    model = YOLO(str(WEIGHTS)); sam = SAM(str(SAM_WEIGHTS))
    for p in (OUT_B, OUT_C):
        if p.exists():
            p.unlink()
    metas, total = [], 0
    header_b = header_c = True

    # one shadow-free baseline (sun geometry irrelevant with no shadow; use sun_az=0)
    B, meta = run_cell(model, sam, None, 0.0, seed0=900000)
    B.to_csv(OUT_B, mode="a", header=header_b, index=False); header_b = False
    pd.DataFrame([meta]).to_csv(OUT_C, mode="a", header=header_c, index=False); header_c = False
    metas.append(meta); total += len(B)
    print(f"[free] n={len(B)} fields={meta['n_fields']} t={time.time()-t0:.0f}s")

    for ii, inc in enumerate(INCIDENCES):
        for jj, sun in enumerate(SUN_AZ):
            B, meta = run_cell(model, sam, inc, sun, seed0=1000 + 1000 * ii + 100 * jj)
            B.to_csv(OUT_B, mode="a", header=header_b, index=False); header_b = False
            pd.DataFrame([meta]).to_csv(OUT_C, mode="a", header=header_c, index=False)
            metas.append(meta); total += len(B)
            fr = meta["n_fail_sam"] / max(meta["n_kept"] + meta["n_fail_sam"], 1)
            print(f"[inc {inc:.0f} sun {sun:.0f}] n={len(B)} failrate={fr:.3f} "
                  f"fields={meta['n_fields']} t={time.time()-t0:.0f}s", flush=True)

    OUT_M.write_text(json.dumps({
        "calibration": {"cap": CAP, "shadow": SHADOW, "height_frac": HF,
                        "incidence_deg": CAL["incidence_deg"], "sun_from_deg": CAL["sun_from_deg"]},
        "grid": {"sun_az": SUN_AZ, "incidences": INCIDENCES, "target_per_cell": TARGET},
        "render": {"size": SIZE, "n_per_field": NPF, "bg": BG, "noise": NOISE,
                   "diam_px": DIAM_PX, "aspect": ASPECT, "aspect_min": ASPECT_MIN},
        "n_cells": len(metas), "n_boulders_total": int(total), "wall_s": round(time.time() - t0, 1),
    }, indent=2))
    print(f"DONE total={total} cells={len(metas)} wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
