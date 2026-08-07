# T1 — synthetic uniform-truth null for the SAM2 orientation-artifact validation plan
# (docs/sam2_orientation_validation_plan.md). Renders fields of synthetic boulders at UNIFORM-random
# known long-axis angles (shadow-free AND shadowed, fixed sun azimuth), spanning the size bands and
# aspect 1.0-3.0, runs the production detect -> SAM2 re-mask path (detect_sam2_mask_sets, configs
# yolo+sam256), reads each aligned mask set with the moment estimator, and matches detections back to
# the known truth by centroid. Caches a tidy per-boulder table so the notebook does the CPU card
# (grid_frac cardinal, f_lower half-plane, asymmetry, signed per-angle deviation, by band x aspect)
# without re-running the GPU. A small sam256/sam1024 equivalence spot-check is saved alongside.
#
# __main__ guard required: ultralytics dataloader workers use Windows spawn and re-import this file.
# Run from inside YOLOv8-BeyondEarth/ so SAM("sam2.1_b.pt") resolves (or the absolute path below).
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

FIELD = 1024            # px per synthetic field (native res; sam256 masks land at slice res = native)
N_PER_FIELD = 400       # target boulders per field (rejection sampling places as many as fit)
DIAM_PX = (3, 20)       # spans DEFAULT_SIZE_BANDS (3-4 ... 14+)
ASPECT = (1.0, 3.0)     # full aspect sweep; production rose filters aspect > 1.35 in analysis
SUN_AZ = 255.0          # fixed sun azimuth for the shadowed arm (render default)
TARGET_PER_COND = 20000 # matched boulders per condition (plan: N ~ 20-40k)
SPOTCHECK_FIELDS = 6    # small yolo/sam256/sam1024 equivalence spot-check (config-invariance)


def _match(truth, df):
    """Greedy one-to-one match of truth boulders to detections by centroid distance, accepted within
    max(0.75*diam_true, 6) px. Returns a DataFrame of matched rows carrying both truth and per-side
    measured columns (truth is uniform by construction, so banding can use the true size/aspect)."""
    if len(df) == 0 or len(truth) == 0:
        return pd.DataFrame()
    tc = truth[["cx", "cy"]].to_numpy()
    dc = df[["cx", "cy"]].to_numpy()
    radius = np.maximum(0.75 * truth["diam_px"].to_numpy(), 6.0)
    pairs = cKDTree(tc).query_ball_tree(cKDTree(dc), r=float(radius.max()))
    cand = []
    for ti, dis in enumerate(pairs):
        for di in dis:
            d = float(np.hypot(*(tc[ti] - dc[di])))
            if d <= radius[ti]:
                cand.append((d, ti, di))
    cand.sort()
    used_t, used_d, rows = set(), set(), []
    for d, ti, di in cand:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti); used_d.add(di)
        tr = truth.iloc[ti]; de = df.iloc[di]
        rows.append({
            "truth_angle": float(tr["angle180"]),
            "aspect_true": float(tr["a_px"] / tr["b_px"]),
            "diam_true_px": float(tr["diam_px"]),
            "match_dist": d,
            "angle_yolo": float(de["angle_yolo"]), "aspect_yolo": float(de["aspect_yolo"]),
            "angle_sam256": float(de["angle_sam256"]), "aspect_sam256": float(de["aspect_sam256"]),
            "dpx_sam256": float(de["dpx_sam256"]),
        })
    return pd.DataFrame(rows)


def _run_arm(model, sam, cond, shadow_frac, seed0, target, configs, detect, angfn):
    """Render fields for one condition until `target` boulders are matched; return matched rows."""
    out, field_i = [], 0
    n_matched = 0
    while n_matched < target and field_i < 400:
        seed = seed0 + field_i
        img, truth = render_field(size=FIELD, n=N_PER_FIELD, diam_px=DIAM_PX, aspect=ASPECT,
                                  shadow_frac=shadow_frac, sun_az_deg=SUN_AZ, seed=seed)
        df, masks = detect(model, sam, img, slice_size=256, imgsz=1024, conf=0.10, device=0,
                           configs=configs, empty_cache_every=8)
        if len(df):
            for side in configs:
                a = angfn(masks[side])
                df[f"angle_{side}"] = a["angle180"].values
                df[f"aspect_{side}"] = a["aspect"].values
                df[f"dpx_{side}"] = a["diameter_px"].values
            m = _match(truth, df)
            if len(m):
                m["cond"] = cond
                m["seed"] = seed
                out.append(m)
                n_matched += len(m)
        field_i += 1
        print(f"  [{cond}] field {field_i} seed {seed}: matched {n_matched}/{target}", flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


if __name__ == "__main__":
    from ultralytics import YOLO, SAM
    from YOLOv8BeyondEarth.orientation_validation import detect_sam2_mask_sets, angles_from_moments
    from YOLOv8BeyondEarth.transforms_test import render_boulder_field as render_field

    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    sam = SAM(str(SAM_WEIGHTS))

    t0 = time.time()
    # --- main run: yolo + sam256, both conditions ---
    arms = []
    for cond, sf, seed0 in (("shadowfree", 0.0, 1000), ("shadow", 0.6, 5000)):
        print(f"== condition {cond} ==", flush=True)
        arms.append(_run_arm(model, sam, cond, sf, seed0, TARGET_PER_COND,
                             ("yolo", "sam256"), detect_sam2_mask_sets, angles_from_moments))
    main_df = pd.concat(arms, ignore_index=True)
    main_df.to_csv(OUT / "t1_synth.csv", index=False)

    # --- sam256/sam1024 equivalence spot-check: a few fields, all three configs (config-invariance;
    #     already the same detections, so no truth match needed — read the aligned triple directly) ---
    spot_rows, fi = [], 0
    while fi < SPOTCHECK_FIELDS:
        seed = 30000 + fi
        img, truth = render_field(size=FIELD, n=N_PER_FIELD, diam_px=DIAM_PX, aspect=ASPECT,
                                  shadow_frac=0.0, sun_az_deg=SUN_AZ, seed=seed)
        df, masks = detect_sam2_mask_sets(model, sam, img, slice_size=256, imgsz=1024, conf=0.10,
                                          device=0, configs=("yolo", "sam256", "sam1024"),
                                          empty_cache_every=8)
        if len(df):
            for side in ("yolo", "sam256", "sam1024"):
                a = angles_from_moments(masks[side])
                df[f"angle_{side}"] = a["angle180"].values
                df[f"aspect_{side}"] = a["aspect"].values
                df[f"dpx_{side}"] = a["diameter_px"].values
            # keep the aligned triple directly (already same detections; no truth match needed)
            keep = df[["angle_yolo", "angle_sam256", "angle_sam1024",
                       "aspect_yolo", "aspect_sam256", "aspect_sam1024"]].copy()
            keep["seed"] = seed
            spot_rows.append(keep)
        fi += 1
        print(f"  [spotcheck] field {fi}/{SPOTCHECK_FIELDS} seed {seed}: n={len(df)}", flush=True)
    spot = pd.concat(spot_rows, ignore_index=True) if spot_rows else pd.DataFrame()
    spot.to_csv(OUT / "t1_synth_spotcheck.csv", index=False)

    wall = time.time() - t0
    meta = {
        "wall_s": wall,
        "n_shadowfree": int((main_df.cond == "shadowfree").sum()),
        "n_shadow": int((main_df.cond == "shadow").sum()),
        "n_spotcheck": int(len(spot)),
        "field_px": FIELD, "diam_px": list(DIAM_PX), "aspect": list(ASPECT),
        "sun_az_deg": SUN_AZ, "configs": ["yolo", "sam256"],
        "slice_size": 256, "imgsz": 1024, "conf": 0.10,
    }
    (OUT / "t1_run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print(f"wall {wall:.0f}s", flush=True)
