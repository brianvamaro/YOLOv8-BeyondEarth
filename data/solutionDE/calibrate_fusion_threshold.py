# Goal-3 phase 1: calibrate the fusion cluster threshold (DIoU) from the three-arm run's
# cross-scale matched pairs (goal3_multiscale_fusion_plan.md, "Small-box matching risk").
# A DIoU threshold is NOT an IoU threshold (DIoU in (-1,1], systematically below IoU), so 0.5
# must not be ported across. Positive class = match_detections pairs (centroid <= 2*res, the
# standing pairing gate) between baseline(256) and each other arm; negative control = nearest
# *distinct* neighbours within the baseline itself (post-NMS, so real separate boulders).
# The chosen threshold gets pinned in tests/test_fusion.py.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
from YOLOv8BeyondEarth.benchmark import match_detections

DE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionDE")
BANDS = [(3, 4), (4, 6), (6, 8), (8, 10), (10, 14), (14, np.inf)]

SCENES = {
    "hirise": dict(native_res=0.5, arms={
        "baseline": ("hirise_baseline", "hirise_crop4k-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg"),
        "e2":       ("hirise_e2", "hirise_crop4k_sr2-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg"),
        "dctl":     ("hirise_dctl", "hirise_crop4k-predictions-ct-010-ss-128-is-1024-ov-020-downscaled-mask-nms.gpkg"),
    }),
    "nac": dict(native_res=0.48357597846618, arms={
        "baseline": ("nac_baseline", "nac_crop2k-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg"),
        "e2":       ("nac_e2", "nac_crop2k_sr2-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg"),
        "dctl":     ("nac_dctl", "nac_crop2k-predictions-ct-010-ss-128-is-1024-ov-020-downscaled-mask-nms.gpkg"),
    }),
}


def boxes_of(gdf):
    return np.array([g.bounds for g in gdf.geometry.values])  # (n, 4) xmin ymin xmax ymax


def pairwise_iou_diou(ba, bb):
    """Element-wise box IoU and DIoU for aligned (n,4) box arrays."""
    ixmin = np.maximum(ba[:, 0], bb[:, 0]); iymin = np.maximum(ba[:, 1], bb[:, 1])
    ixmax = np.minimum(ba[:, 2], bb[:, 2]); iymax = np.minimum(ba[:, 3], bb[:, 3])
    iw = np.clip(ixmax - ixmin, 0, None); ih = np.clip(iymax - iymin, 0, None)
    inter = iw * ih
    area_a = (ba[:, 2] - ba[:, 0]) * (ba[:, 3] - ba[:, 1])
    area_b = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
    union = area_a + area_b - inter
    iou = np.where(union > 0, inter / union, 0.0)
    # DIoU = IoU - d^2/c^2 (Zheng et al. 2020): d = center distance, c = diag of enclosing box
    cxa = (ba[:, 0] + ba[:, 2]) / 2; cya = (ba[:, 1] + ba[:, 3]) / 2
    cxb = (bb[:, 0] + bb[:, 2]) / 2; cyb = (bb[:, 1] + bb[:, 3]) / 2
    d2 = (cxa - cxb) ** 2 + (cya - cyb) ** 2
    exmin = np.minimum(ba[:, 0], bb[:, 0]); eymin = np.minimum(ba[:, 1], bb[:, 1])
    exmax = np.maximum(ba[:, 2], bb[:, 2]); eymax = np.maximum(ba[:, 3], bb[:, 3])
    c2 = (exmax - exmin) ** 2 + (eymax - eymin) ** 2
    return iou, iou - np.where(c2 > 0, d2 / c2, 0.0)


def band_label(dpx):
    for lo, hi in BANDS:
        if lo <= dpx < hi:
            return f"{lo}-{hi if np.isfinite(hi) else 'inf'}"
    return "<3"


if __name__ == "__main__":
    rows = []
    for scene, cfg in SCENES.items():
        res = cfg["native_res"]
        gdfs = {arm: gpd.read_file(DE / "preds" / d / f) for arm, (d, f) in cfg["arms"].items()}
        for g in gdfs.values():
            g["dpx"] = np.sqrt(4 * g.geometry.area / np.pi) / res

        base = gdfs["baseline"]
        bb_base = boxes_of(base)

        # positives: trusted cross-scale pairs (same boulder seen at two scales)
        for arm in ("e2", "dctl"):
            other = gdfs[arm]
            m = match_detections(base, other, max_dist=2.0 * res)
            ba = bb_base[m.idx_a.values]
            bo = boxes_of(other)[m.idx_b.values]
            iou, diou = pairwise_iou_diou(ba, bo)
            for band, i, d in zip(base.dpx.iloc[m.idx_a.values].map(band_label), iou, diou):
                rows.append(dict(scene=scene, kind=f"pair_{arm}", band=band, iou=i, diou=d))

        # negatives: nearest DISTINCT baseline neighbour (post-NMS -> different boulders)
        cx = (bb_base[:, 0] + bb_base[:, 2]) / 2
        cy = (bb_base[:, 1] + bb_base[:, 3]) / 2
        tree = cKDTree(np.column_stack((cx, cy)))
        _, nn = tree.query(np.column_stack((cx, cy)), k=2)
        j = nn[:, 1]
        iou, diou = pairwise_iou_diou(bb_base, bb_base[j])
        for band, i, d in zip(base.dpx.map(band_label), iou, diou):
            rows.append(dict(scene=scene, kind="neighbour", band=band, iou=i, diou=d))

    df = pd.DataFrame(rows)
    df.to_csv(DE / "fusion_threshold_calibration.csv", index=False)

    q = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    print("=== per-pair metric distributions (positives = cross-scale matched pairs; "
          "negatives = nearest distinct baseline neighbour) ===")
    for metric in ("iou", "diou"):
        print(f"\n--- {metric.upper()} ---")
        summ = (df.groupby(["kind", "band"])[metric]
                  .quantile(q).unstack().round(3))
        summ["n"] = df.groupby(["kind", "band"])[metric].size()
        print(summ.to_string())

    # capture-vs-contamination sweep: fraction of positives kept / negatives merged per threshold
    print("\n=== threshold sweep (DIoU): capture = frac of true pairs clustered; "
          "contam = frac of distinct neighbours wrongly clustered ===")
    pos = df[df.kind.str.startswith("pair")]
    neg = df[df.kind == "neighbour"]
    for thr in np.round(np.arange(-0.2, 0.55, 0.05), 2):
        cap = float((pos.diou >= thr).mean())
        cap_small = float((pos[pos.band.isin(["3-4", "4-6"])].diou >= thr).mean())
        con = float((neg.diou >= thr).mean())
        print(f"thr={thr:+.2f}  capture={cap:.3f}  capture(3-6px)={cap_small:.3f}  contam={con:.4f}")
