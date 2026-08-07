# Goal-3 phase 3: duplicate-rate deep dive. The pre-registered gate (plan §Validation 3) is
# about CROSS-SCALE near-pairs surviving fusion (< ~1-2%) — the tiny-box leakage mode fusion
# could ADD. fusion.duplicate_rate counts ALL near-pairs (upper bound; its docstring warns that
# adjacent distinct boulders land in the audit window). This script separates the three
# populations on the fused scale_aware catalog:
#   (a) same-scale near-pairs   — inherited scene geometry (present in the single-scale
#                                 baseline too; NOT fusion damage) -> compared against the
#                                 256-only baseline's own near-pair rate,
#   (b) cross-scale near-pairs  — the gated quantity: same boulder seen at two scales,
#                                 localization shift below the cluster threshold,
#   (c) bug pairs (>= thr)      — must be 0 by construction.
# Writes goal3_dupe_deepdive.csv.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
from YOLOv8BeyondEarth.fusion import (DEFAULT_CLUSTER_THR, fuse_predictions,
                                      equivalent_diameter_px, _candidate_pairs)
from YOLOv8BeyondEarth.benchmark import DEFAULT_SIZE_BANDS, _band_label

DE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionDE")

SCENES = {
    "nac": dict(res=0.48357597846618, arms={
        512: "preds/nac_512/nac_crop2k-predictions-ct-010-ss-512-is-1024-ov-020-downscaled-mask-nms.gpkg",
        256: "preds/nac_baseline/nac_crop2k-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg",
        (256, 2): "preds/nac_e2stream/nac_crop2k-predictions-ct-010-ss-256sr2-is-1024-ov-020-downscaled-mask-nms.gpkg"}),
    "hirise": dict(res=0.5, arms={
        512: "preds/hirise_512/hirise_crop4k-predictions-ct-010-ss-512-is-1024-ov-020-downscaled-mask-nms.gpkg",
        256: "preds/hirise_baseline/hirise_crop4k-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg",
        (256, 2): "preds/hirise_e2stream/hirise_crop4k-predictions-ct-010-ss-256sr2-is-1024-ov-020-downscaled-mask-nms.gpkg"}),
}


def near_pair_table(gdf, res, scale_col=None):
    """Near-pairs (DIoU in (0, thr)) by band; optionally split same- vs cross-scale."""
    boxes = gdf.geometry.bounds.to_numpy(dtype=float)
    ii, jj, vals = _candidate_pairs(boxes, 1e-9, "diou")
    near = vals < DEFAULT_CLUSTER_THR
    ii, jj = ii[near], jj[near]
    dpx = equivalent_diameter_px(gdf, res)
    if scale_col is not None:
        sk = gdf[scale_col].to_numpy()
        cross = sk[ii] != sk[jj]
    else:
        cross = np.zeros(len(ii), dtype=bool)
    rows = []
    for lo, hi in DEFAULT_SIZE_BANDS:
        lab = _band_label(lo, hi)
        in_band = (dpx >= lo) & (dpx < hi)
        pair_in = in_band[ii] | in_band[jj] if len(ii) else np.zeros(0, bool)
        n = int(in_band.sum())
        rows.append(dict(band=lab, n_dets=n,
                         near_same=int((pair_in & ~cross).sum()),
                         near_cross=int((pair_in & cross).sum()),
                         rate_same=float((pair_in & ~cross).sum() / n) if n else np.nan,
                         rate_cross=float((pair_in & cross).sum() / n) if n else np.nan))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = []
    for scene, cfg in SCENES.items():
        res = cfg["res"]
        gdfs = {k: gpd.read_file(DE / p) for k, p in cfg["arms"].items()}
        fused, _ = fuse_predictions(gdfs)

        t = near_pair_table(fused, res, scale_col="scale_key")
        t.insert(0, "catalog", "fused_scale_aware"); t.insert(0, "scene", scene)
        out.append(t)
        fused_e, _ = fuse_predictions(gdfs, engulf_thr=0.7)
        t = near_pair_table(fused_e, res, scale_col="scale_key")
        t.insert(0, "catalog", "fused_scale_aware_engulf"); t.insert(0, "scene", scene)
        out.append(t)
        for k in (256, (256, 2)):
            t = near_pair_table(gdfs[k], res)
            t.insert(0, "catalog", f"arm_{k}"); t.insert(0, "scene", scene)
            out.append(t)

    df = pd.concat(out, ignore_index=True)
    df.to_csv(DE / "goal3_dupe_deepdive.csv", index=False)
    piv = df.pivot_table(index=["scene", "band"], columns="catalog",
                         values=["rate_same", "rate_cross"])
    pd.set_option("display.width", 200)
    print(piv.round(4).to_string())
    print("\nGATE (cross-scale rate on fused, per band < ~0.02):")
    g = df[df.catalog.str.startswith("fused")][["scene", "catalog", "band", "n_dets",
                                                "near_cross", "rate_cross"]]
    print(g.to_string(index=False))
