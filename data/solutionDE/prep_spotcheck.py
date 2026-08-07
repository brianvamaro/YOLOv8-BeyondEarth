# Goal-3 phase 3: precision spot-check prep (plan §Validation 4, maintainer-ratified protocol).
# Two strata from the production-candidate fused catalog (256pref + engulf 0.7):
#   (1) ~100 SR-arm-only SUB-FLOOR detections (3-4 / 4-6 px, both dev crops; n_merged == 0, no
#       other arm saw anything) — the "sub-floor news" whose precision is unproven;
#   (2) ~24 LARGE (>=10 px) non-256-sourced uniques — the ROM precision dip at >=10 px
#       (baseline 0.87 -> fused 0.82) says ~half the added large detections are unverified.
# Each sample renders a blind chip PNG: [native x8 nearest | Lanczos x2 (what the SR arm saw)
# x4 nearest | same + outline]. ~20 chips are duplicated under fresh blind ids (observer-
# repeatability sub-sample; Golombek 2003 / Bickel 2021 multi-observer practice). Outputs in
# spotcheck/: chips + spotcheck_sheet.csv (to fill: verdict = boulder | not_boulder | unsure)
# + spotcheck_key.csv (the unblinding map — do not open before judging).
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import cv2

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
from YOLOv8BeyondEarth.fusion import fuse_predictions, equivalent_diameter_px

DE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionDE")
OUT = DE / "spotcheck"
SEED = 42
N_PER_CELL = 25          # 2 scenes x 2 bands x 25 = 100
N_REPEAT = 20
CTX = 24                 # native px context half-width around the centroid

SCENES = {
    "nac": dict(res=0.48357597846618, tif=DE / "nac_crop2k.tif", arms={
        512: "preds/nac_512/nac_crop2k-predictions-ct-010-ss-512-is-1024-ov-020-downscaled-mask-nms.gpkg",
        256: "preds/nac_baseline/nac_crop2k-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg",
        (256, 2): "preds/nac_e2stream/nac_crop2k-predictions-ct-010-ss-256sr2-is-1024-ov-020-downscaled-mask-nms.gpkg"}),
    "hirise": dict(res=0.5, tif=DE / "hirise_crop4k.tif", arms={
        512: "preds/hirise_512/hirise_crop4k-predictions-ct-010-ss-512-is-1024-ov-020-downscaled-mask-nms.gpkg",
        256: "preds/hirise_baseline/hirise_crop4k-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg",
        (256, 2): "preds/hirise_e2stream/hirise_crop4k-predictions-ct-010-ss-256sr2-is-1024-ov-020-downscaled-mask-nms.gpkg"}),
}
BANDS = {"3-4": (3.0, 4.0), "4-6": (4.0, 6.0)}


def render_chip(src, transform_inv, geom, res, path_with, scale_disp=4, ctx=CTX):
    """[native x2*scale_disp | SRx2 x scale_disp | SRx2 x scale_disp + outline] side by side."""
    cx, cy = geom.centroid.x, geom.centroid.y
    col, row = transform_inv * (cx, cy)
    col, row = int(round(col)), int(round(row))
    w = rasterio.windows.Window(col - ctx, row - ctx, 2 * ctx, 2 * ctx)
    native = src.read(1, window=w, boundless=True, fill_value=0)
    p1, p99 = np.percentile(native[native > 0], [1, 99]) if (native > 0).any() else (0, 255)
    disp = np.clip((native.astype(float) - p1) / max(p99 - p1, 1) * 255, 0, 255).astype(np.uint8)
    sr = cv2.resize(disp, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    z = 2 * scale_disp
    panel_nat = cv2.resize(disp, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
    panel_sr = cv2.resize(sr, None, fx=scale_disp, fy=scale_disp, interpolation=cv2.INTER_NEAREST)
    panel_ol = cv2.cvtColor(panel_sr, cv2.COLOR_GRAY2BGR)
    xs, ys = geom.exterior.xy
    pts = []
    for x, y in zip(xs, ys):
        c, r = transform_inv * (x, y)
        pts.append([int(round((c - (col - ctx)) * z)), int(round((r - (row - ctx)) * z))])
    cv2.polylines(panel_ol, [np.array(pts)], True, (0, 90, 255), 1, cv2.LINE_AA)
    gap = np.full((panel_nat.shape[0], 6, 3), 40, np.uint8)
    canvas = np.hstack([cv2.cvtColor(panel_nat, cv2.COLOR_GRAY2BGR), gap,
                        cv2.cvtColor(panel_sr, cv2.COLOR_GRAY2BGR), gap, panel_ol])
    cv2.imwrite(str(path_with), canvas)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    samples = []
    for scene, cfg in SCENES.items():
        gdfs = {k: gpd.read_file(DE / p) for k, p in cfg["arms"].items()}
        fused, _ = fuse_predictions(gdfs, rule="bands", bands={256: (0.0, np.inf)},
                                    native_res=cfg["res"], engulf_thr=0.7)
        fused = fused.reset_index(drop=True)
        fused["dpx"] = equivalent_diameter_px(fused, cfg["res"])
        sr_only = fused[(fused.scale_key == "(256, 2)") & (fused.n_merged == 0)]
        for band, (lo, hi) in BANDS.items():
            cand = sr_only[(sr_only.dpx >= lo) & (sr_only.dpx < hi)]
            take = min(N_PER_CELL, len(cand))
            sel = cand.sample(take, random_state=int(rng.integers(1e9)))
            for _, r in sel.iterrows():
                samples.append(dict(scene=scene, band=band, dpx=round(r.dpx, 2),
                                    score=round(float(r.score), 3), geometry=r.geometry,
                                    src_index=int(r.src_index)))
            print(f"{scene} {band}px: {len(cand)} candidates -> {take} sampled", flush=True)
        # stratum 2: large non-256-sourced uniques (the >=10 px precision-dip population)
        big = fused[(fused.scale_key != "256") & (fused.dpx >= 10)]
        take = min(16 if scene == "hirise" else 8, len(big))
        sel = big.sample(take, random_state=int(rng.integers(1e9)))
        for _, r in sel.iterrows():
            samples.append(dict(scene=scene, band="10-inf", dpx=round(r.dpx, 2),
                                score=round(float(r.score), 3), geometry=r.geometry,
                                src_index=int(r.src_index)))
        print(f"{scene} large uniques: {len(big)} candidates -> {take} sampled", flush=True)

    df = pd.DataFrame(samples)
    # repeats: 20 random rows re-judged under fresh ids
    rep = df.sample(min(N_REPEAT, len(df)), random_state=SEED).copy()
    rep["is_repeat"] = True
    df["is_repeat"] = False
    allrows = pd.concat([df, rep], ignore_index=True)
    order = rng.permutation(len(allrows))
    allrows = allrows.iloc[order].reset_index(drop=True)
    allrows["chip_id"] = [f"chip_{i:03d}" for i in range(len(allrows))]

    for scene, cfg in SCENES.items():
        sub = allrows[allrows.scene == scene]
        with rasterio.open(cfg["tif"]) as src:
            inv = ~src.transform
            for _, r in sub.iterrows():
                render_chip(src, inv, r.geometry, cfg["res"], OUT / f"{r.chip_id}.png",
                            ctx=max(CTX, int(np.ceil(r.dpx * 2))))

    key_cols = ["chip_id", "scene", "band", "dpx", "score", "src_index", "is_repeat"]
    allrows[key_cols].to_csv(OUT / "spotcheck_key.csv", index=False)
    sheet = allrows[["chip_id"]].copy()
    sheet["verdict"] = ""      # boulder | not_boulder | unsure
    sheet["notes"] = ""
    sheet.to_csv(OUT / "spotcheck_sheet.csv", index=False)
    print(f"\n{len(allrows)} chips written to {OUT} "
          f"({len(df)} unique + {len(rep)} blind repeats). Judge spotcheck_sheet.csv; "
          "panels = [native | SR x2 | SR x2 + outline].")
