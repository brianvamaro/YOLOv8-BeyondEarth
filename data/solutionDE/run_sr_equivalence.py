# Goal-3 phase 2: streamed-SR equivalence check (plan §"SR is streamed, not materialized").
# Runs the new sr_factor knob (slice 128 native -> Lanczos x2 per slice -> 256 -> is 1024) on the
# NATIVE dev crops and compares against the existing whole-crop-SR e2 outputs (same settings as
# run_three_arm.py: ct 0.1, is 1024, ov 0.2, downscale_pred, gpkg). Known non-identities to
# expect: (i) sahi's integer overlap makes the grids differ slightly (stride 103 native streamed
# vs 102.5 native whole-crop); (ii) Lanczos kernel-edge effects at slice borders (4-px support,
# inside the 5% edge zone). So the bar is matched-detection equivalence, not bit-identity.
# Idempotent: skips arms whose output exists.
import json, sys, time
from pathlib import Path

import numpy as np
import geopandas as gpd

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
from YOLOv8BeyondEarth.benchmark import match_detections
from YOLOv8BeyondEarth.fusion import pairwise_box_metric

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data")
DE = BASE / "solutionDE"
WEIGHTS = BASE / "weights" / "yolov8-m-boulder-detection-tmp.pt"

SCENES = [
    ("nac", DE / "nac_crop2k.tif", 0.48357597846618,
     DE / "preds/nac_e2/nac_crop2k_sr2-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg"),
    ("hirise", DE / "hirise_crop4k.tif", 0.5,
     DE / "preds/hirise_e2/hirise_crop4k-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg".replace(
         "hirise_crop4k-", "hirise_crop4k_sr2-")),
]

BANDS = [(3, 4), (4, 6), (6, 8), (8, 10), (10, 14), (14, np.inf)]


def band_label(dpx):
    for lo, hi in BANDS:
        if lo <= dpx < hi:
            return f"{lo}-{hi if np.isfinite(hi) else 'inf'}"
    return "<3"


def stream_gpkg(out_dir, raster):
    return out_dir / (raster.stem + "-predictions-ct-010-ss-256sr2-is-1024-ov-020"
                      "-downscaled-mask-nms.gpkg")


if __name__ == "__main__":
    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth.predict import get_sliced_prediction

    model = None
    timings = {}
    for scene, raster, res, e2_path in SCENES:
        out_dir = DE / "preds" / f"{scene}_e2stream"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = stream_gpkg(out_dir, raster)
        if not out.exists():
            if model is None:
                model = AutoDetectionModel.from_pretrained(
                    model_type="yolov8", model_path=str(WEIGHTS), device="cuda:0",
                    image_size=1024, confidence_threshold=0.1)
            print(f"[run ] {scene} streamed SR x2 on {raster.name}", flush=True)
            t0 = time.time()
            gdf, gdf_nms = get_sliced_prediction(
                raster, detection_model=model, confidence_threshold=0.1,
                output_dir=out_dir, slice_size=256, sr_factor=2, inference_size=1024,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                downscale_pred=True, postprocess=True,
                save_prenms=False, save_bbox=False, output_format="gpkg")
            timings[scene] = dict(wall_s=round(time.time() - t0, 1), n_nms=int(len(gdf_nms)))
            print(f"[done] {scene}: {timings[scene]}", flush=True)
        else:
            print(f"[skip] {scene} (output exists)", flush=True)

    print("\n=== equivalence: whole-crop SR (e2) vs streamed per-slice SR ===")
    for scene, raster, res, e2_path in SCENES:
        e2 = gpd.read_file(e2_path)
        st = gpd.read_file(stream_gpkg(DE / "preds" / f"{scene}_e2stream", raster))
        # CRS grids: e2 was georeferenced off the SR raster, streamed off the native raster —
        # same map grid by construction (prep_de_crops preserved extent); sanity-check anyway
        m = match_detections(e2, st, max_dist=2.0 * res)
        be = e2.geometry.bounds.to_numpy(float)[m.idx_a.values]
        bs = st.geometry.bounds.to_numpy(float)[m.idx_b.values]
        diou = pairwise_box_metric(be, bs, "diou")
        e2["dpx"] = np.sqrt(4 * e2.geometry.area / np.pi) / res
        st["dpx"] = np.sqrt(4 * st.geometry.area / np.pi) / res
        unmatched_e2 = e2.drop(e2.index[m.idx_a.values])
        unmatched_st = st.drop(st.index[m.idx_b.values])
        print(f"\n{scene}: n_e2={len(e2)}  n_stream={len(st)}  matched={len(m)} "
              f"({len(m)/len(e2):.3f} of e2, {len(m)/len(st):.3f} of stream)")
        print(f"  matched-pair DIoU: median={np.median(diou):.3f}  "
              f"p05={np.percentile(diou, 5):.3f}  frac>=0.30: {(diou >= 0.30).mean():.4f}")
        # size agreement on matched pairs (the cross-scale size-consistency audit, same-arm case)
        ratio = st.dpx.to_numpy()[m.idx_b.values] / e2.dpx.to_numpy()[m.idx_a.values]
        print(f"  matched size ratio (stream/e2): median={np.median(ratio):.4f}  "
              f"IQR=[{np.percentile(ratio,25):.4f}, {np.percentile(ratio,75):.4f}]")
        for tag, u in [("e2-only", unmatched_e2), ("stream-only", unmatched_st)]:
            vc = u.dpx.map(band_label).value_counts()
            print(f"  {tag} ({len(u)}): " + ", ".join(f"{b}px: {vc.get(b, 0)}"
                  for b in ["<3", "3-4", "4-6", "6-8", "8-10", "10-14", "14-inf"]))
    if timings:
        print("\ntimings:", json.dumps(timings))
