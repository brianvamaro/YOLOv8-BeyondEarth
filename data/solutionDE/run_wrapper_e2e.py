# Goal-3 phase 3: one end-to-end pass of the production wrapper (get_multiscale_prediction)
# on the NAC dev crop, checked against fusing the same arms loaded from disk. Validates the
# file-writing path + that the wrapper's in-memory fusion equals the from-disk fusion the
# analysis scripts use (column-name round-trip, CRS, provenance). Cheap (~1 min GPU).
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
from YOLOv8BeyondEarth.fusion import fuse_predictions, matched_recall

DE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionDE")
WEIGHTS = DE.parent / "weights" / "yolov8-m-boulder-detection-tmp.pt"
OUT = DE / "preds" / "wrapper_e2e"

ARMS_DISK = {
    512: DE / "preds/nac_512/nac_crop2k-predictions-ct-010-ss-512-is-1024-ov-020-downscaled-mask-nms.gpkg",
    256: DE / "preds/nac_baseline/nac_crop2k-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg",
    (256, 2): DE / "preds/nac_e2stream/nac_crop2k-predictions-ct-010-ss-256sr2-is-1024-ov-020-downscaled-mask-nms.gpkg",
}

if __name__ == "__main__":
    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth.predict import get_multiscale_prediction

    OUT.mkdir(parents=True, exist_ok=True)
    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=str(WEIGHTS), device="cuda:0",
        image_size=1024, confidence_threshold=0.1)
    fused_w, audit_w, per_scale = get_multiscale_prediction(
        DE / "nac_crop2k.tif", detection_model=model, output_dir=OUT,
        confidence_threshold=0.1, inference_size=1024,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2, downscale_pred=True)

    gdfs = {k: gpd.read_file(p) for k, p in ARMS_DISK.items()}
    fused_d, audit_d = fuse_predictions(gdfs)

    print(f"wrapper per-scale ns: " +
          ", ".join(f"{k}: {len(g)}" for k, g in per_scale.items()))
    print(f"disk    per-scale ns: " +
          ", ".join(f"{k}: {len(g)}" for k, g in gdfs.items()))
    print(f"fused: wrapper {len(fused_w)} vs disk {len(fused_d)} | "
          f"audit {len(audit_w)} vs {len(audit_d)}")
    rec_wd, _ = matched_recall(fused_w, fused_d)
    rec_dw, _ = matched_recall(fused_d, fused_w)
    print(f"mutual matched recall wrapper<->disk: {rec_wd:.4f} / {rec_dw:.4f}")
    out_file = OUT / "nac_crop2k-predictions-fused-scale_aware-512-256-256sr2-mask.gpkg"
    print(f"fused layer on disk: {out_file.exists()} ({out_file.name})")
    same_comp = (fused_w.scale_key.value_counts().sort_index()
                 .equals(fused_d.scale_key.value_counts().sort_index()))
    print(f"composition identical: {same_comp}")
