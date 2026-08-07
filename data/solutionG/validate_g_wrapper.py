# Validate the G production wrapper: get_sliced_prediction(remask_fn=sam2_remasker(...)) on the
# HiRISE crop, through the FULL production path (overlap, edge handling, NMS, georef, gpkg). Confirm
# it (a) writes a gpkg and (b) reproduces the dissolution (elongated diag-frac ~0.20 vs YOLO ~0.40)
# now that re-masking runs through the real pipeline, not the no-overlap detect_sam2 harness.
import sys, time, json
from pathlib import Path
import numpy as np
import geopandas as gpd

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data")
CROP = BASE / "solutionDE" / "hirise_crop4k.tif"
WEIGHTS = BASE / "weights" / "yolov8-m-boulder-detection-tmp.pt"
OUT = BASE / "solutionG" / "prod_wrapper"
RES = 0.5

if __name__ == "__main__":
    from sahi import AutoDetectionModel
    from ultralytics import SAM
    from YOLOv8BeyondEarth.predict import get_sliced_prediction
    from YOLOv8BeyondEarth.orientation_validation import sam2_remasker
    from YOLOv8BeyondEarth.orientation import attach_orientations
    from YOLOv8BeyondEarth.benchmark import orientation_by_band
    OUT.mkdir(parents=True, exist_ok=True)

    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=str(WEIGHTS), device="cuda:0",
        image_size=1024, confidence_threshold=0.1)
    sam = SAM("sam2.1_b.pt")
    common = dict(confidence_threshold=0.1, slice_size=256, inference_size=1024,
                  overlap_height_ratio=0.2, overlap_width_ratio=0.2, downscale_pred=True,
                  postprocess=True, save_prenms=False, save_bbox=False, output_format="gpkg")

    # plain YOLO through the same function (baseline)
    t0 = time.time()
    _, yolo_nms = get_sliced_prediction(CROP, detection_model=model, output_dir=OUT, **common)
    t_yolo = time.time() - t0

    # G: same call + SAM re-masker
    remask = sam2_remasker(sam, sam_batch=96, device=0, empty_cache_every=64)
    t0 = time.time()
    _, sam_nms = get_sliced_prediction(CROP, detection_model=model, output_dir=OUT,
                                       remask_fn=remask, mask_tag="sam2", **common)
    t_sam = time.time() - t0

    def diagfrac(gdf):
        g = attach_orientations(gdf, 0.25)
        g["dpx"] = g.diameter_m / RES
        return orientation_by_band(g.angle180, g.aspect_ra, g.dpx, grid_axis="diagonal")

    yt, st = diagfrac(yolo_nms), diagfrac(sam_nms)
    res = dict(
        n_yolo=int(len(yolo_nms)), n_sam=int(len(sam_nms)),
        yolo_diag_all=float(yt.iloc[0]["frac"]), sam_diag_all=float(st.iloc[0]["frac"]),
        yolo_chir_all=float(yt.iloc[0]["chirality"]), sam_chir_all=float(st.iloc[0]["chirality"]),
        wall_yolo_s=t_yolo, wall_sam_s=t_sam)
    (OUT / "g_wrapper_validation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("\nYOLO by band:\n", yt.to_string(index=False))
    print("\nSAM by band:\n", st.to_string(index=False))
    print("\ngpkg files:", [p.name for p in sorted(OUT.glob("*.gpkg"))])
