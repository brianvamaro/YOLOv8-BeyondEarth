# Shared D/E three-arm crop run (E Stage 2 step 2 / D validation run 1), f=2.
# Arms per scene: baseline = native ss256 | E2 = Lanczos x2 image, ss256 (SR px, 128 native
# footprint, trained 4x internal resize) | Dctl = native ss128 (same footprint, 8x resize).
# Uniform settings everywhere: ct 0.1, is 1024, ov 0.2, downscale_pred, NMS iou 0.5 default,
# min_area_threshold None, gpkg, nms-mask layer only. Idempotent: arms with existing output skip.
import json, sys, time
from pathlib import Path

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data")
DE = BASE / "solutionDE"
WEIGHTS = BASE / "weights" / "yolov8-m-boulder-detection-tmp.pt"
TIMINGS = DE / "three_arm_timings.json"

ARMS = [
    # (scene, arm, raster, slice_size)
    ("nac", "baseline", DE / "nac_crop2k.tif", 256),
    ("nac", "e2",       DE / "nac_crop2k_sr2.tif", 256),
    ("nac", "dctl",     DE / "nac_crop2k.tif", 128),
    ("hirise", "baseline", DE / "hirise_crop4k.tif", 256),
    ("hirise", "e2",       DE / "hirise_crop4k_sr2.tif", 256),
    ("hirise", "dctl",     DE / "hirise_crop4k.tif", 128),
]


def nms_gpkg(output_dir, raster, ss):
    return output_dir / (raster.stem + f"-predictions-ct-010-ss-{ss}-is-1024-ov-020"
                         "-downscaled-mask-nms.gpkg")


if __name__ == "__main__":
    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth.predict import get_sliced_prediction

    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=str(WEIGHTS), device="cuda:0",
        image_size=1024, confidence_threshold=0.1)

    timings = json.loads(TIMINGS.read_text()) if TIMINGS.exists() else {}
    for scene, arm, raster, ss in ARMS:
        key = f"{scene}_{arm}"
        out_dir = DE / "preds" / key
        out_dir.mkdir(parents=True, exist_ok=True)
        if nms_gpkg(out_dir, raster, ss).exists():
            print(f"[skip] {key} (output exists)", flush=True)
            continue
        print(f"[run ] {key}: {raster.name} ss={ss}", flush=True)
        t0 = time.time()
        gdf, gdf_nms = get_sliced_prediction(
            raster, detection_model=model, confidence_threshold=0.1,
            output_dir=out_dir, slice_size=ss, inference_size=1024,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
            downscale_pred=True, postprocess=True,
            save_prenms=False, save_bbox=False, output_format="gpkg")
        timings[key] = {"wall_s": time.time() - t0, "slice_size": ss,
                        "raster": raster.name, "n_raw": int(len(gdf)),
                        "n_nms": int(len(gdf_nms))}
        TIMINGS.write_text(json.dumps(timings, indent=2))
        print(f"[done] {key}: {timings[key]}", flush=True)

    print(json.dumps(timings, indent=2))
