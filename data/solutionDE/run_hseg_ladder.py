# H-seg adoption check, GPU half (plan: "H-seg adoption: decision point = end of phase 3";
# the phase-3 crop suite is the acceptance harness for a weight swap). Re-runs the FULL ladder
# under the H-seg weights (yolo26m-h-seg.pt) on both dev crops + the 20 ROM tiles — including
# the 256 arm (the parity baseline resets with the weights). Outputs under preds/hseg_* /
# preds/hseg_rom_ladder mirror the production layout; analysis = analyze_hseg_acceptance.py.
# Context: H-seg = mask mAP50 0.467->0.569 but box mAP 0.762->0.749 (slightly down) — under a
# phase-5 G-remask future SAM2 is prompted from the BOXES, so the acceptance axes are recall +
# box localization, not the mask headline. Idempotent; timings to hseg_timings.json.
import json, sys, time
from pathlib import Path

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data")
DE = BASE / "solutionDE"
WEIGHTS = BASE / "weights" / "yolo26m-h-seg.pt"
TIMINGS = DE / "hseg_timings.json"

ARMS = []
for scene, raster in [("nac", DE / "nac_crop2k.tif"), ("hirise", DE / "hirise_crop4k.tif")]:
    for ss, sr in ((512, 1), (256, 1), (256, 2)):
        ARMS.append((f"hseg_{scene}", raster, ss, sr))
for tile in sorted(DE.glob("hirise_rom??.tif")):
    for ss, sr in ((512, 1), (256, 1), (256, 2)):
        ARMS.append((f"hseg_rom_ladder/{tile.stem}", tile, ss, sr))


def nms_gpkg(out_dir, raster, ss, sr):
    tag = f"{ss}sr{sr}" if sr > 1 else f"{ss}"
    return out_dir / (raster.stem + f"-predictions-ct-010-ss-{tag}-is-1024-ov-020"
                      "-downscaled-mask-nms.gpkg")


if __name__ == "__main__":
    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth.predict import get_sliced_prediction

    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=str(WEIGHTS), device="cuda:0",
        image_size=1024, confidence_threshold=0.1)

    timings = json.loads(TIMINGS.read_text()) if TIMINGS.exists() else {}
    for sub, raster, ss, sr in ARMS:
        key = f"{sub.replace('/', '_')}_ss{ss}" + (f"sr{sr}" if sr > 1 else "")
        out_dir = DE / "preds" / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        if nms_gpkg(out_dir, raster, ss, sr).exists():
            print(f"[skip] {key}", flush=True)
            continue
        print(f"[run ] {key}: {raster.name} ss={ss} sr={sr}", flush=True)
        t0 = time.time()
        gdf, gdf_nms = get_sliced_prediction(
            raster, detection_model=model, confidence_threshold=0.1,
            output_dir=out_dir, slice_size=ss, sr_factor=sr, inference_size=1024,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
            downscale_pred=True, postprocess=True,
            save_prenms=False, save_bbox=False, output_format="gpkg")
        timings[key] = {"wall_s": round(time.time() - t0, 2), "slice_size": ss, "sr_factor": sr,
                        "raster": raster.name, "n_raw": int(len(gdf)), "n_nms": int(len(gdf_nms))}
        TIMINGS.write_text(json.dumps(timings, indent=2))
        print(f"[done] {key}: {timings[key]}", flush=True)

    print(json.dumps(timings, indent=2))
