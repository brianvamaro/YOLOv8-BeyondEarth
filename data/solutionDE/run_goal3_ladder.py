# Goal-3 phase 3: fill in the missing production-ladder arms (plan §Phases 3).
# The ladder is {512 native, 256 native, SR x2 streamed @ 128-native footprint}. Already on disk
# from phases 1-2 (same settings: ct 0.1, is 1024, ov 0.2, downscale_pred, gpkg):
#   preds/{nac,hirise}_baseline (256) and preds/{nac,hirise}_e2stream (256sr2).
# This run adds: the 512 arm on both dev crops, and all three arms on the 20 hirise_rom tiles
# (recall-vs-human item; see prep_rom_crops.py). Idempotent; per-arm wall clocks appended to
# goal3_timings.json (the benchmark-card time axis).
import json, sys, time
from pathlib import Path

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data")
DE = BASE / "solutionDE"
WEIGHTS = BASE / "weights" / "yolov8-m-boulder-detection-tmp.pt"
TIMINGS = DE / "goal3_timings.json"

ARMS = [("nac_512", DE / "nac_crop2k.tif", 512, 1),
        ("hirise_512", DE / "hirise_crop4k.tif", 512, 1)]
for tile in sorted(DE.glob("hirise_rom??.tif")):
    for ss, sr in ((512, 1), (256, 1), (256, 2)):
        ARMS.append((f"rom_ladder/{tile.stem}", tile, ss, sr))


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
        key = f"{sub.split('/')[-1] if '/' in sub else sub}_ss{ss}" + (f"sr{sr}" if sr > 1 else "")
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
