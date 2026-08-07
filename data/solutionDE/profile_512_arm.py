# Goal-3 phase 4: profile the 512 arm (plan §Phase-3 "Wall clock: AT RISK").
# The 512 arm measured ~1.6x the 256 arm on crops instead of the slice-count-predicted +0.25x.
# Hypothesis (plan): per-slice CPU mask->polygon at 512^2/detection — every slice infers at
# imgsz=1024, ultralytics returns masks at 1024^2, and downscale_pred resizes each to
# slice_size, so the 512 arm pays 4x the pixels per detection in resize+count+find_contours.
# This script (a) re-measures wall clocks for {256, 512, SR} x {skimage, cv2} on the same crop
# under identical conditions, (b) attributes the 512-arm time with cProfile on the serial path
# (n_workers=1 — the thread pool hides workers from cProfile). Results -> profile_512.json.
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data")
DE = BASE / "solutionDE"
WEIGHTS = BASE / "weights" / "yolov8-m-boulder-detection-tmp.pt"
CROP = DE / "hirise_crop4k.tif"
OUT = DE / "profile_512.json"
OUT_DIR = DE / "profile_runs"

# (key, slice_size, sr_factor, contour_method, n_workers, profile?)
RUNS = [
    ("ss256_skimage", 256, 1, "skimage", None, False),
    ("ss512_skimage", 512, 1, "skimage", None, False),
    ("ss512_cv2", 512, 1, "cv2", None, False),
    ("ss256sr2_skimage", 256, 2, "skimage", None, False),
    ("ss256sr2_cv2", 256, 2, "cv2", None, False),
    ("ss512_skimage_serial_prof", 512, 1, "skimage", 1, True),
    ("ss256_skimage_serial_prof", 256, 1, "skimage", 1, True),
]

if __name__ == "__main__":
    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth.predict import get_sliced_prediction

    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=str(WEIGHTS), device="cuda:0",
        image_size=1024, confidence_threshold=0.1)

    OUT_DIR.mkdir(exist_ok=True)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for key, ss, sr, cm, nw, prof in RUNS:
        if key in results:
            print(f"[skip] {key}", flush=True)
            continue
        print(f"[run ] {key}", flush=True)
        kwargs = dict(detection_model=model, confidence_threshold=0.1,
                      output_dir=OUT_DIR, slice_size=ss, sr_factor=sr, inference_size=1024,
                      overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                      downscale_pred=True, postprocess=True, contour_method=cm,
                      save_prenms=False, save_bbox=False, output_format="gpkg")
        if nw is not None:
            kwargs["n_workers"] = nw
        # warm-up once per process (CUDA context + torch kernels) so run 1 isn't penalized
        if not results.get("_warmed"):
            get_sliced_prediction(DE / "hirise_rom00.tif", **{**kwargs, "output_dir": OUT_DIR})
            results["_warmed"] = True

        t0 = time.time()
        if prof:
            pr = cProfile.Profile()
            pr.enable()
            gdf, gdf_nms = get_sliced_prediction(CROP, **kwargs)
            pr.disable()
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(45)
            (OUT_DIR / f"{key}.pstats.txt").write_text(s.getvalue())
            pr.dump_stats(OUT_DIR / f"{key}.pstats")
        else:
            gdf, gdf_nms = get_sliced_prediction(CROP, **kwargs)
        wall = time.time() - t0
        results[key] = {"wall_s": round(wall, 2), "slice_size": ss, "sr_factor": sr,
                        "contour_method": cm, "n_workers": nw,
                        "n_raw": int(len(gdf)), "n_nms": int(len(gdf_nms))}
        OUT.write_text(json.dumps(results, indent=2))
        print(f"[done] {key}: {results[key]}", flush=True)

    print(json.dumps(results, indent=2))
