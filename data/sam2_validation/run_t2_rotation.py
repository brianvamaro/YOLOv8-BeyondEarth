# T2 — rotation invariance with SAM2 in the loop (docs/sam2_orientation_validation_plan.md).
# Reads dense crops from the two full scenes (HiRISE ESP_017355_2260 = Mars, NAC M139694087LE =
# Moon), rotates each by alpha in {0..90}, runs the production YOLO-detect + SAM2 re-mask on the
# identical rotated pixels, and reads yolo + sam256 angles with the same moment estimator. YOLO is
# the built-in positive control (a known grid-locker) on the same resampled inputs, so the
# rotation-kernel confound cancels in the YOLO-vs-SAM comparison. Caches a long-format per-alpha
# table so the notebook computes rotation_slope + the raw/geo grid-fraction split without GPU.
#
# __main__ guard required (Windows spawn dataloader workers re-import this file). Run from inside
# YOLOv8-BeyondEarth/ so SAM("sam2.1_b.pt") resolves (absolute path used below regardless).
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth")
sys.path.insert(0, str(BASE / "src"))

OUT = BASE / "data" / "sam2_validation"
WEIGHTS = BASE / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"
SAM_WEIGHTS = BASE / "sam2.1_b.pt"
HIRISE = BASE / "data" / "mars_hirise" / "raster" / "ESP_017355_2260_RED.tif"
NAC = BASE / "data" / "raster" / "M139694087LE.tif"

# crop is a multiple of 256 so detect_sam2_mask_sets' 256-slicing covers it fully. HiRISE is very
# boulder-dense (the artifact scene ~2.4e-3 det/px^2 -> ~1400 dets per 768 crop), so modest crops
# keep the SAM cost bounded while still pooling thousands across crops; NAC is sparser -> more crops.
SCENES = (("hirise_mars", HIRISE), ("nac_moon", NAC))
ALPHAS = [0, 15, 30, 45, 60, 75, 90]     # rotation_slope excludes the 0/90 endpoints from the fit
CROP = 768
N_CROPS = 6
SEED = 2

if __name__ == "__main__":
    from ultralytics import YOLO, SAM
    from YOLOv8BeyondEarth.orientation_validation import sam2_rotation_sweep
    from YOLOv8BeyondEarth.transforms_test import read_dense_crops

    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    sam = SAM(str(SAM_WEIGHTS))

    t0 = time.time()
    frames = []
    for scene, raster in SCENES:
        print(f"== scene {scene}: reading {N_CROPS} dense {CROP}px crops ==", flush=True)
        crops = read_dense_crops(str(raster), crop=CROP, n=N_CROPS, min_valid=0.99, seed=SEED)
        print(f"   got {len(crops)} crops; sweeping alphas {ALPHAS}", flush=True)
        res = sam2_rotation_sweep(model, sam, crops, ALPHAS, device=0,
                                  sides=("yolo", "sam256"))
        realigned = res["sam256"].attrs.get("realigned_chunks", 0)
        print(f"   [{scene}] SAM count-mismatch fallback fired on {realigned} chunk(s)", flush=True)
        for side, df in res.items():
            df.insert(0, "side", side)
            df.insert(0, "scene", scene)
            frames.append(df)
            print(f"   [{scene}/{side}] done: {df.n.sum()} total dets across alphas", flush=True)

    alldf = pd.concat(frames, ignore_index=True)
    alldf.to_csv(OUT / "t2_rotation.csv", index=False)

    wall = time.time() - t0
    meta = {"wall_s": wall, "alphas": ALPHAS, "crop": CROP, "n_crops": N_CROPS,
            "scenes": [s for s, _ in SCENES], "sides": ["yolo", "sam256"]}
    (OUT / "t2_run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print(f"wall {wall:.0f}s", flush=True)
