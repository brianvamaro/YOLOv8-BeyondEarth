# Solution G production benchmark: detector-preserving SAM2 re-masking on the same HiRISE 4096
# artifact crop used for D/E. Produces the three aligned mask sets (yolo/sam256/sam1024) per YOLO
# box, reads angles off all three with one moment estimator, saves raw per-detection table +
# angles to data/solutionG/. Timed for the card's time axis. Card + gating analysis is a separate
# CPU script (analyze_G.py) so the SAM run is not repeated when iterating on the readout.
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data")
G = BASE / "solutionG"
CROP = BASE / "solutionDE" / "hirise_crop4k.tif"           # same artifact scene as D/E
WEIGHTS = BASE / "weights" / "yolov8-m-boulder-detection-tmp.pt"
NATIVE_RES = 0.5

if __name__ == "__main__":
    from ultralytics import YOLO, SAM
    from YOLOv8BeyondEarth.orientation_validation import detect_sam2_mask_sets, angles_from_moments

    model = YOLO(str(WEIGHTS))
    sam = SAM("sam2.1_b.pt")

    with rasterio.open(CROP) as src:
        img = src.read(1)
    print(f"crop {img.shape} dtype={img.dtype}", flush=True)

    t0 = time.time()
    df, masks = detect_sam2_mask_sets(model, sam, img, slice_size=256, imgsz=1024,
                                      conf=0.10, device=0)
    wall = time.time() - t0
    n = len(df)
    print(f"detect_sam2_mask_sets: {n} kept detections in {wall:.1f}s "
          f"({wall / max(n,1) * 1000:.1f} ms/boulder); failed={df.attrs['failed']}", flush=True)

    # angles for each aligned mask set (moment estimator reads all three the same way)
    for side in ("yolo", "sam256", "sam1024"):
        a = angles_from_moments(masks[side])
        df[f"angle_{side}"] = a["angle180"].values
        df[f"aspect_{side}"] = a["aspect"].values
        df[f"dpx_{side}"] = a["diameter_px"].values

    # native size axis from the YOLO (slice-res) area, shared by the aligned triple
    df["diameter_native_m"] = 2.0 * np.sqrt(df["area_yolo"].values / np.pi) * NATIVE_RES
    df["dpx_native"] = df["diameter_native_m"].values / NATIVE_RES

    df.to_csv(G / "G_detections.csv", index=False)
    meta = {"n_detections": int(n), "wall_s": wall, "ms_per_boulder": wall / max(n, 1) * 1000,
            "failed": df.attrs["failed"], "crop": CROP.name, "n_tiles": (img.shape[0] // 256) ** 2}
    (G / "G_run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
