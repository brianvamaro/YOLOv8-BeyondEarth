# Isolate: does the raw model fire on SR-x4 slices (64 native-px footprint) vs SR-x2 (128) vs
# baseline (256)? Bypasses get_sliced_prediction's footprint/within-slice filtering entirely --
# so if counts collapse here too, the f=4 collapse is real context loss, not a pipeline bug.
import sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
from ultralytics import YOLO

DE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionDE")
W = r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\weights\yolov8-m-boulder-detection-tmp.pt"

# Same ground patch (HiRISE crop centre) at each scale; slice px chosen so native footprint =
# 256 (baseline), 128 (f2), 64 (f4). All fed to the model at imgsz 1024 (the trained inference size).
CASES = [
    ("baseline_ss256_native", DE / "hirise_crop4k.tif",     1.0, 256),
    ("e2_ss256_sr2",          DE / "hirise_crop4k_sr2.tif", 2.0, 256),
    ("dctl_ss128_native",     DE / "hirise_crop4k.tif",     1.0, 128),
    ("e4_ss256_sr4",          DE / "hirise_crop4k_sr4.tif", 4.0, 256),
    ("dctl4_ss64_native",     DE / "hirise_crop4k.tif",     1.0,  64),
]

OUT_CSV = DE / "probe_footprint.csv"

if __name__ == "__main__":
    import pandas as pd
    model = YOLO(W)
    _rows = []
    # centre of the native 4096 crop, in native px; convert per scale
    cx_nat, cy_nat = 2048, 2048
    N = 12  # tiles per case, walking a strip so we sample real boulders
    for name, raster, f, ss in CASES:
        with rasterio.open(raster) as src:
            counts = []
            for k in range(N):
                # native strip position -> scaled px; footprint (ss/f) native px per tile
                foot_nat = ss / f
                c0_nat = 1000 + k * int(foot_nat)
                r0_nat = 2048
                c0, r0 = int(c0_nat * f), int(r0_nat * f)
                if c0 + ss > src.width or r0 + ss > src.height:
                    break
                tile = src.read(1, window=Window(c0, r0, ss, ss))
                rgb = np.stack([tile] * 3, axis=-1)
                res = model(rgb, imgsz=1024, verbose=False, conf=0.1, device=0)
                counts.append(len(res[0].boxes))
            print(f"{name:24s} ss={ss:4d} foot_nat={ss/f:5.0f}px  "
                  f"tiles={len(counts)}  dets/tile={np.mean(counts):5.1f}  total={sum(counts)}")
            _rows.append(dict(case=name, slice_px=ss, footprint_native_px=int(ss / f),
                              tiles=len(counts), dets_per_tile=float(np.mean(counts)),
                              total=int(sum(counts))))
    pd.DataFrame(_rows).to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
