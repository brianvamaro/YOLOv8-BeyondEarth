"""H-seg orientation via its MASKS (same ellipse readout as YOLO-seg). Does YOLO26-seg's better
prototype module (mask mAP50 0.467->0.569, same-protocol val) reduce the diagonal lock vs stock
YOLOv8m-seg (0.428)? Also H-seg's §4c-null (uniform-truth field). Caches CSVs to data/solutionH/."""
import sys, numpy as np, pandas as pd, rasterio
sys.path.insert(0, "YOLOv8-BeyondEarth/src")
from ultralytics import YOLO
from YOLOv8BeyondEarth.orientation_validation import detect_orientations
from YOLOv8BeyondEarth.transforms_test import render_boulder_field
from YOLOv8BeyondEarth.orientation import diagonal_fraction, grid_fraction, asymmetry_ratio, grid_fraction_uniform

OUT = "YOLOv8-BeyondEarth/data/solutionH"
CROP = "YOLOv8-BeyondEarth/data/solutionDE/hirise_crop4k.tif"
UNI = grid_fraction_uniform()
img = rasterio.open(CROP).read(1)

models = {"YOLO-seg (stock)": "YOLOv8-BeyondEarth/data/weights/yolov8-m-boulder-detection-tmp.pt",
          "H-seg (yolo26m)": "YOLOv8-BeyondEarth/data/weights/yolo26m-h-seg.pt"}

def line(tag, df):
    el = df[df.aspect > 1.35]
    return (f"{tag:22} n={len(df):>6} elong={len(el):>5}  diag={diagonal_fraction(el.angle180):.3f}  "
            f"grid={grid_fraction(el.angle180):.3f}  chir={asymmetry_ratio(el.angle180):.2f}")

print(f"uniform={UNI:.3f}\n--- real crop (mask->ellipse readout, same estimator) ---")
for tag, w in models.items():
    df = detect_orientations(YOLO(w, task="segment"), img, slice_size=256, imgsz=1024, conf=0.10)
    df.to_csv(f"{OUT}/seg_crop_{'stock' if 'stock' in tag else 'h'}.csv", index=False)
    print(line(tag, df))

print("\n--- H-seg §4c-null (uniform-truth field) ---")
fimg, truth = render_boulder_field(size=2048, n=1600, diam_px=(10, 26), aspect=(1.6, 3.0), sun_az_deg=255.0, seed=1)
dfn = detect_orientations(YOLO(models["H-seg (yolo26m)"], task="segment"), fimg, slice_size=256, imgsz=1024, conf=0.10)
dfn.to_csv(f"{OUT}/seg_null_h.csv", index=False)
print(line("truth (uniform)", truth.assign(aspect=2.0)))
print(line("H-seg OUT on truth", dfn))
