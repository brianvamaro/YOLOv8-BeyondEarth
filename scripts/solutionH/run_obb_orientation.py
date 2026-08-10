"""Cache the H-obb orientation readouts + example tiles for the SolutionH notebook.

Produces (in data/solutionH/, gitignored), with an optional filename SUFFIX:
  obb_crop_angles{SUF}.csv - H-obb angles on the real HiRISE crop (slice 256, full)
  obb_null_pred{SUF}.csv   - H-obb angles on a uniform-truth synthetic field (the §4c-null OUTPUT)
  obb_null_truth{SUF}.csv  - the rendered TRUTH angles of that field (uniform by construction)
  obb_examples{SUF}.npz    - a few example tiles (image + predicted OBB corners; + truth for synthetic)

Heavy GPU work lives here; the notebook loads these and only renders. Run from the repo root.
CLI: [weights] [suffix]  — default = the degrees=0 baseline (original cache, no suffix). Pass the
rotation-aug weight + "_rot" to cache the R-A arm ALONGSIDE the baseline (both kept):
  # baseline (degrees=0):
  conda run -n bouldernet python YOLOv8-BeyondEarth/scripts/solutionH/run_obb_orientation.py
  # R-A (degrees=180) arm:
  conda run -n bouldernet python YOLOv8-BeyondEarth/scripts/solutionH/run_obb_orientation.py \
      YOLOv8-BeyondEarth/data/weights/yolo26m-h-obb-rot.pt _rot
"""
import json, sys, numpy as np, rasterio
sys.path.insert(0, "YOLOv8-BeyondEarth/src")
from ultralytics import YOLO
from YOLOv8BeyondEarth.orientation_validation import detect_obb_angles
from YOLOv8BeyondEarth.transforms_test import render_boulder_field

OUT = "YOLOv8-BeyondEarth/data/solutionH"
CROP = "YOLOv8-BeyondEarth/data/solutionDE/hirise_crop4k.tif"
WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else "YOLOv8-BeyondEarth/data/weights/yolo26m-h-obb.pt"
SUF = sys.argv[2] if len(sys.argv) > 2 else ""
m = YOLO(WEIGHTS, task="obb")


def predict_tile(img):
    """Run H-obb on one square tile (native px, upscaled to 1024) -> OBB corners (n,4,2) col,row."""
    rgb = np.stack([img] * 3, axis=-1)
    r = m.predict(rgb, imgsz=1024, conf=0.10, verbose=False)[0]
    if r.obb is None or len(r.obb) == 0:
        return np.empty((0, 4, 2))
    return r.obb.xyxyxyxy.cpu().numpy()


# 1) real crop, full readout (slice 256) ------------------------------------------------------
img = rasterio.open(CROP).read(1)
crop_df = detect_obb_angles(m, img, slice_size=256, imgsz=1024, conf=0.10)
crop_df.to_csv(f"{OUT}/obb_crop_angles{SUF}.csv", index=False)

# 2) uniform-truth null: render boulders at KNOWN uniform angles, run H-obb -------------------
fimg, truth = render_boulder_field(size=2048, n=1600, diam_px=(10, 26), aspect=(1.6, 3.0),
                                   sun_az_deg=255.0, seed=1)
truth.to_csv(f"{OUT}/obb_null_truth{SUF}.csv", index=False)
null_df = detect_obb_angles(m, fimg, slice_size=256, imgsz=1024, conf=0.10)
null_df.to_csv(f"{OUT}/obb_null_pred{SUF}.csv", index=False)

# 2b) provenance: `edge_dropped` lives in df.attrs and does NOT survive to_csv, so record it.
# Without it a drop_edge run is indistinguishable from a plain one on disk, and the whole point
# of the seam experiment is how much the filter removed.
prov = {"weights": WEIGHTS, "suffix": SUF, "drop_edge": True,
        "crop": {"kept": int(len(crop_df)), "edge_dropped": int(crop_df.attrs.get("edge_dropped", -1))},
        "null": {"kept": int(len(null_df)), "edge_dropped": int(null_df.attrs.get("edge_dropped", -1))}}
with open(f"{OUT}/obb_readout_provenance{SUF}.json", "w") as fh:
    json.dump(prov, fh, indent=2)
print("provenance:", json.dumps(prov))

# 3) example tiles for overlays --------------------------------------------------------------
ex = {}
# a couple of real 512-crops that contain boulders
picks, k = [(1200, 1500), (2600, 900)], 0
for (y0, x0) in picks:
    tile = img[y0:y0 + 512, x0:x0 + 512]
    ex[f"real_img_{k}"] = tile
    ex[f"real_corners_{k}"] = predict_tile(tile)
    k += 1
# a synthetic 512 tile with KNOWN truth (so we can draw true orientation vs predicted box)
sfimg, struth = render_boulder_field(size=512, n=28, diam_px=(14, 30), aspect=(1.8, 3.0),
                                     sun_az_deg=255.0, seed=7)
ex["synth_img"] = sfimg
ex["synth_corners"] = predict_tile(sfimg)
ex["synth_truth"] = struth[["cx", "cy", "a_px", "b_px", "angle180"]].to_numpy()
np.savez_compressed(f"{OUT}/obb_examples{SUF}.npz", **ex)
print(f"wrote obb_crop_angles{SUF}.csv, obb_null_{{pred,truth}}{SUF}.csv, obb_examples{SUF}.npz to", OUT)
