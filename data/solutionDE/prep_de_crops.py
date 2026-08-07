# Prep for the shared D/E three-arm crop run (docs/solutions/D_multiscale_fusion.md /
# E_super_resolution.md Stage 2): fixed crops + Lanczos x2 SR versions, written as GeoTIFFs
# under YOLOv8-BeyondEarth/data/solutionDE/ (gitignored via data/).
#
# Arms (f=2): baseline = native crop ss256 | E-arm = SR x2 crop ss256 (SR px) | D-control =
# native crop ss128. SR transform: pixel size halved, same CRS, same origin.
import json
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from rasterio import Affine
from pathlib import Path

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data")
OUT = BASE / "solutionDE"
OUT.mkdir(exist_ok=True)

NAC_CROP = BASE / "raster" / "crop.tif"           # existing 2048^2 dev crop, 0.484 m/px
HIRISE = BASE / "mars_hirise" / "raster" / "ESP_017355_2260_RED.tif"  # 22490x85922, 0.5 m/px


def write_window(src_path, out_path, col_off, row_off, size):
    with rasterio.open(src_path) as src:
        w = Window(col_off, row_off, size, size)
        arr = src.read(1, window=w)
        prof = src.profile.copy()
        prof.update(width=size, height=size, count=1,
                    transform=src.window_transform(w))
        prof.pop("blockxsize", None); prof.pop("blockysize", None)
        prof["tiled"] = False
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(arr, 1)
    return arr


def write_sr2(src_path, out_path):
    with rasterio.open(src_path) as src:
        arr = src.read(1)
        prof = src.profile.copy()
        sr = cv2.resize(arr, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        prof.update(width=sr.shape[1], height=sr.shape[0],
                    transform=src.transform * Affine.scale(0.5))
        prof.pop("blockxsize", None); prof.pop("blockysize", None)
        prof["tiled"] = False
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(sr, 1)
    return sr


report = {}

# --- NAC: copy the existing dev crop into solutionDE (single canonical location) + SR x2 ---
nac_out = OUT / "nac_crop2k.tif"
with rasterio.open(NAC_CROP) as src:
    arr = src.read(1)
    report["nac"] = {"shape": list(arr.shape), "nonzero_frac": float((arr > 0).mean()),
                     "res": src.res[0], "crs": str(src.crs)}
    prof = src.profile.copy()
    with rasterio.open(nac_out, "w", **prof) as dst:
        dst.write(arr, 1)
sr = write_sr2(nac_out, OUT / "nac_crop2k_sr2.tif")
report["nac_sr2"] = {"shape": list(sr.shape)}

# --- HiRISE: fixed 4096^2 window at the scene centre (footprint is valid there) ---
size = 4096
with rasterio.open(HIRISE) as src:
    W, H = src.width, src.height
col0, row0 = (W - size) // 2, (H - size) // 2
arr = write_window(HIRISE, OUT / "hirise_crop4k.tif", col0, row0, size)
report["hirise"] = {"window": [col0, row0, size], "shape": list(arr.shape),
                    "nonzero_frac": float((arr > 0).mean())}
sr = write_sr2(OUT / "hirise_crop4k.tif", OUT / "hirise_crop4k_sr2.tif")
report["hirise_sr2"] = {"shape": list(sr.shape)}

with rasterio.open(OUT / "hirise_crop4k_sr2.tif") as s:
    report["hirise_sr2"]["res"] = s.res[0]

print(json.dumps(report, indent=2))
