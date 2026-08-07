# Build Lanczos x4 SR versions of the two crops for the f=4 arm of the three-arm run.
import json
import cv2
import rasterio
from rasterio import Affine
from pathlib import Path

DE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionDE")


def write_sr(src_path, out_path, factor):
    with rasterio.open(src_path) as src:
        arr = src.read(1)
        prof = src.profile.copy()
        sr = cv2.resize(arr, None, fx=factor, fy=factor, interpolation=cv2.INTER_LANCZOS4)
        prof.update(width=sr.shape[1], height=sr.shape[0],
                    transform=src.transform * Affine.scale(1.0 / factor))
        prof.pop("blockxsize", None); prof.pop("blockysize", None)
        prof["tiled"] = False
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(sr, 1)
    return sr.shape


rep = {}
rep["nac_sr4"] = write_sr(DE / "nac_crop2k.tif", DE / "nac_crop2k_sr4.tif", 4)
rep["hirise_sr4"] = write_sr(DE / "hirise_crop4k.tif", DE / "hirise_crop4k_sr4.tif", 4)
print(json.dumps({k: list(v) for k, v in rep.items()}, indent=2))
