"""Test 2 — repeat pairs: download RDR JP2 -> 8-bit tif (x255/1023) -> infer (matched config) ->
EllipseModel orientations -> cache CSV. Sequential, idempotent, robust per-scene. 10 scenes."""
import time, urllib.request, traceback
import numpy as np, rasterio, geopandas as gpd, pandas as pd
from rasterio.windows import Window
from pathlib import Path
from sahi import AutoDetectionModel
from YOLOv8BeyondEarth.predict import get_sliced_prediction
from YOLOv8BeyondEarth.orientation import boulder_orientations, grid_fraction, grid_fraction_uniform

ROOT = Path(r"c:/Users/brian/Documents/PhD/BoulderNet/YOLOv8-BeyondEarth")
weights = ROOT / "data/weights/yolov8-m-boulder-detection-tmp.pt"
OBSIDS = ["ESP_071699_2260", "ESP_071923_2260", "ESP_076235_2260", "ESP_055833_2260",
          "ESP_069088_2270", "ESP_067214_2270", "ESP_059499_2270", "ESP_055714_2270",
          "PSP_007703_2265", "PSP_007492_2265"]

def pds_url(obsid):
    phase, orbit = obsid.split("_")[0], int(obsid.split("_")[1])
    lo = (orbit // 100) * 100
    return f"https://hirise-pds.lpl.arizona.edu/PDS/RDR/{phase}/ORB_{lo:06d}_{lo+99:06d}/{obsid}/{obsid}_RED.JP2"

def cdist(a, c): a = np.asarray(a) % 180.0; return np.abs(((a - c + 90) % 180) - 90)

print("loading model...", flush=True)
model = AutoDetectionModel.from_pretrained(model_type="yolov8", model_path=str(weights),
                                           device="cuda:0", image_size=1024, confidence_threshold=0.1)
for obsid in OBSIDS:
    try:
        d = ROOT / f"data/test2_pairs/{obsid}"; d.mkdir(parents=True, exist_ok=True)
        jp2, tif = d / f"{obsid}_RED.JP2", d / f"{obsid}_RED.tif"
        gpkg = d / f"{obsid}_RED-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg"
        csv = d / f"{obsid}_orientations.csv"
        url = pds_url(obsid)
        print(f"\n===== {obsid}  ({url}) =====", flush=True)
        if not jp2.exists():
            print("  downloading...", flush=True); t0 = time.time()
            urllib.request.urlretrieve(url, jp2)
            print(f"  {jp2.stat().st_size/1e6:.0f} MB in {(time.time()-t0)/60:.1f} min", flush=True)
        if not tif.exists():
            print("  converting 10->8 bit...", flush=True); t0 = time.time()
            with rasterio.open(jp2) as src:
                prof = src.profile.copy(); prof.update(driver="GTiff", dtype="uint8", count=1,
                                                       compress="lzw", tiled=True, blockxsize=512, blockysize=512)
                with rasterio.open(tif, "w", **prof) as dst:
                    for r0 in range(0, src.height, 2048):
                        h = min(2048, src.height - r0); win = Window(0, r0, src.width, h)
                        a = src.read(1, window=win).astype(np.float32)
                        dst.write(np.clip(np.round(a * (255.0/1023.0)), 0, 255).astype(np.uint8), 1, window=win)
            print(f"  converted in {(time.time()-t0)/60:.1f} min", flush=True)
        if not gpkg.exists():
            print("  inference...", flush=True); t0 = time.time()
            gdf, gdf_nms = get_sliced_prediction(str(tif), detection_model=model, output_dir=str(d),
                                                 slice_size=256, inference_size=1024, downscale_pred=True,
                                                 min_area_threshold=6, postprocess_match_threshold=0.2,
                                                 contour_method="skimage", n_workers=8,
                                                 save_bbox=False, save_prenms=False, output_format="gpkg")
            print(f"  inference {(time.time()-t0)/60:.1f} min; post-NMS={len(gdf_nms):,}", flush=True)
        if not csv.exists():
            print("  orientations...", flush=True); t0 = time.time()
            res = rasterio.open(tif).res[0]
            df = boulder_orientations(gpd.read_file(gpkg), res=res, n_workers=8)
            df.to_csv(csv, index=False)
            print(f"  oriented {(time.time()-t0)/60:.1f} min; res={res:.3f}", flush=True)
        else:
            df = pd.read_csv(csv)
        el = df[df.aspect_ra > 1.35]; a = el.angle180.values % 180.0
        h, e = np.histogram(a, bins=np.arange(0, 181, 10)); pk = (e[:-1]+e[1:])[np.argmax(h)]/2
        r135, r45 = np.mean(cdist(a, 135) <= 15), np.mean(cdist(a, 45) <= 15)
        print(f"  SUMMARY {obsid}: n_elong={len(a):,} peak~{pk:.0f} grid_fraction={grid_fraction(el.angle180,10):.3f} "
              f"(unif {grid_fraction_uniform(10):.3f}) ratio135/45={r135/r45 if r45 else float('nan'):.2f}", flush=True)
    except Exception:
        print(f"  FAILED {obsid}:\n{traceback.format_exc()}", flush=True)
print("\nBATCH DONE", flush=True)
