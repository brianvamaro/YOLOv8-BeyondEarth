# Goal-3 phase 3: the deferred §4c content-free-field fusion null (plan §Validation 1, the
# standing selection-bias gate; deferred from phase 1 because the test3/4 caches hold no
# polygons). Fields use the Test-4 §4c MARGINAL recipe (angles=None -> uniformly random true
# orientations; near-round low-SNR blobs, the regime where the pipeline chirality lives), sized
# to the ladder: 2048^2 at the same boulder density as the original 1024^2/n=400. Each field is
# written as a GeoTIFF (0.5 m/px, HiRISE CRS) and run through the full production ladder
# {512, 256, SR x2 streamed}; per-scale outputs are cached under preds/null_seed*/.
# Gate: for every fusion rule, the fused rose must MATCH its inputs' — anisotropy metrics
# (cardinal grid-frac, diagonal-frac, 135/45) within the input arms' envelope (with bootstrap
# CIs), no new peak. Composition shift toward the finest arm is expected and allowed; a fused
# metric OUTSIDE the input envelope is a gate failure.
import json, sys, time
from pathlib import Path

import numpy as np
import rasterio
from rasterio import Affine

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data")
DE = BASE / "solutionDE"
WEIGHTS = BASE / "weights" / "yolov8-m-boulder-detection-tmp.pt"
RES = 0.5
SEEDS = [200, 201, 202, 203]
SIZE, N = 2048, 1600                      # same density as §4c's 1024^2 / n=400
SCALES = [512, 256, (256, 2)]


def field_tif(seed):
    return DE / f"null_field_seed{seed}.tif"


def nms_gpkg(seed, ss, sr):
    tag = f"{ss}sr{sr}" if sr > 1 else f"{ss}"
    return (DE / "preds" / f"null_seed{seed}" /
            (field_tif(seed).stem + f"-predictions-ct-010-ss-{tag}-is-1024-ov-020"
             "-downscaled-mask-nms.gpkg"))


def render_fields():
    from YOLOv8BeyondEarth.transforms_test import render_boulder_field
    with rasterio.open(DE / "hirise_crop4k.tif") as src:
        crs = src.crs
    for seed in SEEDS:
        out = field_tif(seed)
        if out.exists():
            continue
        img, truth = render_boulder_field(size=SIZE, n=N, angles=None, diam_px=(4, 8),
                                          aspect=(1.0, 1.12), cap=16, noise=12.0,
                                          shadow_frac=0.0, seed=seed)
        transform = Affine(RES, 0, 0, 0, -RES, 0)   # arbitrary origin, 0.5 m/px
        with rasterio.open(out, "w", driver="GTiff", width=SIZE, height=SIZE, count=1,
                           dtype="uint8", crs=crs, transform=transform) as dst:
            dst.write(img.astype(np.uint8), 1)
        truth.to_csv(out.with_suffix(".truth.csv"), index=False)
        print(f"rendered {out.name} (n={len(truth)})", flush=True)


if __name__ == "__main__":
    render_fields()

    from sahi import AutoDetectionModel
    from YOLOv8BeyondEarth.predict import get_sliced_prediction

    model = None
    for seed in SEEDS:
        for key in SCALES:
            ss, sr = key if isinstance(key, tuple) else (key, 1)
            if nms_gpkg(seed, ss, sr).exists():
                print(f"[skip] seed{seed} ss{ss}sr{sr}", flush=True)
                continue
            if model is None:
                model = AutoDetectionModel.from_pretrained(
                    model_type="yolov8", model_path=str(WEIGHTS), device="cuda:0",
                    image_size=1024, confidence_threshold=0.1)
            out_dir = DE / "preds" / f"null_seed{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[run ] seed{seed} ss{ss}sr{sr}", flush=True)
            t0 = time.time()
            _, gdf_nms = get_sliced_prediction(
                field_tif(seed), detection_model=model, confidence_threshold=0.1,
                output_dir=out_dir, slice_size=ss, sr_factor=sr, inference_size=1024,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                downscale_pred=True, postprocess=True,
                save_prenms=False, save_bbox=False, output_format="gpkg")
            print(f"[done] seed{seed} ss{ss}sr{sr}: n={len(gdf_nms)} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    print("null-field per-scale runs complete; analysis = analyze_fusion_null.py", flush=True)
