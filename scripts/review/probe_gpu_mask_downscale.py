"""Can the per-detection mask downscale move to the GPU without changing geometry?

`profile_production_path.py` found that the two stages that exist only because ultralytics returns
masks at ``imgsz`` — moving a 1024x1024 float32 mask per detection off the GPU, then
``INTER_AREA``-downscaling it on the CPU — are **a third of the production run**:

    slice 256   cv2.resize 19.4% + Tensor.cpu 14.2%  = 33.6%   (18.1 GiB moved for a 2048px crop)
    slice 512   cv2.resize 15.0% + Tensor.cpu 19.4%  = 34.5%   (12.7 GiB)

Both would shrink if the downscale happened on the device: the transfer drops by the square of the
ratio (16x at slice 256, 4x at slice 512) and the CPU resize disappears. The catch is that this is
the *production geometry path*, so "faster" is worthless unless the downscaled mask is identical.

It should be, and for a reason rather than by luck. The mask is thresholded to exactly {0.0, 1.0}
on the GPU *before* the resize, and for an exact integer ratio ``INTER_AREA`` is a plain box mean,
so every output value is ``n / k**2`` with ``n <= k**2`` and ``k`` a power of two — a dyadic
rational that float32 represents exactly, from either implementation. This checks that argument
against real detections instead of trusting it, and reports what the change would actually buy.

    conda run -n bouldernet python scripts/review/probe_gpu_mask_downscale.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raster", default=str(REPO / "data" / "raster" / "crop.tif"))
    ap.add_argument("--weights",
                    default=str(REPO / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"))
    ap.add_argument("--out", default=str(REPO / "data" / "review"))
    ap.add_argument("--slice-sizes", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--max-slices", type=int, default=48)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import cv2
    import torch
    import torch.nn.functional as F
    from sahi import AutoDetectionModel
    from sahi.slicing import slice_image

    from YOLOv8BeyondEarth import presets
    from YOLOv8BeyondEarth.polygon import binary_mask_to_polygon
    from YOLOv8BeyondEarth.predict import _ensure_png, _to_bgr

    imgsz = presets.GOAL1_PRODUCTION["inference_size"]
    conf = presets.GOAL1_PRODUCTION["confidence_threshold"]
    raster = Path(args.raster)
    png = _ensure_png(raster, presets.file_identity(raster))

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=args.weights, device=args.device,
        image_size=imgsz, confidence_threshold=conf)

    summary = {"raster": raster.name, "inference_size": imgsz, "arms": {}}
    for slice_size in args.slice_sizes:
        factor = imgsz // slice_size
        sliced = slice_image(image=str(png), slice_height=slice_size, slice_width=slice_size,
                             overlap_height_ratio=0.2, overlap_width_ratio=0.2, out_ext=".png")
        images = list(sliced.images)[:args.max_slices]

        n_masks = n_value_identical = n_polygon_identical = n_polygons = 0
        max_abs_diff = 0.0
        t_cpu = t_gpu = 0.0
        bytes_cpu = bytes_gpu = 0

        for start in range(0, len(images), 8):
            batch = images[start:start + 8]
            results = model.model([_to_bgr(im) for im in batch], imgsz=imgsz, conf=conf,
                                  verbose=False, device=model.device)
            for res in results:
                if res.masks is None or res.boxes.data.shape[0] == 0:
                    continue
                hard = (res.masks.data >= 0.5).to(torch.float32)      # what production transfers
                assert hard.shape[-1] == imgsz, f"masks came back at {tuple(hard.shape)}"

                # --- current production: transfer at imgsz, then INTER_AREA on the CPU ---------
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                on_cpu = hard.cpu().numpy()
                cpu_small = np.stack([cv2.resize(m, (slice_size, slice_size),
                                                 interpolation=cv2.INTER_AREA) for m in on_cpu])
                t_cpu += time.perf_counter() - t0
                bytes_cpu += hard.numel() * hard.element_size()

                # --- proposed: exact box mean on the device, then transfer at slice size -------
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                pooled = F.avg_pool2d(hard.unsqueeze(1), kernel_size=factor, stride=factor)
                gpu_small = pooled.squeeze(1).cpu().numpy()
                torch.cuda.synchronize()
                t_gpu += time.perf_counter() - t0
                bytes_gpu += pooled.numel() * pooled.element_size()

                n_masks += len(cpu_small)
                same = np.all(cpu_small == gpu_small, axis=(1, 2))
                n_value_identical += int(same.sum())
                max_abs_diff = max(max_abs_diff, float(np.abs(cpu_small - gpu_small).max()))

                # geometry is what actually has to be preserved, so trace both and compare
                for a, b in zip(cpu_small, gpu_small):
                    pa, pb = binary_mask_to_polygon(a), binary_mask_to_polygon(b)
                    n_polygons += 1
                    if pa is None and pb is None:
                        n_polygon_identical += 1
                    elif pa is not None and pb is not None and pa.shape == pb.shape \
                            and np.array_equal(pa, pb):
                        n_polygon_identical += 1

        summary["arms"][str(slice_size)] = {
            "downscale_factor": factor,
            "n_slices": len(images),
            "n_masks": n_masks,
            "n_value_identical": n_value_identical,
            "frac_value_identical": round(n_value_identical / max(n_masks, 1), 6),
            "max_abs_value_diff": max_abs_diff,
            "n_polygons": n_polygons,
            "n_polygon_identical": n_polygon_identical,
            "frac_polygon_identical": round(n_polygon_identical / max(n_polygons, 1), 6),
            "cpu_path_s": round(t_cpu, 3),
            "gpu_path_s": round(t_gpu, 3),
            "speedup": round(t_cpu / t_gpu, 2) if t_gpu else None,
            "transferred_MiB_now": round(bytes_cpu / 1024 ** 2, 1),
            "transferred_MiB_proposed": round(bytes_gpu / 1024 ** 2, 1),
        }
        print(json.dumps({str(slice_size): summary["arms"][str(slice_size)]}, indent=2), flush=True)

    out = Path(args.out) / "gpu_mask_downscale_probe.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
