"""Sliced full-scene inference for the Mask R-CNN (Test 3, step 3).

Runs a trained torchvision Mask R-CNN over a large raster by tiling it (the same
``sahi``-style slice grid our YOLO pipeline uses), stitching detections with cross-tile NMS, and
georeferencing the masks with the raster's affine transform — producing a GeoDataFrame in the
raster CRS, identical in shape to the YOLO ``get_sliced_prediction`` output, so the orientation
rose can be compared directly (``orientation.boulder_orientations``).

Two inference modes support the "run both" comparison
(``docs/hirise_tests/test3_different_model.md``):

- **native** (``inference_size=None``): tile at ``slice_size`` (≈500 px, MRCNN's training patch
  size) and feed tiles as-is — in-distribution, isolating *architecture* as the only change vs
  YOLO. Read the rose SHAPE (peak angle, 45/135 asymmetry), not detection counts.
- **SAHI-matched** (``inference_size=1024, slice_size=256``): mirror YOLO's
  ``ss-256 is-1024 ov-020`` — each 256 px tile is upscaled to 1024 before inference — to test
  whether the 4× upscale itself moves the rose.

Reuses ``polygon.binary_mask_to_polygon`` / ``binary_mask_to_polygon_cv`` (mask→(x,y) vertices) and
``polygon.add_geometries`` (affine georeferencing), so the mask→polygon→world path is identical to
the YOLO pipeline. The never-MRR orientation rule is honoured downstream (EllipseModel in
``orientation.py``).
"""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from scipy.ndimage import binary_fill_holes as _fill_holes
from skimage.measure import label as _sk_label
from torchvision.ops import nms as nms_torch

from YOLOv8BeyondEarth.polygon import binary_mask_to_polygon, binary_mask_to_polygon_cv
from YOLOv8BeyondEarth.polygon import add_geometries
from YOLOv8BeyondEarth.maskrcnn.dataset import BOULDER_LABEL


def _slice_offsets(full, size, overlap):
    """Top-left offsets tiling ``full`` px with window ``size`` and fractional ``overlap``; the
    last tile is clamped to the edge (so it fully covers, with a little extra overlap)."""
    if full <= size:
        return [0]
    step = max(1, int(round(size * (1.0 - overlap))))
    offs = list(range(0, full - size + 1, step))
    if offs[-1] != full - size:
        offs.append(full - size)
    return offs


def _load_model_from_ckpt(ckpt_path, device):
    from YOLOv8BeyondEarth.maskrcnn.train import build_maskrcnn
    model = build_maskrcnn(weights=None)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval().to(device)
    return model


def _global_nms(boxes, scores, iou, use_lsnms=True):
    """Cross-tile NMS on global-pixel xyxy boxes -> keep indices. Prefers spatial ``lsnms``
    (R-tree, ~O(n log n)), which is essential at the SAHI config's millions of near-disjoint
    boxes where torchvision's greedy CPU NMS is O(n^2) (multi-hour). Falls back to torchvision."""
    if len(boxes) == 0:
        return np.empty(0, dtype=int)
    if use_lsnms:
        try:
            from lsnms import nms as _lsnms
            return np.asarray(_lsnms(boxes, scores, iou_threshold=iou), dtype=int)
        except Exception:
            pass
    return nms_torch(torch.as_tensor(boxes, dtype=torch.float32),
                     torch.as_tensor(scores, dtype=torch.float32), iou).numpy()


@torch.no_grad()
def sliced_predict(raster_path, model, device=None, slice_size=500, inference_size=None,
                   overlap=0.2, conf=0.10, batch_size=8, mask_thresh=0.5, nms_iou=0.5,
                   min_area_px=4, contour_method="skimage", n_workers=8, progress=True,
                   checkpoint_path=None, use_lsnms=True):
    """Tile ``raster_path``, run ``model`` per tile, stitch with global NMS, georeference.

    ``model`` may be a loaded module or a path to a ``best.pt`` checkpoint. Returns a
    GeoDataFrame (raster CRS) with columns ``polygon`` (global-pixel (x,y) vertices), ``score``,
    ``bbox``, ``geometry``.

    Masks are binarized on the GPU (one transfer per batch); the per-detection
    largest-component / fill-holes / contour work — the CPU bottleneck — is threaded over
    ``n_workers`` (those skimage/scipy/cv2 calls release the GIL). ``n_workers=1`` forces serial.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if isinstance(model, (str, Path)):
        model = _load_model_from_ckpt(model, device)
    to_poly = binary_mask_to_polygon_cv if contour_method == "cv2" else binary_mask_to_polygon
    worker = partial(_process_det, inference_size=inference_size, min_area_px=min_area_px,
                     to_poly=to_poly)
    executor = ThreadPoolExecutor(max_workers=n_workers) if n_workers and n_workers > 1 else None

    polygons, scores, boxes = [], [], []
    ckpt = Path(checkpoint_path) if checkpoint_path else None
    if ckpt and ckpt.exists():
        import pickle
        with open(ckpt, "rb") as f:
            d = pickle.load(f)
        polygons, scores, boxes = d["polygons"], d["scores"], d["boxes"]
        if progress:
            print(f"[resume] loaded {len(polygons)} raw detections from {ckpt.name} "
                  "-> skipping tiling", flush=True)
    else:
        with rasterio.open(raster_path) as src:
            H, W = src.height, src.width
            y_offs = _slice_offsets(H, slice_size, overlap)
            x_offs = _slice_offsets(W, slice_size, overlap)
            tiles = [(x0, y0) for y0 in y_offs for x0 in x_offs]
            buf_imgs, buf_meta = [], []

            def flush():
                if not buf_imgs:
                    return
                preds = model([im.to(device) for im in buf_imgs])
                items = []  # (binary_mask uint8 at inference size, score, x0, y0, sw, sh)
                for pred, (x0, y0, sw, sh) in zip(preds, buf_meta):
                    keep = pred["scores"] >= conf
                    if not bool(keep.any()):
                        continue
                    bmask = (pred["masks"][keep][:, 0] >= mask_thresh).to(torch.uint8).cpu().numpy()
                    scs = pred["scores"][keep].cpu().numpy()
                    labs = pred["labels"][keep].cpu().numpy()
                    for binm, s, lab in zip(bmask, scs, labs):
                        if lab == BOULDER_LABEL:
                            items.append((binm, float(s), x0, y0, sw, sh))
                if items:
                    results = executor.map(worker, items) if executor else map(worker, items)
                    for res in results:
                        if res is not None:
                            polygons.append(res[0])
                            scores.append(res[1])
                            boxes.append(res[2])
                buf_imgs.clear()
                buf_meta.clear()

            try:
                for i, (x0, y0) in enumerate(tiles):
                    sw = min(slice_size, W - x0)
                    sh = min(slice_size, H - y0)
                    arr = src.read(1, window=Window(x0, y0, sw, sh))  # (sh, sw) uint8
                    if inference_size and (sh, sw) != (inference_size, inference_size):
                        arr = cv2.resize(arr, (inference_size, inference_size),
                                         interpolation=cv2.INTER_LINEAR)
                    t = torch.from_numpy(np.ascontiguousarray(arr)).float().div_(255.0)
                    t = t.unsqueeze(0).repeat(3, 1, 1)  # 1ch -> 3ch
                    buf_imgs.append(t)
                    buf_meta.append((x0, y0, sw, sh))
                    if len(buf_imgs) >= batch_size:
                        flush()
                    if progress and (i % 500 == 0):
                        print(f"  tile {i}/{len(tiles)}  kept {len(polygons)}", flush=True)
                flush()
            finally:
                if executor:
                    executor.shutdown()
        if ckpt:  # checkpoint raw detections BEFORE the (expensive) NMS, so a killed/slept
            import pickle  # finalize never costs the multi-hour tiling again
            with open(ckpt, "wb") as f:
                pickle.dump({"polygons": polygons, "scores": scores, "boxes": boxes}, f)
            if progress:
                print(f"[checkpoint] saved {len(polygons)} raw detections -> {ckpt.name}",
                      flush=True)

    df = pd.DataFrame({"polygon": polygons, "score": scores})
    if not polygons:
        return add_geometries(raster_path, df)

    # class-agnostic cross-tile NMS on global-pixel boxes (spatial lsnms; torchvision's greedy
    # CPU NMS is O(n^2) and intractable at the SAHI config's millions of near-disjoint boxes)
    keep = _global_nms(np.asarray(boxes, dtype=np.float32),
                       np.asarray(scores, dtype=np.float32), nms_iou, use_lsnms)
    df = df.iloc[keep].reset_index(drop=True)
    return add_geometries(raster_path, df)


def _process_det(item, inference_size, min_area_px, to_poly):
    """One detection's binarized mask -> (global-pixel polygon, score, xyxy box) or None.

    ``item`` = (binary_mask uint8 at inference size, score, x0, y0, sw, sh). Runs the CPU-heavy
    steps (resize-back, largest-component, fill-holes, contour) — this is what gets threaded.
    MRCNN masks can be fragmented or holed; keeping the largest 4-connected component (4-conn
    matches find_contours' 0.5-isoline topology) and filling holes yields a single-ring outline,
    matching the YOLO pipeline's connected/hole-free masks.
    """
    binm, score, x0, y0, sw, sh = item
    if inference_size:
        binm = cv2.resize(binm, (sw, sh), interpolation=cv2.INTER_NEAREST)
    if int(binm.sum()) < min_area_px:
        return None
    lbl = _sk_label(binm, connectivity=1)
    if lbl.max() > 1:
        counts = np.bincount(lbl.ravel())
        counts[0] = 0
        binm = (lbl == counts.argmax()).astype(np.uint8)
    binm = _fill_holes(binm).astype(np.uint8)
    poly = to_poly(binm)  # (N,2) (x,y) in slice-pixel frame, or None
    if poly is None or len(poly) < 3:
        return None
    poly = np.asarray(poly, dtype=np.float64)
    poly[:, 0] += x0  # -> global pixel coords
    poly[:, 1] += y0
    box = [poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()]
    return poly, score, box


def predict_scene(raster_path, ckpt, out_dir, name, device=None, res=None,
                  slice_size=500, inference_size=None, overlap=0.2, conf=0.10,
                  batch_size=4, nms_iou=0.5, contour_method="skimage", sample=None,
                  checkpoint_path=None, use_lsnms=True):
    """End-to-end: sliced inference -> georeferenced gpkg -> orientation CSV
    (``angle180, aspect_ra, diameter_m``, matching the YOLO baselines).

    ``res`` (m/px) defaults to the raster's pixel size. ``name`` labels the outputs.
    ``checkpoint_path`` (optional): pickle the raw pre-NMS detections there — a re-run with the
    same path skips the (multi-hour) tiling and finalises straight from the checkpoint. Returns
    ``(gdf, orientations_df)``.
    """
    from YOLOv8BeyondEarth import orientation

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if res is None:
        with rasterio.open(raster_path) as src:
            res = abs(src.transform.a)

    gdf = sliced_predict(raster_path, ckpt, device=device, slice_size=slice_size,
                         inference_size=inference_size, overlap=overlap, conf=conf,
                         batch_size=batch_size, nms_iou=nms_iou, contour_method=contour_method,
                         checkpoint_path=checkpoint_path, use_lsnms=use_lsnms)
    tag = f"{name}-mrcnn-ss{slice_size}" + (f"-is{inference_size}" if inference_size else "-native")
    gpkg = out_dir / f"{tag}.gpkg"
    gdf.drop(columns=["polygon"]).to_file(gpkg, driver="GPKG")
    print(f"[{name}] {len(gdf)} detections -> {gpkg.name}", flush=True)

    orient = orientation.boulder_orientations(gdf, res=res, sample=sample)
    csv = out_dir / f"{tag}_orientations.csv"
    orient.to_csv(csv, index=False)
    print(f"[{name}] orientations -> {csv.name}", flush=True)
    return gdf, orient
