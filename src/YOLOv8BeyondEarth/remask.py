"""Post-fusion SAM2 re-masking — Goal-3 phase 5a.

The fused catalog (:func:`YOLOv8BeyondEarth.predict.get_multiscale_prediction`) ships **YOLO
masks**: good detections, grid-locked orientations. This module swaps the mask producer *after*
fusion — one SAM2 box prompt per **kept** fused boulder, read out of raster windows around the
fused boxes.

Why after fusion, not per scale: the in-slice ``remask_fn`` path (Solution G,
:func:`YOLOv8BeyondEarth.orientation_validation.sam2_remasker`) pays SAM on every detection of
every arm — including the ones fusion later discards. One call per kept boulder is strictly
cheaper, and there is no YOLO inference at all.

**What is preserved.** Detection content is non-negotiable (standing gate), so this never drops a
row: every fused detection survives into the output. A boulder whose re-mask fails the
background-grab guard, or that sits below ``min_dpx``, keeps its **original YOLO geometry** and is
flagged in ``mask_src`` / ``remask_note``. That differs deliberately from the in-slice G path,
which *drops* guard failures — dropping is safe pre-NMS, but post-fusion it would silently change
the catalog's detection content.

**Scale note (this is what entry gate 2 measures).** SAM's encoder resizes whatever image it is
given to a fixed input, so a boulder in a 256 px window is seen at a different effective
resolution than the same boulder in a 1024 px window. ``window_size`` therefore defaults to
**256** — matching the validated production ``sam256`` config, where SAM saw one 256 px slice at a
time. Changing it changes SAM's input scale, not just the read pattern.

Plan: ``docs/goal3_multiscale_fusion_plan.md`` §Phase-5 plan / 5a.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.windows import Window
from tqdm import tqdm

from YOLOv8BeyondEarth.polygon import (binary_mask_to_polygon, binary_mask_to_polygon_cv,
                                       shift_polygon, add_geometries)
from YOLOv8BeyondEarth.orientation_validation import largest_blob_fill, sam_masks_in_order

# mask_src values
SRC_SAM = "sam2"
SRC_YOLO = "yolo"


def _pixel_bboxes(bounds_xy, transform):
    """Map-coordinate bounds -> pixel bboxes in the ``add_geometries`` convention.

    ``add_geometries`` maps pixel ``(col + 0.5, row + 0.5)`` through the affine, so the inverse
    affine returns ``col + 0.5`` and the polygon frame is that minus 0.5. Both map corners are
    transformed and min/max'd because a north-up raster flips the y axis (and a rotated one mixes
    the axes).

    ``bounds_xy`` is an (N, 4) array of ``[minx, miny, maxx, maxy]``; returns (N, 4) float
    ``[x0, y0, x1, y1]`` in pixel coordinates.
    """
    inv = ~transform
    b = np.asarray(bounds_xy, dtype=float).reshape(-1, 4)
    # the two opposite corners of each bbox
    c0 = np.array([inv * (x, y) for x, y in zip(b[:, 0], b[:, 1])], dtype=float) - 0.5
    c1 = np.array([inv * (x, y) for x, y in zip(b[:, 2], b[:, 3])], dtype=float) - 0.5
    x0 = np.minimum(c0[:, 0], c1[:, 0]); x1 = np.maximum(c0[:, 0], c1[:, 0])
    y0 = np.minimum(c0[:, 1], c1[:, 1]); y1 = np.maximum(c0[:, 1], c1[:, 1])
    return np.stack([x0, y0, x1, y1], axis=-1)


def _read_window_rgb(src, col_off, row_off, width, height):
    """Read a window as HWC uint8 RGB — the representation the sliced path feeds SAM.

    The production path converts the raster to a 1-band PNG and lets SAHI hand slices to the model
    as 3-channel arrays; for the 8-bit single-band scenes this pipeline runs on, replicating band 1
    into three channels is the same pixels. Multi-band rasters use their first three bands.
    """
    win = Window(col_off, row_off, width, height)
    arr = src.read(window=win, boundless=True, fill_value=0)
    if arr.shape[0] >= 3:
        rgb = np.transpose(arr[:3], (1, 2, 0))
    else:
        rgb = np.repeat(np.transpose(arr[:1], (1, 2, 0)), 3, axis=2)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _group_windows(pix_boxes, window_size, pad, raster_width, raster_height):
    """Assign boxes to a fixed grid of windows by box centre; yield ``(window, member_indices)``.

    The window is the grid cell, expanded to fully contain its members plus ``pad`` and clipped to
    the raster — so an interior cell whose boulders sit well inside it is exactly
    ``window_size`` square (SAM sees the production scale), and only cells with a boulder
    straddling the boundary grow. Cells with no kept boulder are never read.
    """
    cx = (pix_boxes[:, 0] + pix_boxes[:, 2]) / 2.0
    cy = (pix_boxes[:, 1] + pix_boxes[:, 3]) / 2.0
    gx = np.floor(cx / window_size).astype(np.int64)
    gy = np.floor(cy / window_size).astype(np.int64)

    order = np.lexsort((gx, gy))
    keys = np.stack([gy[order], gx[order]], axis=-1)
    # boundaries between runs of equal (gy, gx)
    if len(order) == 0:
        return
    starts = np.flatnonzero(np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)])
    ends = np.r_[starts[1:], len(order)]

    for s, e in zip(starts, ends):
        idx = order[s:e]
        cell_y, cell_x = int(keys[s, 0]), int(keys[s, 1])
        # start from the grid cell, grow to cover members + pad, clip to the raster
        x0 = min(cell_x * window_size, int(np.floor(pix_boxes[idx, 0].min() - pad)))
        y0 = min(cell_y * window_size, int(np.floor(pix_boxes[idx, 1].min() - pad)))
        x1 = max((cell_x + 1) * window_size, int(np.ceil(pix_boxes[idx, 2].max() + pad)))
        y1 = max((cell_y + 1) * window_size, int(np.ceil(pix_boxes[idx, 3].max() + pad)))
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(raster_width, x1); y1 = min(raster_height, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        yield (x0, y0, x1 - x0, y1 - y0), idx


def equivalent_diameter_px(gdf, native_res):
    """Equivalent-circle diameter in **native pixels** from polygon area (map units)."""
    area = gdf.geometry.area.to_numpy(dtype=float)
    return 2.0 * np.sqrt(np.maximum(area, 0.0) / np.pi) / float(native_res)


def remask_windows(in_raster, fused, sam_model, *, pad=16, window_size=256, sam_batch=96,
                   max_box_blowup=1.5, device=0, empty_cache_every=64, min_dpx=None,
                   contour_method="skimage", native_res=None, progress=True):
    """Re-mask a fused catalog with SAM2, one box prompt per kept boulder (phase 5a).

    Parameters
    ----------
    in_raster : path to the scene the fused catalog was detected on.
    fused : GeoDataFrame — the fused catalog (map coordinates, provenance columns).
    sam_model : a loaded ultralytics SAM model (the production ``sam2.1_b`` / ``sam256`` config).
    pad : native px of context added around a window's members before reading.
    window_size : native px grid cell. **Defaults to 256 to match the validated in-slice
        ``sam256`` config's input scale** — see the module docstring.
    min_dpx : None (default) = re-mask everything. Otherwise an equivalent-diameter floor in
        native px; boulders below it keep their YOLO mask and are flagged ``below_floor``.
        **Deliberately un-calibrated** — the binding threshold comes from the planned SAM
        orientation trust-threshold analysis (``docs/README.md`` Coming next), so this ships as a
        knob rather than a baked-in default.
    native_res : map units per native px; read from the raster when None.

    Returns
    -------
    GeoDataFrame — one row per input row, in input order, with the SAM2 geometry where the
    re-mask succeeded and the original geometry elsewhere. Adds ``mask_src`` (``"sam2"`` /
    ``"yolo"``) and ``remask_note`` (``""``, ``"below_floor"``, ``"blowup"``, ``"no_mask"``,
    ``"no_polygon"``). ``.attrs`` carries ``n_windows``, ``n_prompted``, ``n_sam``,
    ``realigned_chunks`` and the config used.
    """
    if len(fused) == 0:
        out = fused.copy()
        out["mask_src"] = pd.Series(dtype=object)
        out["remask_note"] = pd.Series(dtype=object)
        out.attrs.update(n_windows=0, n_prompted=0, n_sam=0, realigned_chunks=0)
        return out

    to_poly = binary_mask_to_polygon_cv if contour_method == "cv2" else binary_mask_to_polygon

    with rio.open(in_raster) as src:
        transform = src.transform
        width, height = src.width, src.height
        if native_res is None:
            native_res = abs(transform.a)

        pix = _pixel_bboxes(fused.geometry.bounds.to_numpy(), transform)

        note = np.full(len(fused), "", dtype=object)
        eligible = np.ones(len(fused), dtype=bool)
        if min_dpx is not None:
            small = equivalent_diameter_px(fused, native_res) < float(min_dpx)
            eligible &= ~small
            note[small] = "below_floor"

        # polygons in FULL-RASTER pixel coordinates, so add_geometries reproduces the production
        # georeferencing exactly (same +0.5 pixel-centre convention, same affine).
        polys = np.full(len(fused), None, dtype=object)
        n_windows = n_prompted = n_sam = realigned = 0

        idx_eligible = np.flatnonzero(eligible)
        groups = list(_group_windows(pix[idx_eligible], window_size, pad, width, height))
        for (col_off, row_off, w, h), local_idx in tqdm(groups, disable=not progress,
                                                        desc="remask windows"):
            rows = idx_eligible[local_idx]
            image = _read_window_rgb(src, col_off, row_off, w, h)
            boxes = pix[rows].copy()
            boxes[:, [0, 2]] -= col_off
            boxes[:, [1, 3]] -= row_off
            # a box must be inside the window for SAM to prompt on it; the window was grown to
            # cover its own members, so this only clips sub-pixel overhang at the raster edge.
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

            masks, realigned_here = sam_masks_in_order(sam_model, image, boxes,
                                                       sam_batch=sam_batch, device=device)
            realigned += realigned_here
            n_windows += 1
            n_prompted += len(rows)

            for k, gi in enumerate(rows):
                mb = None if masks[k] is None else largest_blob_fill(masks[k].astype(np.uint8))
                if mb is None:
                    note[gi] = "no_mask"
                    continue
                box_area = max((boxes[k, 2] - boxes[k, 0]) * (boxes[k, 3] - boxes[k, 1]), 1e-9)
                if mb.sum() > max_box_blowup * box_area:
                    note[gi] = "blowup"
                    continue
                p = to_poly(mb)
                if p is None or len(p) < 3:
                    note[gi] = "no_polygon"
                    continue
                polys[gi] = shift_polygon(np.asarray(p, dtype=float), col_off, row_off)
                n_sam += 1

            if empty_cache_every and (n_windows % empty_cache_every == 0):
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

    got = np.array([p is not None for p in polys])
    out = fused.copy()
    out["mask_src"] = np.where(got, SRC_SAM, SRC_YOLO)
    out["remask_note"] = note

    if got.any():
        sam_geom = add_geometries(in_raster, pd.DataFrame({"polygon": list(polys[got])})).geometry
        new_geom = out.geometry.to_numpy().copy()
        new_geom[got] = sam_geom.to_numpy()
        out = out.set_geometry(gpd.GeoSeries(new_geom, index=out.index, crs=out.crs))

    out.attrs.update(n_windows=n_windows, n_prompted=n_prompted, n_sam=n_sam,
                     realigned_chunks=realigned, window_size=window_size, pad=pad,
                     min_dpx=min_dpx, max_box_blowup=max_box_blowup)
    return out
