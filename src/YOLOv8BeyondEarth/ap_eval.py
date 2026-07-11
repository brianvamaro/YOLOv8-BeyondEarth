"""Polygon-catalog AP50 against YOLO-seg labels (Goal 3, validation item 9).

The fused multi-scale catalog is polygons, not model outputs, so ultralytics ``model.val()``
(:func:`benchmark.seg_map_eval`) cannot score it. This module evaluates any polygon catalog
(baseline 256-only, fused ladder, ...) against YOLO-seg label files with a COCO-style AP at a
single IoU threshold, identically for every arm — the *comparison* between arms is the number
that matters. The absolute value is contextual next to the published anchors (Amaro 2026 test
AP50 0.58, Prieur 2023 0.52): it is NOT computed by the same evaluator as the papers, and mAP
against existing labels cannot credit sub-floor detections (truth stops at the annotation
floor, so genuinely new small boulders score as FPs) — which is why per-band P&R + SFD remain
the primary Goal-3 axes (plan §Validation 9).

Geometry convention: predictions and truths are shapely polygons in the same per-image
coordinate frame (any frame — pixel or map — as long as both sides share it). ``geom="box"``
evaluates on envelopes (box AP), ``geom="mask"`` on the polygons themselves (mask AP).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon


def load_yolo_seg_labels(txt_path, width, height, transform=None):
    """Read one YOLO-seg label file -> list of shapely Polygons.

    Each line is ``class x1 y1 x2 y2 ...`` with coordinates normalized to the *labelled image*
    (``width`` x ``height``). Polygons come back in pixel coordinates (x = col, y = row,
    y down), or in map coordinates if a rasterio affine ``transform`` is given (applied as
    ``transform * (col, row)`` — use the transform of the raster the predictions were run on,
    including any padding offset). Degenerate entries (<3 vertices, zero area) are dropped;
    invalid rings are repaired with ``buffer(0)``.
    """
    polys = []
    for line in open(txt_path).read().splitlines():
        parts = line.split()
        if len(parts) < 7:  # class + >=3 vertices
            continue
        coords = np.asarray(parts[1:], dtype=float).reshape(-1, 2)
        px = coords[:, 0] * width
        py = coords[:, 1] * height
        if transform is not None:
            px, py = transform * (px, py)
        g = Polygon(np.column_stack([px, py]))
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty or g.area <= 0:
            continue
        polys.append(g)
    return polys


def _clean(geoms):
    out = []
    for g in geoms:
        if not g.is_valid:
            g = g.buffer(0)
        out.append(g)
    return out


def catalog_ap(preds, gts, iou_thr=0.5, geom="mask"):
    """COCO-style single-class average precision at one IoU threshold, over a multi-image catalog.

    Parameters
    ----------
    preds : DataFrame with columns ``image_id``, ``score``, ``geometry`` (shapely polygons).
    gts : dict ``image_id -> list of shapely polygons`` (images with no predictions still count
        their truths as misses; predictions for unknown image_ids count as FPs).
    iou_thr : match threshold on polygon (or envelope) IoU.
    geom : ``"mask"`` (polygon IoU) or ``"box"`` (envelope IoU).

    Detections are ranked by score globally (COCO convention) and greedily matched, each to the
    highest-IoU not-yet-matched truth of its image at IoU >= ``iou_thr``. AP is the 101-point
    interpolated area under the precision-recall curve (COCOeval's grid). Returns a dict:
    ``ap``, ``n_gt``, ``n_pred``, ``n_tp``, ``precision`` / ``recall`` (curve endpoints, i.e.
    the operating point using every detection).
    """
    if geom not in ("mask", "box"):
        raise ValueError(f"geom must be 'mask' or 'box', got {geom!r}")
    n_gt = int(sum(len(v) for v in gts.values()))
    n_pred = int(len(preds))
    if n_pred == 0 or n_gt == 0:
        return {"ap": 0.0 if n_gt else np.nan, "n_gt": n_gt, "n_pred": n_pred,
                "n_tp": 0, "precision": np.nan, "recall": 0.0 if n_gt else np.nan}

    # per-image GT indexes: geometries (envelope for box AP), STRtree, matched flags
    gt_geoms, gt_trees, gt_used = {}, {}, {}
    for img, polys in gts.items():
        gg = _clean(polys)
        if geom == "box":
            gg = [g.envelope for g in gg]
        gt_geoms[img] = gg
        gt_trees[img] = shapely.STRtree(gg) if gg else None
        gt_used[img] = np.zeros(len(gg), dtype=bool)

    order = np.argsort(-preds["score"].to_numpy(dtype=float), kind="stable")
    image_ids = preds["image_id"].to_numpy()
    geoms = preds["geometry"].to_numpy() if "geometry" in preds else preds.geometry.to_numpy()

    tp = np.zeros(n_pred, dtype=bool)
    for rank, k in enumerate(order):
        img = image_ids[k]
        tree = gt_trees.get(img)
        if tree is None:
            continue
        g = geoms[k]
        if not g.is_valid:
            g = g.buffer(0)
        if geom == "box":
            g = g.envelope
        cand = tree.query(g, predicate="intersects")
        if len(cand) == 0:
            continue
        best_iou, best_j = 0.0, -1
        for j in cand:
            if gt_used[img][j]:
                continue
            other = gt_geoms[img][j]
            inter = g.intersection(other).area
            union = g.area + other.area - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thr:
            gt_used[img][best_j] = True
            tp[rank] = True   # tp is in rank order (score-sorted), ready for cumsum

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(~tp)
    recall = cum_tp / n_gt
    precision = cum_tp / (cum_tp + cum_fp)
    # 101-point interpolation on COCOeval's recall grid, with the standard precision envelope
    prec_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    ap = float(np.mean(np.interp(np.linspace(0, 1, 101), recall, prec_envelope,
                                 left=prec_envelope[0], right=0.0)))
    return {"ap": ap, "n_gt": n_gt, "n_pred": n_pred, "n_tp": int(cum_tp[-1]),
            "precision": float(precision[-1]), "recall": float(recall[-1])}


def size_gated(preds, min_diam_px, res=1.0):
    """Subset a prediction catalog to equivalent-area diameter >= ``min_diam_px`` (native px).

    The plan's admission-gate readout: mAP against labels cannot credit sub-floor detections,
    so the fused arm is also scored with its sub-floor admissions removed (labels themselves
    follow the Prieur 5x5-px minimum-size convention). ``res`` = map units per px.
    """
    areas = np.array([g.area for g in (preds["geometry"] if "geometry" in preds
                                       else preds.geometry)])
    dpx = np.sqrt(4.0 * areas / np.pi) / res
    return preds.loc[dpx >= min_diam_px]
