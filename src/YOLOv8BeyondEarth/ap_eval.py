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

Size convention: one definition, used for truth and prediction alike — **equivalent-area diameter
in native px**, ``sqrt(4 * area / pi) / res``, the same quantity as
:func:`fusion.equivalent_diameter_px`. See ``size_range`` in :func:`catalog_ap` and review item
G3-04: the earlier ">= 5 px AP50" runner applied the size filter to *predictions only* and left all
8,990 truths in the denominator, 2,904 of them (32.3%) below 5 px, so a detector was charged for
missing boulders the evaluation had declared out of scope.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon

#: COCOeval's 101-point recall grid.
RECALL_GRID = np.linspace(0.0, 1.0, 101)


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


def equivalent_diameter(geoms, res=1.0):
    """Equivalent-area diameter in native px — THE size definition (see the module docstring)."""
    areas = np.asarray([g.area for g in geoms], dtype=float)
    return np.sqrt(4.0 * areas / np.pi) / res


def _in_range(dpx, size_range):
    if size_range is None:
        return np.ones(len(dpx), dtype=bool)
    lo, hi = size_range
    return (dpx >= lo) & (dpx < hi)


def _ap_from_flags(tp, ignored, n_gt_kept):
    """COCOeval's precision sampling: step lookup on the 101-point recall grid, not interpolation.

    ``tp``/``ignored`` are in score-rank order. Ignored detections contribute to neither cumsum,
    which is exactly how COCOeval removes them.
    """
    counted = ~ignored
    cum_tp = np.cumsum(tp & counted)
    cum_fp = np.cumsum((~tp) & counted)
    if n_gt_kept == 0:
        return np.nan, cum_tp, cum_fp
    recall = cum_tp / n_gt_kept
    precision = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(float).eps)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    # COCOeval: q[ri] = pr[searchsorted(rc, recThr, 'left')], 0 past the end of the curve
    idx = np.searchsorted(recall, RECALL_GRID, side="left")
    sampled = np.where(idx < len(envelope), envelope[np.minimum(idx, len(envelope) - 1)], 0.0)
    return float(sampled.mean()), cum_tp, cum_fp


def catalog_ap(preds, gts, iou_thr=0.5, geom="mask", size_range=None, res=1.0):
    """COCO-style single-class average precision at one IoU threshold, over a multi-image catalog.

    Parameters
    ----------
    preds : DataFrame with columns ``image_id``, ``score``, ``geometry`` (shapely polygons).
    gts : dict ``image_id -> list of shapely polygons`` (images with no predictions still count
        their truths as misses; predictions for unknown image_ids count as FPs).
    iou_thr : match threshold on polygon (or envelope) IoU.
    geom : ``"mask"`` (polygon IoU) or ``"box"`` (envelope IoU).
    size_range : ``None`` (evaluate everything) or ``(lo, hi)`` half-open equivalent-area diameter
        interval in native px. Applied to **truth and prediction alike**, with COCO's ignore
        semantics (review item G3-04):

        - out-of-range truths are *ignored*, not counted as misses — they leave the recall
          denominator instead of penalising a detector for skipping them;
        - a detection matched to an ignored truth is ignored too, so it is neither a TP nor a FP:
          finding an out-of-scope boulder is not an error;
        - an *unmatched* out-of-range detection is ignored as well.

        Matching prefers in-range truths, so an in-range truth is never lost to an ignored one.
    res : map units per native px, for converting polygon area to a diameter in px.

    Detections are ranked by score globally (COCO convention) and greedily matched, each to the
    highest-IoU not-yet-matched truth of its image at IoU >= ``iou_thr``. AP is the 101-point
    step-sampled area under the precision-recall envelope, as COCOeval computes it.

    Returns a dict: ``ap``, ``n_gt`` (total truths), ``n_gt_counted`` (the recall denominator
    actually used), ``n_pred``, ``n_pred_ignored``, ``n_tp``, ``n_fp``, ``precision`` / ``recall``
    (curve endpoints, i.e. the operating point using every detection), and ``size_range``.
    """
    if geom not in ("mask", "box"):
        raise ValueError(f"geom must be 'mask' or 'box', got {geom!r}")
    if size_range is not None and not (size_range[0] < size_range[1]):
        raise ValueError(f"size_range must be (lo, hi) with lo < hi, got {size_range!r}")

    # per-image GT indexes: geometries (envelope for box AP), STRtree, matched + ignore flags.
    # Size is always measured on the polygon, never the envelope, so the convention does not
    # change between mask AP and box AP.
    gt_geoms, gt_trees, gt_used, gt_ignore = {}, {}, {}, {}
    for img, polys in gts.items():
        cleaned = _clean(polys)
        ignore = ~_in_range(equivalent_diameter(cleaned, res), size_range)
        gg = [g.envelope for g in cleaned] if geom == "box" else cleaned
        gt_geoms[img] = gg
        gt_trees[img] = shapely.STRtree(gg) if gg else None
        gt_used[img] = np.zeros(len(gg), dtype=bool)
        gt_ignore[img] = ignore

    n_gt = int(sum(len(v) for v in gts.values()))
    n_gt_counted = int(sum(int((~v).sum()) for v in gt_ignore.values()))
    n_pred = int(len(preds))
    if n_pred == 0 or n_gt_counted == 0:
        return {"ap": 0.0 if n_gt_counted else np.nan, "n_gt": n_gt,
                "n_gt_counted": n_gt_counted, "n_pred": n_pred, "n_pred_ignored": 0,
                "n_tp": 0, "n_fp": 0, "precision": np.nan,
                "recall": 0.0 if n_gt_counted else np.nan, "size_range": size_range}

    order = np.argsort(-preds["score"].to_numpy(dtype=float), kind="stable")
    image_ids = preds["image_id"].to_numpy()
    geoms = preds["geometry"].to_numpy() if "geometry" in preds else preds.geometry.to_numpy()
    pred_in_range = _in_range(equivalent_diameter(geoms, res), size_range)

    tp = np.zeros(n_pred, dtype=bool)
    ignored = np.zeros(n_pred, dtype=bool)
    for rank, k in enumerate(order):
        img = image_ids[k]
        tree = gt_trees.get(img)
        g = geoms[k]
        if not g.is_valid:
            g = g.buffer(0)
        if geom == "box":
            g = g.envelope
        best_j, matched_ignored = -1, False
        if tree is not None:
            cand = tree.query(g, predicate="intersects")
            # COCO sorts truths so non-ignored come first and stops at the ignored block: an
            # in-range truth must never be consumed by a detection that an ignored truth would do.
            for prefer_ignored in (False, True):
                best_iou = 0.0
                for j in cand:
                    if gt_used[img][j] or bool(gt_ignore[img][j]) != prefer_ignored:
                        continue
                    other = gt_geoms[img][j]
                    inter = g.intersection(other).area
                    union = g.area + other.area - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_j >= 0 and best_iou >= iou_thr:
                    matched_ignored = prefer_ignored
                    break
                best_j = -1
        if best_j >= 0:
            gt_used[img][best_j] = True
            tp[rank] = not matched_ignored
            ignored[rank] = matched_ignored        # credited to an out-of-scope truth: not a FP
        else:
            ignored[rank] = not pred_in_range[k]   # out-of-range miss: not a FP either

    ap, cum_tp, cum_fp = _ap_from_flags(tp, ignored, n_gt_counted)
    return {"ap": ap, "n_gt": n_gt, "n_gt_counted": n_gt_counted, "n_pred": n_pred,
            "n_pred_ignored": int(ignored.sum()), "n_tp": int(cum_tp[-1]),
            "n_fp": int(cum_fp[-1]),
            "precision": float(cum_tp[-1] / max(cum_tp[-1] + cum_fp[-1], 1)),
            "recall": float(cum_tp[-1] / n_gt_counted), "size_range": size_range}


def paired_ap_ci(preds_a, preds_b, gts, n_boot=1000, seed=0, alpha=0.05, **kw):
    """Bootstrap CI for ``AP(b) - AP(a)``, resampling **images** rather than detections.

    The two arms are scored on the *same* resampled image set every iteration, which is what makes
    the interval paired: the shared per-image difficulty cancels instead of inflating the spread.
    Images are the cluster, because detections within one tile are not independent evidence
    (review items G3-04 and G2-06).

    Returns ``{"ap_a", "ap_b", "delta", "lo", "hi", "n_images", "n_boot"}``. An interval that
    excludes 0 is the claim "the fused arm's AP differs from the baseline's"; one that includes 0
    means the observed difference is inside sampling noise for this many images.
    """
    images = sorted(gts)
    n_img = len(images)
    if n_img == 0:
        raise ValueError("gts is empty")
    full_a = catalog_ap(preds_a, gts, **kw)
    full_b = catalog_ap(preds_b, gts, **kw)

    by_img_a = {img: preds_a[preds_a["image_id"] == img] for img in images}
    by_img_b = {img: preds_b[preds_b["image_id"] == img] for img in images}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_img, n_img)
        # relabel duplicates so a twice-drawn image contributes twice, as a bootstrap requires
        sub_gts, frames_a, frames_b = {}, [], []
        for rep, idx in enumerate(pick):
            img = images[idx]
            key = f"{img}#{rep}"
            sub_gts[key] = gts[img]
            for frames, by_img in ((frames_a, by_img_a), (frames_b, by_img_b)):
                part = by_img[img]
                if len(part):
                    part = part.copy()
                    part["image_id"] = key
                    frames.append(part)
        empty = preds_a.iloc[:0]
        sub_a = pd.concat(frames_a, ignore_index=True) if frames_a else empty
        sub_b = pd.concat(frames_b, ignore_index=True) if frames_b else empty
        ap_a = catalog_ap(sub_a, sub_gts, **kw)["ap"]
        ap_b = catalog_ap(sub_b, sub_gts, **kw)["ap"]
        if np.isfinite(ap_a) and np.isfinite(ap_b):
            deltas.append(ap_b - ap_a)
    deltas = np.asarray(deltas)
    lo, hi = (np.nanquantile(deltas, [alpha / 2, 1 - alpha / 2]) if len(deltas)
              else (np.nan, np.nan))
    return {"ap_a": full_a["ap"], "ap_b": full_b["ap"],
            "delta": full_b["ap"] - full_a["ap"], "lo": float(lo), "hi": float(hi),
            "n_images": n_img, "n_boot": int(len(deltas))}


def size_gated(preds, min_diam_px, res=1.0):
    """Subset a prediction catalog to equivalent-area diameter >= ``min_diam_px`` (native px).

    .. warning::
       **This is a prediction-side filter only and is not a size-range AP.** Using it alone leaves
       every truth in the recall denominator, including the out-of-range ones, so the detector is
       charged for missing boulders the evaluation has declared out of scope — the G3-04 defect
       (2,904 of 8,990 truths, 32.3%, sat below the 5 px gate). For a size-restricted AP pass
       ``size_range=(min_diam_px, inf)`` to :func:`catalog_ap`, which applies the same convention
       to both sides with COCO ignore semantics. This helper remains for building size-banded
       *catalogs*, not size-banded metrics.

    ``res`` = map units per px.
    """
    geoms = preds["geometry"] if "geometry" in preds else preds.geometry
    return preds.loc[equivalent_diameter(list(geoms), res) >= min_diam_px]
