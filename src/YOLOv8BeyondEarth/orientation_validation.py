"""Validation drivers for the HiRISE 135-deg orientation investigation (Goal 2).

These answer "is the off-grid (~135 deg, NW-SE) boulder long-axis peak a real ground signal or a
model/segmentation artifact?" with measurements that do NOT depend on the YOLO mask *shape*:

- :func:`per_boulder_structure_tensor` — per-boulder YOLO-free orientation from the raw image
  patch (structure tensor) paired with the mask EllipseModel angle. (Illumination-confounded; see
  the caveat in :mod:`YOLOv8BeyondEarth.orientation`.)
- :func:`regional_orientation_hist` — coherence-weighted structure-tensor orientation of whole raw
  crops (terrain texture; no boulders, no masks).
- :func:`centroid_direction_hist` — nearest-neighbour direction histogram of boulder *centroids*
  (positions only): illumination-immune and mask-shape-free.

Heavy IO deps (rasterio, pyogrio) are imported lazily so importing the orientation primitives
stays cheap. Companion notebook: ``notebooks/Goal2_orientation_HiRISE_investigation.ipynb``.
"""
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import presets
from .orientation import ellipse_angle180, structure_tensor_orientation, orientation_field
from .polygon import _mask_bounds

logger = logging.getLogger(__name__)


def matched_area_ratio(model_gdf, ref_gdf, rom=None, min_radius_m=1.0):
    """Per-boulder mask-area ratio of ``model_gdf`` predictions against a ``ref_gdf`` reference
    (human ROM outlines, or a second model), matched by nearest centroid — the mask-quality /
    over-vs-under-segmentation readout for the Goal-2 re-masking work (Test 3c treatment stage).

    Reads ``ratio`` as **model area vs human-annotation area**, NOT vs true boulder area: on shadowed
    scenes (e.g. Mars) the human outline's relation to the true rock is uncertain — near-boulder dark
    pixels mix the boulder's own shadowed side (genuine boulder, which a sunlit-keyed model mask may
    even under-capture) with cast ground shadow (not boulder), so the sign of the human-vs-truth bias
    is unknown. Use a near-shadow-free scene (e.g. Moon Censorinus, incidence ~9°) or explicit shadow
    handling to anchor the truth; here just report the model-vs-annotation ratio.

    Both GeoDataFrames must already be on the SAME grid (use ``set_crs(..., allow_override=True)``,
    never ``to_crs`` — the Prieur shp vs project gpkg carry cosmetically-different CRS strings on the
    same equirectangular grid; reprojecting throws a bogus datum shift). If ``rom`` (a GeoDataFrame
    or geometry) is given, both sets are first clipped to boulders whose centroid falls inside it.

    Each reference polygon is matched to **at most one** model detection, within
    ``max(ref_equiv_radius, min_radius_m)`` (map units), by an optimal one-to-one assignment
    (:func:`matching.unique_centroid_match`). It used to be a nearest-centroid query, which is
    many-to-one: several references could all claim the same prediction, so a model that merged two
    neighbouring boulders into one still scored full recall and its merged mask was counted twice
    in the ratio distribution (review item G2-05, which flags this as a possible contributor to
    the 1.37x -> 1.02x result).

    Returns ``(df, stats)`` where ``df`` has one row per matched reference boulder
    (``ref_area, model_area, ratio, ref_dia``) and ``stats`` is a dict with the full match/unmatched
    flow counts on both sides. Ratio > 1 ⇒ model masks larger than the reference annotation;
    < 1 ⇒ tighter. (Whether > 1 means overshoot vs the *true* boulder depends on the reference's
    own accuracy — see the shadow caveat above.)
    """
    from scipy.spatial import cKDTree
    import shapely

    from .matching import unique_centroid_match

    m, r = model_gdf.copy(), ref_gdf.copy()
    if rom is not None:
        geom = rom.union_all() if hasattr(rom, "union_all") else (
            rom.unary_union if hasattr(rom, "unary_union") else rom)
        m = m[m.geometry.centroid.within(geom)].copy()
        r = r[r.geometry.centroid.within(geom)].copy()
    for g in (m, r):
        g["_area"] = g.geometry.area
        g["_dia"] = 2 * np.sqrt(g["_area"].to_numpy() / np.pi)
    mc = np.c_[m.geometry.centroid.x.to_numpy(), m.geometry.centroid.y.to_numpy()]
    rc = np.c_[r.geometry.centroid.x.to_numpy(), r.geometry.centroid.y.to_numpy()]
    ref_rad = r["_dia"].to_numpy() / 2
    ref_idx, mod_idx = unique_centroid_match(rc, mc, np.maximum(ref_rad, min_radius_m))
    ref_a = r["_area"].to_numpy()[ref_idx]
    mod_a = m["_area"].to_numpy()[mod_idx]
    df = pd.DataFrame(dict(ref_area=ref_a, model_area=mod_a, ratio=mod_a / ref_a,
                           ref_dia=r["_dia"].to_numpy()[ref_idx],
                           ref_index=ref_idx, model_index=mod_idx))
    # over-detection: model polys with no reference within their own radius
    rdist, _ = cKDTree(rc).query(mc, k=1) if len(rc) and len(mc) else (np.full(len(mc), np.inf), None)
    over = rdist > np.maximum(m["_dia"].to_numpy() / 2, min_radius_m)
    n_matched = int(len(ref_idx))
    stats = dict(n_ref=len(r), n_model=len(m), n_matched=n_matched,
                 n_ref_unmatched=int(len(r) - n_matched),
                 n_model_unmatched=int(len(m) - n_matched),
                 recall=float(n_matched / len(r)) if len(r) else np.nan,
                 precision=float(n_matched / len(m)) if len(m) else np.nan,
                 median_ratio=float(np.median(df.ratio)) if n_matched else np.nan,
                 over_detection_frac=float(over.mean()) if len(m) else np.nan)
    assert df.ref_index.is_unique and df.model_index.is_unique, "matching is not one-to-one"
    return df, stats


#: Area definitions that appear in this project. Only the first is the production convention.
AREA_CONVENTIONS = ("polygon_skimage", "polygon_cv2", "pixel_count")


def mask_areas(masks, convention="polygon_skimage", res=1.0):
    """Area of each mask in map units squared under an explicitly named boundary convention.

    ``polygon_skimage`` — the production convention: clean the mask, trace the 0.5 level set, take
    the polygon's continuous area. A solid 5x5 block measures 24.5 px.
    ``polygon_cv2`` — trace boundary pixel *centres* instead. The same block measures 16.0 px.
    ``pixel_count`` — the integer count of supported pixels. The same block measures 25.

    These are not interchangeable, and mixing them is review item G2-04: the published SAM/human
    (1.06-1.27x) and YOLO/human (1.58-2.13x) factors compared a model **pixel count** against a
    human **continuous polygon area**, so part of every one of them is the convention gap rather
    than a real difference in how big the masks are.
    """
    from .orientation import mask_to_polygon_geom
    from .polygon import mask_support_area

    if convention not in AREA_CONVENTIONS:
        raise ValueError(f"convention must be one of {AREA_CONVENTIONS}, got {convention!r}")
    px_area = float(res) ** 2
    out = []
    for m in masks:
        if convention == "pixel_count":
            out.append(mask_support_area(m) * px_area)
            continue
        method = "cv2" if convention == "polygon_cv2" else "skimage"
        geom, _ = mask_to_polygon_geom(m, contour_method=method)
        out.append(np.nan if geom is None else geom.area * px_area)
    return np.asarray(out, dtype=float)


def convention_sensitivity(masks, res=1.0, conventions=AREA_CONVENTIONS):
    """How far the *same* masks move when only the area convention changes (G2-04 acceptance).

    Returns one row per convention with the median area and its ratio to the production
    convention. Any calibration factor has to be quoted against a stated convention, and this is
    the number that says how much that statement matters.
    """
    areas = {c: mask_areas(masks, convention=c, res=res) for c in conventions}
    base = areas[AREA_CONVENTIONS[0]]
    rows = []
    for c, a in areas.items():
        ok = np.isfinite(a) & np.isfinite(base) & (base > 0)
        rows.append({"convention": c, "n": int(ok.sum()),
                     "median_area": float(np.nanmedian(a)) if ok.any() else np.nan,
                     "ratio_to_production": float(np.median(a[ok] / base[ok])) if ok.any() else np.nan})
    return pd.DataFrame(rows)


def area_calibration(model_gdf, ref_gdf, rom=None, min_radius_m=1.0, groups=None,
                     n_boot=1000, seed=0):
    """Model/reference area ratio on a **one-to-one** cohort, both sides continuous polygons.

    This is what a calibration factor for Goal 4 has to be: the same quantity measured the same way
    on both sides, over a cohort where each reference is used once. Both inputs must already carry
    production-convention polygons — for a mask-based producer, trace it with
    :func:`orientation.mask_to_polygon_geom` first rather than summing pixels.

    ``groups`` labels the cluster each *reference* belongs to (tile, ROM, scene); when given, the
    bootstrap resamples clusters rather than boulders, because boulders inside one tile are not
    independent evidence (G2-06).

    Returns ``(df, stats)``: the per-boulder table from :func:`matched_area_ratio`, and its stats
    plus ``median_ratio_lo/hi``, ``n_clusters`` and the ``convention`` the factor is quoted for.
    """
    df, stats = matched_area_ratio(model_gdf, ref_gdf, rom=rom, min_radius_m=min_radius_m)
    stats = dict(stats)
    stats["convention"] = "polygon_skimage (production)"
    ratios = df["ratio"].to_numpy(dtype=float)
    ratios = ratios[np.isfinite(ratios)]
    if len(ratios) == 0:
        stats.update(median_ratio_lo=np.nan, median_ratio_hi=np.nan, n_clusters=None)
        return df, stats

    rng = np.random.default_rng(seed)
    if groups is None:
        boot = [np.median(rng.choice(ratios, len(ratios), replace=True)) for _ in range(n_boot)]
        n_clusters = None
    else:
        labels = np.asarray(groups)[df["ref_index"].to_numpy()]
        members = [np.flatnonzero(labels == g) for g in pd.unique(labels)]
        k = len(members)
        boot = [np.median(ratios[np.concatenate([members[i] for i in rng.integers(0, k, k)])])
                for _ in range(n_boot)]
        n_clusters = k
    lo, hi = np.percentile(boot, [2.5, 97.5])
    stats.update(median_ratio_lo=float(lo), median_ratio_hi=float(hi), n_clusters=n_clusters)
    return df, stats


def per_boulder_structure_tensor(gpkg_path, raster_path, res, n_fids=4000, aspect_min=1.35,
                                 pad_frac=1.5, pad_min_m=4.0, smooth=1.0, seed=1,
                                 n_total=None):
    """Mask vs YOLO-free (structure-tensor) long-axis angle for a random sample of elongated
    boulders.

    Reads ``n_fids`` random features from the prediction gpkg, keeps those with
    ``aspect_ra > aspect_min``, and for each reads a square raw-image window (side
    ``2*max(pad_frac*radius, pad_min_m)`` metres) centred on the boulder, weights it with a
    centred Gaussian (sigma = boulder radius), and measures the structure-tensor orientation.

    Returns a DataFrame ``[mask_ang, aspect, diameter_m, st_ang, st_coh, cx, cy]``. The mask angle
    is the unbiased EllipseModel azimuth; ``st_ang`` is the raw-pixel structure-tensor azimuth
    (same convention). On sunlit imagery ``st_ang`` is pulled toward the illumination edge axis.
    """
    import rasterio
    from rasterio.windows import from_bounds
    from pyogrio import read_dataframe, read_info

    if n_total is None:
        n_total = read_info(gpkg_path)["features"]
    rng = np.random.default_rng(seed)
    fids = np.sort(rng.choice(n_total, size=min(n_fids, n_total), replace=False))
    g = read_dataframe(gpkg_path, fids=fids)

    rows = []
    with rasterio.open(raster_path) as r:
        for geom in g.geometry.values:
            ang, asp = ellipse_angle180(geom, res)
            if not np.isfinite(asp) or asp < aspect_min:
                continue
            minx, miny, maxx, maxy = geom.bounds
            cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
            rad = max(maxx - minx, maxy - miny) / 2
            pad = max(rad * pad_frac, pad_min_m)
            win = from_bounds(cx - pad, cy - pad, cx + pad, cy + pad, r.transform)
            patch = r.read(1, window=win, boundless=True, fill_value=0).astype(float)
            if patch.shape[0] < 7 or patch.shape[1] < 7:
                continue
            h, w = patch.shape
            yy, xx = np.mgrid[0:h, 0:w]
            sig = max(rad / res, 2.0)
            wgt = np.exp(-(((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * sig ** 2)))
            st_ang, st_coh = structure_tensor_orientation(patch, weight=wgt, smooth=smooth)
            rows.append((ang, asp, 2 * np.sqrt(geom.area / np.pi), st_ang, st_coh, cx, cy))
    return pd.DataFrame(rows, columns=["mask_ang", "aspect", "diameter_m",
                                       "st_ang", "st_coh", "cx", "cy"])


def regional_orientation_hist(raster_path, crop=1024, n_crops=24, scales=((1, 4), (2, 8), (4, 16)),
                              coh_pct=75, min_valid=0.95, seed=7, bin_deg=10):
    """Coherence-weighted structure-tensor orientation histogram of raw terrain crops (no masks).

    Samples ``n_crops`` valid (>= ``min_valid`` non-zero) square crops, computes the per-pixel
    orientation field at each ``(grad_sigma, integ_sigma)`` in ``scales``, and accumulates a
    coherence-weighted azimuth histogram over the most-coherent ``100-coh_pct`` percent of pixels.

    Returns ``(centers, {scale: percent_hist})`` averaged over crops.
    """
    import rasterio
    from rasterio.windows import Window

    edges = np.arange(0, 181, bin_deg)
    centers = (edges[:-1] + edges[1:]) / 2
    agg = {s: np.zeros(len(centers)) for s in scales}
    rng = np.random.default_rng(seed)
    with rasterio.open(raster_path) as r:
        W, H = r.width, r.height
        n = 0; tried = 0
        while n < n_crops and tried < n_crops * 8:
            tried += 1
            c = rng.integers(0, W - crop); rr = rng.integers(0, H - crop)
            img = r.read(1, window=Window(c, rr, crop, crop)).astype(float)
            valid = img > 0
            if valid.mean() < min_valid:
                continue
            n += 1
            for s in scales:
                az, coh = orientation_field(img, s[0], s[1])
                thr = np.percentile(coh, coh_pct)
                m = (coh >= thr) & valid
                h, _ = np.histogram(az[m] % 180, bins=edges, weights=coh[m])
                agg[s] += 100 * h / h.sum()
    return centers, {s: agg[s] / max(n, 1) for s in scales}


def slice_masks_to_slice_res(prediction_result, slice_size):
    """Model masks -> slice-resolution binary masks, exactly as production does it.

    ultralytics returns ``masks.data`` at ``imgsz``, not at ``slice_size``, so this is a genuine
    ``INTER_AREA`` downscale (see G3-10) — the same one ``_result_to_df`` performs under
    ``downscale_pred=True``.
    """
    import cv2

    if prediction_result.masks is None or len(prediction_result.masks.data) == 0:
        return []
    out = []
    for m in prediction_result.masks.data.cpu().numpy():
        mm = (m >= 0.5).astype(np.float32)
        if mm.shape[0] != slice_size:
            mm = cv2.resize(mm, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
        out.append((mm >= 0.5).astype(np.uint8))
    return out


#: Ultralytics' own per-image detection cap (``cfg/default.yaml``), which applies to every call
#: that does not pass ``max_det``. This is review item **X-06**, fixed in the production path
#: 2026-08-04 and found again *in this module* on 2026-08-06: none of the tiled scanners below
#: passed it, so the SAM2 equal-area rescan behind ``data/summary/fixed_area_scenes.csv`` ran
#: capped. Across its 5,382 tiles the per-tile count walls at 295 with a mode of 111 tiles at 278,
#: and the control site ``ESP_030222_1220`` averages 275.6 det/tile with 57.4% of its detections
#: inside capped tiles.
#:
#: What that does and does not contaminate: the cap keeps the top N **by confidence**, and the
#: analysed subset of that rescan is the >=8 px / >=4 m tail (5.7% of rows), which is the
#: high-confidence end — so the **roses are not plausibly moved**, but **every boulders/km2 from a
#: capped run is a scene-dependent lower bound**.
#:
#: Duplicated from :data:`predict.ULTRALYTICS_DEFAULT_MAX_DET` rather than imported: ``predict``
#: pulls in cv2, geopandas and torch at module level and this module is deliberately cheap to
#: import. ``tests/test_orientation_validation.py`` asserts the two never drift apart. Kept as the
#: *function* default here for the same reason ``predict`` keeps it — an un-parameterised call
#: reproduces the historical artifacts exactly; pass ``presets.GOAL1_PRODUCTION["max_det"]`` (600)
#: for a fresh measurement.
ULTRALYTICS_DEFAULT_MAX_DET = 300

#: Mirror of :data:`predict._NEAR_MAX_DET_FRACTION`. A tile *at* the cap proves truncation; a tile
#: just below it is where a tile that truncated and then lost a detection or two to ultralytics'
#: empty-mask filter shows up instead, so both are recorded.
_NEAR_MAX_DET_FRACTION = 0.9


def _new_truncation(max_det):
    """Fresh per-scan truncation tally. See :data:`ULTRALYTICS_DEFAULT_MAX_DET`."""
    return {"max_det": int(max_det), "tiles": 0, "tiles_at_max_det": 0,
            "tiles_near_max_det": 0, "max_det_returned": 0}


def _note_tile_truncation(result, max_det, trunc):
    """Record whether one tile came back at ultralytics' detection cap.

    ``tiles_at_max_det`` is a **lower** bound, exactly as in
    :func:`predict._note_slice_truncation`: ultralytics applies ``max_det`` inside NMS and
    ``SegmentationPredictor.construct_result`` then drops detections whose upsampled mask is empty,
    so a tile that truncated at N can arrive holding fewer than N. Handles both segmentation
    (``result.boxes``) and OBB (``result.obb``) results.

    The count degrades to 0 rather than raising if a result object exposes neither ``len()`` nor a
    ``.data`` array: this is a diagnostic, and a diagnostic that can abort the scan it is watching is
    worse than no diagnostic. (Real ultralytics ``Boxes``/``OBB`` support both; the fallback exists
    for the stand-in models the test suite drives.)
    """
    n = 0
    for attr in ("boxes", "obb"):
        v = getattr(result, attr, None)
        if v is None:
            continue
        try:
            n = int(len(v))
        except TypeError:
            data = getattr(v, "data", None)
            n = 0 if data is None else int(getattr(data, "shape", [len(data)])[0])
        break
    trunc["tiles"] += 1
    trunc["max_det_returned"] = max(trunc["max_det_returned"], n)
    if n >= max_det:
        trunc["tiles_at_max_det"] += 1
    elif n >= max(1, int(round(_NEAR_MAX_DET_FRACTION * max_det))):
        trunc["tiles_near_max_det"] += 1
    return trunc


def _warn_if_truncated(trunc, where):
    """Make a capped scan loud. Silence is what made X-06 survive two reviews."""
    if trunc.get("tiles_at_max_det"):
        warnings.warn(
            f"{where}: {trunc['tiles_at_max_det']} of {trunc['tiles']} tiles returned the full "
            f"max_det={trunc['max_det']}, so the model's output was cut off in each of them and "
            "the faintest detections are missing. Densities from this scan are LOWER BOUNDS; "
            "angle statistics on a high-confidence subset are far less exposed. Raise max_det "
            "(presets.GOAL1_PRODUCTION uses 600) and re-scan. This count is a lower bound.",
            RuntimeWarning, stacklevel=3)
    if trunc.get("tiles"):
        logger.info("%s: tile truncation record %s", where, trunc)
    return trunc


def detect_orientations(model, img, slice_size=256, imgsz=1024, conf=0.10, device=0,
                        aspect_min=1.0, estimator="ellipse", drop_edge=True,
                        contour_method="skimage", max_det=ULTRALYTICS_DEFAULT_MAX_DET):
    """Tile a raw image array into ``slice_size`` windows, run YOLO on each, and return a
    DataFrame ``[angle180, aspect, diameter_px]`` for every detection (geographic long-axis
    azimuth measured in the *image's own* frame).

    A reimplementation of the sliced pipeline for a single in-memory array, with two things the
    review found it had wrong (G2-02):

    - the angle comes from :func:`orientation.mask_readout`, the *one* mask-cleanup -> polygon ->
      ``EllipseModel`` readout production uses, rather than a local copy that could drift;
    - ``drop_edge=True`` applies production's slice-truncation rule. It used to keep tile-edge
      objects, whose masks are clipped by the tile boundary and therefore carry a spurious
      straight edge aligned to the grid — a cardinal bias, fed into tests measuring cardinal bias.

    Tiles are non-overlapping, so a boulder clipped here has no whole counterpart to recover it;
    dropping is the honest choice and matches what production emits away from a footprint edge.
    Pass ``drop_edge=False`` to reproduce the pre-fix numbers.

    ``max_det`` is passed explicitly and its saturation recorded in ``df.attrs["truncation"]``
    (review item X-06 — see :data:`ULTRALYTICS_DEFAULT_MAX_DET`).
    """
    from .orientation import is_mask_within_slice, mask_readout

    H, W = img.shape
    rows = []
    trunc = _new_truncation(max_det)
    for y0 in range(0, H - slice_size + 1, slice_size):
        for x0 in range(0, W - slice_size + 1, slice_size):
            sl = img[y0:y0 + slice_size, x0:x0 + slice_size]
            r0 = model(np.stack([sl] * 3, axis=-1), imgsz=imgsz, conf=conf, verbose=False,
                       device=device, max_det=max_det)[0]
            _note_tile_truncation(r0, max_det, trunc)
            for mb in slice_masks_to_slice_res(r0, slice_size):
                if drop_edge and not is_mask_within_slice(mb, slice_size):
                    continue
                a, asp, dpx = mask_readout(mb, estimator=estimator,
                                           contour_method=contour_method)
                if np.isfinite(asp) and asp >= aspect_min:
                    rows.append((a, asp, dpx))
    df = pd.DataFrame(rows, columns=["angle180", "aspect", "diameter_px"])
    df.attrs["truncation"] = _warn_if_truncated(trunc, "detect_orientations")
    return df


def detect_binary_masks(model, img, slice_size=256, imgsz=1024, conf=0.10, device=0,
                        drop_edge=True, max_det=ULTRALYTICS_DEFAULT_MAX_DET):
    """Tile ``img`` like :func:`detect_orientations` but return every detection's processed
    **binary mask** (bbox-cropped) instead of tracing it — so different mask→polygon tracers can
    be compared on the *same* masks (Test 3d, the contour-tracer chirality swap).

    Mask handling is identical to :func:`detect_orientations` — same slice-resolution downscale,
    same ``largest_blob_fill`` cleanup, same ``drop_edge`` rule (G2-02) — so the two helpers see
    the same population. Cropping to the blob's bounding box (both tracers crop internally anyway)
    keeps a cached run small; orientation is translation-invariant.

    ``max_det`` is passed explicitly (X-06); the return is a plain list, so saturation is reported
    through a ``RuntimeWarning`` and the module logger rather than an ``attrs`` channel.
    """
    from .orientation import is_mask_within_slice

    H, W = img.shape
    masks = []
    trunc = _new_truncation(max_det)
    for y0 in range(0, H - slice_size + 1, slice_size):
        for x0 in range(0, W - slice_size + 1, slice_size):
            sl = img[y0:y0 + slice_size, x0:x0 + slice_size]
            r0 = model(np.stack([sl] * 3, axis=-1), imgsz=imgsz, conf=conf, verbose=False,
                       device=device, max_det=max_det)[0]
            _note_tile_truncation(r0, max_det, trunc)
            for mb in slice_masks_to_slice_res(r0, slice_size):
                if drop_edge and not is_mask_within_slice(mb, slice_size):
                    continue
                clean = largest_blob_fill(mb)
                if clean is None:
                    continue
                ys, xs = np.nonzero(clean)
                masks.append(clean[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy())
    _warn_if_truncated(trunc, "detect_binary_masks")
    return masks


def detect_mask_pairs(model, img, slice_size=256, imgsz=1024, conf=0.10, device=0,
                      max_det=ULTRALYTICS_DEFAULT_MAX_DET):
    """Like :func:`detect_binary_masks` but returns **(native, downscaled)** mask pairs per
    detection from a single forward pass — Test 3g, isolating the production *downscale +
    re-threshold* stage.

    native     = ultralytics' binary mask at inference resolution (``imgsz``), i.e. BEFORE the
                 production resize-back-to-slice; blob/fill applied at native res.
    downscaled = the production path (resize float mask to ``slice_size`` with INTER_AREA,
                 re-threshold >= 0.5, blob/fill) — byte-identical handling to
                 :func:`detect_binary_masks` / :func:`detect_orientations`.

    Both sides are bbox-cropped. A detection is kept only if BOTH sides yield a non-empty mask,
    so the two lists align pairwise. Orientation is scale-free, so the 4x pixel-scale difference
    between the sides does not bias the angle comparison.
    """
    import cv2
    from scipy.ndimage import label, binary_fill_holes

    def _clean_crop(mb):
        lab, n = label(mb)
        if n == 0:
            return None
        if n > 1:
            big = 1 + int(np.argmax([(lab == j).sum() for j in range(1, n + 1)]))
            mb = (lab == big).astype(np.uint8)
        mb = binary_fill_holes(mb).astype(np.uint8)
        ys, xs = np.nonzero(mb)
        return mb[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()

    H, W = img.shape
    native, down = [], []
    trunc = _new_truncation(max_det)
    for y0 in range(0, H - slice_size + 1, slice_size):
        for x0 in range(0, W - slice_size + 1, slice_size):
            sl = img[y0:y0 + slice_size, x0:x0 + slice_size]
            res = model(np.stack([sl] * 3, axis=-1), imgsz=imgsz, conf=conf, verbose=False,
                        device=device, max_det=max_det)
            r0 = res[0]
            _note_tile_truncation(r0, max_det, trunc)
            if r0.masks is None or len(r0.masks.data) == 0:
                continue
            for m in r0.masks.data.cpu().numpy():
                mm = (m >= 0.5).astype(np.float32)
                nat = _clean_crop((mm >= 0.5).astype(np.uint8))
                if mm.shape[0] != slice_size:
                    mm = cv2.resize(mm, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
                dn = _clean_crop((mm >= 0.5).astype(np.uint8))
                if nat is None or dn is None:
                    continue
                native.append(nat)
                down.append(dn)
    _warn_if_truncated(trunc, "detect_mask_pairs")
    return native, down


from contextlib import contextmanager


@contextmanager
def _capture_process_mask(store, proto_store=None):
    """Temporarily wrap ``ultralytics.utils.ops.process_mask`` so each call ALSO appends the
    per-detection **pre-threshold probability field** to ``store`` (Test 3e — the soft mask that
    ultralytics discards: v8.4.75 thresholds the interpolated logits at 0 inside ``process_mask``,
    so nothing softer than a byte mask ever reaches ``Results``).

    The patched body replicates the original op-for-op (coeff @ protos -> box crop -> bilinear
    upsample -> ``gt_(0)``), so the returned binary masks are unchanged; the stored array is
    ``sigmoid(logits)`` gated by the interpolated box indicator (``>= 0.5``) — the gate matters
    because ``crop_mask`` zeroes *logits* outside the box, and a zero logit is probability 0.5,
    which would smear a phantom half-weight halo over the whole box. Within ~1 proto px of the box
    edge the bilinear blend with the zeroed outside still pulls probabilities toward 0.5; the box
    is the same for the hard and soft sides, so this edge zone is shared, not a confound.

    ``proto_store``, if given, additionally receives the field **before** the bilinear upsample —
    i.e. at the prototype basis's own resolution (``imgsz / 4``). That is the array the seg head
    actually produces, and therefore the one that shows *why* small-boulder axes snap to the grid:
    at the production ``slice_size=256, imgsz=1024`` there are ``256/256 = 1`` prototype cells per
    native raster pixel, so a 7-px boulder is drawn on roughly a 7x7 grid. Everything after this
    point is interpolation. Passing ``None`` (the default) leaves the original behaviour untouched.

    ``store`` may be ``None`` to skip the upsampled field entirely. That matters on dense tiles: the
    upsampled capture is an ``imgsz x imgsz`` float per detection (~4 MB each at 1024 px), so a slice
    with a hundred boulders allocates hundreds of MB of GPU memory and OOMs an 8 GB card. Callers that
    only want the prototype-resolution field (16x smaller) should pass ``store=None``.
    """
    import torch
    import torch.nn.functional as F
    import ultralytics.utils.ops as uops

    orig = uops.process_mask

    def patched(protos, masks_in, bboxes, shape, upsample=False):
        c, mh, mw = protos.shape
        logits = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
        wr, hr = mw / shape[1], mh / shape[0]
        ratios = torch.tensor([[wr, hr, wr, hr]], device=bboxes.device)
        logits = uops.crop_mask(logits, bboxes * ratios)
        ind = uops.crop_mask(torch.ones_like(logits), bboxes * ratios)
        if proto_store is not None:
            proto_store.append((logits.sigmoid() * (ind >= 0.5)).cpu().numpy())
        if upsample:
            logits = F.interpolate(logits[None], shape, mode="bilinear")[0]
            ind = F.interpolate(ind[None], shape, mode="bilinear")[0]
        if store is not None:
            store.append((logits.sigmoid() * (ind >= 0.5)).cpu().numpy())
        return logits.gt_(0.0).byte()

    uops.process_mask = patched
    try:
        yield
    finally:
        uops.process_mask = orig


def detect_soft_mask_triples(model, img, slice_size=256, imgsz=1024, conf=0.10, device=0,
                             soft_floor=0.02, max_det=ULTRALYTICS_DEFAULT_MAX_DET):
    """Tile ``img`` like :func:`detect_binary_masks` and return, per detection, the aligned triple
    ``(production, native, soft)`` from a **single forward pass** — Test 3e, isolating the 0.5
    binarisation from the estimator:

    production = the pipeline's binary mask (resize float to slice, re-threshold, largest blob,
                 fill) — byte-identical handling to :func:`detect_binary_masks`; slice res.
    native     = the binary mask at inference res (``imgsz``), before the production downscale
                 (as :func:`detect_mask_pairs`); blob/fill applied. This is exactly
                 ``soft >= 0.5`` up to the box-edge blend, at the same resolution as ``soft``.
    soft       = the pre-threshold probability field captured by :func:`_capture_process_mask`
                 (float, ``imgsz`` res, zero outside the detection box). NO blob selection or
                 hole-fill — all in-box probability mass counts, which is what a soft readout
                 would see in production.

    All three are bbox-cropped independently (orientation is translation-invariant); ``soft`` is
    cropped to its ``>= soft_floor`` support united with the native blob. A detection is kept only
    if all three sides are non-empty, so the lists align pairwise.
    """
    import cv2
    from scipy.ndimage import label, binary_fill_holes

    def _clean_crop(mb):
        lab, n = label(mb)
        if n == 0:
            return None
        if n > 1:
            big = 1 + int(np.argmax([(lab == j).sum() for j in range(1, n + 1)]))
            mb = (lab == big).astype(np.uint8)
        mb = binary_fill_holes(mb).astype(np.uint8)
        ys, xs = np.nonzero(mb)
        return mb[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()

    H, W = img.shape
    prod_l, nat_l, soft_l = [], [], []
    store = []
    trunc = _new_truncation(max_det)
    with _capture_process_mask(store):
        for y0 in range(0, H - slice_size + 1, slice_size):
            for x0 in range(0, W - slice_size + 1, slice_size):
                store.clear()
                sl = img[y0:y0 + slice_size, x0:x0 + slice_size]
                res = model(np.stack([sl] * 3, axis=-1), imgsz=imgsz, conf=conf, verbose=False,
                            device=device, max_det=max_det)
                r0 = res[0]
                _note_tile_truncation(r0, max_det, trunc)
                if r0.masks is None or len(r0.masks.data) == 0:
                    continue
                md = r0.masks.data.cpu().numpy()
                if not store or store[-1].shape[0] != md.shape[0]:
                    raise RuntimeError("process_mask capture out of sync with Results "
                                       f"({[s.shape[0] for s in store]} vs {md.shape[0]} masks)")
                for m, s in zip(md, store[-1]):
                    nat = _clean_crop((m >= 0.5).astype(np.uint8))
                    mm = (m >= 0.5).astype(np.float32)
                    if mm.shape[0] != slice_size:
                        mm = cv2.resize(mm, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
                    prod = _clean_crop((mm >= 0.5).astype(np.uint8))
                    sup = (s >= soft_floor) | (m >= 0.5)
                    if nat is None or prod is None or not sup.any():
                        continue
                    ys, xs = np.nonzero(sup)
                    prod_l.append(prod)
                    nat_l.append(nat)
                    soft_l.append(s[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                                  .astype(np.float32).copy())
    _warn_if_truncated(trunc, "detect_soft_mask_triples")
    return prod_l, nat_l, soft_l


def _largest_filled(mb):
    """Largest connected blob of a binary mask, holes filled; ``None`` if empty.

    The same clean-up every mask path in this module applies (kept module-level so
    :func:`detect_segmenter_view` uses byte-identical handling to the production sides rather than a
    lookalike). Shape is preserved — no bbox crop.
    """
    from scipy.ndimage import binary_fill_holes, label
    lab, n = label(mb)
    if n == 0:
        return None
    if n > 1:
        big = 1 + int(np.argmax([(lab == j).sum() for j in range(1, n + 1)]))
        mb = (lab == big).astype(np.uint8)
    return binary_fill_holes(mb).astype(np.uint8)


def detect_segmenter_view(model, img, slice_size=256, imgsz=1024, conf=0.10, device=0, pad_px=6,
                          max_records=None, empty_cache_every=8,
                          max_det=ULTRALYTICS_DEFAULT_MAX_DET):
    """Everything the segmenter sees for one boulder, at every resolution in the chain.

    This exists to make the orientation artifact's *mechanism* visible instead of argued. For each
    detection it returns the same rock at the four stages the pipeline puts it through:

    ``native``  the raw slice pixels around the box (``slice_size`` resolution = the raster's own);
    ``input``   the same region of the tile after the bilinear upscale to ``imgsz`` — literally the
                array the network is fed (at the production 256 -> 1024 that is a 4x resize, so it
                carries no detail the native slice did not already have);
    ``proto``   the seg head's probability field at the **prototype basis resolution**
                (``imgsz / 4``), captured before the bilinear upsample. This is the honest answer to
                "how many mask cells does the network get to describe this boulder?" — at the
                production config, ``proto_px_per_native_px == 1``, so a 7-px boulder is drawn on a
                ~7x7 grid, and its long axis can only point where such a grid allows;
    ``final``   the production binary mask (upsampled, thresholded, largest blob, filled) — the thing
                the polygon and the reported angle come from.

    Also returned per detection: ``dpx`` (native equivalent diameter), ``angle`` /``aspect`` from the
    final mask via :func:`angles_from_moments`, ``proto_angle`` from the proto field (same estimator,
    so the two are directly comparable), ``proto_cells`` (how many proto cells exceed 0.5), and the
    box. Returns ``(records, meta)``; ``meta`` records the resolution ratios so a figure can state
    them rather than hard-code them.
    """
    import cv2
    import torch

    records = []
    proto_store = []
    up = imgsz / slice_size
    proto_res = imgsz / 4.0
    H, W = img.shape
    n_slices = 0
    trunc = _new_truncation(max_det)
    # `store=None`: only the prototype-resolution field is captured. The upsampled field is 16x
    # larger and unused here, and materialising it per detection OOMs an 8 GB card on dense tiles.
    with _capture_process_mask(None, proto_store):
        for y0 in range(0, H - slice_size + 1, slice_size):
            if max_records and len(records) >= max_records:
                break
            for x0 in range(0, W - slice_size + 1, slice_size):
                if max_records and len(records) >= max_records:
                    break
                proto_store.clear()
                n_slices += 1
                if empty_cache_every and n_slices % empty_cache_every == 0 \
                        and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                sl = img[y0:y0 + slice_size, x0:x0 + slice_size]
                big = cv2.resize(sl, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
                r0 = model(np.stack([sl] * 3, axis=-1), imgsz=imgsz, conf=conf, verbose=False,
                           device=device, max_det=max_det)[0]
                _note_tile_truncation(r0, max_det, trunc)
                if r0.masks is None or len(r0.masks.data) == 0 or not proto_store:
                    continue
                md = r0.masks.data.cpu().numpy()
                # ultralytics rescales `boxes.xyxy` back to the ORIGINAL array it was handed, which
                # here is the native `slice_size` array — NOT imgsz. (Getting this wrong silently
                # crops the wrong window: the panels still look like imagery, just not of the
                # boulder, and the mask contour vanishes.) Each column then scales from native.
                boxes = r0.boxes.xyxy.cpu().numpy()
                if proto_store[-1].shape[0] != md.shape[0]:
                    raise RuntimeError("proto capture out of sync with Results "
                                       f"({proto_store[-1].shape[0]} vs {md.shape[0]})")
                for k in range(len(md)):
                    fin = _largest_filled((md[k] >= 0.5).astype(np.uint8))
                    if fin is None:
                        continue
                    mm = cv2.resize(fin.astype(np.float32), (slice_size, slice_size),
                                    interpolation=cv2.INTER_AREA)
                    final = _largest_filled((mm >= 0.5).astype(np.uint8))
                    if final is None:
                        continue
                    x1, y1, x2, y2 = boxes[k]                      # already native slice coords
                    a, b = int(max(0, x1 - pad_px)), int(max(0, y1 - pad_px))
                    c, d = (int(min(slice_size, x2 + pad_px)), int(min(slice_size, y2 + pad_px)))
                    if c - a < 4 or d - b < 4:
                        continue
                    pr = proto_store[-1][k]
                    sc = pr.shape[0] / slice_size                  # proto cells per native px
                    ang = angles_from_moments([final]).iloc[0]
                    pang = angles_from_moments([pr]).iloc[0]
                    pb, pd_ = int(np.floor(b * sc)), int(np.ceil(d * sc))
                    pa, pc = int(np.floor(a * sc)), int(np.ceil(c * sc))
                    records.append(dict(
                        native=sl[b:d, a:c].copy(),
                        input=big[int(b * up):int(d * up), int(a * up):int(c * up)].copy(),
                        proto=pr[pb:pd_, pa:pc].copy(),
                        final=final[b:d, a:c].copy(),
                        proto_cells_per_native_px=float(sc),
                        box_native=(float(x1 - a), float(y1 - b), float(x2 - a), float(y2 - b)),
                        dpx=float(ang.diameter_px), angle=float(ang.angle180),
                        aspect=float(ang.aspect), proto_angle=float(pang.angle180),
                        proto_aspect=float(pang.aspect),
                        proto_cells=int((pr >= 0.5).sum()), score=float(r0.boxes.conf[k]),
                        slice_xy=(x0, y0)))
    meta = dict(slice_size=slice_size, imgsz=imgsz, upscale=up, proto_res=proto_res,
                proto_px_per_native_px=proto_res / slice_size, n=len(records),
                n_slices=n_slices, truncated=bool(max_records and len(records) >= max_records),
                truncation=_warn_if_truncated(trunc, "detect_segmenter_view"))
    return records, meta


def detect_sam2_mask_sets(model, sam, img, slice_size=256, imgsz=1024, conf=0.10, device=0,
                          max_box_blowup=1.5, sam_batch=96,
                          configs=("yolo", "sam256", "sam1024"), empty_cache_every=1,
                          max_det=ULTRALYTICS_DEFAULT_MAX_DET):
    """Tile ``img`` like :func:`detect_binary_masks`, detect with YOLO, then **re-mask each
    detection with SAM2 box prompts** — Test 3c, the option-G (detector-preserving re-masking)
    prototype and its production path. Up to three aligned mask sets per kept detection, all from
    YOLO's OWN boxes:

    yolo    = YOLO's production binary mask (float >=0.5 -> resize to slice -> largest blob ->
              fill; byte-identical handling to :func:`detect_binary_masks`); slice res.
    sam256  = SAM2 prompted with the box on the **native tile** (masks at slice res — SAM2's
              decoder replacing YOLO's prototype head, same effective px budget).
    sam1024 = SAM2 prompted on the tile **bilinearly upscaled to** ``imgsz`` (boxes scaled with
              it) — the "more mask px per boulder" configuration that §3f's dose-response
              motivates; masks at ``imgsz`` res (orientation is scale-free).

    ``configs`` selects which sides to compute (default all three, the validation setting).
    **Production uses ``configs=("yolo", "sam256")``** — the 2026-07-06 benchmark showed
    sam256 ≈ sam1024 on orientation, and dropping the expensive 1024-px pass is ~6× faster
    (6.5 vs 39 ms/boulder → ~9 h/scene ungated vs ~51 h). ``empty_cache_every`` controls how often
    the CUDA cache is dropped (every tile by default; set higher, e.g. 64, for the sam256-only
    production run — the per-tile ``empty_cache`` sync is a measurable cost and unnecessary at the
    lighter sam256 memory footprint). Encoder-reuse via the SAM predictor's ``set_image`` was tried
    and is **slower** (framework overhead per prompt call dominates the saved re-encode), so the
    chunked ``sam(rgb, bboxes=...)`` path is kept.

    SAM2 masks get the same largest-blob + fill handling, plus a background-grab guard: a re-mask
    larger than ``max_box_blowup`` x its prompt-box area is treated as failed. A detection is kept
    only if **every requested side** yields a usable mask, so the lists align pairwise. Box prompts
    are fed to SAM2 in chunks of ``sam_batch`` (real HiRISE tiles carry 100+ detections; one giant
    prompt batch OOMs an 8 GB card). Returns ``(df, masks)`` with ``df = [cx, cy, score] +
    [area_<c> for c in configs]`` (centres in GLOBAL image px; areas in each side's own px) and
    ``masks = {c: [...] for c in configs}``. Per-side failure counts are in ``df.attrs["failed"]``.

    ``max_det`` (review item **X-06**, extended here 2026-08-06) is passed to the detector
    explicitly and its saturation recorded in ``df.attrs["truncation"]``. It was **not** passed
    before, so ultralytics' 300 applied to every tile of every scan this function produced —
    including the SAM2 equal-area rescan behind ``data/summary/fixed_area_scenes.csv``, which is
    saturated on at least one control site. See :data:`ULTRALYTICS_DEFAULT_MAX_DET` for the
    measured extent and for what it does and does not contaminate.
    """
    import cv2
    import torch
    from scipy.ndimage import label, binary_fill_holes

    realigned = {"chunks": 0}

    def _sam_masks(rgb_img, bxs):
        """One SAM mask per box in ``bxs`` **in order**, as a list of (H, W) arrays or ``None``.

        SAM2 normally returns exactly one mask per prompted box, but on some tiles (observed on
        rotated / edge crops) a batched call returns a *different* count — a dropped or merged prompt.
        Indexing a stacked array by box then either crashes or, worse, **silently misaligns** every
        mask after the drop. So on any per-chunk count mismatch we re-prompt that chunk one box at a
        time, which pins each mask to its box (``None`` where a box yields no mask). The count-matched
        path is unchanged (same masks, same order) — a no-op for the validated G runs where it never
        fires — so this only adds a slow fallback on the rare offending tile."""
        out = []
        for i in range(0, len(bxs), sam_batch):
            chunk = bxs[i:i + sam_batch]
            r = sam(rgb_img, bboxes=chunk, verbose=False, device=device)[0]
            mdc = None if r.masks is None else r.masks.data.cpu().numpy()
            if mdc is not None and len(mdc) == len(chunk):
                out.extend(list(mdc))
            else:
                realigned["chunks"] += 1
                for b in chunk:
                    rb = sam(rgb_img, bboxes=b[None], verbose=False, device=device)[0]
                    mb = None if rb.masks is None else rb.masks.data.cpu().numpy()
                    out.append(mb[0] if (mb is not None and len(mb)) else None)
        return out

    def _clean(mb):
        lab, n = label(mb)
        if n == 0:
            return None
        if n > 1:
            big = 1 + int(np.argmax([(lab == j).sum() for j in range(1, n + 1)]))
            mb = (lab == big).astype(np.uint8)
        return binary_fill_holes(mb).astype(np.uint8)

    def _crop(mb):
        ys, xs = np.nonzero(mb)
        return mb[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()

    up = imgsz // slice_size
    H, W = img.shape
    need_s256, need_s1024 = "sam256" in configs, "sam1024" in configs
    masks = {c: [] for c in configs}
    failed = {c: 0 for c in configs}
    trunc = _new_truncation(max_det)
    rows, tile_i = [], 0
    for y0 in range(0, H - slice_size + 1, slice_size):
        for x0 in range(0, W - slice_size + 1, slice_size):
            tile_i += 1
            sl = img[y0:y0 + slice_size, x0:x0 + slice_size]
            rgb = np.stack([sl] * 3, axis=-1)
            r0 = model(rgb, imgsz=imgsz, conf=conf, verbose=False, device=device,
                       max_det=max_det)[0]
            _note_tile_truncation(r0, max_det, trunc)
            if r0.masks is None or len(r0.masks.data) == 0:
                continue
            boxes = r0.boxes.xyxy.cpu().numpy()
            scores = r0.boxes.conf.cpu().numpy()
            md = r0.masks.data.cpu().numpy()
            s256 = _sam_masks(rgb, boxes) if need_s256 else None
            if need_s1024:
                rgb_up = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
                s1024 = _sam_masks(rgb_up, boxes * up)
            for k in range(len(boxes)):
                mm = (md[k] >= 0.5).astype(np.float32)
                if mm.shape[0] != slice_size:
                    mm = cv2.resize(mm, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
                box_area = ((boxes[k, 2] - boxes[k, 0]) * (boxes[k, 3] - boxes[k, 1]))
                spec = {"yolo": ((mm >= 0.5).astype(np.uint8), 1.0)}
                if need_s256:
                    spec["sam256"] = (None if s256[k] is None else s256[k].astype(np.uint8), 1.0)
                if need_s1024:
                    spec["sam1024"] = (None if s1024[k] is None else s1024[k].astype(np.uint8),
                                       float(up))
                sides = {}
                for name in configs:
                    m, scale = spec[name]
                    mb = None if m is None else _clean(m)
                    if mb is None or (name != "yolo"
                                      and mb.sum() > max_box_blowup * box_area * scale ** 2):
                        failed[name] += 1
                        sides = None
                        break
                    sides[name] = mb
                if sides is None:
                    continue
                for name in configs:
                    masks[name].append(_crop(sides[name]))
                rows.append((x0 + (boxes[k, 0] + boxes[k, 2]) / 2,
                             y0 + (boxes[k, 1] + boxes[k, 3]) / 2, float(scores[k]))
                            + tuple(int(sides[name].sum()) for name in configs))
            if torch.cuda.is_available() and (tile_i % empty_cache_every == 0):
                torch.cuda.empty_cache()
    df = pd.DataFrame(rows, columns=["cx", "cy", "score"] + [f"area_{c}" for c in configs])
    df.attrs["failed"] = failed
    df.attrs["realigned_chunks"] = realigned["chunks"]
    df.attrs["truncation"] = _warn_if_truncated(trunc, "detect_sam2_mask_sets")
    return df, masks


def largest_blob_fill(mb):
    """Largest connected component of a binary mask, holes filled; ``None`` if empty.

    Module-level since 2026-07-30 so the Goal-3 phase-5a window re-masker
    (:func:`YOLOv8BeyondEarth.remask.remask_windows`) applies the *same* mask cleanup as the
    validated in-slice G path — extracted verbatim from ``sam2_remasker``'s local ``_clean``.

    **Cropped to the mask's own bounding box first** (2026-08-04). This ran ``label`` +
    ``binary_fill_holes`` over the whole ``slice_size**2`` array for every boulder, serially and
    outside the mask->polygon thread pool, which the 2026-08-02/03 optimisation research measured at
    **1.97 ms of Solution G's 6.54 ms/boulder** — about 30% of G's CPU wall, spent almost entirely on
    background. A production boulder is a ~5-15 native-px blob inside a 256x256 mask, so the tight
    bbox is two orders of magnitude smaller than the array; it is the same ``_mask_bounds`` trick the
    tracer already uses (and this did not).

    **Bit-identical by construction, not by luck**, on three counts:

    - *Components are crop-invariant.* Everything outside the bbox is zero, so no component can be
      connected through the region removed, and the component sets are the same. Cropping preserves
      raster order, so ``label``'s first-encounter numbering is order-isomorphic and the
      lowest-index-wins tie-break between equal-area components picks the same one.
    - *Hole classification is crop-invariant.* ``binary_fill_holes`` fills background not connected
      to the array border. A background pixel on the tight bbox border is, in the full array,
      adjacent to the all-zero exterior — so it reaches the border either way and is filled in
      neither.
    - *Connectivity is untouched.* ``scipy.ndimage``'s default 2D structure is 4-connectivity for
      both calls; switching to ``cv2.connectedComponentsWithStats`` would have meant matching that
      (cv2 defaults to 8) *and* matching the tie-break, so scipy stays and the win comes from the
      crop alone. Measured on 200k synthetic masks: zero differing outputs.

    ``bincount`` replaces the per-label ``(lab == j).sum()`` loop — one pass instead of ``n``, same
    counts and the same first-max tie-break. Returns a full-size ``uint8`` {0, 1} array, as before.
    """
    from scipy.ndimage import label, binary_fill_holes
    arr = np.asarray(mb)
    bounds = _mask_bounds(arr)
    if bounds is None:                       # no foreground at all: the old `n == 0` branch
        return None
    y0, y1, x0, x1 = bounds
    crop = arr[y0:y1, x0:x1]
    lab, n = label(crop)
    if n > 1:
        counts = np.bincount(lab.ravel())
        crop = (lab == int(np.argmax(counts[1:])) + 1)
    out = np.zeros(arr.shape, dtype=np.uint8)
    out[y0:y1, x0:x1] = binary_fill_holes(crop)
    return out


#: Confidence passed to SAM2 on a **box-prompted** re-mask. Ultralytics' ``SAM.predict`` defaults it
#: to 0.25 (`models/sam/model.py:106`) and applies it to the mask decoder's *predicted-IoU* head —
#: ``idx = pred_scores > self.args.conf`` (`models/sam/predict.py:523`) — so a prompted box whose mask
#: SAM is merely *unsure* about comes back with **no mask at all**. That is wrong here on two counts:
#:
#: * **It contradicts what Solution G is.** G is *detector-preserving*: YOLO decides that a boulder is
#:   there, SAM only supplies its outline. A predicted-IoU threshold lets the mask producer silently
#:   overrule the detector, which is the very coupling review item G2-03 removed on the other side
#:   (a failed re-mask keeps the detection). Those boulders end up carrying a YOLO mask — tagged
#:   honestly as ``mask_src="yolo"``, but downgraded because of SAM's *self-report*, not because
#:   anything checked the mask.
#: * **It is redundant with a better guard.** Both G paths already gate on mask *geometry* — the
#:   background-grab guard (``mask area > max_box_blowup x box area``), ``largest_blob_fill``, and a
#:   degenerate-polygon check. Those inspect the actual mask; the IoU head only reports a hunch.
#:
#: It also carried a latent cost spike: dropping even one mask from a chunk makes the returned count
#: disagree with the prompt count, and the realignment path below then re-prompts that chunk **one box
#: at a time** — 1 + ``sam_batch`` encodes (97 at the default) where 1 would have done.
#:
#: 0.0 disables the filter without disabling anything else: ``iou_prediction_use_sigmoid=True``
#: (`models/sam/build.py`), so ``pred_scores`` is strictly positive and ``> 0.0`` always passes. The
#: realignment path stays as a safety net; it simply should no longer fire for this reason.
SAM_PROMPT_CONF = 0.0


def sam_masks_in_order(sam_model, image, boxes, sam_batch=96, device=0, conf=SAM_PROMPT_CONF):
    """One SAM mask per box **in order** — ``(masks, n_realigned_chunks)``.

    ``masks`` is a list with ``None`` where a box yielded none. SAM2 usually returns one mask per
    prompted box, but a batched call can return a *different* count on some slices — indexing a
    stacked array then silently misaligns every later box. On a per-chunk count mismatch the chunk
    is re-prompted one box at a time so alignment is exact; the count-matched path is unchanged
    (same masks, same order).

    Module-level since 2026-07-30 so the phase-5a window re-masker shares this exact prompting +
    realignment path with the validated in-slice G path — extracted verbatim from
    ``sam2_remasker``'s local ``_sam_masks`` (which now calls it).

    ``conf`` is forwarded to the model; see :data:`SAM_PROMPT_CONF` for why the default is 0.0 and
    not ultralytics' 0.25. Passing it is what makes the count mismatch — and the 97-encode
    realignment it triggers — stop happening for a reason other than luck.

    The ``boxes`` copy is not incidental. ``Predictor._prepare_prompts`` does
    ``torch.as_tensor(bboxes, dtype=..., device=...)`` then ``bboxes *= r``, and ``as_tensor`` shares
    memory whenever dtype *and* device already match — so a caller handing in a float32 CUDA tensor
    (or any float32 array on a CPU-device run) would have its own boxes rescaled in place, and the
    **second chunk** would then prompt with coordinates scaled twice. Production happens to escape
    because a host→device transfer copies, but that is an accident of where the array lives, not an
    invariant anyone stated (review item G2-11).
    """
    boxes = np.array(boxes, dtype=np.float64, copy=True)
    out, realigned = [], 0
    for i in range(0, len(boxes), sam_batch):
        chunk = boxes[i:i + sam_batch]
        r = sam_model(image, bboxes=chunk, verbose=False, device=device, conf=conf)[0]
        mdc = None if r.masks is None else r.masks.data.cpu().numpy()
        if mdc is not None and len(mdc) == len(chunk):
            out.extend(list(mdc))
        else:
            realigned += 1
            for b in chunk:
                rb = sam_model(image, bboxes=b[None], verbose=False, device=device, conf=conf)[0]
                mb = None if rb.masks is None else rb.masks.data.cpu().numpy()
                out.append(mb[0] if (mb is not None and len(mb)) else None)
    return out, realigned


def sam2_remasker(sam_model, sam_batch=96, max_box_blowup=1.5, device=0, empty_cache_every=64,
                  sam_conf=SAM_PROMPT_CONF):
    """Build a ``remask_fn`` for :func:`YOLOv8BeyondEarth.predict.get_sliced_prediction` (Solution G).

    Returns a callable ``remask(slice_image, boxes_xyxy) -> list`` that re-masks each YOLO box with
    SAM2 (the native-tile ``sam256`` production config) and returns one slice-resolution binary mask
    per box, or ``None`` where the re-mask fails a **background-grab guard** (mask area >
    ``max_box_blowup`` x its prompt-box area) or is degenerate — those detections are dropped by the
    caller. Same handling as :func:`detect_sam2_mask_sets` (largest-blob + fill), so a full-scene
    ``get_sliced_prediction(..., remask_fn=sam2_remasker(sam), mask_tag="sam2")`` reproduces the
    validated re-masking through the production slicing/edge/NMS/georef path. Box prompts are chunked
    (``sam_batch``) to fit VRAM; the CUDA cache is dropped every ``empty_cache_every`` slices (the
    per-slice ``empty_cache`` sync is a measurable cost at the light sam256 footprint).

    ``sam_conf`` is the threshold SAM applies to its own predicted-IoU head; it defaults to
    :data:`SAM_PROMPT_CONF` (0.0) rather than ultralytics' 0.25, so a prompted box is never left
    unmasked merely because SAM was unsure. The mask-quality decision stays with the geometric
    guards below, which look at the mask instead of at a self-report.

    Cost note: the "6.5 ms/boulder -> ~9 h/scene" pair from the 2026-07-06 benchmark is **not** a
    usable scene estimate. The 9 h scaled a dense-crop wall by *area*, and correcting it by boulder
    count is also wrong, because G pays a **per-slice** encode as well as a per-boulder decode and the
    6.5 ms is a blended rate from a crop 4.35x denser than a real scene. See
    ``docs/cost_model_optimization_plan.md`` (2026-08-04 audit) for the reconstructed figures.
    """
    import torch

    state = {"n": 0, "realigned": 0}

    def remask(image, boxes_xyxy):
        boxes = np.asarray(boxes_xyxy, dtype=float)
        if len(boxes) == 0:
            return []
        s, realigned = sam_masks_in_order(sam_model, image, boxes, sam_batch=sam_batch,
                                          device=device, conf=sam_conf)
        state["realigned"] += realigned
        masks = []
        for k in range(len(boxes)):
            mb = None if s[k] is None else largest_blob_fill(s[k].astype(np.uint8))
            box_area = (boxes[k, 2] - boxes[k, 0]) * (boxes[k, 3] - boxes[k, 1])
            if mb is None or mb.sum() > max_box_blowup * box_area:
                masks.append(None)                       # dropped by _process_detection
            else:
                masks.append(mb.astype(np.float32))      # slice-res {0,1}, same as YOLO masks_np
        state["n"] += 1
        if torch.cuda.is_available() and state["n"] % empty_cache_every == 0:
            torch.cuda.empty_cache()
        return masks

    # Provenance (review item A1). Solution G *is* the re-masking configuration, so which SAM and
    # which guards produced a catalog's masks has to survive onto disk — and it cannot be added
    # retroactively once the run has written. `presets._jsonable` reads `.config` in preference to
    # `__name__`, which is what puts these into the `config_hash` and so back under G1-07's
    # overwrite guard; `.inputs` is merged into the manifest's `inputs` by the caller.
    #
    # Only the *result-defining* parameters go in `.config`. `sam_batch`, `device` and
    # `empty_cache_every` are deliberately left out: `sam_masks_in_order` returns the same masks in
    # the same order whatever the chunking (that is what the one-box-at-a-time realignment exists
    # for), so folding them in would make a resume at a different batch size raise a false conflict
    # — and the documented remedy for a false conflict is `on_conflict="overwrite"`, which switches
    # the guard off altogether. A spurious conflict here costs real provenance elsewhere.
    ckpt = getattr(sam_model, "ckpt_path", None) or getattr(sam_model, "model_name", None)
    identity = presets.file_identity(ckpt) if ckpt else {"path": None, "exists": False}
    remask.config = {
        "factory": "sam2_remasker",
        # Content, not location or mtime: a checkpoint that moves between machines must keep its
        # hash, or every re-run on a second machine reads as a different configuration.
        "checkpoint": {"name": Path(ckpt).name if ckpt else None,
                       "bytes": identity.get("bytes"), "sha256": identity.get("sha256")},
        "sam_conf": float(sam_conf),
        "max_box_blowup": float(max_box_blowup),
    }
    remask.inputs = {"sam_checkpoint": identity}
    return remask


def sam2_rotation_sweep(model, sam, crops, alphas, aspect_min=1.35, small_px=8.0, sigma=4.0,
                        n_boot=300, order=3, seed=0, device=0, slice_size=256, imgsz=1024,
                        conf=0.10, sides=("yolo", "sam256")):
    """T2 — rotation invariance with SAM2 in the loop (readout B, ``remask_fn``-aware).

    For each rotation ``alpha``, rotate every crop in ``crops`` (a list of 2D uint8 arrays), run the
    production YOLO-detect + SAM2 re-mask (:func:`detect_sam2_mask_sets`) on each, read angles off
    each requested side with the SAME moment estimator (:func:`angles_from_moments`), and pool the
    elongated (``aspect > aspect_min``) angles across crops. Angles are measured in the ROTATED
    (image) frame; the geographic angle un-rotates by ``+alpha``. Both ``yolo`` and ``sam256`` are
    read from the **identical** rotated+resampled pixels, so **YOLO is the built-in positive control**
    (a known grid-locker, geo-slope ≈ +1): the resampling confound is applied to both and cancels in
    the comparison — if SAM stays content-locked where YOLO grid-locks, the difference is the mask
    producer (SAM2 decoder vs YOLO prototype head), not the rotation kernel.

    Returns ``{side: DataFrame}`` with one row per alpha:
    ``[alpha, n, raw_peak, geo_peak, geo_lo, geo_hi, grid_raw, grid_geo, diag_raw, diag_geo, fl_raw,
    fl_geo, n_sm, grid_raw_sm, grid_geo_sm, fl_raw_sm, fl_geo_sm]``. The ``raw_peak``/``geo_*``
    columns match :func:`transforms_test.population_rotation_sweep` so
    :func:`transforms_test.rotation_slope` / :func:`plot_rotation_sweep` apply unchanged (geo-slope
    0 = tracks content, +1 = re-imposed on the grid). ``grid_raw``/``grid_geo`` are the cardinal
    :func:`orientation.grid_fraction` in the image and geographic frames: a **grid-locked** producer
    pins the cardinal excess to pixels (``grid_raw`` flat, ``grid_geo`` varies with α); a
    **content-locked** one does the reverse (``grid_geo`` flat). ``diag_*`` is the 45/135 analogue
    (YOLO's axis) and ``fl_*`` the half-plane :func:`orientation.half_plane_fraction` (``f_lower``,
    0–90 vs 90–180) in each frame — same frame logic: a pixel-locked half-plane imbalance keeps
    ``fl_raw`` off-0.5 while ``fl_geo`` relaxes, a ground-locked one the reverse (the rotation
    complement to T4a's mirror-flip test). ``*_sm`` restricts to the small (< ``small_px`` native-px)
    subset where T1's cardinal excess lives — the sharpest test of whether *that* residual grid-locks.
    """
    from .transforms_test import rotate_image
    from .orientation import (bootstrap_peak_ci, grid_fraction, diagonal_fraction,
                              half_plane_fraction)

    out = {s: [] for s in sides}
    realigned_total = 0
    for a in alphas:
        pooled = {s: [] for s in sides}
        for crop in crops:
            rimg = rotate_image(crop, a, order=order)
            df, masks = detect_sam2_mask_sets(model, sam, rimg, slice_size=slice_size, imgsz=imgsz,
                                              conf=conf, device=device, configs=sides)
            realigned_total += int(df.attrs.get("realigned_chunks", 0))
            for s in sides:
                ang = angles_from_moments(masks[s])
                el = ang[(ang.aspect > aspect_min) & np.isfinite(ang.angle180)]
                if len(el):
                    pooled[s].append((el.angle180.values, el.diameter_px.values))
        for s in sides:
            if pooled[s]:
                raw_ang = np.concatenate([p[0] for p in pooled[s]])
                dpx = np.concatenate([p[1] for p in pooled[s]])
            else:
                raw_ang = np.array([]); dpx = np.array([])
            if len(raw_ang) == 0:
                out[s].append((float(a), 0) + (np.nan,) * 10 + (0,) + (np.nan,) * 4)
                continue
            geo_ang = (raw_ang + a) % 180.0
            raw, lo, hi = bootstrap_peak_ci(raw_ang, n_boot=n_boot, sigma=sigma, seed=seed)
            sm = dpx < small_px
            row = [float(a), len(raw_ang), float(raw), float((raw + a) % 180.0),
                   float((lo + a) % 180.0), float((hi + a) % 180.0),
                   grid_fraction(raw_ang), grid_fraction(geo_ang),
                   diagonal_fraction(raw_ang), diagonal_fraction(geo_ang),
                   half_plane_fraction(raw_ang), half_plane_fraction(geo_ang), int(sm.sum())]
            if sm.sum() >= 20:
                row += [grid_fraction(raw_ang[sm]), grid_fraction(geo_ang[sm]),
                        half_plane_fraction(raw_ang[sm]), half_plane_fraction(geo_ang[sm])]
            else:
                row += [np.nan, np.nan, np.nan, np.nan]
            out[s].append(tuple(row))
    cols = ["alpha", "n", "raw_peak", "geo_peak", "geo_lo", "geo_hi", "grid_raw", "grid_geo",
            "diag_raw", "diag_geo", "fl_raw", "fl_geo", "n_sm", "grid_raw_sm", "grid_geo_sm",
            "fl_raw_sm", "fl_geo_sm"]
    res = {s: pd.DataFrame(out[s], columns=cols) for s in sides}
    for s in sides:
        res[s].attrs["realigned_chunks"] = realigned_total
    return res


def detect_obb_angles(model, img, slice_size=256, imgsz=1024, conf=0.10, device=0,
                      max_det=ULTRALYTICS_DEFAULT_MAX_DET):
    """Solution H-obb orientation readout. Tile ``img`` exactly like :func:`detect_sam2_mask_sets`
    (non-overlapping ``slice_size`` windows, each upscaled to ``imgsz``, same ``conf``), run the
    trained **OBB** model per tile, and read each detection's long-axis azimuth **straight from the
    predicted oriented box** (``result.obb.xyxyxyxy`` → :func:`orientation.obb_angle180`). There is
    NO mask or polygon anywhere in the angle path — the whole point of the OBB arm.

    Returns ``[angle180, aspect, diameter_px]`` in the same geographic frame as the mask readouts
    (North=0, East=90, [0,180)), so the diagonal-fraction is directly comparable to YOLO's and G's
    numbers on the same scene. ``diameter_px`` is the box's equivalent-area diameter in native px
    (``2*sqrt(w*h/pi)`` from ``obb.xywhr``); use it with the elongated + well-resolved subset cut.
    """
    from .orientation import obb_angle180
    H, W = img.shape
    rows = []
    trunc = _new_truncation(max_det)
    for y0 in range(0, H - slice_size + 1, slice_size):
        for x0 in range(0, W - slice_size + 1, slice_size):
            sl = img[y0:y0 + slice_size, x0:x0 + slice_size]
            rgb = np.stack([sl] * 3, axis=-1)
            r0 = model(rgb, imgsz=imgsz, conf=conf, verbose=False, device=device,
                       max_det=max_det)[0]
            _note_tile_truncation(r0, max_det, trunc)
            if r0.obb is None or len(r0.obb) == 0:
                continue
            corners = r0.obb.xyxyxyxy.cpu().numpy()      # (n, 4, 2) (col,row) in slice coords
            wh = r0.obb.xywhr.cpu().numpy()[:, 2:4]       # (n, 2) w,h in native slice px
            for k in range(len(corners)):
                ang, asp = obb_angle180(corners[k])
                if not np.isfinite(ang):
                    continue
                dpx = 2.0 * np.sqrt(max(float(wh[k, 0] * wh[k, 1]), 0.0) / np.pi)
                rows.append((ang, asp, dpx))
    df = pd.DataFrame(rows, columns=["angle180", "aspect", "diameter_px"])
    df.attrs["truncation"] = _warn_if_truncated(trunc, "detect_obb_angles")
    return df


def angles_from_moments(masks):
    """Orientation from **weighted second moments** — no contour, no ellipse fit, and no threshold
    requirement, so it reads binary and soft (float-probability) masks with the SAME estimator.
    Returns ``[angle180, aspect, diameter_px]`` in the frame of :func:`angles_from_masks`
    (geographic azimuth from North in the image's own frame, [0, 180)); rows align with ``masks``
    (degenerate inputs are NaN, not dropped) for pairwise comparison across mask flavours.

    The long axis is the principal eigenvector of the weight-covariance of pixel centres,
    ``aspect = sqrt(l1/l2)``, and ``diameter_px = 2*sqrt(sum(w)/pi)`` (for binary weights the
    usual area-equivalent diameter; for soft weights the probability mass plays the role of area).
    """
    rows = []
    for w in masks:
        w = np.asarray(w, dtype=float)
        tot = w.sum()
        if tot <= 0:
            rows.append((np.nan, np.nan, np.nan))
            continue
        ys, xs = np.mgrid[0:w.shape[0], 0:w.shape[1]]
        yg = -ys.astype(float)                      # geographic y = -row (North up)
        mx = (w * xs).sum() / tot
        my = (w * yg).sum() / tot
        cxx = (w * (xs - mx) ** 2).sum() / tot
        cyy = (w * (yg - my) ** 2).sum() / tot
        cxy = (w * (xs - mx) * (yg - my)).sum() / tot
        half = np.hypot((cxx - cyy) / 2, cxy)
        l1 = (cxx + cyy) / 2 + half
        l2 = (cxx + cyy) / 2 - half
        if l1 <= 0:
            rows.append((np.nan, np.nan, np.nan))
            continue
        phi = 0.5 * np.arctan2(2 * cxy, cxx - cyy)  # major axis from +x, CCW, geographic frame
        rows.append((float((90.0 - np.degrees(phi)) % 180.0),
                     float(np.sqrt(l1 / max(l2, 1e-12))),
                     float(2 * np.sqrt(tot / np.pi))))
    return pd.DataFrame(rows, columns=["angle180", "aspect", "diameter_px"])


def angles_from_masks(masks, method="skimage"):
    """Trace each binary mask with the chosen tracer and return ``[angle180, aspect, diameter_px]``.

    The CPU-only second half of :func:`detect_orientations`, factored out so the SAME masks can be
    measured under both tracers — skimage marching-squares (`binary_mask_to_polygon`) vs cv2
    border-following (`binary_mask_to_polygon_cv`). Rows align with ``masks`` (failed traces/fits
    are NaN, not dropped) so the two tracers can be compared **pairwise per mask**.

    NOTE: unlike :func:`detect_orientations` this does NOT drop polygons with < 5 vertices — cv2's
    CHAIN_APPROX_SIMPLE legitimately returns 4-corner outlines for the tiny blocky blobs that are
    exactly the Test 3d subject, and ``ellipse_angle180`` densifies (segmentize) before fitting, so
    a 4-vertex square is a valid input, not a degenerate one.

    Routed through :func:`orientation.mask_readout` (G2-02), so the tracer comparison runs on the
    production readout rather than a local copy of it.
    """
    from .orientation import mask_readout

    rows = [mask_readout(mb, estimator="ellipse", contour_method=method) for mb in masks]
    return pd.DataFrame(rows, columns=["angle180", "aspect", "diameter_px"])


def _central_blob(bw, cy, cx):
    """Connected component of boolean ``bw`` at (or nearest) the centre pixel ``(cy, cx)`` — the
    target boulder rather than a clipped neighbour that crept into the window. Returns ``bw``
    unchanged when there is 0 or 1 component."""
    from scipy import ndimage as ndi

    lab, n = ndi.label(bw)
    if n <= 1:
        return bw
    cid = 0
    if 0 <= cy < lab.shape[0] and 0 <= cx < lab.shape[1]:
        cid = int(lab[cy, cx])
    if cid == 0:                                    # centre off a blob -> nearest labelled pixel
        ys, xs = np.nonzero(lab)
        j = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        cid = int(lab[ys[j], xs[j]])
    return lab == cid


def rock_trace_orientations(img, truth, bg, cap, cap_frac=0.5, win_frac=2.5):
    """Rock-only "human-prior" proxy orientations for a synthetic field (the illumination control).

    A human tracing a boulder outlines the sunlit **rock** and discounts its cast shadow; this proxy
    does the same on a :func:`~YOLOv8BeyondEarth.transforms_test.render_boulder_field` image by
    keeping only pixels brighter than ``bg + cap_frac * cap`` — a cut above both the background
    (~``bg``) and the darker cast shadow (~``bg + shadow``), so the shadow is excluded by
    construction. For each **known** truth boulder it takes the connected blob at the truth centroid
    within a window of half-size ``win_frac * a_px``, fills holes, and reads it with
    :func:`angles_from_moments` — the *same* estimator used on the SAM and YOLO masks, so the three
    maskers are directly comparable. Returns a DataFrame ``[angle180, aspect, diameter_px]`` aligned
    row-for-row with ``truth`` (empty/degenerate → NaN): what a shadow-discounting reader recovers.
    Its departure from ``truth.angle180`` is the shaping any rock-tracer still incurs, and
    SAM-minus-proxy is the SAM-specific excess.

    ⚠ **This proxy is NOT the shadow-free endpoint it was billed as** (review pass 9, 2026-08-07).
    "The shadow is excluded by construction" is true of the *threshold* and false of the *result*:
    the proxy's departure from truth appears **only when a shadow is drawn**. Re-rendered
    proxy-minus-truth pull (n ≈ 2,930 per incidence, se ≈ 0.08) is **+0.070 ± 0.085 with no shadow** —
    consistent with zero — against **−0.158 / −0.373 / −0.395 / −0.107°** at incidence 30 / 42 / 58 /
    68°. So ≈0.23° of its pull is a shadow-induced artifact of the proxy itself, and only ~55–60% is
    the detection-selection effect it is credited with.

    **Consequence for callers:** ``SAM − proxy`` is *not* a clean illumination-only metric — it
    **overstates** the SAM-specific excess by ≈0.23°, so published values around 0.5–0.7° correspond
    to ~0.3–0.5° against truth. The error is conservative, so conclusions bounded by it survive.
    Where truth is available (it is, on synthetic fields), prefer **SAM − truth**.
    """
    from scipy import ndimage as ndi

    im = np.asarray(img, dtype=float)
    H, W = im.shape
    thr = float(bg) + float(cap_frac) * float(cap)
    masks = []
    for r in truth.itertuples(index=False):
        half = max(int(round(win_frac * float(r.a_px))), 3)
        cx, cy = int(round(float(r.cx))), int(round(float(r.cy)))
        x0, x1 = max(cx - half, 0), min(cx + half + 1, W)
        y0, y1 = max(cy - half, 0), min(cy + half + 1, H)
        crop = im[y0:y1, x0:x1]
        bw = crop >= thr
        if not bw.any():
            masks.append(np.zeros(crop.shape, dtype=np.uint8))
            continue
        blob = _central_blob(bw, cy - y0, cx - x0)
        masks.append(ndi.binary_fill_holes(blob).astype(np.uint8))
    return angles_from_moments(masks)


def center_orientation(model, patch, imgsz=1024, conf=0.10, device=0, max_center_dist=40):
    """Run YOLO on a square ``patch`` and return ``(angle180, aspect)`` of the detection nearest
    the patch centre, or ``None``. The geographic azimuth is measured in the *patch's own* frame.

    Used by the paired transform/equivariance test (§8.5): feed the same boulder's patch, its
    flip, and its 90° rotation, then map the angles back to compare. Same mask handling as
    :func:`true_mask_for_point` (>=0.5 -> resize -> largest blob -> fill holes -> contour).
    """
    import cv2
    import shapely
    from scipy.ndimage import label, binary_fill_holes
    from .polygon import binary_mask_to_polygon

    ss = patch.shape[0]
    res = model(np.stack([patch] * 3, axis=-1), imgsz=imgsz, conf=conf, verbose=False, device=device)
    r0 = res[0]
    if r0.masks is None or len(r0.masks.data) == 0:
        return None
    md = r0.masks.data.cpu().numpy()
    bx = r0.boxes.xywh.cpu().numpy()
    k = int(np.argmin(np.hypot(bx[:, 0] - ss / 2, bx[:, 1] - ss / 2)))
    if np.hypot(bx[k, 0] - ss / 2, bx[k, 1] - ss / 2) > max_center_dist:
        return None
    m = (md[k] >= 0.5).astype(np.float32)
    if m.shape[0] != ss:
        m = cv2.resize(m, (ss, ss), interpolation=cv2.INTER_AREA)
    mb = (m >= 0.5).astype(np.uint8)
    lab, n = label(mb)
    if n == 0:
        return None
    if n > 1:
        big = 1 + int(np.argmax([(lab == j).sum() for j in range(1, n + 1)]))
        mb = (lab == big).astype(np.uint8)
    mb = binary_fill_holes(mb).astype(np.uint8)
    poly = binary_mask_to_polygon(mb)
    if poly is None or len(poly) < 5:
        return None
    ang, asp = ellipse_angle180(shapely.geometry.Polygon(np.c_[poly[:, 0], -poly[:, 1]]), 1.0)
    return (ang, asp) if np.isfinite(ang) else None


def true_mask_for_point(model, dataset, x, y, slice_size=256, imgsz=1024, conf=0.10, device=0,
                        max_center_dist=None):
    """Re-run YOLO on a ``slice_size`` window centred on map point ``(x, y)`` and return the
    **true mask** (as the pipeline uses it) of the detection nearest the window centre — for the
    §9 worked-example panels (mask → polygon → ellipse → orientation).

    Reproduces the production path: feed a ``slice_size`` crop at ``imgsz``, threshold the mask at
    0.5, resize to ``slice_size`` (the pipeline's ``downscale_pred``), then contour it. Geographic
    long-axis azimuth + aspect are taken with the same :func:`ellipse_angle180` used everywhere
    else (north-up square-pixel raster, so the pixel frame's azimuth equals the map azimuth).

    Returns a dict ``{patch, mask, polygon (col,row), centroid (col,row), angle180, aspect,
    ellipse=(xc,yc,a,b,theta) in image space, score}`` or ``None`` if nothing is detected /
    contourable near the centre. ``model`` is an ultralytics ``YOLO``; ``dataset`` an open rasterio
    handle.
    """
    import cv2
    import shapely
    from rasterio.windows import Window
    from scipy.ndimage import label, binary_fill_holes
    from skimage.measure import EllipseModel
    from .polygon import binary_mask_to_polygon

    col0, row0 = ~dataset.transform * (x, y)
    c0, r0 = int(round(col0)), int(round(row0))
    half = slice_size // 2
    patch = dataset.read(1, window=Window(c0 - half, r0 - half, slice_size, slice_size),
                         boundless=True, fill_value=0).astype(np.uint8)
    rgb = np.stack([patch] * 3, axis=-1)                  # 3-channel grayscale (BGR==RGB)
    res = model(rgb, imgsz=imgsz, conf=conf, verbose=False, device=device)
    r0res = res[0]
    if r0res.masks is None or len(r0res.masks.data) == 0:
        return None
    md = r0res.masks.data.cpu().numpy()
    bx = r0res.boxes.xywh.cpu().numpy()
    d = np.hypot(bx[:, 0] - slice_size / 2, bx[:, 1] - slice_size / 2)
    k = int(np.argmin(d))
    if max_center_dist is not None and d[k] > max_center_dist:
        return None
    m = (md[k] >= 0.5).astype(np.float32)
    if m.shape[0] != slice_size:
        m = cv2.resize(m, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
    mask = (m >= 0.5).astype(np.uint8)
    lab, nlab = label(mask)                               # keep only the largest connected blob
    if nlab == 0:
        return None
    if nlab > 1:
        biggest = 1 + int(np.argmax([(lab == j).sum() for j in range(1, nlab + 1)]))
        mask = (lab == biggest).astype(np.uint8)
    mask = binary_fill_holes(mask).astype(np.uint8)      # solid boulder -> single clean contour
    poly = binary_mask_to_polygon(mask)                  # (col, row) in patch frame
    if poly is None or len(poly) < 5:
        return None
    col, row = poly[:, 0], poly[:, 1]
    ang, asp = ellipse_angle180(shapely.geometry.Polygon(np.c_[col, -row]), 1.0)  # y=North -> -row
    em = EllipseModel()
    ell = em.params if em.estimate(np.c_[col, row]) else None   # image-space draw params
    cen = (float(ell[0]), float(ell[1])) if ell is not None else (float(col.mean()), float(row.mean()))
    return dict(patch=patch, mask=mask, polygon=poly, centroid=cen, angle180=ang, aspect=asp,
                ellipse=ell, score=float(r0res.boxes.conf[k]))


def centroid_direction_hist(gpkg_path, scene_bounds, win=2000.0, n_win=12, k=8,
                            dmin=1.5, dmax=30.0, min_count=2000, seed=3, bin_deg=10):
    """Nearest-neighbour direction histogram of boulder centroids (positions only).

    Reads all boulders in ``n_win`` dense ``win``-metre windows (no subsampling — NN geometry
    needs complete local sampling), and for each boulder accumulates the azimuths of its ``k``
    nearest neighbours within the distance band ``[dmin, dmax]`` metres. Illumination-immune and
    independent of mask shape: a real structural fabric -> directional excess; image/scan
    artifacts -> 0/90. ``scene_bounds`` = ``(left, bottom, right, top)`` in the gpkg CRS.

    Returns ``(centers, percent_hist, per_window_peaks)``.
    """
    from pyogrio import read_dataframe
    from scipy.spatial import cKDTree

    left, bottom, right, top = scene_bounds
    edges = np.arange(0, 181, bin_deg)
    centers = (edges[:-1] + edges[1:]) / 2
    rng = np.random.default_rng(seed)
    agg = np.zeros(len(centers)); peaks = []; n = 0
    for _ in range(n_win * 4):
        if n >= n_win:
            break
        x0 = rng.uniform(left, right - win); y0 = rng.uniform(bottom, top - win)
        g = read_dataframe(gpkg_path, bbox=(x0, y0, x0 + win, y0 + win), columns=["id"])
        if len(g) < min_count:
            continue
        n += 1
        cen = np.c_[g.geometry.centroid.x.values, g.geometry.centroid.y.values]
        tree = cKDTree(cen)
        dist, idx = tree.query(cen, k=k + 1)               # col 0 is self
        dx = cen[idx[:, 1:], 0] - cen[:, [0]]              # East component
        dy = cen[idx[:, 1:], 1] - cen[:, [1]]              # North component
        band = (dist[:, 1:] >= dmin) & (dist[:, 1:] <= dmax)
        az = np.degrees(np.arctan2(dx[band], dy[band])) % 180.0
        h, _ = np.histogram(az, bins=edges)
        hpct = 100 * h / h.sum()
        agg += hpct; peaks.append(float(centers[np.argmax(hpct)]))
    return centers, agg / max(n, 1), peaks
