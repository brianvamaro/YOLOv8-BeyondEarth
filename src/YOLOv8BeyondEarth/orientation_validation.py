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
import numpy as np
import pandas as pd

from .orientation import ellipse_angle180, structure_tensor_orientation, orientation_field


def matched_area_ratio(model_gdf, ref_gdf, rom=None, min_radius_m=1.0):
    """Per-boulder mask-area ratio of ``model_gdf`` predictions against a ``ref_gdf`` reference
    (human ROM outlines, or a second model), matched by nearest centroid — the mask-quality /
    over-vs-under-segmentation readout for the Goal-2 re-masking work (Test 3c treatment stage).

    Both GeoDataFrames must already be on the SAME grid (use ``set_crs(..., allow_override=True)``,
    never ``to_crs`` — the Prieur shp vs project gpkg carry cosmetically-different CRS strings on the
    same equirectangular grid; reprojecting throws a bogus datum shift). If ``rom`` (a GeoDataFrame
    or geometry) is given, both sets are first clipped to boulders whose centroid falls inside it.

    Each reference polygon is matched to the nearest model centroid, accepted if within
    ``max(ref_equiv_radius, min_radius_m)`` (map units). Returns
    ``(df, stats)`` where ``df`` has one row per matched reference boulder
    (``ref_area, model_area, ratio, ref_dia``) and ``stats`` is a dict with match/recall/
    over-detection fractions. Ratio > 1 ⇒ model masks larger than reference (YOLO overshoot);
    < 1 ⇒ tighter/under-segmenting. NOTE: Mars human outlines include shadow, inflating ``ref_area``
    — so the *true* boulder-area ratio is larger than reported here (read as a lower bound).
    """
    from scipy.spatial import cKDTree
    import shapely

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
    tree = cKDTree(mc)
    dist, idx = tree.query(rc, k=1)
    ref_rad = r["_dia"].to_numpy() / 2
    ok = dist <= np.maximum(ref_rad, min_radius_m)
    ref_a = r["_area"].to_numpy()[ok]
    mod_a = m["_area"].to_numpy()[idx[ok]]
    df = pd.DataFrame(dict(ref_area=ref_a, model_area=mod_a, ratio=mod_a / ref_a,
                           ref_dia=r["_dia"].to_numpy()[ok]))
    # over-detection: model polys with no reference within their own radius
    rdist, _ = cKDTree(rc).query(mc, k=1)
    over = rdist > np.maximum(m["_dia"].to_numpy() / 2, min_radius_m)
    stats = dict(n_ref=len(r), n_model=len(m), n_matched=int(ok.sum()),
                 recall=float(ok.mean()), median_ratio=float(np.median(df.ratio)),
                 over_detection_frac=float(over.mean()))
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


def detect_orientations(model, img, slice_size=256, imgsz=1024, conf=0.10, device=0,
                        aspect_min=1.0):
    """Tile a raw image array into ``slice_size`` windows, run YOLO on each, and return a
    DataFrame ``[angle180, aspect, diameter_px]`` for every detection (geographic long-axis
    azimuth measured in the *image's own* frame).

    A minimal reimplementation of the sliced pipeline (mask >=0.5 -> resize to slice -> largest
    connected blob -> contour -> EllipseModel), without NMS/edge handling — enough to get an
    orientation *distribution* for transform / rotation tests on a single array. ``img`` is a 2D
    uint8/0-255 array; non-overlapping tiles keep double-counting negligible.
    """
    import cv2
    import shapely
    from scipy.ndimage import label, binary_fill_holes
    from .polygon import binary_mask_to_polygon

    H, W = img.shape
    rows = []
    for y0 in range(0, H - slice_size + 1, slice_size):
        for x0 in range(0, W - slice_size + 1, slice_size):
            sl = img[y0:y0 + slice_size, x0:x0 + slice_size]
            res = model(np.stack([sl] * 3, axis=-1), imgsz=imgsz, conf=conf, verbose=False, device=device)
            r0 = res[0]
            if r0.masks is None or len(r0.masks.data) == 0:
                continue
            for m in r0.masks.data.cpu().numpy():
                mm = (m >= 0.5).astype(np.float32)
                if mm.shape[0] != slice_size:
                    mm = cv2.resize(mm, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
                mb = (mm >= 0.5).astype(np.uint8)
                lab, n = label(mb)
                if n == 0:
                    continue
                if n > 1:
                    big = 1 + int(np.argmax([(lab == j).sum() for j in range(1, n + 1)]))
                    mb = (lab == big).astype(np.uint8)
                mb = binary_fill_holes(mb).astype(np.uint8)        # solid boulder -> single contour
                poly = binary_mask_to_polygon(mb)
                if poly is None or len(poly) < 5:
                    continue
                a, asp = ellipse_angle180(shapely.geometry.Polygon(np.c_[poly[:, 0], -poly[:, 1]]), 1.0)
                if np.isfinite(asp) and asp >= aspect_min:
                    rows.append((a, asp, 2 * np.sqrt(int(mb.sum()) / np.pi)))
    return pd.DataFrame(rows, columns=["angle180", "aspect", "diameter_px"])


def detect_binary_masks(model, img, slice_size=256, imgsz=1024, conf=0.10, device=0):
    """Tile ``img`` like :func:`detect_orientations` but return every detection's processed
    **binary mask** (bbox-cropped) instead of tracing it — so different mask→polygon tracers can
    be compared on the *same* masks (Test 3d, the contour-tracer chirality swap).

    Mask handling is identical to :func:`detect_orientations` (>=0.5 -> resize to slice ->
    largest blob -> fill holes). Cropping to the blob's bounding box (both tracers crop
    internally anyway) keeps a cached run small; orientation is translation-invariant.
    """
    import cv2
    from scipy.ndimage import label, binary_fill_holes

    H, W = img.shape
    masks = []
    for y0 in range(0, H - slice_size + 1, slice_size):
        for x0 in range(0, W - slice_size + 1, slice_size):
            sl = img[y0:y0 + slice_size, x0:x0 + slice_size]
            res = model(np.stack([sl] * 3, axis=-1), imgsz=imgsz, conf=conf, verbose=False,
                        device=device)
            r0 = res[0]
            if r0.masks is None or len(r0.masks.data) == 0:
                continue
            for m in r0.masks.data.cpu().numpy():
                mm = (m >= 0.5).astype(np.float32)
                if mm.shape[0] != slice_size:
                    mm = cv2.resize(mm, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
                mb = (mm >= 0.5).astype(np.uint8)
                lab, n = label(mb)
                if n == 0:
                    continue
                if n > 1:
                    big = 1 + int(np.argmax([(lab == j).sum() for j in range(1, n + 1)]))
                    mb = (lab == big).astype(np.uint8)
                mb = binary_fill_holes(mb).astype(np.uint8)
                ys, xs = np.nonzero(mb)
                masks.append(mb[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy())
    return masks


def detect_mask_pairs(model, img, slice_size=256, imgsz=1024, conf=0.10, device=0):
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
    for y0 in range(0, H - slice_size + 1, slice_size):
        for x0 in range(0, W - slice_size + 1, slice_size):
            sl = img[y0:y0 + slice_size, x0:x0 + slice_size]
            res = model(np.stack([sl] * 3, axis=-1), imgsz=imgsz, conf=conf, verbose=False,
                        device=device)
            r0 = res[0]
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
    return native, down


from contextlib import contextmanager


@contextmanager
def _capture_process_mask(store):
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
        if upsample:
            logits = F.interpolate(logits[None], shape, mode="bilinear")[0]
            ind = F.interpolate(ind[None], shape, mode="bilinear")[0]
        store.append((logits.sigmoid() * (ind >= 0.5)).cpu().numpy())
        return logits.gt_(0.0).byte()

    uops.process_mask = patched
    try:
        yield
    finally:
        uops.process_mask = orig


def detect_soft_mask_triples(model, img, slice_size=256, imgsz=1024, conf=0.10, device=0,
                             soft_floor=0.02):
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
    with _capture_process_mask(store):
        for y0 in range(0, H - slice_size + 1, slice_size):
            for x0 in range(0, W - slice_size + 1, slice_size):
                store.clear()
                sl = img[y0:y0 + slice_size, x0:x0 + slice_size]
                res = model(np.stack([sl] * 3, axis=-1), imgsz=imgsz, conf=conf, verbose=False,
                            device=device)
                r0 = res[0]
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
    return prod_l, nat_l, soft_l


def detect_sam2_mask_sets(model, sam, img, slice_size=256, imgsz=1024, conf=0.10, device=0,
                          max_box_blowup=1.5):
    """Tile ``img`` like :func:`detect_binary_masks`, detect with YOLO, then **re-mask each
    detection with SAM2 box prompts** — Test 3c, the option-G (detector-preserving re-masking)
    prototype. Three aligned mask sets per kept detection, all from YOLO's OWN boxes:

    yolo    = YOLO's production binary mask (float >=0.5 -> resize to slice -> largest blob ->
              fill; byte-identical handling to :func:`detect_binary_masks`); slice res.
    sam256  = SAM2 prompted with the box on the **native tile** (masks at slice res — SAM2's
              decoder replacing YOLO's prototype head, same effective px budget).
    sam1024 = SAM2 prompted on the tile **bilinearly upscaled to** ``imgsz`` (boxes scaled with
              it) — the "more mask px per boulder" configuration that §3f's dose-response
              motivates; masks at ``imgsz`` res (orientation is scale-free).

    SAM2 masks get the same largest-blob + fill handling, plus a background-grab guard: a re-mask
    larger than ``max_box_blowup`` x its prompt-box area is treated as failed. A detection is kept
    only if all three sides yield a usable mask, so the lists align pairwise. Returns
    ``(df, masks)`` with ``df = [cx, cy, score, area_yolo, area_sam256, area_sam1024]`` (centres
    in GLOBAL image px, for truth-matching on known-orientation fields; areas in each side's own
    px) and ``masks = {"yolo": [...], "sam256": [...], "sam1024": [...]}``. Per-side failure
    counts (before completeness pairing) are in ``df.attrs["failed"]``.
    """
    import cv2
    from scipy.ndimage import label, binary_fill_holes

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
    rows, masks = [], {"yolo": [], "sam256": [], "sam1024": []}
    failed = {"yolo": 0, "sam256": 0, "sam1024": 0}
    for y0 in range(0, H - slice_size + 1, slice_size):
        for x0 in range(0, W - slice_size + 1, slice_size):
            sl = img[y0:y0 + slice_size, x0:x0 + slice_size]
            rgb = np.stack([sl] * 3, axis=-1)
            r0 = model(rgb, imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
            if r0.masks is None or len(r0.masks.data) == 0:
                continue
            boxes = r0.boxes.xyxy.cpu().numpy()
            scores = r0.boxes.conf.cpu().numpy()
            md = r0.masks.data.cpu().numpy()
            rgb_up = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            s256 = sam(rgb, bboxes=boxes, verbose=False, device=device)[0]
            s1024 = sam(rgb_up, bboxes=boxes * up, verbose=False, device=device)[0]
            s256 = s256.masks.data.cpu().numpy()
            s1024 = s1024.masks.data.cpu().numpy()
            for k in range(len(boxes)):
                mm = (md[k] >= 0.5).astype(np.float32)
                if mm.shape[0] != slice_size:
                    mm = cv2.resize(mm, (slice_size, slice_size), interpolation=cv2.INTER_AREA)
                box_area = ((boxes[k, 2] - boxes[k, 0]) * (boxes[k, 3] - boxes[k, 1]))
                sides = {}
                for name, m, scale in (("yolo", (mm >= 0.5).astype(np.uint8), 1.0),
                                       ("sam256", s256[k].astype(np.uint8), 1.0),
                                       ("sam1024", s1024[k].astype(np.uint8), float(up))):
                    mb = _clean(m)
                    if mb is None or (name != "yolo"
                                      and mb.sum() > max_box_blowup * box_area * scale ** 2):
                        failed[name] += 1
                        sides = None
                        break
                    sides[name] = mb
                if sides is None:
                    continue
                for name in masks:
                    masks[name].append(_crop(sides[name]))
                rows.append((x0 + (boxes[k, 0] + boxes[k, 2]) / 2,
                             y0 + (boxes[k, 1] + boxes[k, 3]) / 2, float(scores[k]),
                             int(sides["yolo"].sum()), int(sides["sam256"].sum()),
                             int(sides["sam1024"].sum())))
    df = pd.DataFrame(rows, columns=["cx", "cy", "score",
                                     "area_yolo", "area_sam256", "area_sam1024"])
    df.attrs["failed"] = failed
    return df, masks


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
    """
    import shapely
    from .polygon import binary_mask_to_polygon, binary_mask_to_polygon_cv
    from .orientation import ellipse_angle180

    trace = binary_mask_to_polygon if method == "skimage" else binary_mask_to_polygon_cv
    rows = []
    for mb in masks:
        poly = trace(mb)
        if poly is None or len(poly) < 3:
            rows.append((np.nan, np.nan, np.nan))
            continue
        ang, asp = ellipse_angle180(shapely.geometry.Polygon(np.c_[poly[:, 0], -poly[:, 1]]), 1.0)
        rows.append((ang, asp, 2 * np.sqrt(int(mb.sum()) / np.pi)))
    return pd.DataFrame(rows, columns=["angle180", "aspect", "diameter_px"])


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
