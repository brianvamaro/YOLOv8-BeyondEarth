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
