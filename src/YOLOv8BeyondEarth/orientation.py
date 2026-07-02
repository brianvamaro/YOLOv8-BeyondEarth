"""Boulder long-axis orientation from predicted polygons, plus rose-diagram / grid-fraction
helpers. Consolidates logic that was previously inline in the Goal-2 notebooks.

Orientation is measured with skimage ``EllipseModel`` (a least-squares ellipse fit), which is the
**unbiased** estimator. The minimum-rotated-rectangle azimuth used by ``geomorph.boulder`` is
grid-biased on pixelated masks and must NOT be used for orientation (see
``docs/orientation_investigation.md``). ``angle180`` here matches the shptools
``ellipse -> geomorph.boulder`` convention: long-axis azimuth from **North**, clockwise, in
[0, 180) degrees (North=0, East=90).
"""
import numpy as np
import pandas as pd
import shapely
from skimage.measure import EllipseModel
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor


def ellipse_angle180(geom, res):
    """Fit an EllipseModel to a boulder polygon and return ``(angle180, aspect_ratio)``.

    The polygon is first densified to ``res`` spacing (``shapely.segmentize``) so the fit is not
    dominated by uneven vertex spacing — important to compare extractors with different vertex
    densities (e.g. skimage vs cv2). Returns ``(nan, nan)`` if the fit fails.

    angle180 : long-axis azimuth from North, in [0, 180) degrees.
    aspect_ratio : major/minor semi-axis ratio (>= 1).
    """
    if geom is None or geom.is_empty:
        return np.nan, np.nan
    g = shapely.segmentize(geom, res)
    if g is None or g.is_empty:
        return np.nan, np.nan
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda p: p.area)         # largest part
    if g.geom_type != "Polygon" or g.exterior is None:
        return np.nan, np.nan
    xy = np.asarray(g.exterior.coords)
    xy = xy - xy.mean(axis=0)
    model = EllipseModel()
    if not model.estimate(xy):
        return np.nan, np.nan
    _, _, a, b, theta = model.params
    if not (a > 0 and b > 0):
        return np.nan, np.nan
    # major-axis direction, then convert math-angle theta -> azimuth-from-North in [0,180)
    theta_major = theta if a >= b else theta + np.pi / 2.0
    angle180 = np.degrees(np.arctan2(np.cos(theta_major), np.sin(theta_major))) % 180.0
    aspect = max(a, b) / min(a, b)
    return angle180, aspect


def boulder_orientations(gdf, res, sample=None, seed=0, n_workers=8):
    """Compute orientations for a GeoDataFrame of boulder polygons.

    Parameters
    ----------
    gdf : GeoDataFrame in a projected (metric) CRS.
    res : float — densify spacing for the ellipse fit (use the raster pixel size, e.g. 0.5 m).
    sample : int or None — randomly subsample to this many boulders first (the orientation
        *distribution* is well-estimated from ~1e5; full ~1e6 fits are unnecessary).
    n_workers : int — threads for the per-boulder fit (EllipseModel releases the GIL in its
        BLAS calls). Pass 1 for serial.

    Returns
    -------
    DataFrame with columns ``[angle180, aspect_ra, diameter_m]`` (failed fits dropped).
    """
    g = gdf.sample(sample, random_state=seed) if (sample is not None and len(gdf) > sample) else gdf
    geoms = g.geometry.values
    diameter_m = 2.0 * np.sqrt(g.geometry.area.values / np.pi)  # equivalent-area diameter

    def _fit(geom):
        return ellipse_angle180(geom, res)

    if n_workers and n_workers > 1 and len(geoms) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            out = list(ex.map(_fit, geoms))
    else:
        out = [_fit(geom) for geom in geoms]

    df = pd.DataFrame({
        "angle180": np.array([o[0] for o in out]),
        "aspect_ra": np.array([o[1] for o in out]),
        "diameter_m": diameter_m,
    })
    return df.dropna(subset=["angle180"]).reset_index(drop=True)


def attach_orientations(gdf, res, n_workers=8):
    """Like :func:`boulder_orientations` but **preserves every input column** — returns a copy of
    ``gdf`` with ``[angle180, aspect_ra, diameter_m]`` added (same rows, order, and index), so
    per-detection metadata (e.g. YOLO ``score``, or a true-/false-positive label from a spatial match)
    stays alongside the orientation. Failed fits get NaN ``angle180``/``aspect_ra``; filter downstream
    with :func:`well_resolved_subset` (which drops them via the aspect/diameter cuts) or ``dropna``.
    Used by the Test 7 detection-selection readouts, where the TP/FP split must travel with the angle.
    """
    geoms = gdf.geometry.values
    diameter_m = 2.0 * np.sqrt(gdf.geometry.area.values / np.pi)
    if n_workers and n_workers > 1 and len(geoms) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            out = list(ex.map(lambda g: ellipse_angle180(g, res), geoms))
    else:
        out = [ellipse_angle180(g, res) for g in geoms]
    df = gdf.copy()
    df["angle180"] = np.array([o[0] for o in out])
    df["aspect_ra"] = np.array([o[1] for o in out])
    df["diameter_m"] = diameter_m
    return df


def grid_fraction(angle180, tol_deg=10.0):
    """Fraction of long-axis azimuths within ``tol_deg`` of a grid direction (0/90/180).

    For a uniform (unbiased) distribution this equals ``2 * tol_deg / 90`` (e.g. 22.2% at
    tol=10) — values above that indicate snapping toward the raster's row/column axes.
    """
    a = np.asarray(angle180, dtype=float) % 180.0
    d0 = np.minimum(a, 180.0 - a)         # distance to 0 / 180 (N-S)
    d90 = np.abs(a - 90.0)                # distance to 90 (E-W)
    return float((np.minimum(d0, d90) <= tol_deg).mean())


def grid_fraction_uniform(tol_deg=10.0):
    """The grid_fraction a perfectly uniform distribution would give (the unbiased baseline)."""
    return 2.0 * tol_deg / 90.0


def diagonal_fraction(angle180, tol_deg=10.0):
    """Fraction of long-axis azimuths within ``tol_deg`` of a pixel **diagonal** (45/135).

    The diagonal analogue of :func:`grid_fraction` (which measures the 0/90 cardinals). In a
    north-up scene the pixel anti-diagonal is 135°, so this is the direct measure of the HiRISE
    YOLO lock. Uniform (unbiased) baseline is the same ``2*tol_deg/90`` (:func:`grid_fraction_uniform`).
    Used by Test 3 (different-model) to show YOLO snaps to the diagonal while Mask R-CNN does not.
    """
    a = np.asarray(angle180, dtype=float) % 180.0
    return float((np.minimum(axial_distance(a, 45.0), axial_distance(a, 135.0)) <= tol_deg).mean())


def asymmetry_ratio(angle180, tol_deg=10.0):
    """The 135/45 asymmetry: ``count(within tol of 135) / count(within tol of 45)``.

    A process symmetric under 90° rotation / reflection gives ~1 (it cannot tell the NE–SW 45°
    diagonal from the NW–SE 135° one); ``>1`` means the 135° diagonal genuinely dominates. Returns
    ``nan`` if no azimuth falls near 45°.
    """
    a = np.asarray(angle180, dtype=float) % 180.0
    n45 = int((axial_distance(a, 45.0) <= tol_deg).sum())
    n135 = int((axial_distance(a, 135.0) <= tol_deg).sum())
    return float(n135) / n45 if n45 else float("nan")


# --- Rose-diagram statistics: well-resolved subset, smoothed density, peak + bootstrap CI -----
# Consolidates the small analysis helpers previously inlined across the Goal-2 / HiRISE-test
# notebooks (well-resolved subset selection, smoothed axial density, the refined peak, its
# bootstrap CI, axial distance). Azimuths are axial (mod 180); North=0, East=90.

DPX_MIN = 8   # a boulder narrower than this many pixels (diameter_m / m_per_px) is too pixelated
              # to carry a meaningful long-axis orientation, so it's dropped from distributions.


def well_resolved_subset(df, res, aspect_min=1.35, dpx_min=DPX_MIN):
    """Subset for orientation analysis: elongated AND well-resolved boulders.

    elongated = ``aspect_ra > aspect_min`` (near-circular masks have no real long axis);
    well-resolved = equivalent diameter >= ``dpx_min`` pixels (``diameter_m / res``), since the
    long axis of a near-pixel-scale mask is dominated by discretisation. ``df`` must have
    ``aspect_ra`` and ``diameter_m`` columns (as returned by :func:`boulder_orientations`).
    """
    return df[(df["aspect_ra"] > aspect_min) & (df["diameter_m"] / res >= dpx_min)]


def axial_distance(angles, center):
    """Axial (mod-180) angular distance from each azimuth to ``center``, in [0, 90]."""
    a = np.asarray(angles, dtype=float) % 180.0
    return np.abs(((a - center + 90.0) % 180.0) - 90.0)


def _circular_smooth(counts, sigma=4.0):
    """Wrap-around Gaussian smoothing of a length-N circular histogram (period N)."""
    c = np.asarray(counts, dtype=float)
    n = len(c)
    x = np.arange(n)
    d = np.minimum(np.abs(x - x[:, None]), n - np.abs(x - x[:, None]))
    k = np.exp(-0.5 * (d / sigma) ** 2)
    return (k / k.sum(1, keepdims=True)) @ c


def azimuth_density(angles, sigma=4.0, nbins=180):
    """Smoothed axial-azimuth density. Returns ``(bin_centers_deg, density_percent)``.

    A ``nbins``-bin histogram over [0, 180) (1 deg wide at the default) with wrap-around Gaussian
    smoothing (``sigma`` in bins), normalised to sum to 100%.
    """
    h, e = np.histogram(np.asarray(angles, dtype=float) % 180.0,
                        bins=np.linspace(0.0, 180.0, nbins + 1))
    d = _circular_smooth(h.astype(float), sigma)
    centers = (e[:-1] + e[1:]) / 2.0
    total = d.sum()
    return centers, (d / total * 100.0 if total else d)


def refine_peak(angles, sigma=4.0):
    """Mode of the smoothed azimuth density with sub-bin parabolic interpolation; deg in [0,180)."""
    c, d = azimuth_density(angles, sigma)
    n = len(d)
    i = int(np.argmax(d))
    y0, y1, y2 = d[(i - 1) % n], d[i], d[(i + 1) % n]
    den = y0 - 2 * y1 + y2
    return float((c[i] + (0.5 * (y0 - y2) / den if den else 0.0)) % 180.0)


def bootstrap_peak_ci(angles, n_boot=400, ci=(2.5, 97.5), sigma=4.0, seed=0):
    """Refined peak azimuth and its bootstrap CI. Returns ``(peak, lo, hi)`` in degrees.

    CAVEAT: the percentile CI is taken on the linear angle values, so it is only meaningful when
    the peak sits away from the 0/180 wrap (true for the HiRISE diagonal peaks ~115-145).
    """
    a = np.asarray(angles, dtype=float) % 180.0
    n = len(a)
    if n == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.array([refine_peak(a[rng.integers(0, n, n)], sigma) for _ in range(n_boot)])
    lo, hi = np.percentile(boot, ci)
    return refine_peak(a, sigma), float(lo), float(hi)


def meridian_convergence(tif_path):
    """Meridian convergence (deg) at a raster's centre: the grid-bearing of true North.

    Convert a grid-frame azimuth to geographic with ``geo = grid - convergence``. Returns 0.0 for
    equirectangular (``eqc``) north-up products by construction. Needs rasterio + pyproj (imported
    lazily so the rest of this module stays dependency-light).
    """
    import rasterio
    from pyproj import CRS, Transformer
    with rasterio.open(tif_path) as ds:
        crs = CRS.from_wkt(ds.crs.to_wkt())
        if crs.to_dict().get("proj") == "eqc":
            return 0.0
        cx = (ds.bounds.left + ds.bounds.right) / 2.0
        cy = (ds.bounds.top + ds.bounds.bottom) / 2.0
    geog = crs.geodetic_crs
    inv = Transformer.from_crs(geog, crs, always_xy=True)
    lon, lat = Transformer.from_crs(crs, geog, always_xy=True).transform(cx, cy)
    d = 0.0005
    x1, y1 = inv.transform(lon, lat)
    x2, y2 = inv.transform(lon, lat + d)
    return float(np.degrees(np.arctan2(x2 - x1, y2 - y1)))


def plot_rose(ax, angle180, bins=36, color="tab:blue", title=None, density=True, label=None):
    """Draw an axial rose (plots theta and theta+180) on a polar Axes (North up, clockwise).

    ``ax`` must be created with ``projection="polar"``. ``bins`` is the number of bins over
    [0, 180); the rose mirrors them to 360 for the conventional symmetric look.
    """
    a180 = np.asarray(angle180, dtype=float) % 180.0
    edges = np.linspace(0.0, 2 * np.pi, 2 * bins + 1)
    both = np.deg2rad(np.concatenate([a180, a180 + 180.0]))
    counts, _ = np.histogram(both, bins=edges)
    if density and counts.sum() > 0:
        counts = counts / counts.sum()
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
           color=color, edgecolor="k", linewidth=0.3, alpha=0.8, label=label)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


# --- YOLO-free orientation from raw pixels (structure tensor) -------------------------------
# These measure intensity-gradient structure directly from the imagery, with NO segmentation
# mask in the loop. CAVEAT for sunlit planetary scenes: the dominant gradient is the
# bright-cap -> shadow terminator, so the returned axis is pulled toward the illumination edge
# (perpendicular to the sun line), NOT necessarily the rock's true long axis. Use these to
# locate the *imaging* axes (illumination / shadow) and as a YOLO-free cross-check, not as
# ground-truth boulder orientation. Azimuth convention matches ``ellipse_angle180`` (North=0,
# East=90, clockwise, [0,180)). Validated on synthetic lines (N-S->0, E-W->90, NW-SE diag->135).

def structure_tensor_orientation(patch, weight=None, smooth=1.0):
    """Long-axis azimuth ([0,180) from North) and coherence ([0,1]) of a raw image patch via
    the gradient structure tensor.

    weight : optional same-shape per-pixel weights (e.g. a centred Gaussian to focus on a boulder).
    smooth : Gaussian pre-smoothing sigma (px) applied before the gradient.
    """
    a = gaussian_filter(np.asarray(patch, dtype=float), smooth)
    gy, gx = np.gradient(a)                       # gx = d/dcol (East), gy = d/drow (South)
    w = np.ones_like(a) if weight is None else np.asarray(weight, dtype=float)
    Jxx = float(np.sum(w * gx * gx)); Jyy = float(np.sum(w * gy * gy)); Jxy = float(np.sum(w * gx * gy))
    evals, evecs = np.linalg.eigh(np.array([[Jxx, Jxy], [Jxy, Jyy]]))   # ascending eigenvalues
    coh = float((evals[1] - evals[0]) / (evals[1] + evals[0] + 1e-12))
    vx, vy = evecs[:, 0]                          # smallest-eigenvalue evec = structure (long) axis
    az = float(np.degrees(np.arctan2(vx, -vy)) % 180.0)   # East=vx, North=-vy
    return az, coh


def orientation_field(img, grad_sigma=1.0, integ_sigma=4.0):
    """Per-pixel structure-tensor azimuth-from-North ([0,180)) and coherence for a whole image.

    Same convention/caveats as :func:`structure_tensor_orientation`. ``grad_sigma`` sets the
    texture scale of the gradient; ``integ_sigma`` the neighbourhood the tensor is averaged over.
    Returns ``(azimuth, coherence)`` arrays the shape of ``img``.
    """
    a = gaussian_filter(np.asarray(img, dtype=float), grad_sigma)
    gy, gx = np.gradient(a)
    Jxx = gaussian_filter(gx * gx, integ_sigma); Jyy = gaussian_filter(gy * gy, integ_sigma)
    Jxy = gaussian_filter(gx * gy, integ_sigma)
    tr = Jxx + Jyy
    disc = np.sqrt(np.maximum((Jxx - Jyy) ** 2 + 4 * Jxy * Jxy, 0.0))
    l_max = (tr + disc) / 2; l_min = (tr - disc) / 2
    coh = (l_max - l_min) / (l_max + l_min + 1e-12)
    vx = Jxy; vy = l_min - Jxx                    # eigenvector for the smaller eigenvalue
    az = np.degrees(np.arctan2(vx, -vy)) % 180.0
    return az, coh
