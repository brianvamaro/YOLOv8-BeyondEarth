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


def _fit_ellipse(geom, res):
    """Shared EllipseModel fit for a boulder polygon. Returns ``(xc, yc, a, b, theta)`` in the
    polygon's own coordinates, or ``None`` if the fit fails.

    The polygon is first densified to ``res`` spacing (``shapely.segmentize``) so the fit is not
    dominated by uneven vertex spacing — important to compare extractors with different vertex
    densities (e.g. skimage vs cv2). ``theta`` is the skimage math-angle (radians) of the ``a``
    semi-axis; ``a``/``b`` are the semi-axes (not necessarily major/minor-ordered).

    The fit runs on mean-shifted coordinates for conditioning, and the returned centre is the
    **fitted** one shifted back. It used to return the coordinate mean itself and throw the fitted
    offset away, which is only the same thing for a polygon whose vertices are symmetric about the
    ellipse centre. Across 26,830 validation polygons the error was 0.157 px median, 0.702 px p99
    and 7.216 px maximum, with 2.1% above half a pixel (review item G2-08). Orientation was never
    affected — ``ellipse_angle180`` uses only ``theta`` — but every OBB label built from this was
    displaced.
    """
    if geom is None or geom.is_empty:
        return None
    g = shapely.segmentize(geom, res)
    if g is None or g.is_empty:
        return None
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda p: p.area)         # largest part
    if g.geom_type != "Polygon" or g.exterior is None:
        return None
    xy = np.asarray(g.exterior.coords)
    shift = xy.mean(axis=0)
    model = EllipseModel()
    if not model.estimate(xy - shift):
        return None
    xc, yc, a, b, theta = model.params
    if not (a > 0 and b > 0):
        return None
    return float(xc + shift[0]), float(yc + shift[1]), float(a), float(b), float(theta)


def ellipse_angle180(geom, res):
    """Fit an EllipseModel to a boulder polygon and return ``(angle180, aspect_ratio)``.

    The polygon is first densified to ``res`` spacing (``shapely.segmentize``) so the fit is not
    dominated by uneven vertex spacing — important to compare extractors with different vertex
    densities (e.g. skimage vs cv2). Returns ``(nan, nan)`` if the fit fails.

    angle180 : long-axis azimuth from North, in [0, 180) degrees.
    aspect_ratio : major/minor semi-axis ratio (>= 1).
    """
    fit = _fit_ellipse(geom, res)
    if fit is None:
        return np.nan, np.nan
    _, _, a, b, theta = fit
    # major-axis direction, then convert math-angle theta -> azimuth-from-North in [0,180)
    theta_major = theta if a >= b else theta + np.pi / 2.0
    angle180 = np.degrees(np.arctan2(np.cos(theta_major), np.sin(theta_major))) % 180.0
    aspect = max(a, b) / min(a, b)
    return angle180, aspect


def ellipse_obb(geom, res):
    """Oriented bounding box from the unbiased EllipseModel fit of a boulder polygon.

    Returns ``(corners, angle180, aspect_ratio)`` where ``corners`` is a ``(4, 2)`` array of the
    rotated-rectangle vertices in the polygon's own coordinates, ordered consistently
    (+major+minor, +major-minor, -major-minor, -major+minor). The box half-extents are the fitted
    semi-axes ``a`` (major) and ``b`` (minor) — i.e. the ellipse's ``2a x 2b`` bounding rectangle,
    aligned to the fitted long axis. This is the label geometry Solution H trains the OBB head on:
    it inherits the EllipseModel's grid-neutrality, unlike a minimum-rotated-rectangle of the
    pixelated polygon (which snaps to 0/90 — never use MRR for orientation labels). Returns
    ``(None, nan, nan)`` if the fit fails.
    """
    fit = _fit_ellipse(geom, res)
    if fit is None:
        return None, np.nan, np.nan
    xc, yc, a, b, theta = fit
    # order semi-axes so (amaj, theta_major) is the long axis
    if a >= b:
        amaj, amin, theta_major = a, b, theta
    else:
        amaj, amin, theta_major = b, a, theta + np.pi / 2.0
    u = amaj * np.array([np.cos(theta_major), np.sin(theta_major)])   # along major
    v = amin * np.array([-np.sin(theta_major), np.cos(theta_major)])  # along minor
    centre = np.array([xc, yc])
    corners = np.array([centre + u + v, centre + u - v, centre - u - v, centre - u + v])
    angle180 = np.degrees(np.arctan2(np.cos(theta_major), np.sin(theta_major))) % 180.0
    aspect = amaj / amin
    return corners, angle180, aspect


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


def obb_angle180(corners):
    """Long-axis azimuth + aspect of an oriented bounding box, from its 4 corners.

    ``corners`` is a ``(4, 2)`` array of ``(col, row)`` image-pixel coordinates (e.g. ultralytics
    ``result.obb.xyxyxyxy[k]``). Returns ``(angle180, aspect_ratio)`` in the SAME geographic
    convention as :func:`ellipse_angle180` — azimuth from North (up), clockwise, [0, 180); North=0,
    East=90 — via the ``y = -row`` flip used throughout the orientation-validation code (image rows
    increase downward, so North is ``-row``). The long-axis direction ``(dcol, drow)`` maps to
    ``angle180 = atan2(dcol, -drow) mod 180``. This is the Solution-H OBB orientation readout: the
    angle comes straight from the predicted box, with NO mask/polygon step in the path.
    """
    c = np.asarray(corners, dtype=float)
    e1 = c[1] - c[0]                       # consecutive edges of the rotated rectangle
    e2 = c[2] - c[1]
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    long_e, short_len, long_len = (e1, l2, l1) if l1 >= l2 else (e2, l1, l2)
    if long_len <= 0:
        return np.nan, np.nan
    dcol, drow = float(long_e[0]), float(long_e[1])
    angle180 = np.degrees(np.arctan2(dcol, -drow)) % 180.0
    aspect = long_len / short_len if short_len > 0 else np.nan
    return angle180, aspect


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


def half_plane_fraction(angle180):
    """Fraction of long-axis azimuths in the **lower half-plane** ``[0, 90)`` — the SAM2
    half-plane-imbalance metric (``f_lower`` in ``docs/sam2_orientation_validation_plan.md``).

    ``f_lower = mean((angle180 % 180) < 90)``. The two halves ``[0, 90)`` and ``[90, 180)`` map onto
    each other under a mirror flip about a cardinal axis (θ → 180 − θ), so a reflection-symmetric
    producer on reflection-symmetric input splits them 50/50: the **null is 0.5**. A persistent
    departure requires a *directional* cause (illumination from a fixed sun azimuth, a real fabric,
    or a decoder handedness) — which the T4 rotation/mirror + cross-sun probes separate. Unlike the
    axial :func:`grid_fraction`/:func:`diagonal_fraction` (both blind to which side of an axis the
    mass sits on), this is the one metric that resolves the left/right bulk asymmetry, of which the
    narrow 45-vs-135 :func:`asymmetry_ratio` is only one facet. NaNs are dropped before the mean;
    returns ``nan`` for an all-NaN/empty input.
    """
    a = np.asarray(angle180, dtype=float)
    a = a[np.isfinite(a)] % 180.0
    return float((a < 90.0).mean()) if a.size else float("nan")


def half_plane_fraction_uniform():
    """The :func:`half_plane_fraction` a reflection-symmetric (unbiased) distribution gives: 0.5."""
    return 0.5


def half_plane_fraction_about(angle180, ref_axis):
    """:func:`half_plane_fraction` generalised to an **arbitrary reference axis** ``ref_axis`` (deg).

    Fraction of azimuths whose signed axial position sits in the half turned clockwise of
    ``ref_axis``: ``mean(((angle180 - ref_axis) % 180) < 90)``. The plain
    :func:`half_plane_fraction` is exactly the ``ref_axis = 0`` case (split about the N–S cardinal).
    Passing the **sun azimuth** as ``ref_axis`` makes this the "which side of the sun line does the
    apparent long axis lean" probe. Because a pull *toward* the sun axis is reflection-symmetric
    about it, a purely axial illumination concentration leaves this at the **null 0.5** — so a
    departure here is a *handedness* about the sun (decoder or shadow asymmetry), complementary to
    the fixed-frame :func:`half_plane_fraction` which instead tracks *where* the sun axis sits
    relative to the cardinals (that is the "f_lower tracks the sun by construction" signal). NaNs
    dropped; ``nan`` for empty input.
    """
    a = np.asarray(angle180, dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        return float("nan")
    return float((((a - float(ref_axis)) % 180.0) < 90.0).mean())


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


def signed_axial_deviation(angles, ref):
    """Signed axial deviation of each azimuth from ``ref``, in (-90, 90] (deg).

    The signed companion of :func:`axial_distance` (whose absolute value it equals). Positive means
    ``angles`` lies clockwise of ``ref`` (larger azimuth, the short way round the mod-180 circle),
    negative counter-clockwise; ``((angles - ref + 90) % 180) - 90`` so it wraps correctly (1° vs
    ref=179° gives +2°) and treats θ and θ+180 as one axis. NaNs propagate (not dropped), matching
    :func:`axial_distance`. Two uses in the illumination probe: the **signed apparent-angle error**
    (``ref`` = the known truth angle) and the **signed boulder position relative to the sun**
    (``ref`` = the sun azimuth); regressing the former on the latter is the shaping curve — a pull
    toward the sun axis is a negative slope through the origin. ``ref`` may be a scalar or an
    array broadcastable to ``angles`` (e.g. a per-boulder sun azimuth).
    """
    a = np.asarray(angles, dtype=float)
    r = np.asarray(ref, dtype=float)
    return ((a - r + 90.0) % 180.0) - 90.0


def axis_alignment_fraction(angle180, ref_axis, tol_deg=45.0):
    """Fraction of long-axis azimuths within ``tol_deg`` of the reference axis ``ref_axis`` (axial).

    With the default ``tol_deg = 45`` this splits the axial circle into the 90°-wide half **centered
    on** ``ref_axis`` (axes lying *along* it) versus the half centered on its perpendicular (axes
    lying *across* it) — so, passing the **sun axis**, it answers directly: *are more boulders
    oriented along the sun line than across it?* The **null for a uniform population is
    ``tol_deg / 90`` = 0.5 at tol 45**; a pull *toward* ``ref_axis`` (the illumination artifact)
    pushes it above 0.5. Smaller ``tol_deg`` measures tighter concentration about the axis (e.g. tol
    10 → a :func:`grid_fraction`-style share about an arbitrary axis, null 0.111). Unlike
    :func:`half_plane_fraction_about` (which splits *at* the axis and so measures left/right
    handedness, null 0.5 and blind to a symmetric axial pull), this splits *across* the axis and so is
    sensitive to the concentration itself. NaNs dropped; ``nan`` for empty input.
    """
    a = np.asarray(angle180, dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        return float("nan")
    return float((axial_distance(a, ref_axis) <= float(tol_deg)).mean())


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


def bootstrap_peak_ci(angles, n_boot=400, ci=(2.5, 97.5), sigma=4.0, seed=0, groups=None):
    """Refined peak azimuth and its bootstrap CI. Returns ``(peak, lo, hi)`` in degrees.

    ``groups``: optional per-boulder cluster label (tile, ROM, scene). **Pass it whenever the
    boulders are not independent** — which is essentially always, since thousands of detections
    typically come from a handful of tiles or scenes (review item G2-06). Resampling individual
    boulders treats every detection as independent evidence and produces an interval far narrower
    than the data supports: the current Mars matched comparison is 96 rocks from six tiles, so the
    honest sample size is six. With ``groups``, whole clusters are resampled with replacement.

    CAVEAT: the percentile CI is taken on the linear angle values, so it is only meaningful when
    the peak sits away from the 0/180 wrap (true for the HiRISE diagonal peaks ~115-145).
    """
    a = np.asarray(angles, dtype=float) % 180.0
    n = len(a)
    if n == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    if groups is None:
        boot = [refine_peak(a[rng.integers(0, n, n)], sigma) for _ in range(n_boot)]
    else:
        members = _group_members(groups, n)
        k = len(members)
        boot = []
        for _ in range(n_boot):
            pick = np.concatenate([members[i] for i in rng.integers(0, k, k)])
            boot.append(refine_peak(a[pick], sigma))
    lo, hi = np.percentile(np.asarray(boot, dtype=float), ci)
    return refine_peak(a, sigma), float(lo), float(hi)


def _group_members(groups, n):
    """List of index arrays, one per distinct cluster label."""
    groups = np.asarray(groups)
    if len(groups) != n:
        raise ValueError(f"groups has {len(groups)} labels for {n} angles")
    return [np.flatnonzero(groups == g) for g in pd.unique(groups)]


def peak_ci_report(angles, groups=None, n_boot=400, ci=(2.5, 97.5), sigma=4.0, seed=0):
    """Peak azimuth with cluster-aware uncertainty **and the diagnostics that qualify it**.

    A headline peak is only as strong as the number of *independent* units behind it, so this
    reports that number alongside the interval, and re-estimates the peak with each cluster left
    out in turn. A leave-one-out spread comparable to the interval width means the result rests on
    one tile or one scene and the interval is understating the real uncertainty (review item
    G2-06).

    Returns a dict: ``peak``, ``lo``, ``hi``, ``n``, ``n_clusters``,
    ``leave_one_out`` (per-cluster peak with that cluster removed), ``loo_spread`` (max - min),
    and ``cluster_bootstrap`` (whether clustering was applied at all).
    """
    a = np.asarray(angles, dtype=float) % 180.0
    peak, lo, hi = bootstrap_peak_ci(a, n_boot=n_boot, ci=ci, sigma=sigma, seed=seed,
                                     groups=groups)
    out = {"peak": peak, "lo": lo, "hi": hi, "n": int(len(a)),
           "cluster_bootstrap": groups is not None}
    if groups is None:
        out.update(n_clusters=None, leave_one_out={}, loo_spread=np.nan)
        return out
    members = _group_members(groups, len(a))
    labels = list(pd.unique(np.asarray(groups)))
    loo = {}
    for label, idx in zip(labels, members):
        keep = np.setdiff1d(np.arange(len(a)), idx, assume_unique=False)
        loo[label] = refine_peak(a[keep], sigma) if len(keep) else np.nan
    finite = [v for v in loo.values() if np.isfinite(v)]
    out.update(n_clusters=len(members), leave_one_out=loo,
               loo_spread=float(max(finite) - min(finite)) if len(finite) > 1 else 0.0)
    return out


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

def _tensor_azimuth_coherence(Jxx, Jyy, Jxy):
    """Structure-tensor long-axis azimuth ([0,180) from North) and coherence ([0,1]).

    Shared by the scalar and whole-image entry points so the two cannot disagree, and written as a
    **half-angle formula** rather than an explicit eigenvector (review item G2-07).

    The field implementation used to build the minor eigenvector as ``(Jxy, l_min - Jxx)``. That is
    the right eigenvector only while the tensor is non-diagonal: for ``Jxy == 0`` with
    ``Jxx <= Jyy`` — any purely horizontal structure, and every axis-aligned synthetic test case —
    it collapses to ``(0, 0)``, ``arctan2`` returns 0, and a horizontal line reads the same
    azimuth as a vertical one. The half-angle form has no degenerate branch: the dominant-gradient
    direction is ``0.5 * atan2(2 Jxy, Jxx - Jyy)`` and the structure axis is its perpendicular,
    both well defined as ``Jxy -> 0``.

    Accepts scalars or arrays and returns the same shape.
    """
    Jxx = np.asarray(Jxx, dtype=float)
    Jyy = np.asarray(Jyy, dtype=float)
    Jxy = np.asarray(Jxy, dtype=float)
    disc = np.sqrt(np.maximum((Jxx - Jyy) ** 2 + 4.0 * Jxy * Jxy, 0.0))
    coh = disc / (Jxx + Jyy + 1e-12)              # (l_max - l_min) / (l_max + l_min)

    phi = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)  # dominant gradient direction
    vx, vy = -np.sin(phi), np.cos(phi)            # perpendicular = the structure (long) axis
    az = np.degrees(np.arctan2(vx, -vy)) % 180.0  # East = vx, North = -vy
    return az, coh


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
    az, coh = _tensor_azimuth_coherence(Jxx, Jyy, Jxy)
    return float(az), float(coh)


def orientation_field(img, grad_sigma=1.0, integ_sigma=4.0):
    """Per-pixel structure-tensor azimuth-from-North ([0,180)) and coherence for a whole image.

    Same convention/caveats as :func:`structure_tensor_orientation`, and now the same maths — see
    :func:`_tensor_azimuth_coherence` for the degenerate-eigenvector defect this carried (review
    item G2-07: horizontal and vertical structure both read azimuth 0, which invalidated the
    regional raw-texture panel).

    ``grad_sigma`` sets the texture scale of the gradient; ``integ_sigma`` the neighbourhood the
    tensor is averaged over. Returns ``(azimuth, coherence)`` arrays the shape of ``img``.
    """
    a = gaussian_filter(np.asarray(img, dtype=float), grad_sigma)
    gy, gx = np.gradient(a)
    Jxx = gaussian_filter(gx * gx, integ_sigma); Jyy = gaussian_filter(gy * gy, integ_sigma)
    Jxy = gaussian_filter(gx * gy, integ_sigma)
    return _tensor_azimuth_coherence(Jxx, Jyy, Jxy)
