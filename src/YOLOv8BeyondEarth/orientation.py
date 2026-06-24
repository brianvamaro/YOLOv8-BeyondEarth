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
