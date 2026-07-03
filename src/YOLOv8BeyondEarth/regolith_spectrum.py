"""Test 5 — flat-regolith directional power spectrum (HiRISE 135-deg investigation).

Model-free probe of the H_grid mechanism: does the boulder-free ground texture of the model's
input `.tif` carry a directional (diagonal) smear, or is the diagonal locking added downstream by
the segmentation pipeline? Design: ``docs/hirise_tests/test5_regolith_spectrum.md``.

Everything is measured in the **pixel/grid frame** (array axes), because both a resampling smear
and the mask bias are grid-locked. Azimuth convention matches :mod:`YOLOv8BeyondEarth.orientation`
(from North = up-rows, clockwise, East = +cols, axial mod 180).

FRAME CAVEAT (wavevector vs structure): the angular power profile is binned by **wavevector**
azimuth. Power at wavevector azimuth theta means intensity varies *along* theta — so elongated
structures lying along theta put their power at theta+90, and a directional **blur along** theta
*suppresses* power at theta (a dip at the smear direction). The injection calibration
(:func:`directional_blur` at a known angle) empirically pins where each effect lands, so no
reading of the profile ever rests on getting this conversion right by hand.

Heavy IO deps (rasterio, pyogrio, cv2, torch/ultralytics) are imported lazily; the spectral
primitives need only numpy/scipy.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .orientation import structure_tensor_orientation, axial_distance


# --- Spectral primitives ---------------------------------------------------------------------

def _freq_grids(n):
    """fftshifted (radius, azimuth-from-North-deg-axial) grids for an n x n FFT plane.

    Rows index South (+row = South), cols index East, so the wavevector's North component is
    ``-ky_row``. Radius is in cycles/px.
    """
    f = np.fft.fftshift(np.fft.fftfreq(n))
    ky = f[:, None]                                   # + = increasing row = South
    kx = f[None, :]                                   # + = increasing col = East
    radius = np.hypot(kx, ky)
    azim = np.degrees(np.arctan2(np.broadcast_to(kx, (n, n)),
                                 np.broadcast_to(-ky, (n, n)))) % 180.0
    return radius, azim


def angular_power_profile(patch, band_px=(2.5, 8.0), nbins=36, detrend_sigma=None):
    """Angular power profile of one patch in the high-frequency annulus.

    detrend (subtract a Gaussian low-pass, sigma = patch/8 by default) -> Hann window -> |FFT|^2
    -> mean power per **wavevector-azimuth** bin over the annulus of wavelengths ``band_px``
    (px/cycle), normalised to mean 1 over the annulus (so patches with different texture power
    contribute equally to an ensemble mean).

    Returns ``(bin_centers_deg, profile)``; profile entries are NaN for empty bins (shouldn't
    happen at the default 64 px / 5-deg binning).
    """
    a = np.asarray(patch, dtype=float)
    n = a.shape[0]
    if detrend_sigma is None:
        detrend_sigma = n / 8.0
    a = a - gaussian_filter(a, detrend_sigma)
    w = np.hanning(n)
    P = np.abs(np.fft.fftshift(np.fft.fft2(a * w[:, None] * w[None, :]))) ** 2
    radius, azim = _freq_grids(n)
    lo, hi = 1.0 / band_px[1], 1.0 / band_px[0]       # wavelengths -> cycles/px
    ann = (radius >= lo) & (radius <= hi)
    edges = np.linspace(0.0, 180.0, nbins + 1)
    idx = np.clip(np.digitize(azim[ann], edges) - 1, 0, nbins - 1)
    pw = P[ann]
    sums = np.bincount(idx, weights=pw, minlength=nbins)
    cnts = np.bincount(idx, minlength=nbins)
    with np.errstate(invalid="ignore"):
        prof = sums / cnts
    m = np.nanmean(prof)
    return (edges[:-1] + edges[1:]) / 2.0, prof / m if m else prof


def ensemble_profiles(patches, band_px=(2.5, 8.0), nbins=36, detrend_sigma=None):
    """Stack per-patch angular profiles. Returns ``(centers, profiles[n_patch, nbins])``."""
    profs = []
    centers = None
    for p in patches:
        centers, pr = angular_power_profile(p, band_px, nbins, detrend_sigma)
        profs.append(pr)
    return centers, np.asarray(profs)


def bootstrap_profile(profiles, n_boot=400, ci=(2.5, 97.5), seed=0):
    """Mean angular profile with a patch-resampling bootstrap band. Returns ``(mean, lo, hi)``."""
    pr = np.asarray(profiles, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(pr)
    boots = np.array([np.nanmean(pr[rng.integers(0, n, n)], axis=0) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, ci, axis=0)
    return np.nanmean(pr, axis=0), lo, hi


def axis_contrast(centers, profile, axes=(45.0, 135.0), tol=10.0):
    """Signed power contrast of the given axes vs the rest of the profile.

    ``mean(profile near axes) / mean(profile elsewhere) - 1``: positive = excess wavevector power
    at the axes, negative = a deficit (what a smear *along* the axes produces). Use
    ``axes=(45,135)`` for the diagonal question, ``axes=(0,90)`` for the JP2/cardinal one.
    """
    c = np.asarray(centers, dtype=float)
    near = np.zeros(len(c), dtype=bool)
    for ax in axes:
        near |= axial_distance(c, ax) <= tol
    inside, outside = np.nanmean(profile[near]), np.nanmean(profile[~near])
    return float(inside / outside - 1.0)


def profile_summary(centers, profile, tol=10.0):
    """Convenience dict: paired + per-axis contrasts, peak/trough azimuths, modulation depth.

    The per-axis contrasts (``c0/c45/c90/c135``) matter because a one-sided smear (e.g. along 135
    only) gives a *trough* at 135 and a *peak* at 45 — averaging them into one 'diagonal' number
    can cancel exactly the signature we're looking for.
    """
    prof = np.asarray(profile, dtype=float)
    out = dict(
        diag_contrast=axis_contrast(centers, prof, (45.0, 135.0), tol),
        card_contrast=axis_contrast(centers, prof, (0.0, 90.0), tol),
        peak_deg=float(centers[np.nanargmax(prof)]),
        trough_deg=float(centers[np.nanargmin(prof)]),
        modulation=float((np.nanmax(prof) - np.nanmin(prof)) / np.nanmean(prof)),
    )
    for ax in (0.0, 45.0, 90.0, 135.0):
        out[f"c{int(ax)}"] = axis_contrast(centers, prof, (ax,), tol)
    return out


# --- Patch harvesting from a scene ------------------------------------------------------------

def load_prediction_bounds(gpkg_path):
    """All per-detection bounding boxes of a prediction gpkg as a float array [n, 4]
    (minx, miny, maxx, maxy in map coords). One-time read (~1 min for the 1.1 M-row treatment
    gpkg); afterwards a candidate window is vetted boulder-free with a vectorised overlap test.
    """
    from pyogrio import read_dataframe
    g = read_dataframe(str(gpkg_path), columns=[])
    return g.geometry.bounds.to_numpy(dtype=float)


def sample_flat_patches(raster_path, bounds, n_patches=800, patch=64, buffer_px=8,
                        valid_min=0.999, std_min=1.5, shadow_dn=10, shadow_frac=0.005,
                        outlier_z=4.0, outlier_frac=0.002, coh_max=0.35, seed=0,
                        max_tries=None):
    """Harvest flat, boulder/shadow-free regolith patches from a scene.

    A candidate ``patch``-px window is accepted iff, in order:
    - fully valid (> ``valid_min`` nonzero pixels — off-swath HiRISE margins are 0);
    - **boulder-free with margin**: no predicted-boulder bbox (``bounds``, from
      :func:`load_prediction_bounds`) overlaps the window grown by ``buffer_px``;
    - has usable texture (std >= ``std_min`` DN) — dead-flat patches carry no signal;
    - shadow-free: < ``shadow_frac`` of pixels below ``shadow_dn``;
    - no residual bright/dark blobs (undetected boulders): < ``outlier_frac`` of |z| > ``outlier_z``;
    - not coherently lineated (dunes/ridges would fake anisotropy): whole-patch structure-tensor
      coherence < ``coh_max``.

    Returns ``(patches [n, patch, patch] float, meta DataFrame)`` with per-patch position and the
    screening stats (plus per-reason rejection counts in ``meta.attrs['rejects']``).
    """
    import rasterio
    from rasterio.windows import Window

    if max_tries is None:
        max_tries = n_patches * 200
    rng = np.random.default_rng(seed)
    rejects = dict(valid=0, boulder=0, std=0, shadow=0, outlier=0, coherence=0)
    patches, rows = [], []
    with rasterio.open(raster_path) as r:
        W, H = r.width, r.height
        px_x, px_y = r.transform.a, -r.transform.e
        tries = 0
        while len(patches) < n_patches and tries < max_tries:
            tries += 1
            c0 = int(rng.integers(0, W - patch)); r0 = int(rng.integers(0, H - patch))
            img = r.read(1, window=Window(c0, r0, patch, patch)).astype(float)
            if (img > 0).mean() < valid_min:
                rejects["valid"] += 1; continue
            x0, y0 = r.transform * (c0 - buffer_px, r0 + patch + buffer_px)   # lower-left (map)
            x1, y1 = r.transform * (c0 + patch + buffer_px, r0 - buffer_px)   # upper-right
            minx, maxx = min(x0, x1), max(x0, x1)
            miny, maxy = min(y0, y1), max(y0, y1)
            if np.any((bounds[:, 0] < maxx) & (bounds[:, 2] > minx) &
                      (bounds[:, 1] < maxy) & (bounds[:, 3] > miny)):
                rejects["boulder"] += 1; continue
            sd = float(img.std())
            if sd < std_min:
                rejects["std"] += 1; continue
            if (img < shadow_dn).mean() > shadow_frac:
                rejects["shadow"] += 1; continue
            hp = img - gaussian_filter(img, patch / 8.0)
            if (np.abs(hp) > outlier_z * hp.std()).mean() > outlier_frac:
                rejects["outlier"] += 1; continue
            _, coh = structure_tensor_orientation(img, smooth=1.0)
            if coh > coh_max:
                rejects["coherence"] += 1; continue
            cx, cy = r.transform * (c0 + patch / 2.0, r0 + patch / 2.0)
            patches.append(img)
            rows.append((cx, cy, c0, r0, sd, coh))
    meta = pd.DataFrame(rows, columns=["x", "y", "col", "row", "std", "coherence"])
    meta.attrs["rejects"] = rejects
    meta.attrs["tries"] = tries
    return np.asarray(patches), meta


def sample_boulder_patches(raster_path, gpkg_path, n_patches=400, patch=64,
                           diam_px=(4.0, 16.0), res=0.5, seed=0, n_total=None):
    """Positive-control patches centred on random *predicted boulders* (bright cap + shadow in
    frame). Expect the spectrum to find the sun/shadow axis — proof the method sees a real
    direction. ``diam_px`` selects boulders small enough to fit with context in ``patch``.
    Returns ``(patches, meta)`` like :func:`sample_flat_patches`.
    """
    import rasterio
    from rasterio.windows import Window
    from pyogrio import read_dataframe, read_info

    if n_total is None:
        n_total = read_info(str(gpkg_path))["features"]
    rng = np.random.default_rng(seed)
    fids = np.sort(rng.choice(n_total, size=min(n_patches * 4, n_total), replace=False))
    g = read_dataframe(str(gpkg_path), fids=fids, columns=[])
    patches, rows = [], []
    with rasterio.open(raster_path) as r:
        for geom in g.geometry.values:
            if len(patches) >= n_patches:
                break
            d_px = 2.0 * np.sqrt(geom.area / np.pi) / res
            if not (diam_px[0] <= d_px <= diam_px[1]):
                continue
            col, row = ~r.transform * (geom.centroid.x, geom.centroid.y)
            c0, r0 = int(round(col - patch / 2)), int(round(row - patch / 2))
            if c0 < 0 or r0 < 0 or c0 + patch > r.width or r0 + patch > r.height:
                continue
            img = r.read(1, window=Window(c0, r0, patch, patch)).astype(float)
            if (img > 0).mean() < 0.999:
                continue
            patches.append(img)
            rows.append((geom.centroid.x, geom.centroid.y, d_px))
    meta = pd.DataFrame(rows, columns=["x", "y", "diam_px"])
    return np.asarray(patches), meta


def sample_windows(raster_path, n=64, size=256, valid_min=0.999, std_min=1.5, seed=0):
    """Plain valid-and-textured windows (NO boulder-free requirement) — inputs for
    :func:`isotropize` (phase randomisation destroys the boulders anyway). Returns [n, size, size].
    """
    import rasterio
    from rasterio.windows import Window

    rng = np.random.default_rng(seed)
    out = []
    with rasterio.open(raster_path) as r:
        W, H = r.width, r.height
        tries = 0
        while len(out) < n and tries < n * 100:
            tries += 1
            c0 = int(rng.integers(0, W - size)); r0 = int(rng.integers(0, H - size))
            img = r.read(1, window=Window(c0, r0, size, size)).astype(float)
            if (img > 0).mean() >= valid_min and img.std() >= std_min:
                out.append(img)
    return np.asarray(out)


# --- Controls: injection, isotropisation, warp simulation -------------------------------------

def directional_blur(patch, sigma_px, angle_deg, sigma_perp=0.3):
    """Blur ``patch`` with an anisotropic Gaussian whose long axis lies **along** azimuth
    ``angle_deg`` (from North, cw) — the injected 'smear'. ``sigma_perp`` keeps the kernel a
    band rather than a grid-aliased 1-px line. Returns the blurred patch (same shape).
    """
    from scipy.ndimage import convolve

    k = int(np.ceil(6.0 * max(sigma_px, sigma_perp))) | 1
    half = k // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    th = np.radians(angle_deg)
    ex, ny = np.sin(th), np.cos(th)                   # unit vector: East, North components
    t = xx * ex + yy * (-ny)                          # along-axis coord (row = -North)
    s = xx * ny + yy * ex                             # perpendicular coord
    ker = np.exp(-0.5 * ((t / max(sigma_px, 1e-6)) ** 2 + (s / sigma_perp) ** 2))
    ker /= ker.sum()
    return convolve(np.asarray(patch, dtype=float), ker, mode="reflect")


def isotropize(patch, seed=0, quantize=True):
    """Synthetic patch with the input's **radially averaged** power spectrum and random phases:
    stationary, isotropic-by-construction, boulder-free-by-construction (phase randomisation
    destroys localized structures). Used for (i) the method's null angular response, (ii) the
    warp-simulation substrate, (iii) the Test-7d pure-noise YOLO input.

    ``quantize`` re-scales to the input's mean/std and rounds to uint8-range integers, matching
    the 8-bit model input (quantisation noise is white/isotropic).
    """
    a = np.asarray(patch, dtype=float)
    n = a.shape[0]
    P = np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean()))) ** 2
    radius, _ = _freq_grids(n)
    rbin = np.round(radius * n).astype(int)           # integer radial bins in FFT-pixel units
    nb = rbin.max() + 1
    prof = np.bincount(rbin.ravel(), weights=P.ravel(), minlength=nb) / \
        np.maximum(np.bincount(rbin.ravel(), minlength=nb), 1)
    amp = np.sqrt(prof[rbin])                         # radially-symmetric amplitude target
    rng = np.random.default_rng(seed)
    wn = np.fft.fftshift(np.fft.fft2(rng.standard_normal((n, n))))
    out = np.fft.ifft2(np.fft.ifftshift(wn * amp)).real
    out = (out - out.mean()) / (out.std() + 1e-12) * a.std() + a.mean()
    if quantize:
        out = np.clip(np.round(out), 0, 255)
    return out


def warp_rotate(patch, angle_deg, kernel="bilinear"):
    """Rotate ``patch`` about its centre with a resampling ``kernel`` (cv2: nearest / bilinear /
    cubic / lanczos) and return the central half-size crop (free of border fill) — one generic
    map-projection resample. Feed isotropised patches and measure the anisotropy the kernel alone
    imposes: the empirical 'plausible H_grid magnitude' the null must be able to detect.
    """
    import cv2

    interp = dict(nearest=cv2.INTER_NEAREST, bilinear=cv2.INTER_LINEAR,
                  cubic=cv2.INTER_CUBIC, lanczos=cv2.INTER_LANCZOS4)[kernel]
    n = patch.shape[0]
    M = cv2.getRotationMatrix2D((n / 2.0, n / 2.0), float(angle_deg), 1.0)
    out = cv2.warpAffine(np.asarray(patch, dtype=np.float32), M, (n, n), flags=interp,
                         borderMode=cv2.BORDER_REFLECT)
    q = n // 4
    return out[q:q + n // 2, q:q + n // 2].astype(float)


# --- Test 7d rider: YOLO on boulder-free (synthetic) input -------------------------------------

def yolo_detections_on_patches(model, patches, imgsz=1024, conf=0.10, device=0):
    """Run YOLO on each (256-px, uint8-range) patch at the production settings and return every
    detection's orientation: DataFrame ``[patch_idx, angle180, aspect, diameter_px, score]``.
    Mask handling reproduces the pipeline (>=0.5 -> resize to patch -> largest blob -> fill ->
    contour -> EllipseModel), as in ``orientation_validation.detect_orientations``.
    """
    import cv2
    import shapely
    from scipy.ndimage import label, binary_fill_holes
    from .polygon import binary_mask_to_polygon
    from .orientation import ellipse_angle180

    rows = []
    for i, p in enumerate(patches):
        img = np.clip(np.asarray(p), 0, 255).astype(np.uint8)
        ss = img.shape[0]
        res = model(np.stack([img] * 3, axis=-1), imgsz=imgsz, conf=conf, verbose=False,
                    device=device)
        r0 = res[0]
        if r0.masks is None or len(r0.masks.data) == 0:
            continue
        scores = r0.boxes.conf.cpu().numpy()
        for m, sc in zip(r0.masks.data.cpu().numpy(), scores):
            mm = (m >= 0.5).astype(np.float32)
            if mm.shape[0] != ss:
                mm = cv2.resize(mm, (ss, ss), interpolation=cv2.INTER_AREA)
            mb = (mm >= 0.5).astype(np.uint8)
            lab, nl = label(mb)
            if nl == 0:
                continue
            if nl > 1:
                big = 1 + int(np.argmax([(lab == j).sum() for j in range(1, nl + 1)]))
                mb = (lab == big).astype(np.uint8)
            mb = binary_fill_holes(mb).astype(np.uint8)
            poly = binary_mask_to_polygon(mb)
            if poly is None or len(poly) < 5:
                continue
            ang, asp = ellipse_angle180(
                shapely.geometry.Polygon(np.c_[poly[:, 0], -poly[:, 1]]), 1.0)
            if np.isfinite(ang):
                rows.append((i, ang, asp, 2 * np.sqrt(int(mb.sum()) / np.pi), float(sc)))
    return pd.DataFrame(rows, columns=["patch_idx", "angle180", "aspect", "diameter_px", "score"])
