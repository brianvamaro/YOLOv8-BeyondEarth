"""Test 4 (transforms) drivers for the HiRISE 135-deg orientation investigation (Goal 2).

Decides the T4 column of ``docs/hirise_artifact_resolution_plan.md``: is the ~135 deg long-axis
peak **re-imposed on the current pixel grid** by the YOLO segmentation (``H_seg`` -> rotation slope
~ +1) or **baked into the pixel content** (``H_geo``/``H_img``/``H_grid`` -> slope ~ 0)? Because YOLO
is not rotation-equivariant (``degrees=0`` in training) and non-90 deg rotations resample, the
rotation readout is only interpretable **against a control null** -- so this module also renders
synthetic boulders of *known* orientation to measure the model's intrinsic non-equivariance residual.

Readouts (subplan ``docs/hirise_tests/test4_transforms.md``):
- A. per-boulder paired rotation residual (``per_boulder_rotation_residual``) -- cleanest; extends
  notebook section 8.5.
- B. population slope of geographic peak vs rotation angle (``population_rotation_sweep``).
- the control null on rendered ellipses (``render_boulder_field`` + the same two readouts).

Convention: ``angle180`` is the long-axis azimuth from North (image -row), clockwise, in [0, 180) --
identical to :func:`YOLOv8BeyondEarth.orientation.ellipse_angle180`, so rendered truth angles compare
directly to :func:`YOLOv8BeyondEarth.orientation_validation.detect_orientations` output. Heavy IO deps
(rasterio, pyogrio) are imported lazily. Companion notebook: ``notebooks/HiRISE_Test4_transforms.py``.
"""
import numpy as np
import pandas as pd


def _circular_diff(a, b):
    """Smallest unsigned angular distance in [0, 90] between two azimuths in [0, 180) (deg)."""
    d = np.abs((np.asarray(a, float) - np.asarray(b, float)) % 180.0)
    return np.minimum(d, 180.0 - d)


def render_boulder_field(size=512, n=80, angles=None, diam_px=(8, 22), aspect=(1.5, 3.0),
                         bg=110, noise=12.0, cap=80.0, shadow=-55.0, shadow_frac=0.6,
                         sun_az_deg=255.0, margin=20, min_sep_frac=1.4, seed=0):
    """Render a field of synthetic boulders at **known** long-axis orientations.

    Each boulder is a filled bright ellipse (the sunlit rock, value ``bg+cap``) with an optional
    cast shadow (a same-shape ellipse offset along the anti-sun direction, value ``bg+shadow``)
    drawn underneath, on a mid-grey Gaussian-noise background -- enough structure for the detector
    to fire while the *true* orientation is exactly controlled.

    The drawing azimuth ``phi`` is in the project convention (from North = image -row, clockwise);
    cv2's ellipse angle is ``phi - 90`` (cv2 measures from +col toward +row). Returns
    ``(img uint8 [size, size], truth DataFrame [cx, cy, a_px, b_px, angle180, diam_px])`` where
    ``angle180`` is each boulder's rendered azimuth and ``diam_px`` its equivalent-area diameter.

    ``angles`` (deg, project convention) sets the orientations; if None they are drawn uniformly.
    Boulders are rejected if they fall within ``min_sep_frac x`` their own radius of an existing one,
    so masks stay separable. ``sun_az_deg`` is the azimuth the light comes *from* (shadow cast
    opposite).
    """
    import cv2

    rng = np.random.default_rng(seed)
    img = np.clip(bg + rng.normal(0, noise, (size, size)), 0, 255)
    if angles is None:
        angles = rng.uniform(0, 180, n)
    else:
        angles = np.asarray(angles, float)
        n = len(angles)

    # shadow is cast in the anti-sun direction; sun_az is "from", measured like phi (N, clockwise)
    anti = np.deg2rad(sun_az_deg + 180.0)
    sdx, sdy = np.sin(anti), -np.cos(anti)            # (col, row) unit vector toward the shadow

    placed = []                                       # (cx, cy, rad) for separation checks
    rows = []
    for phi in angles:
        a_px = rng.uniform(*diam_px) / 2.0            # major semi-axis (px)
        asp = rng.uniform(*aspect)
        b_px = a_px / asp                             # minor semi-axis
        rad = a_px
        for _ in range(40):                           # rejection-sample a non-overlapping centre
            cx = rng.uniform(margin, size - margin)
            cy = rng.uniform(margin, size - margin)
            if all(np.hypot(cx - px, cy - py) > min_sep_frac * (rad + pr) for px, py, pr in placed):
                break
        else:
            continue
        placed.append((cx, cy, rad))
        cv2_ang = float(phi - 90.0)
        axes = (max(int(round(a_px)), 1), max(int(round(b_px)), 1))
        if shadow_frac > 0:                           # cast shadow underneath, offset anti-sun
            soff = shadow_frac * a_px
            sc = (int(round(cx + sdx * soff)), int(round(cy + sdy * soff)))
            cv2.ellipse(img, sc, axes, cv2_ang, 0, 360, float(bg + shadow), -1)
        cv2.ellipse(img, (int(round(cx)), int(round(cy))), axes, cv2_ang, 0, 360,
                    float(bg + cap), -1)
        rows.append((cx, cy, a_px, b_px, phi % 180.0, 2.0 * np.sqrt(a_px * b_px)))

    img = np.clip(img, 0, 255).astype(np.uint8)
    truth = pd.DataFrame(rows, columns=["cx", "cy", "a_px", "b_px", "angle180", "diam_px"])
    return img, truth


def render_boulder_patch(size=256, phi=0.0, diam_px=12.0, aspect=2.0, bg=110, noise=12.0,
                         cap=80.0, shadow=-55.0, shadow_frac=0.6, sun_az_deg=255.0, seed=0):
    """Render a single synthetic boulder at azimuth ``phi`` centred in a ``size`` square patch --
    the per-boulder (readout A) synthetic-null analogue of :func:`render_boulder_field`. Returns the
    uint8 patch. Same drawing convention (cv2 angle ``phi - 90``, shadow cast anti-sun)."""
    import cv2

    rng = np.random.default_rng(seed)
    img = np.clip(bg + rng.normal(0, noise, (size, size)), 0, 255)
    a_px = diam_px / 2.0 * np.sqrt(aspect)            # keep equiv-diameter ~ diam_px
    b_px = a_px / aspect
    c = size // 2
    axes = (max(int(round(a_px)), 1), max(int(round(b_px)), 1))
    cv2_ang = float(phi - 90.0)
    if shadow_frac > 0:
        anti = np.deg2rad(sun_az_deg + 180.0)
        soff = shadow_frac * a_px
        sc = (int(round(c + np.sin(anti) * soff)), int(round(c - np.cos(anti) * soff)))
        cv2.ellipse(img, sc, axes, cv2_ang, 0, 360, float(bg + shadow), -1)
    cv2.ellipse(img, (c, c), axes, cv2_ang, 0, 360, float(bg + cap), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def rotate_image(img, alpha_deg, order=3, cval=0):
    """Rotate ``img`` counter-clockwise by ``alpha_deg`` (same handedness as ``np.rot90``).

    Multiples of 90 deg use :func:`numpy.rot90` (lossless, no resampling); other angles use a
    cubic-spline :func:`scipy.ndimage.rotate` with ``reshape=False`` (so the array size is kept; the
    corners rotate out and ``cval`` fills in). With CCW rotation by ``alpha``, a ground feature's
    azimuth measured in the rotated frame is ``phi - alpha``; recover the geographic azimuth with
    ``(measured + alpha) % 180`` (validated on the synthetic field, where truth is known).
    """
    a = float(alpha_deg) % 360.0
    if a % 90.0 == 0.0:
        return np.rot90(img, int(a // 90) % 4)
    from scipy.ndimage import rotate
    return rotate(img, a, reshape=False, order=order, mode="constant", cval=cval, prefilter=True)


def population_rotation_sweep(model, img, alphas, aspect_min=1.35, bin_deg=10, sigma=4.0,
                             n_boot=400, order=3, seed=0, device=0):
    """Readout B: rotate ``img`` by each ``alpha``, detect, and express the orientation peak in the
    geographic (un-rotated) frame. Returns a DataFrame ``[alpha, n, raw_peak, geo_peak, geo_lo,
    geo_hi]`` (bootstrap CI on the geographic peak), one row per alpha.

    The geographic peak un-rotates the raw peak by ``alpha``; a slope of ``geo_peak`` vs ``alpha``
    near 0 means the signal rotates with the pixels (in-pixels), near +1 means it is re-imposed on
    the current grid (``H_seg``). Fit the slope on the **lossless** 90 deg multiples first, then read
    the resampled angles against the synthetic null.
    """
    from .orientation import bootstrap_peak_ci
    from .orientation_validation import detect_orientations

    rows = []
    for a in alphas:
        det = detect_orientations(model, rotate_image(img, a, order=order), device=device,
                                  aspect_min=aspect_min)
        el = det[det.aspect >= aspect_min] if len(det) else det
        if len(el) == 0:
            rows.append((float(a), 0, np.nan, np.nan, np.nan, np.nan)); continue
        raw, lo, hi = bootstrap_peak_ci(el.angle180.values, n_boot=n_boot, sigma=sigma, seed=seed)
        geo = (raw + a) % 180.0
        rows.append((float(a), len(el), float(raw), float(geo),
                     float((lo + a) % 180.0), float((hi + a) % 180.0)))
    return pd.DataFrame(rows, columns=["alpha", "n", "raw_peak", "geo_peak", "geo_lo", "geo_hi"])


def per_boulder_rotation_residual(model, patches, alphas, max_center_dist=50, order=3,
                                  truth_angles=None, device=0):
    """Readout A: for each boulder ``patch`` (a square 2D uint8 array, boulder centred), measure the
    orientation in the patch, then in the patch rotated by each ``alpha``; map the rotated angle back
    (``+alpha``) and record the residual.

    Works for real boulders (patches read from the raster around each centroid) and the synthetic
    null (single rendered boulders, with ``truth_angles`` known). Returns a DataFrame
    ``[idx, alpha, ang0, ang_back, residual, truth, residual_truth, grid_dist0]`` where ``ang0`` is
    the unrotated detection, ``ang_back = (ang_rot + alpha) % 180``, ``residual`` is the circular
    distance ``|ang0 - ang_back|``, ``grid_dist0`` is ``ang0``'s distance to the nearest grid axis
    (0/45/90/135) for the snapping signature, and (if ``truth_angles`` given) ``residual_truth`` is
    ``|truth - ang_back|``. Rows with a failed/edge detection are skipped.
    """
    from .orientation_validation import center_orientation

    def grid_dist(x):
        d = np.abs((x - np.array([0, 45, 90, 135, 180])))
        return float(np.min(d))

    rows = []
    for i, patch in enumerate(patches):
        o0 = center_orientation(model, patch, max_center_dist=max_center_dist, device=device)
        if o0 is None:
            continue
        a0 = o0[0]
        tr = None if truth_angles is None else float(truth_angles[i]) % 180.0
        for a in alphas:
            orot = center_orientation(model, rotate_image(patch, a, order=order),
                                      max_center_dist=max_center_dist, device=device)
            if orot is None:
                continue
            back = (orot[0] + a) % 180.0
            rows.append((i, float(a), float(a0), float(back),
                         float(_circular_diff(a0, back)),
                         tr, (np.nan if tr is None else float(_circular_diff(tr, back))),
                         grid_dist(a0)))
    return pd.DataFrame(rows, columns=["idx", "alpha", "ang0", "ang_back", "residual",
                                       "truth", "residual_truth", "grid_dist0"])


# --- real-data readers (HiRISE side of the same two readouts) --------------------------------

def read_boulder_patches(gpkg_path, raster_path, res, n=200, aspect_min=1.6, diam_range=(5, 16),
                         slice_size=256, n_fids=30000, n_total=None, seed=7):
    """Read ``n`` centred square (``slice_size``) raw-image patches around elongated real boulders,
    for the per-boulder rotation residual (readout A) on HiRISE. Mirrors notebook section 8.5:
    sample ``n_fids`` features, keep ``aspect>aspect_min`` and ``diam in diam_range`` (metres),
    return ``(patches, gpkg_angles, meta_df)`` where ``patches`` is a list of uint8 arrays,
    ``gpkg_angles`` the production-mask EllipseModel azimuths, and ``meta_df`` has cx/cy/aspect/diam.
    """
    import rasterio
    from rasterio.windows import Window
    from pyogrio import read_dataframe, read_info
    from .orientation import ellipse_angle180

    if n_total is None:
        n_total = read_info(gpkg_path)["features"]
    rng = np.random.default_rng(seed)
    fids = np.sort(rng.choice(n_total, size=min(n_fids, n_total), replace=False))
    g = read_dataframe(gpkg_path, fids=fids)
    cx = g.geometry.centroid.x.values; cy = g.geometry.centroid.y.values
    diam = 2 * np.sqrt(g.geometry.area.values / np.pi)
    ang_asp = np.array([ellipse_angle180(gm, res) for gm in g.geometry.values])
    asp = ang_asp[:, 1]; ang = ang_asp[:, 0]
    sel = np.where((asp > aspect_min) & (diam >= diam_range[0]) & (diam <= diam_range[1]))[0]
    rng.shuffle(sel); sel = sel[:n]

    half = slice_size // 2
    patches, angs, rows = [], [], []
    with rasterio.open(raster_path) as ds:
        for i in sel:
            c0, r0 = ~ds.transform * (cx[i], cy[i]); c0, r0 = int(round(c0)), int(round(r0))
            p = ds.read(1, window=Window(c0 - half, r0 - half, slice_size, slice_size),
                        boundless=True, fill_value=0).astype(np.uint8)
            if (p > 0).mean() < 0.9:                  # skip edge/nodata patches
                continue
            patches.append(p); angs.append(float(ang[i]))
            rows.append((float(cx[i]), float(cy[i]), float(asp[i]), float(diam[i])))
    meta = pd.DataFrame(rows, columns=["cx", "cy", "aspect", "diam_m"])
    return patches, np.array(angs), meta


def read_dense_crops(raster_path, crop=1280, n=4, min_valid=0.99, seed=2):
    """Sample ``n`` dense (>= ``min_valid`` non-zero) square ``crop`` windows from a raster, as the
    representative crops for the population rotation sweep (readout B). Returns a list of uint8
    arrays."""
    import rasterio
    from rasterio.windows import Window

    rng = np.random.default_rng(seed)
    out = []
    with rasterio.open(raster_path) as ds:
        for _ in range(n * 30):
            if len(out) >= n:
                break
            c = rng.integers(0, ds.width - crop); r = rng.integers(0, ds.height - crop)
            a = ds.read(1, window=Window(c, r, crop, crop)).astype(np.uint8)
            if (a > 0).mean() >= min_valid:
                out.append(a)
    return out


# --- plotting helpers (the notebook calls these; logic stays in src) -------------------------

def rotation_slope(sweep, exclude_90_multiples=True):
    """Slope of the rotation sweep (readout B), the T4 discriminator. Fits ``raw_peak`` (image-frame
    peak) vs ``alpha``; returns ``dict(raw_slope, geo_slope, intercept, n)``.

    ``geo_slope = raw_slope + 1``: **in-pixels** (content rotates with the image) → raw_slope ≈ -1,
    geo_slope ≈ 0; **H_seg** (peak re-imposed on the grid) → raw_slope ≈ 0, geo_slope ≈ +1.

    By default the lossless 90 deg multiples are **excluded** from the fit: empirically YOLO's
    non-equivariance makes ``rot90`` outputs bimodal/out-of-distribution (notebook section 8.5), so
    the clean estimator is the small-angle resampled arm — and it is controlled by running the
    identical resample on the synthetic null.
    """
    t = sweep.dropna(subset=["raw_peak"]).sort_values("alpha")
    if exclude_90_multiples:
        t = t[t.alpha % 90 != 0]
    if len(t) < 2:
        return dict(raw_slope=np.nan, geo_slope=np.nan, intercept=np.nan, n=len(t))
    raw = np.rad2deg(np.unwrap(np.deg2rad(t.raw_peak.values * 2)) / 2)   # unwrap on 180-periodicity
    raw = raw - raw[0] + t.raw_peak.values[0]
    s, icpt = np.polyfit(t.alpha.values, raw, 1)
    return dict(raw_slope=float(s), geo_slope=float(s + 1), intercept=float(icpt), n=len(t))


def plot_rotation_sweep(ax, sweep, label="", color="tab:blue", exclude_90_multiples=True):
    """Readout B figure: **raw (image-frame) peak vs rotation angle** — the "is the peak pinned to the
    grid?" view. A flat line = pinned to the image grid (``H_seg``, geo-slope +1); a -1 descending
    line = tracks the content (in-pixels, geo-slope 0). Lossless 90 deg multiples are drawn hollow
    (confounded by non-equivariance) and excluded from the fit. Annotates the geo-slope."""
    t = sweep.dropna(subset=["raw_peak"]).sort_values("alpha")
    if len(t) == 0:
        return np.nan
    is90 = (t.alpha % 90 == 0).values
    ax.scatter(t.alpha[~is90], t.raw_peak[~is90], color=color, s=36, label=label, zorder=3)
    ax.scatter(t.alpha[is90], t.raw_peak[is90], facecolors="none", edgecolors=color, s=46,
               label="90° mult. (confounded)", zorder=3)
    fit = rotation_slope(sweep, exclude_90_multiples=exclude_90_multiples)
    xs = np.array([t.alpha.min(), t.alpha.max()])
    ax.plot(xs, fit["raw_slope"] * xs + fit["intercept"], "--", color=color, lw=1.4)
    a0 = fit["intercept"]
    ax.axhline(a0, color="grey", ls=":", lw=1, zorder=0)                   # H_seg: pinned
    ax.plot(xs, -1.0 * xs + a0, color="grey", ls="-.", lw=1, zorder=0)     # in-pixels: tracks content
    ax.set_xlabel("rotation alpha (deg)"); ax.set_ylabel("raw image-frame peak (deg)")
    ax.set_title(f"{label}  geo-slope={fit['geo_slope']:+.2f}  (H_seg→+1, in-pixels→0)", fontsize=10)
    return fit["geo_slope"]


def plot_residual_hist(ax, residuals_by_name, bins=np.arange(0, 91, 5)):
    """Readout A figure: overlaid per-boulder residual histograms. ``residuals_by_name`` is a dict
    ``{label: (residual_array, color)}``; annotates each median."""
    for name, (res, color) in residuals_by_name.items():
        res = np.asarray(res, float)
        ax.hist(res, bins=bins, density=True, histtype="step", lw=2, color=color,
                label=f"{name} (med {np.median(res):.1f})")
    ax.axvline(15, color="grey", ls=":", lw=1)
    ax.set_xlabel("|orig - back-mapped| residual (deg)"); ax.set_ylabel("density")
    ax.legend(fontsize=8); ax.set_title("Per-boulder rotation residual (A)", fontsize=10)


def plot_snapping(ax, snap_by_name, edges=np.array([0, 7.5, 15, 22.5])):
    """Snapping-signature figure: median residual binned by distance to the nearest grid axis
    (0/45/90/135). Flat = no snapping (null); rising toward the off-grid bin = grid-snapping (H_seg).
    ``snap_by_name`` is a dict ``{label: (res_df, color)}`` of :func:`per_boulder_rotation_residual`
    tables."""
    centers = (edges[:-1] + edges[1:]) / 2
    for name, (df, color) in snap_by_name.items():
        med = [df.residual[(df.grid_dist0 >= lo) & (df.grid_dist0 < hi)].median()
               for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(centers, med, "o-", color=color, label=name)
    ax.set_xlabel("distance of orientation to nearest grid axis (deg)")
    ax.set_ylabel("median residual (deg)")
    ax.legend(fontsize=8); ax.set_title("Snapping signature (A)", fontsize=10)


def show_crop(ax, img, title=None):
    """Show a grayscale crop on ``ax`` (uniform styling for the rotation/flip demo panels)."""
    ax.imshow(img, cmap="gray", vmin=0, vmax=255); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def mask_for_patch(model, patch, imgsz=1024, conf=0.10, device=0, max_center_dist=50):
    """Run YOLO on a square ``patch`` and return the **mask + polygon + fitted long-axis** of the
    detection nearest the centre, for *visualising* what the segmentation produces (None if nothing
    near the centre). Same mask handling as :func:`center_orientation`. Returns a dict
    ``{patch, mask (HxW 0/1), poly (col,row), cx, cy, angle180, aspect}`` — feed a patch and its
    rotated/flipped copy to see whether the mask shape tracks the rock or snaps to the image diagonal.
    """
    import cv2
    import shapely
    from scipy.ndimage import label, binary_fill_holes
    from .polygon import binary_mask_to_polygon
    from .orientation import ellipse_angle180

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
        mb = (lab == 1 + int(np.argmax([(lab == j).sum() for j in range(1, n + 1)]))).astype(np.uint8)
    mb = binary_fill_holes(mb).astype(np.uint8)
    poly = binary_mask_to_polygon(mb)
    if poly is None or len(poly) < 5:
        return None
    ang, asp = ellipse_angle180(shapely.geometry.Polygon(np.c_[poly[:, 0], -poly[:, 1]]), 1.0)
    return dict(patch=patch, mask=mb, poly=poly, cx=float(poly[:, 0].mean()),
                cy=float(poly[:, 1].mean()), angle180=float(ang), aspect=float(asp))


def plot_mask_overlay(ax, res, title=None, axis_color="tab:red", ref_angle=None, zoom_half=None):
    """Show a patch with its YOLO mask outline + fitted long-axis drawn on top (``res`` from
    :func:`mask_for_patch`). The long-axis line is drawn at ``angle180`` (azimuth from North=-row),
    so you can read off whether it follows the rock or the image diagonal. If ``ref_angle`` is given,
    a dashed yellow line is drawn at that azimuth — used by :func:`plot_rotation_strip` to show where
    the axis *would* point if it tracked the rock (so a snap is read as the red line peeling off the
    dashed one). ``zoom_half`` (px) crops the view to a ``2*zoom_half`` window centred on the **patch
    centre** (preserved under rotation, so every frame stays at the same scale and the rock stays put);
    None shows the whole patch."""
    ax.imshow(res["patch"], cmap="gray", vmin=0, vmax=255)
    p = res["poly"]
    ax.plot(np.r_[p[:, 0], p[0, 0]], np.r_[p[:, 1], p[0, 1]], color="cyan", lw=1.2)
    L = 0.45 * res["patch"].shape[0]
    if ref_angle is not None:                         # "expected if it tracked the rock" reference
        rphi = np.deg2rad(ref_angle)
        ax.plot([res["cx"] - L * np.sin(rphi), res["cx"] + L * np.sin(rphi)],
                [res["cy"] + L * np.cos(rphi), res["cy"] - L * np.cos(rphi)],
                color="yellow", lw=1.4, ls="--", zorder=2)
    phi = np.deg2rad(res["angle180"])
    dcol, drow = np.sin(phi), -np.cos(phi)            # azimuth from North (up=-row), clockwise
    ax.plot([res["cx"] - L * dcol, res["cx"] + L * dcol],
            [res["cy"] - L * drow, res["cy"] + L * drow], color=axis_color, lw=2, zorder=3)
    h, w = res["patch"].shape[:2]
    if zoom_half is not None:                          # crop to a centred window (rotation-invariant)
        cx0, cy0 = w / 2.0, h / 2.0
        ax.set_xlim(cx0 - zoom_half, cx0 + zoom_half); ax.set_ylim(cy0 + zoom_half, cy0 - zoom_half)
    else:
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def boulder_rotation_strip(model, patch, alphas, max_center_dist=50, order=3):
    """One boulder across rotations — readout A made concrete for a *single* rock. Detect the centred
    boulder in ``patch`` (the α=0 reference), then in the patch rotated by each ``alpha``; back-map
    each rotated angle (``+alpha``) and record the residual against the reference.

    Returns a list of dicts (one per frame, α=0 first): ``{alpha, res, ang_back, residual, ref_rot}``
    where ``res`` is the :func:`mask_for_patch` dict in *that* frame (or None if nothing centred),
    ``ang_back = (measured + alpha) % 180`` the geographic angle, ``residual = |ref - ang_back|``, and
    ``ref_rot = (ref - alpha) % 180`` where the long-axis should point in the rotated frame if it
    tracked the rock (the dashed reference). A tracker keeps ``residual ≈ 0`` across α; a snapper's
    residual grows because the mask re-draws on the image diagonal regardless of the rock."""
    base = mask_for_patch(model, patch, max_center_dist=max_center_dist)
    ref = None if base is None else base["angle180"]
    out = [dict(alpha=0.0, res=base, ang_back=ref, residual=(np.nan if ref is None else 0.0),
                ref_rot=ref)]
    for a in alphas:
        r = mask_for_patch(model, rotate_image(patch, a, order=order),
                           max_center_dist=max_center_dist)
        ref_rot = None if ref is None else (ref - a) % 180.0
        if r is None:
            out.append(dict(alpha=float(a), res=None, ang_back=None, residual=np.nan, ref_rot=ref_rot))
            continue
        back = (r["angle180"] + a) % 180.0
        resid = np.nan if ref is None else float(_circular_diff(ref, back))
        out.append(dict(alpha=float(a), res=r, ang_back=back, residual=resid, ref_rot=ref_rot))
    return out


def plot_rotation_strip(axes, strip, color="tab:red", zoom_half=None, drift_px=12.0,
                        area_lo=0.6, area_hi=1.8):
    """Plot a :func:`boulder_rotation_strip` as a row of mask overlays (one panel per rotation frame).
    Each panel shows the rock in its rotated frame with the YOLO mask (cyan), the fitted long-axis
    (``color``), and the dashed reference (where the axis would point if it tracked). Titles give α,
    the back-mapped geographic angle, the residual, and the **mask area + centroid drift** so you can
    see whether the detection is still the *same* boulder. A panel is **flagged** (red frame +
    "mask unstable") when the centroid drifts > ``drift_px`` from the patch centre or the mask area
    departs the α=0 reference by more than [``area_lo``, ``area_hi``]× — i.e. the segmentation no
    longer tracks one boulder under rotation, itself a non-equivariance/``H_seg`` signature (distinct
    from a stable mask whose axis merely snaps to the grid). ``zoom_half`` (px) crops every panel to
    the same centred window (passed to :func:`plot_mask_overlay`)."""
    import matplotlib.patches as mpatches

    ref = strip[0]["res"] if strip else None
    area0 = None if ref is None else float(np.asarray(ref["mask"]).sum())
    for ax, fr in zip(np.atleast_1d(axes), strip):
        r = fr["res"]
        if r is None:
            ax.axis("off"); ax.set_title(f"α={fr['alpha']:.0f}°\n(no detection)", fontsize=9)
            continue
        h, w = r["patch"].shape[:2]
        area = float(np.asarray(r["mask"]).sum())
        drift = float(np.hypot(r["cx"] - w / 2.0, r["cy"] - h / 2.0))
        ratio = (area / area0) if area0 else np.nan
        if fr["alpha"] == 0:
            t = f"α=0   axis={r['angle180']:.0f}°  (reference)\narea={area:.0f}px"
            plot_mask_overlay(ax, r, title=t, axis_color=color, zoom_half=zoom_half)
            continue
        unstable = (drift > drift_px) or (area0 is not None and (ratio < area_lo or ratio > area_hi))
        t = (f"α={fr['alpha']:.0f}°   meas={r['angle180']:.0f}° → {fr['ang_back']:.0f}°   "
             f"resid {fr['residual']:.0f}°\narea={area:.0f}px ({ratio:.1f}×)  drift={drift:.0f}px"
             + ("   ⚠ mask unstable" if unstable else ""))
        plot_mask_overlay(ax, r, title=t, axis_color=color, ref_angle=fr["ref_rot"],
                          zoom_half=zoom_half)
        if unstable:
            ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                            edgecolor="red", lw=3, zorder=10, clip_on=False))


def crop_around(img, cx, cy, half=14):
    """Square crop of ~``half``-radius pixels around ``(cx, cy)`` (clamped to the image), for zoomed
    single-boulder insets of the synthetic fields. Returns a 2D array (a view into ``img``)."""
    cx, cy = int(round(cx)), int(round(cy))
    r0, r1 = max(cy - half, 0), min(cy + half, img.shape[0])
    c0, c1 = max(cx - half, 0), min(cx + half, img.shape[1])
    return img[r0:r1, c0:c1]


# --- corroborating probes C (resolution), D (kernel), E (grid-corrected estimator) -----------

def resolution_sweep(model, crop, factors, base_res, aspect_min=1.35, n_boot=200, seed=0, device=0):
    """Readout C: resample ``crop`` by each ``factor`` (``<1`` coarser via area-average, ``>1`` finer
    via cubic), detect, and read the peak. The scene is north-up so the image frame *is* the
    geographic frame -- no un-rotation. Returns ``[factor, eff_mpp, n, peak, lo, hi]`` (effective
    m/px = ``base_res/factor``). Peak that moves with ``eff_mpp`` -> resolution/pixel-tied artifact;
    peak that stays -> NOT clean evidence against an artifact (Panozzo Heilbronner: the grid lock is
    scale-invariant), so read this one-directionally."""
    import cv2
    from .orientation import bootstrap_peak_ci
    from .orientation_validation import detect_orientations

    H, W = crop.shape
    rows = []
    for f in factors:
        interp = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
        r = cv2.resize(crop, (max(int(W * f), 32), max(int(H * f), 32)), interpolation=interp)
        det = detect_orientations(model, r, aspect_min=aspect_min, device=device)
        el = det[det.aspect >= aspect_min] if len(det) else det
        if len(el) == 0:
            rows.append((float(f), base_res / f, 0, np.nan, np.nan, np.nan)); continue
        pk, lo, hi = bootstrap_peak_ci(el.angle180.values, n_boot=n_boot, seed=seed)
        rows.append((float(f), base_res / f, len(el), float(pk), float(lo), float(hi)))
    return pd.DataFrame(rows, columns=["factor", "eff_mpp", "n", "peak", "lo", "hi"])


def kernel_sweep(model, crop, orders=(0, 1, 3), alpha=22.5, aspect_min=1.35, tol=10.0, device=0):
    """Readout D (bonus H_grid probe): resample ``crop`` by rotating a fixed off-grid ``alpha`` with
    different spline ``orders`` (0=nearest, 1=bilinear, 3=cubic), detect in the rotated frame, and
    measure the fraction of long axes within ``tol`` of the grid diagonal (45/135). Strength rising
    with kernel support => an interpolation-injected diagonal smear (H_grid). Returns
    ``[order, n, frac_diag, peak]``."""
    from scipy.ndimage import rotate as ndrotate
    from .orientation import refine_peak
    from .orientation_validation import detect_orientations

    rows = []
    for o in orders:
        r = ndrotate(crop, alpha, reshape=False, order=o, mode="constant", cval=0, prefilter=(o > 1))
        det = detect_orientations(model, r.astype(np.uint8), aspect_min=aspect_min, device=device)
        el = det[det.aspect >= aspect_min] if len(det) else det
        if len(el) == 0:
            rows.append((o, 0, np.nan, np.nan)); continue
        d = _circular_diff(el.angle180.values, 45.0)
        d = np.minimum(d, _circular_diff(el.angle180.values, 135.0))
        rows.append((o, len(el), float((d <= tol).mean()), float(refine_peak(el.angle180.values))))
    return pd.DataFrame(rows, columns=["order", "n", "frac_diag", "peak"])


def rotation_combined_angles(model, img, alphas, aspect_min=1.35, device=0):
    """MHHC de-bias combine (Šilhavý et al. 2016): pool the geographic (un-rotated) long-axis angles
    across rotations of ``img`` by each ``alpha``. A grid-locked peak **spreads/dissolves** when
    pooled (each rotation puts it at image-135 -> geographic 135+alpha); an in-pixel peak reinforces.
    Returns one 1-D array of geographic azimuths."""
    from .orientation_validation import detect_orientations

    out = []
    for a in alphas:
        det = detect_orientations(model, rotate_image(img, a), aspect_min=aspect_min, device=device)
        el = det[det.aspect >= aspect_min] if len(det) else det
        if len(el):
            out.append((el.angle180.values + a) % 180.0)
    return np.concatenate(out) if out else np.array([])


def grid_corrected_angles(gpkg_path, res, smooth_px=2.0, n=20000, aspect_min=1.0, seed=0,
                          n_total=None):
    """Readout E: re-measure each predicted polygon's long-axis azimuth with a **boundary-smoothing**
    step before EllipseModel (Panozzo Heilbronner 1988 -- a smoothing spline on the digitised
    boundary "altogether avoids" the grid-discretisation distortion), versus the raw EllipseModel
    angle on the same polygon. Returns ``[angle_raw, angle_smooth, aspect]``.

    The boundary is resampled to ``res`` spacing then circularly smoothed with a Gaussian of
    ``smooth_px`` pixels before the fit. If the ~135 peak shrinks under smoothing, the lock lives in
    the contour discretisation (corroborates H_seg / is a candidate Goal-2 fix); if it persists, it
    is in the mask shape itself.
    """
    from pyogrio import read_dataframe, read_info

    if n_total is None:
        n_total = read_info(gpkg_path)["features"]
    rng = np.random.default_rng(seed)
    fids = np.sort(rng.choice(n_total, size=min(n, n_total), replace=False))
    g = read_dataframe(gpkg_path, fids=fids)

    rows = []
    for geom in g.geometry.values:
        sb = smooth_boundary(geom, res, smooth_px)
        if sb is None or not np.isfinite(sb["aspect"]) or sb["aspect"] < aspect_min:
            continue
        rows.append((sb["raw_ang"], sb["smooth_ang"], sb["aspect"]))
    return pd.DataFrame(rows, columns=["angle_raw", "angle_smooth", "aspect"])


def smooth_boundary(geom, res, smooth_px):
    """Boundary-smoothing primitive for readout E (and its §9 viz, so both use identical code).
    Densify ``geom`` to ``res`` spacing, circularly Gaussian-smooth the exterior (sigma ``smooth_px``
    vertices), and return ``dict(raw_xy, smooth_xy, raw_ang, smooth_ang, aspect)`` — coords in map
    units (y=North), angles in the :func:`ellipse_angle180` convention. ``None`` if unusable."""
    import shapely
    from scipy.ndimage import gaussian_filter1d
    from skimage.measure import EllipseModel
    from .orientation import ellipse_angle180

    raw_ang, asp = ellipse_angle180(geom, res)
    if not np.isfinite(asp):
        return None
    gg = shapely.segmentize(geom, res)
    if gg.geom_type == "MultiPolygon":
        gg = max(gg.geoms, key=lambda p: p.area)
    if gg.geom_type != "Polygon" or gg.exterior is None:
        return None
    xy = np.asarray(gg.exterior.coords)[:-1]              # drop the closing duplicate
    if len(xy) < 8:
        return dict(raw_xy=xy, smooth_xy=xy, raw_ang=raw_ang, smooth_ang=raw_ang, aspect=asp)
    sx = gaussian_filter1d(xy[:, 0], smooth_px, mode="wrap")
    sy = gaussian_filter1d(xy[:, 1], smooth_px, mode="wrap")
    sm = np.c_[sx, sy]
    model = EllipseModel()
    if not model.estimate(sm - sm.mean(0)):              # map coords (y=North), match ellipse_angle180
        return dict(raw_xy=xy, smooth_xy=sm, raw_ang=raw_ang, smooth_ang=raw_ang, aspect=asp)
    _, _, a, b, th = model.params
    thm = th if a >= b else th + np.pi / 2
    sang = np.degrees(np.arctan2(np.cos(thm), np.sin(thm))) % 180.0
    return dict(raw_xy=xy, smooth_xy=sm, raw_ang=raw_ang, smooth_ang=float(sang), aspect=asp)
