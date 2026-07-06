"""Shared benchmark harness for the Goal-2 orientation *solutions* (``docs/solutions/``).

Every candidate fix (super-resolution E, multi-scale fusion D, SAM2 re-masking G, ...) is scored on
the **same three axes** so results are comparable across solutions (the "benchmark card"):

1. **Time cost** -- added wall-clock per unit (per boulder / per mask / per image) vs the
   detection-only baseline. Recorded by the per-treatment timers here and summarised by
   :func:`time_cost_summary`.
2. **Orientation quality** -- grid/diagonal fraction, the 135/45 (or 0/90) asymmetry, and median
   ``|Δangle|`` vs a known truth. Computed by :func:`orientation_summary` (thin wrapper over the
   metrics already in :mod:`YOLOv8BeyondEarth.orientation`).
3. **Segmentation quality** -- box/mask mAP50 and mAP50-95 on the Prieur et al. (2023)
   ``boulder2024`` validation split via ultralytics ``model.val()`` (:func:`seg_map_eval`).

The **degrade -> SR -> measure-vs-truth** experiment (option E validation #2, ``docs/solutions``)
lives here too: it renders sharp synthetic boulders of known orientation, degrades them to native
sensor resolution, applies a super-resolution / interpolation treatment, and measures the recovered
long-axis angle the *same way production does* (mask -> :func:`polygon.binary_mask_to_polygon` ->
:func:`orientation.ellipse_angle180`). It is entirely CPU (Lanczos / cubic / EDSR-CPU); the GAN arm
(BSRGAN) and the seg-mAP eval are the only GPU pieces.

Angle convention throughout: ``angle180`` = long-axis azimuth from North (image -row), clockwise,
in [0, 180) -- identical to :func:`YOLOv8BeyondEarth.orientation.ellipse_angle180`.
"""
import time

import numpy as np
import pandas as pd

from .orientation import (
    ellipse_angle180, grid_fraction, grid_fraction_uniform,
    diagonal_fraction, asymmetry_ratio, axial_distance,
)
from .polygon import binary_mask_to_polygon, binary_mask_to_polygon_cv
from .transforms_test import render_boulder_patch


# --- mask -> orientation (the production readout, on an arbitrary binary mask) ----------------

def mask_angle180(mask, res=0.5, method="skimage"):
    """Long-axis azimuth ``(angle180, aspect_ratio)`` of a binary mask, via the production path.

    ``mask`` -> largest-contour polygon (:func:`polygon.binary_mask_to_polygon`, skimage default; or
    ``method="cv2"``) -> :func:`orientation.ellipse_angle180`. ``res`` is the densify spacing in the
    mask's own (pixel) units; angle is scale-invariant so 0.5 px matches the production 0.5 densify.
    Returns ``(nan, nan)`` when the mask is empty or the contour has <3 vertices.

    The contour's row coordinate is negated before the fit: contours are image-frame (row grows
    downward) while ``ellipse_angle180`` expects world-frame y (North up) — same flip as
    ``transforms_test.mask_for_patch``. Without it the azimuth comes back mirrored (180 - phi).
    """
    from shapely.geometry import Polygon
    to_poly = binary_mask_to_polygon_cv if method == "cv2" else binary_mask_to_polygon
    coords = to_poly(np.asarray(mask, dtype=bool))
    if coords is None or len(coords) < 3:
        return np.nan, np.nan
    g = Polygon(np.c_[coords[:, 0], -coords[:, 1]])
    if not g.is_valid:
        g = g.buffer(0)
    if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
        return np.nan, np.nan
    return ellipse_angle180(g, res)


def largest_blob(mask):
    """Keep only the largest 4-connected component of a boolean mask (drops noise specks)."""
    from scipy.ndimage import label
    m = np.asarray(mask, dtype=bool)
    lab, n = label(m)
    if n <= 1:
        return m
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def mask_from_intensity(img, thresh, keep_largest=True):
    """Binary mask of the bright object: ``img >= thresh``, optionally largest connected blob."""
    m = np.asarray(img) >= thresh
    return largest_blob(m) if keep_largest else m


# --- degrade -> SR -> measure (option E validation #2) ---------------------------------------

def upscale(img, method, factor):
    """Upscale ``img`` by integer ``factor`` with a named interpolation / SR method.

    Faithful (add no hallucinated detail, CPU): ``nearest`` (control ~ blocky), ``bilinear``,
    ``cubic``, ``lanczos``. Learned faithful (CPU): ``edsr`` (cv2 ``dnn_superres``; needs an
    ``EDSR_x{factor}.pb`` model on the cv2 model path -- raises with a clear message if absent).
    GAN (GPU, hallucinates detail): ``bsrgan`` -- deferred, raises ``NotImplementedError``.
    """
    import cv2
    if factor == 1:
        return np.asarray(img)
    interp = {
        "nearest": cv2.INTER_NEAREST, "bilinear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4,
    }
    if method in interp:
        h, w = img.shape[:2]
        return cv2.resize(img, (w * factor, h * factor), interpolation=interp[method])
    if method == "edsr":
        if not hasattr(cv2, "dnn_superres"):
            raise ModuleNotFoundError(
                "cv2.dnn_superres missing: the env ships opencv-python (not -contrib), and "
                "swapping in opencv-contrib-python risks the ultralytics cv2 pin. The "
                "learned-faithful SR arm is deferred (see docs/solutions/E_super_resolution.md).")
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        model_path = f"EDSR_x{factor}.pb"           # expected on CWD / cv2 model path
        try:
            sr.readModel(model_path)
        except cv2.error as e:                       # pragma: no cover - env dependent
            raise FileNotFoundError(
                f"EDSR_x{factor}.pb not found; download from the OpenCV model zoo to enable the "
                f"learned-faithful SR arm (cv2.dnn_superres). Original: {e}")
        sr.setModel("edsr", factor)
        rgb = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out = sr.upsample(rgb)
        return out if img.ndim == 3 else cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if method == "bsrgan":
        raise NotImplementedError("BSRGAN is the GPU/GAN arm; run it in the GPU-gated notebook cell.")
    raise ValueError(f"unknown upscale method {method!r}")


def degrade(hr, factor):
    """Downsample a high-res patch by ``factor`` with area averaging (sensor-like integration)."""
    import cv2
    h, w = hr.shape[:2]
    return cv2.resize(hr, (w // factor, h // factor), interpolation=cv2.INTER_AREA)


def measure_treatment(lr, method, sr_factor, thresh, res=0.5, contour="skimage",
                      domain="intensity"):
    """Measure ``(angle180, aspect, seconds)`` for one SR treatment of a low-res patch.

    ``method``/``sr_factor`` name the upscaler (``sr_factor=1`` + ``method="nearest"`` = the blocky
    baseline: measure the native LR mask). Timer covers only the upscale+mask+fit (the added cost a
    solution pays per boulder). ``thresh`` binarises the bright object.

    ``domain`` selects *what* gets upscaled — the mechanistic split from the solution plan:
    - ``"intensity"``: upscale the LR **image**, then threshold. The soft grey edge carries
      sub-pixel boundary information, so a faithful upscaler can genuinely localise the boundary
      better (what option E's SR-before-detection does, minus the detector).
    - ``"mask"``: threshold at native LR first, then upscale the **binary mask** and re-threshold
      at 0.5 — no new information by construction (the documented ``retina_masks``/upsample
      dead-end); the control arm any real fix must beat.
    """
    t0 = time.perf_counter()
    if domain == "mask":
        m_lr = mask_from_intensity(lr, thresh)
        if sr_factor == 1:
            mask = m_lr
        else:
            up = upscale(m_lr.astype(np.float32), method, sr_factor)
            mask = up >= 0.5
    else:
        up = np.asarray(lr) if sr_factor == 1 else upscale(lr, method, sr_factor)
        mask = mask_from_intensity(up, thresh)
    angle, aspect = mask_angle180(mask, res=res, method=contour)
    dt = time.perf_counter() - t0
    return angle, aspect, dt


def run_degrade_sr_experiment(treatments, n=400, diam_px=(6, 12), aspect=(1.7, 3.0),
                              degrade_factor=8, bg=110, cap=80.0, noise=6.0, lr_noise=0.0,
                              patch_px=320, contour="skimage", seed=0):
    """Render ``n`` synthetic boulders at uniform-random known angles, degrade each to native sensor
    resolution, and measure recovered orientation under each treatment. Returns a tidy DataFrame.

    ``treatments`` : list of ``(name, method, sr_factor)`` or ``(name, method, sr_factor, domain)``
    -- e.g. ``[("blocky", "nearest", 1), ("lanczos_x4", "lanczos", 4),
    ("mask_lanczos_x4", "lanczos", 4, "mask")]``; ``domain`` defaults to ``"intensity"``
    (see :func:`measure_treatment` for the intensity-vs-mask mechanistic split).
    Each boulder is rendered sharp at ``degrade_factor``x the target size (shadow off, low noise --
    isolate geometry), area-downsampled by ``degrade_factor`` to give an LR patch whose boulder spans
    ``diam_px`` native px (the small/marginal regime where the grid bias lives). Threshold is the
    bright-cap midpoint ``bg + cap/2``. Also fits the *sharp HR* mask as an ``hr_oracle`` sanity row.
    ``noise`` is rendered at HR (the area downsample averages it down ~``degrade_factor``x);
    ``lr_noise`` adds Gaussian sigma to the LR patch *after* degradation -- use it for the
    marginal-SNR sweep, where sensor noise is per native pixel and no upscaler can average it away.

    Columns: ``[i, truth, diam_px, aspect_true, treatment, angle180, aspect_meas, abs_err, seconds]``.
    """
    rng = np.random.default_rng(seed)
    thresh = bg + cap / 2.0
    rows = []
    for i in range(n):
        phi = float(rng.uniform(0, 180))
        d_lr = float(rng.uniform(*diam_px))
        asp = float(rng.uniform(*aspect))
        hr = render_boulder_patch(size=patch_px, phi=phi, diam_px=d_lr * degrade_factor,
                                  aspect=asp, bg=bg, cap=cap, noise=noise, shadow_frac=0.0,
                                  seed=int(rng.integers(0, 2**31 - 1)))
        lr = degrade(hr, degrade_factor)
        if lr_noise > 0:
            lr = np.clip(lr.astype(np.float64) + rng.normal(0, lr_noise, lr.shape),
                         0, 255).astype(np.uint8)

        # HR oracle: fit the sharp mask (should recover phi to <1 deg, no grid bias)
        a_hr, asp_hr = mask_angle180(mask_from_intensity(hr, thresh), res=0.5, method=contour)
        rows.append((i, phi, d_lr, asp, "hr_oracle", a_hr, asp_hr,
                     float(axial_distance(a_hr, phi)) if np.isfinite(a_hr) else np.nan, np.nan))

        for t in treatments:
            name, method, sr_factor = t[:3]
            domain = t[3] if len(t) > 3 else "intensity"
            angle, aspect_m, dt = measure_treatment(lr, method, sr_factor, thresh,
                                                    res=0.5, contour=contour, domain=domain)
            err = float(axial_distance(angle, phi)) if np.isfinite(angle) else np.nan
            rows.append((i, phi, d_lr, asp, name, angle, aspect_m, err, dt))

    return pd.DataFrame(rows, columns=["i", "truth", "diam_px", "aspect_true", "treatment",
                                       "angle180", "aspect_meas", "abs_err", "seconds"])


# --- metric summaries (the benchmark card) ----------------------------------------------------

def orientation_summary(df, angle_col="angle180", group="treatment", tol_deg=10.0,
                        grid_axis="cardinal"):
    """Per-group orientation metrics -> DataFrame (one row per group).

    ``grid_axis``: ``"cardinal"`` (0/90, NAC) uses :func:`orientation.grid_fraction`; ``"diagonal"``
    (45/135, HiRISE north-up) uses :func:`orientation.diagonal_fraction`. ``median_abs_err`` needs an
    ``abs_err`` column (present for the degrade experiment); it is NaN otherwise. ``uniform`` is the
    unbiased baseline both fractions should approach.
    """
    frac = grid_fraction if grid_axis == "cardinal" else diagonal_fraction
    out = []
    for name, g in df.groupby(group, sort=False):
        a = g[angle_col].dropna().values
        rec = {
            group: name, "n": len(a),
            "grid_frac" if grid_axis == "cardinal" else "diag_frac": frac(a, tol_deg),
            "asymmetry_135_45": asymmetry_ratio(a, tol_deg),
        }
        if "abs_err" in g.columns:
            rec["median_abs_err"] = float(np.nanmedian(g["abs_err"].values))
            rec["p90_abs_err"] = float(np.nanpercentile(g["abs_err"].dropna().values, 90)) \
                if g["abs_err"].notna().any() else np.nan
        out.append(rec)
    res = pd.DataFrame(out)
    res.attrs["uniform"] = grid_fraction_uniform(tol_deg)
    return res


def match_detections(gdf_a, gdf_b, max_dist):
    """Spatially match two detection sets by centroid distance -> DataFrame ``[idx_a, idx_b, dist]``.

    Greedy one-to-one matching: candidate pairs within ``max_dist`` (map units) are taken in
    increasing-distance order; each detection matches at most once. This is the standing
    "spatially-matched pairing" gate for real-scene before/after claims (docs/solutions/README.md) —
    population shift confounded the retina/slice aggregate comparisons, so cross-treatment
    orientation deltas must be computed on these pairs, not on the raw per-run populations.
    ``idx_a``/``idx_b`` are positional indices into ``gdf_a``/``gdf_b`` (use ``.iloc``).
    """
    from scipy.spatial import cKDTree

    ca = np.column_stack((gdf_a.geometry.centroid.x.values, gdf_a.geometry.centroid.y.values))
    cb = np.column_stack((gdf_b.geometry.centroid.x.values, gdf_b.geometry.centroid.y.values))
    if len(ca) == 0 or len(cb) == 0:
        return pd.DataFrame(columns=["idx_a", "idx_b", "dist"])
    pairs = cKDTree(ca).query_ball_tree(cKDTree(cb), r=max_dist)
    cand = [(np.hypot(*(ca[i] - cb[j])), i, j) for i, js in enumerate(pairs) for j in js]
    cand.sort()
    used_a, used_b, rows = set(), set(), []
    for d, i, j in cand:
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j)
        rows.append((i, j, d))
    return pd.DataFrame(rows, columns=["idx_a", "idx_b", "dist"])


# --- size-band orientation readouts (the Stage-2 / G card axis) --------------------------------

# Native-px size bands shared by the D/E three-arm and G re-masking benchmark cards. The artifact
# lives in the low-px bands (the 3f dose-response), so the bands are fine-grained there and pool the
# well-resolved tail. Edges are half-open [lo, hi); the last band is [14, inf).
DEFAULT_SIZE_BANDS = ((3, 4), (4, 6), (6, 8), (8, 10), (10, 14), (14, np.inf))


def _band_label(lo, hi):
    return f"{lo}-{'inf' if not np.isfinite(hi) else hi}"


def orientation_by_band(angle180, aspect, dpx, grid_axis="diagonal", aspect_min=1.35,
                        tol_deg=10.0, bands=DEFAULT_SIZE_BANDS, min_n=30):
    """Grid/diagonal fraction + 135/45 chirality per native-px size band, on the elongated subset.

    The per-arm (D/E) and per-side (G) card readout: given aligned ``angle180`` / ``aspect`` /
    ``dpx`` (native-px equivalent diameter) arrays, restrict to ``aspect > aspect_min`` and finite
    angle, then report :func:`orientation.diagonal_fraction` (``grid_axis="diagonal"``, HiRISE
    north-up 45/135) or :func:`orientation.grid_fraction` (``"cardinal"``, NAC 0/90) and
    :func:`orientation.asymmetry_ratio` for the ``"all"`` row and each band with >= ``min_n``
    members. Returns a DataFrame ``[band, n, frac, chirality]`` (``frac`` column is the fraction
    either way; the caller knows the axis). ``uniform`` is on ``.attrs``.
    """
    frac = diagonal_fraction if grid_axis == "diagonal" else grid_fraction
    a = np.asarray(angle180, float); asp = np.asarray(aspect, float); d = np.asarray(dpx, float)
    el = (asp > aspect_min) & np.isfinite(a)
    rows = [{"band": "all", "n": int(el.sum()),
             "frac": frac(a[el], tol_deg), "chirality": asymmetry_ratio(a[el], tol_deg)}]
    for lo, hi in bands:
        m = el & (d >= lo) & (d < hi)
        if m.sum() >= min_n:
            rows.append({"band": _band_label(lo, hi), "n": int(m.sum()),
                         "frac": frac(a[m], tol_deg), "chirality": asymmetry_ratio(a[m], tol_deg)})
    res = pd.DataFrame(rows)
    res.attrs["uniform"] = grid_fraction_uniform(tol_deg)
    return res


def remask_gating_sweep(dpx, aspect_gate, angle_base, angle_remask, taus,
                        grid_axis="diagonal", aspect_min=1.35, tol_deg=10.0):
    """Sweep a size gate for detector-preserving re-masking (Solution G production question).

    Builds a *combined* rose that uses the re-masked angle where ``dpx < tau`` and the baseline
    (YOLO) angle otherwise, for each threshold in ``taus`` — the operating curve for "re-mask only
    the small subset". The elongated subset is selected by ``aspect_gate`` (the baseline aspect,
    since the production gate is applied *before* re-masking). Returns a DataFrame
    ``[tau, remask_frac, frac, chirality, n_elong]`` where ``remask_frac`` is over all detections.
    Pass ``np.inf`` in ``taus`` for the re-mask-everything row. ``uniform`` is on ``.attrs``.
    """
    frac_fn = diagonal_fraction if grid_axis == "diagonal" else grid_fraction
    d = np.asarray(dpx, float); asp = np.asarray(aspect_gate, float)
    ab = np.asarray(angle_base, float); ar = np.asarray(angle_remask, float)
    rows = []
    for tau in taus:
        use_remask = d < tau
        ang = np.where(use_remask, ar, ab)
        el = (asp > aspect_min) & np.isfinite(ang)
        rows.append({"tau": ("inf" if not np.isfinite(tau) else tau),
                     "remask_frac": float(use_remask.mean()),
                     "frac": frac_fn(ang[el], tol_deg),
                     "chirality": asymmetry_ratio(ang[el], tol_deg),
                     "n_elong": int(el.sum())})
    res = pd.DataFrame(rows)
    res.attrs["uniform"] = grid_fraction_uniform(tol_deg)
    return res


def time_cost_summary(df, group="treatment", per_unit="boulder"):
    """Per-group timing -> DataFrame: mean/median seconds per unit and the multiple vs the fastest
    group. ``per_unit`` is a label only (the rows are per whatever unit ``seconds`` was timed over)."""
    out = []
    for name, g in df.groupby(group, sort=False):
        s = g["seconds"].dropna().values
        if s.size == 0:
            continue
        out.append({group: name, f"mean_s_per_{per_unit}": float(s.mean()),
                    f"median_s_per_{per_unit}": float(np.median(s)), "n": int(s.size)})
    res = pd.DataFrame(out)
    if len(res):
        base = res[f"mean_s_per_{per_unit}"].min()
        res["x_vs_fastest"] = res[f"mean_s_per_{per_unit}"] / base if base > 0 else np.nan
    return res


# --- segmentation mAP on the Prieur boulder2024 val split (GPU) -------------------------------

def seg_map_eval(weights, data_yaml, split="val", imgsz=1024, device=0, conf=0.001, iou=0.6,
                 **val_kwargs):
    """Box + mask mAP for a YOLO-seg model on a ``data_yaml`` split via ultralytics ``model.val()``.

    Returns ``dict`` with ``box_map50``, ``box_map`` (50-95), ``mask_map50``, ``mask_map``, plus the
    per-image ``speed`` breakdown (ms). GPU. This is the segmentation-quality axis every solution's
    card reports; run once for the stock model to get the reference the fixes are compared against.
    """
    from ultralytics import YOLO
    m = YOLO(weights)
    r = m.val(data=str(data_yaml), split=split, imgsz=imgsz, device=device, conf=conf, iou=iou,
              plots=False, verbose=False, **val_kwargs)
    box, seg = r.box, getattr(r, "seg", None)
    out = {"box_map50": float(box.map50), "box_map": float(box.map), "speed_ms": dict(r.speed)}
    if seg is not None:
        out.update({"mask_map50": float(seg.map50), "mask_map": float(seg.map)})
    return out
