"""The pilot's pixel noise, measured beside hirise2ctx's R11 (their Stage 1 note, section 6, question 1).

hirise2ctx measured each delivered raster's pixel noise as 1.4826 MAD of the 3 x 3 kernel
[[1,-2,1],[-2,4,-2],[1,-2,1]], / 6, in 300 seeded 128 px windows (``hirise2ctx.delivery_views.pixel_noise``,
read here, not imported). They found the six images that lost label noise-dominated. This checks that reading on
the same windows, in three ways.

1. **Reproduce.** Their sampler and statistic, re-implemented, against their ``r11/noise.json``.
2. **Other estimators.**
   - The continuous Immerkaer form, sqrt(pi/2) mean|kernel| / 6. Their MAD form moves in steps of 0.247 DN on
     8-bit data.
   - Robust first differences across columns (dx) and down columns (dy). The kernel is the outer product of
     [1, -2, 1] with itself, so its response is 16 sin^2(pi kx) sin^2(pi ky): zero for anything constant along
     either image axis, column stripes included.
3. **Physical units.** I/F = DN10 x SCALING_FACTOR + OFFSET (the label), and an 8-bit DN is 4 10-bit DN
   (``min((DN + 2) >> 2, 255)``). That gives the noise in I/F, the window's contrast in I/F (its 1-99 % span) and
   the scene's mean I/F (the prep sidecar's 10-bit histogram). So "noise-dominated" (noise / span) can be split
   into more noise and less contrast.

Then the column pattern hirise2ctx saw in ESP_049730_2225, at two scales, for every scene:
- **fine:** the narrowest peaks of the scene's average 2D power spectrum over the same windows (plane removed,
  Hann taper), as power over the median of its |k| annulus; periods 2-64 px, with orientation;
- **coarse:** persistent across-track brightness steps (CCD and channel seams), found as in
  ``sherlock_pilot_analysis.section_steps`` (1 km along-track segments, a step = a jump of >= 0.5 DN with the
  same sign in >= 85 % of >= 10 segments, >= 150 m apart).

Read-only on the delivery. A description of image quality, not a correction, and nothing here changes a label.

    conda run -n bouldernet python -I scripts/review/pilot_noise_check.py --out data/review/pilot_noise_check.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
DELIVERY = REPO.parent / "data" / "deliveries" / "hirise2ctx_v3" / "pilot-1"
H2C_NOISE = Path(r"C:/Users/brian/Documents/PhD/HiRiseToCTXBoulders/v3_data/reports/stage1_acceptance/"
                 r"pilot-1/r11/noise.json")
INCIDENCE = REPO / "data" / "review" / "incidence_pairs_pilot.json"
KERNEL = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])
SQRT_PI_2 = math.sqrt(math.pi / 2.0)


def mad(x):
    x = np.asarray(x, dtype=float).ravel()
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def label_fields(lbl):
    from YOLOv8BeyondEarth import pds_prep
    f = pds_prep.read_label(lbl)
    get = lambda k: pds_prep._lookup(f, k, False)                            # noqa: E731
    num = lambda k: float(re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", str(get(k)))[0])   # noqa: E731
    ints = lambda k: [int(x) for x in re.findall(r"-?\d+", str(get(k) or ""))]          # noqa: E731
    flags = re.findall(r"ON|OFF", str(get("MRO:CCD_FLAG")))
    on = [k for k in range(min(10, len(flags))) if flags[k] == "ON"]
    tdi, binning = ints("MRO:TDI"), ints("MRO:BINNING")
    # the HiRISE pipeline's own per-CCD flag: CUBENORM is its column normalisation, ISIS's fix for
    # vertical striping
    special = re.findall(r"[A-Z_]+", str(get("MRO:SPECIAL_PROCESSING_FLAG") or ""))
    special_on = [special[k] for k in on if k < len(special)]
    return {"special_processing_on": sorted(set(special_on)),
            "cubenorm": bool(special_on) and all(x == "CUBENORM" for x in special_on),
            "scaling_factor": num("SCALING_FACTOR"), "offset": num("OFFSET"),
            "incidence": num("INCIDENCE_ANGLE"), "emission": num("EMISSION_ANGLE"), "phase": num("PHASE_ANGLE"),
            "ls": num("SOLAR_LONGITUDE"), "local_time": num("LOCAL_TIME"), "start_time": str(get("START_TIME")),
            "center_latitude": num("CENTER_LATITUDE"), "map_scale": num("MAP_SCALE"),
            "stretch_dn10": [num("MRO:MINIMUM_STRETCH"), num("MRO:MAXIMUM_STRETCH")],
            "red_on": [f"RED{k}" for k in on], "tdi_on": sorted({tdi[k] for k in on if k < len(tdi)}),
            "binning_on": sorted({binning[k] for k in on if k < len(binning)})}


def scene_dn10(prep):
    """Mean, sd and 1 / 99 % of the scene's valid 10-bit DN, from the prep sidecar's histogram.

    The histogram (``histogram_nonzero``) keeps a DN-0 bin, which is the fill (it equals
    ``dn_counts.null_fill``, 18-43 % of these rasters); it is dropped here."""
    h = json.loads(Path(prep).read_text())["dn_counts"]["histogram_nonzero"]
    h = {k: v for k, v in h.items() if int(k) > 0}
    dn = np.array([int(k) for k in h], dtype=float)
    n = np.array(list(h.values()), dtype=float)
    o = np.argsort(dn)
    dn, n = dn[o], n[o]
    c = np.cumsum(n) / n.sum()
    mean = float((dn * n).sum() / n.sum())
    return {"mean": mean, "sd": float(np.sqrt((n * (dn - mean) ** 2).sum() / n.sum())),
            "p01": float(dn[np.searchsorted(c, 0.01)]), "p99": float(dn[np.searchsorted(c, 0.99)]),
            "n_valid": int(n.sum())}


def sample_windows(tif, n_windows=300, window=128, seed=0, decimation=64):
    """hirise2ctx's sampler (delivery_views.pixel_noise), re-implemented call for call.

    Each window is read twice. Their ``read_window`` defaults to ``resampling="average"`` in a boundless read,
    and that read returns slightly resampled values (their spans are not whole DN). So their statistic is
    reproduced on that read, and every other measure here uses the raw pixels the detector sees."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    from scipy import ndimage
    with rasterio.Env(GDAL_PAM_ENABLED="NO"), rasterio.open(tif) as src:
        h, w = src.height, src.width
        ov = src.read(1, window=Window(0, 0, w, h), out_shape=(h // decimation, w // decimation), boundless=True,
                      fill_value=0, resampling=Resampling.nearest)
        cand = np.argwhere(ndimage.binary_erosion(ov > 0, iterations=max(1, int(math.ceil(window / decimation)) + 2)))
        rng = np.random.default_rng(seed)
        if len(cand):
            cand = cand[rng.choice(len(cand), size=min(n_windows, len(cand)), replace=False)]
        out, origins, theirs = [], [], []
        for r, c in cand:
            win = Window(int(c * decimation), int(r * decimation), window, window)
            t = src.read(1, window=win, boundless=True, fill_value=0, resampling=Resampling.average)
            if not (t == 0).any():
                theirs.append(t)
            a = src.read(1, window=win, boundless=True, fill_value=0)
            if (a == 0).any():
                continue
            out.append(a)
            origins.append((int(c * decimation), int(r * decimation)))
    return out, ov, (h, w), origins, theirs


def window_stats(a):
    from scipy import ndimage
    a = a.astype(float)
    lap = ndimage.convolve(a, KERNEL, mode="reflect")[1:-1, 1:-1]
    lo, hi = np.percentile(a, [1, 99])
    dx, dy = np.diff(a, axis=1), np.diff(a, axis=0)
    return {"sigma_lap_mad": 1.4826 * float(np.median(np.abs(lap - np.median(lap)))) / 6.0,
            "sigma_lap_meanabs": SQRT_PI_2 * float(np.mean(np.abs(lap))) / 6.0,
            "sigma_dx_mad": mad(dx) / math.sqrt(2), "sigma_dy_mad": mad(dy) / math.sqrt(2),
            # continuous (mean-absolute) forms: the MAD of integer differences moves in steps of 1.05 DN
            "sigma_dx_meanabs": SQRT_PI_2 * float(np.mean(np.abs(dx - np.median(dx)))) / math.sqrt(2),
            "sigma_dy_meanabs": SQRT_PI_2 * float(np.mean(np.abs(dy - np.median(dy)))) / math.sqrt(2),
            "span": float(hi - lo), "mean": float(a.mean())}


def spectrum(windows, swath_deg=None, half_width_deg=4.0):
    """Average 2D power over the windows (plane removed, Hann taper), its narrowest peaks, and the power in
    narrow wedges: across track (where stripes constant along the detector's columns put it) and along
    track, each over the median of its |k| annulus, for 0.05 <= |k| <= 0.45 cycles per pixel."""
    n = windows[0].shape[0]
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    A = np.column_stack([np.ones(n * n), xx.ravel(), yy.ravel()])
    pinv = np.linalg.pinv(A)
    hann = np.outer(np.hanning(n), np.hanning(n))
    P = np.zeros((n, n))
    for a in windows:
        a = a.astype(float)
        resid = a.ravel() - A @ (pinv @ a.ravel())
        P += np.abs(np.fft.fft2(resid.reshape(n, n) * hann)) ** 2
    P /= len(windows)
    k = np.fft.fftfreq(n)
    KX, KY = np.meshgrid(k, k)
    KR = np.hypot(KX, KY)
    rbin = np.round(KR * n).astype(int)
    ratio = np.zeros_like(P)
    for b in np.unique(rbin):
        m = rbin == b
        ratio[m] = P[m] / np.median(P[m])
    ok = (KR >= 2.0 / n) & ((KY > 0) | ((KY == 0) & (KX > 0)))          # one of each +-k pair, no DC
    idx = np.argsort(ratio[ok])[::-1][:6]
    peaks = []
    for i in idx:
        kx, ky, r = KX[ok][i], KY[ok][i], ratio[ok][i]
        peaks.append({"kx": float(kx), "ky": float(ky), "period_px": float(1.0 / math.hypot(kx, ky)),
                      "wavevector_deg_from_x": float(math.degrees(math.atan2(ky, kx))),
                      "power_over_annulus_median": float(r)})
    # the strongest response on the two Nyquist lines, where an odd/even column or line pattern would sit
    nyq_x = ratio[:, n // 2]
    nyq_y = ratio[n // 2, :]
    # radially averaged power (Hann-windowed, plane removed), in log-spaced |k| bins: blur steepens it, while
    # less contrast only lowers it
    edges = np.geomspace(1.0 / n, 0.5 * math.sqrt(2), 25)
    rad = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (KR >= lo) & (KR < hi)
        if m.any():
            rad.append({"k_lo": float(lo), "k_hi": float(hi), "power": float(P[m].mean())})
    wedges = None
    if swath_deg is not None:
        # numpy's ky runs down the rows; columns stripes' wavevector is the across-track unit vector, at the
        # swath's angle from the x axis in (kx, ky-down) coordinates (swath_deg: long axis from vertical,
        # positive = top leans east)
        ang = np.degrees(np.arctan2(KY, KX))
        band = (KR >= 0.05) & (KR <= 0.45)

        def wedge(center):
            """Median over annuli (broadband stripes) and the maximum, with where it sits (narrowband)."""
            dev = np.abs(((ang - center + 90.0) % 180.0) - 90.0)
            vals, ks = [], []
            for b in np.unique(rbin[band]):
                m = band & (rbin == b)
                w = m & (dev <= half_width_deg)
                if w.any():
                    vals.append(float(P[w].mean() / np.median(P[m])))
                    ks.append(float(b) / n)
            i = int(np.argmax(vals))
            return {"median_ratio": float(np.median(vals)), "max_ratio": float(vals[i]),
                    "max_at_k": ks[i], "max_at_period_px": 1.0 / ks[i], "n_annuli": len(vals),
                    "n_annuli_over_3": int(sum(v > 3.0 for v in vals))}
        wedges = {"across_track_column_stripes": wedge(swath_deg), "along_track_row_stripes": wedge(swath_deg + 90.0),
                  "image_x_axis": wedge(0.0), "image_y_axis": wedge(90.0), "half_width_deg": half_width_deg}
    return {"peaks": peaks, "max_on_kx_nyquist": float(nyq_x.max()), "max_on_ky_nyquist": float(nyq_y.max()),
            "radial_power": rad, "wedges": wedges}


def swath_angle(ov):
    """Degrees of the valid footprint's long axis from image vertical (positive = top leans east)."""
    rr, cc = np.nonzero(ov > 0)
    xy = np.column_stack([cc, -rr]).astype(float)
    xy -= xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy[:: max(1, len(xy) // 100000)], full_matrices=False)
    vx, vy = vt[0]
    if vy < 0:
        vx, vy = -vx, -vy
    return float(math.degrees(math.atan2(vx, vy)))


def across_swath(ov, origins, sigma, span, n_bins=12, decimation=64, window=128):
    """The windows' noise in bins of across-track position: a channel or CCD that is noisier than the rest
    (bit flips add noise in some channels) stands out here and is averaged away in the scene's median."""
    rr, cc = np.nonzero(ov > 0)
    xy = np.column_stack([cc, rr]).astype(float) * decimation
    mu = xy.mean(axis=0)
    _, _, vt = np.linalg.svd((xy - mu)[:: max(1, len(xy) // 100000)], full_matrices=False)
    across = vt[1] if vt[1][0] >= 0 else -vt[1]
    ext = (xy - mu) @ across
    w = (np.asarray(origins, float) + window / 2.0 - mu) @ across
    edges = np.linspace(ext.min(), ext.max(), n_bins + 1)
    b = np.clip(np.digitize(w, edges) - 1, 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        rows.append({"u_frac": float((edges[k] + edges[k + 1]) / 2 - ext.min()) / float(ext.max() - ext.min()),
                     "n": int(m.sum()), "sigma_lap_mad": float(np.median(sigma[m])) if m.sum() >= 5 else None,
                     "noise_over_span": float(np.median(sigma[m] / np.maximum(span[m], 1.0))) if m.sum() >= 5 else None})
    v = [r["sigma_lap_mad"] for r in rows if r["sigma_lap_mad"] is not None]
    if not v:
        return {"bins": rows, "max_over_min_sigma": None, "max_over_median_of_rest": None}
    k = int(np.argmax(v))
    rest = v[:k] + v[k + 1:]
    return {"bins": rows, "max_over_min_sigma": float(max(v) / min(v)) if min(v) > 0 else None,
            "max_over_median_of_rest": float(v[k] / np.median(rest)) if rest and np.median(rest) > 0 else None}


def across_track_steps(tif, f=8, seg_m=1000.0, win_m=20.0, min_jump=0.5, min_agree=0.85, min_seg=10, sep=150):
    """Persistent across-track DN steps, as section_steps finds them (its rule, any scene)."""
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.Env(GDAL_PAM_ENABLED="NO"), rasterio.open(tif) as r:
        T, H, W = r.transform, r.height, r.width
        dn = r.read(1, out_shape=(H // f, W // f), resampling=Resampling.nearest)
    s = T.a
    # the frame from a subsample, then the (segment, 1 m bin) sums block by block: S1's 6.31 Gpx at 1/8
    # would otherwise hold ~1e8 coordinates at once
    rr, cc = np.nonzero(dn[::4, ::4] > 0)
    xy = np.column_stack([cc, rr]).astype(float) * 4 * f
    mu = xy.mean(axis=0)
    _, _, vt = np.linalg.svd((xy - mu)[:: max(1, len(xy) // 200000)], full_matrices=False)
    along, across = vt[0], vt[1]
    corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], float) - mu
    u_min, t_min = (corners @ across * s).min(), (corners @ along * s).min()
    nU = int(math.ceil((corners @ across * s).max() - u_min)) + 1
    nT = int(math.ceil(((corners @ along * s).max() - t_min) / seg_m)) + 1
    S_, N_ = np.zeros(nT * nU), np.zeros(nT * nU)
    u_lo, u_hi = np.inf, -np.inf
    xs = np.arange(dn.shape[1], dtype=float) * f
    for r0 in range(0, dn.shape[0], 256):
        blk = dn[r0:r0 + 256]
        rr, cc = np.nonzero(blk > 0)
        if not len(rr):
            continue
        x = xs[cc] - mu[0]
        y = (rr + r0).astype(float) * f - mu[1]
        u = (x * across[0] + y * across[1]) * s
        t = (x * along[0] + y * along[1]) * s
        u_lo, u_hi = min(u_lo, u.min()), max(u_hi, u.max())
        flat = np.floor((t - t_min) / seg_m).astype(np.int64) * nU + np.floor(u - u_min).astype(np.int64)
        S_ += np.bincount(flat, weights=blk[rr, cc].astype(float), minlength=nT * nU)
        N_ += np.bincount(flat, minlength=nT * nU)
    S_, N_ = S_.reshape(nT, nU), N_.reshape(nT, nU)
    used = np.flatnonzero(N_.sum(axis=0) > 0)
    S_, N_ = S_[:, used.min():used.max() + 1], N_[:, used.min():used.max() + 1]
    u_min += used.min()
    nU = S_.shape[1]
    w = int(win_m)
    cS, cN = np.cumsum(np.pad(S_, ((0, 0), (1, 0))), axis=1), np.cumsum(np.pad(N_, ((0, 0), (1, 0))), axis=1)
    idx = np.arange(w, nU - w)
    left = (cS[:, idx] - cS[:, idx - w]) / np.maximum(cN[:, idx] - cN[:, idx - w], 1)
    right = (cS[:, idx + w] - cS[:, idx]) / np.maximum(cN[:, idx + w] - cN[:, idx], 1)
    # a w-metre window in a full segment holds w / (f s) x seg_m / (f s) samples; section_steps wrote the first
    # factor as 0.5 per metre, right at 0.25 m/px (S2) and twice too strict at 0.5 m/px
    need = 0.9 * (w / (f * s)) * (seg_m / (f * s))
    full = ((cN[:, idx] - cN[:, idx - w]) >= need) & ((cN[:, idx + w] - cN[:, idx]) >= need)
    jump = np.where(full, right - left, np.nan)
    fin = np.isfinite(jump)
    nseg = fin.sum(axis=0)
    med = np.full(jump.shape[1], np.nan)
    med[nseg > 0] = np.nanmedian(jump[:, nseg > 0], axis=0)
    agree = np.where(nseg > 0, (fin & (np.sign(np.nan_to_num(jump)) == np.sign(med)[None, :])).sum(axis=0)
                     / np.maximum(nseg, 1), 0.0)
    edge_ok = (idx >= sep) & (idx <= nU - sep)
    score_ = np.where((agree >= min_agree) & (nseg >= min_seg) & edge_ok, np.abs(np.nan_to_num(med)), 0.0)
    steps = []
    for k in np.argsort(score_)[::-1]:
        if score_[k] < min_jump:
            break
        if all(abs(k - j) >= sep for j in steps):
            steps.append(int(k))
    steps = sorted(steps)
    pos = [float(u_min + idx[k]) for k in steps]
    return {"swath_width_m": float(u_hi - u_lo), "n_segments": int(nT),
            "steps": [{"u_m": p, "jump_dn": float(med[k]), "agree": float(agree[k]), "n_seg": int(nseg[k])}
                      for p, k in zip(pos, steps)],
            "spacing_m": [float(b - a) for a, b in zip(pos, pos[1:])]}


def comparisons():
    """The eleven same-ground comparisons: (key, reference, other, label ratio at 0.25)."""
    j = json.loads(INCIDENCE.read_text())
    return [(k, c["reference"], c["other"], c["bootstrap"]["ratio_ref_over_other"]["0.25"]["point"])
            for k, c in j["comparisons"].items()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--delivery", default=str(DELIVERY))
    ap.add_argument("--h2c-noise", default=str(H2C_NOISE))
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="+", help="these ObsIds only (no comparisons): for development")
    args = ap.parse_args()
    from YOLOv8BeyondEarth import presets
    delivery = Path(args.delivery)
    theirs = json.loads(Path(args.h2c_noise).read_text())["scenes"]
    scenes = {}
    for d in sorted((delivery / "scenes").iterdir()):
        obs = d.name
        if args.only and obs not in args.only:
            continue
        tif, lbl, prep = d / f"{obs}_RED.tif", d / f"{obs}_RED.LBL", d / f"{obs}_RED.prep.json"
        L = label_fields(lbl)
        sf = L["scaling_factor"]
        wins, ov, shape, origins, theirs_wins = sample_windows(tif)
        st = [window_stats(a) for a in wins]
        col = lambda k: np.array([x[k] for x in st])                       # noqa: E731
        span = np.maximum(col("span"), 1.0)
        m = {k: float(np.median(col(k))) for k in ("sigma_lap_mad", "sigma_lap_meanabs", "sigma_dx_mad",
                                                     "sigma_dy_mad", "sigma_dx_meanabs", "sigma_dy_meanabs",
                                                     "span", "mean")}
        frac = {k: float(np.median(col(k) / span)) for k in ("sigma_lap_mad", "sigma_lap_meanabs",
                                                             "sigma_dx_mad", "sigma_dy_mad")}
        sc = scene_dn10(prep)
        to_if = 4.0 * sf
        win_if_mean = col("mean") * to_if + L["offset"]
        rec = {"label": L, "n_windows": len(st), "window_dn8_median": m, "noise_over_span_median": frac,
               "dx_over_dy_median": float(np.median(col("sigma_dx_mad") / np.maximum(col("sigma_dy_mad"), 1e-9))),
               "dx_over_dy_meanabs_median": float(np.median(col("sigma_dx_meanabs")
                                                            / np.maximum(col("sigma_dy_meanabs"), 1e-9))),
               "in_if": {"sigma_lap_mad": m["sigma_lap_mad"] * to_if,
                         "sigma_lap_meanabs": m["sigma_lap_meanabs"] * to_if,
                         "span": m["span"] * to_if,
                         "window_mean": float(np.median(win_if_mean)),
                         "scene_mean": sc["mean"] * sf + L["offset"],
                         "scene_sd": sc["sd"] * sf,
                         "scene_dn10_mean_sd": [sc["mean"], sc["sd"]],
                         "scene_p01_p99": [sc["p01"] * sf + L["offset"], sc["p99"] * sf + L["offset"]],
                         "snr_window": float(np.median(win_if_mean / np.maximum(col("sigma_lap_meanabs") * to_if,
                                                                               1e-12))),
                         "dn10_per_if": 1.0 / sf},
               "spectrum": spectrum(wins, swath_angle(ov)), "swath_angle_deg_from_vertical": swath_angle(ov),
               "raster_shape": list(shape),
               "noise_across_swath": across_swath(ov, origins, col("sigma_lap_mad"), col("span"))}
        t = theirs.get(obs)
        if t:
            ts = [window_stats(a) for a in theirs_wins]
            tsig = np.array([x["sigma_lap_mad"] for x in ts])
            tspan = np.array([x["span"] for x in ts])
            mine = {"n_windows": len(ts), "sigma_n_dn": float(np.median(tsig)), "span_dn": float(np.median(tspan)),
                    "noise_frac": float(np.median(tsig / np.maximum(tspan, 1.0)))}
            rec["hirise2ctx"] = {"theirs": {k: t[k] for k in ("n_windows", "sigma_n_dn", "span_dn", "noise_frac")},
                                 "recomputed_their_way": mine,
                                 "reproduced": bool(t["n_windows"] == mine["n_windows"]
                                                    and all(abs(t[k] - mine[k]) < 1e-9
                                                            for k in ("sigma_n_dn", "span_dn", "noise_frac")))}
        rec["steps"] = across_track_steps(tif)
        scenes[obs] = rec
        print(f"{obs}: n {len(st)} sigma(their) {m['sigma_lap_mad']:.3f} meanabs {m['sigma_lap_meanabs']:.3f} "
              f"dx/dy {rec['dx_over_dy_meanabs_median']:.3f} col-wedge {rec['spectrum']['wedges']['across_track_column_stripes']['median_ratio']:.2f} span {m['span']:.1f} | I/F sigma {rec['in_if']['sigma_lap_meanabs']:.2e} "
              f"span {rec['in_if']['span']:.2e} mean {rec['in_if']['scene_mean']:.4f} | TDI {L['tdi_on']} | "
              f"top peak {rec['spectrum']['peaks'][0]['period_px']:.2f} px x{rec['spectrum']['peaks'][0]['power_over_annulus_median']:.1f} "
              f"| steps {len(rec['steps']['steps'])} | theirs reproduced {rec.get('hirise2ctx', {}).get('reproduced')}",
              flush=True)
    comps = []
    for key, ref, other, ratio in ([] if args.only else comparisons()):
        a, b = scenes[ref], scenes[other]
        comps.append({"comparison": key, "reference": ref, "other": other, "label_ratio_0.25": ratio,
                      "noise_over_span_other_over_ref": {
                          k: b["noise_over_span_median"][k] / a["noise_over_span_median"][k]
                          for k in a["noise_over_span_median"]},
                      "sigma_if_other_over_ref": b["in_if"]["sigma_lap_meanabs"] / a["in_if"]["sigma_lap_meanabs"],
                      "span_if_other_over_ref": b["in_if"]["span"] / a["in_if"]["span"],
                      "relative_contrast_other_over_ref": (b["in_if"]["span"] / b["in_if"]["window_mean"])
                                                          / (a["in_if"]["span"] / a["in_if"]["window_mean"]),
                      "window_mean_if_other_over_ref": b["in_if"]["window_mean"] / a["in_if"]["window_mean"],
                      "both_cubenorm": [a["label"]["cubenorm"], b["label"]["cubenorm"]],
                      "scene_mean_if_other_over_ref": b["in_if"]["scene_mean"] / a["in_if"]["scene_mean"],
                      "scene_sd_if_other_over_ref": b["in_if"]["scene_sd"] / a["in_if"]["scene_sd"],
                      "scene_sd_dn10_other_over_ref": b["in_if"]["scene_dn10_mean_sd"][1] / a["in_if"]["scene_dn10_mean_sd"][1],
                      "snr_other_over_ref": b["in_if"]["snr_window"] / a["in_if"]["snr_window"]})
    out = {"what": "the pilot's pixel noise beside hirise2ctx's R11: their measure reproduced, two others, the same "
                   "in I/F, and column patterns at two scales (read-only on the delivery)",
           "code": presets.code_identity(), "delivery": str(delivery), "hirise2ctx_noise_json": str(args.h2c_noise),
           "caveats": ["windows are drawn over each image's own footprint, not the ground two images share, so a "
                       "comparison's ratios mix terrain with acquisition",
                       "scene statistics exclude the fill (DN 0)",
                       "nothing here is a test: 11 comparisons on 9 grounds, no truth"],
           "scenes": scenes, "comparisons": comps}
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
