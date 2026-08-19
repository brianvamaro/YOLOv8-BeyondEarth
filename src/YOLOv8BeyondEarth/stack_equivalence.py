"""Do two compute stacks produce the same *science*, or merely the same bytes?

Why this module exists
----------------------
`scripts/review/measure_stack_parity.py` answers a **bit-level** question: are the catalogs identical?
Measured 2026-08-19 across the laptop (torch 2.11.0+cu128, RTX 5070) and Sherlock (torch 2.6.0+cu124,
RTX 3090) they are not — 1–16% of polygons differ, and a handful of detections differ in membership.

But that bar was borrowed from G1-02's ``batch_size`` comparison, where both sides ran the same
libraries on the same card and exact equality was achievable. Across GPU architectures identical floats
are not achievable *in principle*, so exact-equality metrics can only ever fail, and failing tells you
nothing about whether it matters. The differences actually measured were **sub-pixel** (median centroid
offset 0.002–0.046 px) and confined **to the confidence floor** in membership (all 10 differing rows at
score ≈ 0.100).

So the question that decides anything is: **do the numbers we publish move?** This module measures that
directly — per-boulder orientation and size deltas on matched boulders, the aggregate rose statistics
each stack yields with their own bootstrap intervals, whether boulders cross the Goal-4 fabric gate's
cuts, and the size-frequency counts.

What it can and cannot establish
--------------------------------
It can **falsify** equivalence, and it can **bound** the difference. It cannot *prove* equivalence: the
paired data available is the 2048² dev crop, so a null result means "no difference detectable at this
sample size", not "no difference". State the bound, not a pass.

Tolerances are inherited rather than invented
---------------------------------------------
* Per-boulder ``|Δangle|`` is compared against **2.69°**, the *tighter* of the two SAM crop-position
  jitter medians (Moon 2.69°, Mars 4.65°) that this project already accepts as random noise which
  "roses survive". A cross-stack median below that is no worse than a difference already tolerated.
* Aggregate statistics are compared against **their own bootstrap CI**. If the two stacks differ by less
  than the sampling uncertainty of either, they are indistinguishable at this sample size — which is the
  honest statement, and it degrades gracefully as the sample shrinks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import presets
from .fusion import fuse_predictions
from .matching import unique_centroid_match
from .orientation import (attach_orientations, axial_resultant, axial_resultant_uniform,
                          grid_fraction, grid_fraction_uniform, peak_ci_report, rayleigh_p,
                          well_resolved_subset)
from .orientation_trust import axial_abs_delta

#: The tighter SAM crop-position-jitter median (Moon 2.69°, Mars 4.65°) — a per-boulder angle
#: difference this project already treats as random and survivable.
ACCEPTED_DANGLE_MEDIAN_DEG = 2.69

#: Equivalent-diameter band edges in native pixels, matching the bands the Goal-3 work reports.
SFD_EDGES_PX = (3.0, 4.0, 6.0, 8.0, 10.0, 14.0, np.inf)


def fuse_arms(arms, native_res):
    """Fuse one stack's arm catalogs with the **production** rule.

    ``arms`` maps ``(slice_size, sr_factor)`` to a GeoDataFrame. Uses ``GOAL3_PRODUCTION`` rather than
    ``fuse_predictions``' own defaults, because the generic defaults reproduce no benchmarked result
    (review item G3-01). Returns ``(fused, audit)``.
    """
    cfg = presets.GOAL3_PRODUCTION
    return fuse_predictions(arms, rule=cfg["rule"], bands=cfg["bands"], engulf_thr=cfg["engulf_thr"],
                            metric=cfg["metric"], cross_arm_only=cfg["cross_arm_only"],
                            native_res=native_res)


def paired_deltas(a, b, res, tol_px=2.0, n_workers=8):
    """Match two catalogs of the same ground one-to-one; report per-boulder differences.

    Pairing is what gives this test its power: comparing two independent aggregates of ~2,000 boulders
    is nearly blind, while comparing the *same* boulder measured twice is not. ``res`` is the raster
    pixel size in metres; ``tol_px`` the centroid tolerance for calling two rows the same boulder.
    """
    oa = attach_orientations(a, res, n_workers=n_workers)
    ob = attach_orientations(b, res, n_workers=n_workers)
    ca = np.c_[oa.geometry.centroid.x.to_numpy(), oa.geometry.centroid.y.to_numpy()]
    cb = np.c_[ob.geometry.centroid.x.to_numpy(), ob.geometry.centroid.y.to_numpy()]
    ia, ib = unique_centroid_match(ca, cb, tol_px * res)

    pa, pb = oa.iloc[ia].reset_index(drop=True), ob.iloc[ib].reset_index(drop=True)
    ok = pa["angle180"].notna() & pb["angle180"].notna()
    dang = axial_abs_delta(pa.loc[ok, "angle180"].to_numpy(), pb.loc[ok, "angle180"].to_numpy())
    dd = (pb["diameter_m"].to_numpy() - pa["diameter_m"].to_numpy()) / pa["diameter_m"].to_numpy()
    dasp = pb["aspect_ra"].to_numpy() - pa["aspect_ra"].to_numpy()

    def _unmatched(df, idx):
        un = np.setdiff1d(np.arange(len(df)), idx)
        if not len(un):
            return {"n": 0}
        s = df["score"].to_numpy()[un] if "score" in df else np.full(len(un), np.nan)
        return {"n": int(len(un)), "score_min": float(np.nanmin(s)),
                "score_median": float(np.nanmedian(s)), "score_max": float(np.nanmax(s))}

    return {
        "n_a": int(len(a)), "n_b": int(len(b)), "n_matched": int(len(ia)),
        "n_angle_pairs": int(ok.sum()),
        "unmatched_a": _unmatched(oa, ia), "unmatched_b": _unmatched(ob, ib),
        "dangle_median_deg": float(np.median(dang)) if len(dang) else np.nan,
        "dangle_p95_deg": float(np.percentile(dang, 95)) if len(dang) else np.nan,
        "dangle_max_deg": float(dang.max()) if len(dang) else np.nan,
        "frac_dangle_gt_1deg": float((dang > 1.0).mean()) if len(dang) else np.nan,
        "frac_dangle_gt_5deg": float((dang > 5.0).mean()) if len(dang) else np.nan,
        "ddiam_rel_median": float(np.nanmedian(dd)), "ddiam_rel_p95": float(np.nanpercentile(dd, 95)),
        "daspect_median": float(np.nanmedian(dasp)),
        "within_accepted_dangle": bool(np.median(dang) <= ACCEPTED_DANGLE_MEDIAN_DEG) if len(dang) else None,
        "_paired": (pa, pb),
    }


def aggregate_stats(gdf, res, aspect_min=1.35, dpx_min=8.0, n_boot=400, seed=0, n_workers=8,
                    already_oriented=False):
    """The rose statistics one stack yields from its own catalog, with a bootstrap peak CI.

    Uses the shared orientation readout and the same ``well_resolved_subset`` gate the Goal-4 fabric
    gate uses, so the population is defined the way the science defines it rather than ad hoc.
    """
    df = gdf if already_oriented else attach_orientations(gdf, res, n_workers=n_workers)
    sub = well_resolved_subset(df.dropna(subset=["angle180"]), res,
                               aspect_min=aspect_min, dpx_min=dpx_min)
    ang = sub["angle180"].to_numpy()
    n = len(ang)
    if n < 3:
        return {"n": n}
    r = float(axial_resultant(ang))
    peak = peak_ci_report(ang, n_boot=n_boot, seed=seed)
    return {
        "n": n,
        "grid_fraction": float(grid_fraction(ang)),
        "grid_fraction_isotropic": float(grid_fraction_uniform()),
        "R": r,
        "R_isotropic": float(axial_resultant_uniform(n)),
        "rayleigh_p": float(rayleigh_p(r, n)),
        # peak_ci_report returns `peak`/`lo`/`hi`, plus leave-one-out diagnostics we do not use here
        # because a single crop is one cluster and the LOO spread is undefined for it.
        "peak_deg": float(peak["peak"]),
        "peak_ci": [float(peak["lo"]), float(peak["hi"])],
    }


def indistinguishable(stat_a, stat_b, key, ci_key="peak_ci"):
    """Is the two stacks' difference in ``key`` inside either stack's own bootstrap interval?

    The honest comparison at a given sample size: a difference smaller than the sampling uncertainty of
    the statistic is not evidence of a difference. Only defined where a CI is available.
    """
    if key not in stat_a or key not in stat_b:
        return None
    lo_a, hi_a = stat_a.get(ci_key, (np.nan, np.nan))
    lo_b, hi_b = stat_b.get(ci_key, (np.nan, np.nan))
    if not np.isfinite([lo_a, hi_a, lo_b, hi_b]).all():
        return None
    return bool(lo_a <= stat_b[key] <= hi_a or lo_b <= stat_a[key] <= hi_b)


def gate_crossing(pa, pb, res, aspect_min=1.35, dpx_min=8.0):
    """Of *matched* boulders, how many change side of the fabric gate's cuts?

    This is the mechanism a per-boulder summary would miss. The Goal-4 gate selects on ``aspect_ra`` and
    diameter, so a sub-pixel geometry difference can flip a boulder across a cut and change the gated
    *population* even when every individual measurement is fine.
    """
    def _sel(df):
        return ((df["aspect_ra"] > aspect_min) & (df["diameter_m"] / res >= dpx_min)).to_numpy()

    sa, sb = _sel(pa), _sel(pb)
    return {"n_pairs": int(len(sa)), "n_in_a": int(sa.sum()), "n_in_b": int(sb.sum()),
            "n_crossed": int((sa != sb).sum()),
            "frac_crossed": float((sa != sb).mean()) if len(sa) else np.nan,
            "aspect_min": aspect_min, "dpx_min": dpx_min}


def sfd_counts(df, res, edges=SFD_EDGES_PX):
    """Counts per equivalent-diameter band, in native pixels. ``df`` needs ``diameter_m``."""
    dpx = df["diameter_m"].to_numpy() / res
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        label = f"{lo:g}-{'inf' if not np.isfinite(hi) else f'{hi:g}'}"
        out[label] = int(((dpx >= lo) & (dpx < hi)).sum())
    return out


def compare_sfd(a, b, res, edges=SFD_EDGES_PX):
    """Per-band counts on both stacks and the relative difference, as a tidy frame."""
    ca, cb = sfd_counts(a, res, edges), sfd_counts(b, res, edges)
    rows = [{"band_px": k, "n_a": ca[k], "n_b": cb[k], "delta": cb[k] - ca[k],
             "rel_delta": (cb[k] - ca[k]) / ca[k] if ca[k] else np.nan} for k in ca]
    return pd.DataFrame(rows)
