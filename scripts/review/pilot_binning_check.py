"""Binning inside one scene (``docs/sherlock_pilot_plan.md`` section 8.G), and where ROM's 0.25 m/px tiles lie.

ESP_069088_2270 (P3's 50.8 deg reference) is the pilot's one mixed-binning image. Its label bins RED0-3 and
RED7-8 2x at TDI 64, and RED4-6 not, at TDI 128. Its strips are therefore 0.5 m data resampled onto the 0.25 m
grid beside true 0.25 m data. Comparing detections either side of the two seams measures what that does to the
label within one scene, at no GPU cost: input for hirise2ctx's D13 and S15 (the mixed-binning rule).
The comparison is bin 2 at TDI 64 against bin 1 at TDI 128, as flown; it cannot separate binning from TDI.

- **Locating the strips** in the prepared raster, not from instrument geometry. The swath's along-track axis
  is the principal axis of the valid footprint, so each pixel has an along-track t and an across-track u.
  Sharpness is |k * DN|, with k the 3 x 3 kernel [[1,-2,1],[-2,4,-2],[1,-2,1]], averaged in 1 m bins of u per
  1 km along-track segment. Resampled 0.5 m data carries almost nothing near the 0.25 m grid's Nyquist
  frequency, so a binned strip reads low. A seam is where the profile crosses halfway between the two strips'
  levels: per segment, then the median.
- **Either side of each seam,** in bands 128 m wide with a 4 m guard (as section 8.D), at floors 0.10, 0.25 and
  0.40 and sizes >= 1.41 m (1.41-2, 2-4, 4-6, >= 6 m):
  - detections per km2 by centroid, and polygon area per band area (a sum, not the label's union);
  - the kept share (>= floor over >= 0.10) and the median score;
  - the size distribution's rollover at 0.25 (the peak of the differential distribution in log bins), which
    is what hirise2ctx's S15 estimates.
- **The statistic:** binned over unbinned, with a block bootstrap over 1 km along-track blocks paired across
  the seam. **The null:** the same comparison at pseudo-seams inside strips of one binning (>= 150 m from either
  seam and from the swath edges), giving the terrain-only spread.
- **ROM's 0.25 m/px image ESP_028537_2270 is mixed too** (RED0-2 and RED7-8 2x). The same profile locates its
  strips in its prepared raster (pixel-identical to Prieur's training raster), and each ROM tile is assigned to
  a strip by its centroid. hirise2ctx noted this as unchecked; it bears on that image's ROM-based evidence.

Read-only on the delivery and on the ROM data. A measurement for D13 and S15; it changes no label.

    conda run -n bouldernet python -I scripts/review/pilot_binning_check.py --out data/review/pilot_binning_check.json
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
OBS = "ESP_069088_2270"
ROM_IMAGE = "ESP_028537_2270"
ROM_RASTER = REPO / "data" / "pds_scenes" / ROM_IMAGE / f"{ROM_IMAGE}_RED.tif"
ROM_SHP = REPO.parent / "data_prieur" / "raw_data" / "mars" / ROM_IMAGE / "shp" / f"{ROM_IMAGE}-ROM.shp"
KERNEL = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])
FLOORS = (0.10, 0.25, 0.40)
SIZES = ((1.4105, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, np.inf))
D0 = 1.4105
SEG_M, BAND_M, GUARD_M, SEP_M = 1000.0, 128.0, 4.0, 150.0


def label_ccds(lbl):
    from YOLOv8BeyondEarth import pds_prep
    f = pds_prep.read_label(lbl)
    ints = lambda k: [int(x) for x in re.findall(r"-?\d+", str(pds_prep._lookup(f, k, False) or ""))]   # noqa: E731
    flags = re.findall(r"ON|OFF", str(pds_prep._lookup(f, "MRO:CCD_FLAG", False)))
    b, t = ints("MRO:BINNING"), ints("MRO:TDI")
    return {f"RED{k}": {"binning": b[k], "tdi": t[k]} for k in range(10) if k < len(flags) and flags[k] == "ON"}


def swath_frame(tif, f=16):
    """Along/across unit vectors and centre of the valid footprint, in full-resolution pixel coordinates."""
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(tif) as r:
        H, W, T = r.height, r.width, r.transform
        dn = r.read(1, out_shape=(H // f, W // f), resampling=Resampling.nearest)
    rr, cc = np.nonzero(dn > 0)
    xy = np.column_stack([cc, rr]).astype(float) * f
    mu = xy.mean(axis=0)
    _, _, vt = np.linalg.svd((xy - mu)[:: max(1, len(xy) // 200000)], full_matrices=False)
    along, across = vt[0], vt[1]
    if across[0] < 0:                          # u grows eastwards, whatever the SVD's sign
        across = -across
    if along[1] < 0:                           # t grows southwards (down the rows)
        along = -along
    return {"mu": mu, "along": along, "across": across, "res": abs(T.a), "shape": (H, W), "transform": T}


def sharpness_profile(tif, fr, chunk=256):
    """Mean |kernel response| and mean DN per (1 km segment, 1 m across-track bin), full resolution."""
    import rasterio
    from rasterio.windows import Window
    from scipy import ndimage
    H, W = fr["shape"]
    s = fr["res"]
    corners = np.array([[0, 0], [W, 0], [0, H], [W, H]], float) - fr["mu"]
    ulim = corners @ fr["across"] * s
    tlim = corners @ fr["along"] * s
    u0, t0 = ulim.min(), tlim.min()
    nU, nT = int(math.ceil(ulim.max() - u0)) + 1, int(math.ceil((tlim.max() - t0) / SEG_M)) + 1
    S, D, N = (np.zeros((nT, nU)) for _ in range(3))
    xs = np.arange(W, dtype=float)
    with rasterio.open(tif) as r:
        for y0 in range(0, H, chunk):
            ya, yb = max(0, y0 - 1), min(H, y0 + chunk + 1)
            a = r.read(1, window=Window(0, ya, W, yb - ya)).astype(float)
            lap = np.abs(ndimage.convolve(a, KERNEL, mode="nearest"))
            valid = ndimage.minimum_filter(a, size=3, mode="constant", cval=0) > 0     # no fill in the 3x3
            rows = slice(y0 - ya, y0 - ya + min(chunk, H - y0))
            a, lap, valid = a[rows], lap[rows], valid[rows]
            ys = np.arange(y0, y0 + a.shape[0], dtype=float)
            X, Y = np.meshgrid(xs - fr["mu"][0], ys - fr["mu"][1])
            u = (X * fr["across"][0] + Y * fr["across"][1]) * s
            t = (X * fr["along"][0] + Y * fr["along"][1]) * s
            ub = np.floor(u[valid] - u0).astype(np.int64)
            tb = np.floor((t[valid] - t0) / SEG_M).astype(np.int64)
            flat = tb * nU + ub
            S += np.bincount(flat, weights=lap[valid], minlength=nT * nU).reshape(nT, nU)
            D += np.bincount(flat, weights=a[valid], minlength=nT * nU).reshape(nT, nU)
            N += np.bincount(flat, minlength=nT * nU).reshape(nT, nU)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"sharp": S / N, "dn": D / N, "n": N, "u0": u0, "t0": t0}


def find_seams(prof, ccds):
    """Seams between binned and unbinned strips: per segment, where the smoothed sharpness profile crosses the
    midpoint of the two levels; the median over segments."""
    sharp, n = prof["sharp"], prof["n"]
    ok = n >= 0.5 * np.nanmedian(n[n > 0])
    med = np.nanmedian(np.where(ok, sharp, np.nan), axis=0)
    k = np.ones(25) / 25.0
    sm = np.convolve(np.nan_to_num(med, nan=np.nanmedian(med)), k, mode="same")
    order = [c for c in sorted(ccds, key=lambda c: int(c[3:]))]
    bins = [ccds[c]["binning"] for c in order]
    lo, hi = np.nanpercentile(sm[np.isfinite(med)], [10, 90])
    mid = 0.5 * (lo + hi)
    above = sm > mid
    edges = np.flatnonzero(np.diff(above.astype(int)) != 0)
    valid_u = np.flatnonzero(np.isfinite(med))
    edges = [int(e) for e in edges if valid_u.min() + 100 < e < valid_u.max() - 100]
    per_seg = []
    for e in edges:
        pos = []
        for j in range(sharp.shape[0]):
            row = np.convolve(np.nan_to_num(sharp[j], nan=mid), k, mode="same")
            if not ok[j, max(0, e - 200):e + 200].all():
                continue
            w = slice(max(0, e - 200), e + 200)
            cr = np.flatnonzero(np.diff((row[w] > mid).astype(int)) != 0)
            if len(cr):
                pos.append(w.start + cr[np.argmin(np.abs(w.start + cr - e))])
        per_seg.append(pos)
    seams = [{"u_m": float(prof["u0"] + np.median(p)) if p else float(prof["u0"] + e),
              "n_segments": len(p), "spread_m_iqr": float(np.subtract(*np.percentile(p, [75, 25]))) if p else None,
              "sharpness_left": float(np.nanmedian(med[max(0, e - 300):e - 50])),
              "sharpness_right": float(np.nanmedian(med[e + 50:e + 300]))}
             for e, p in zip(edges, per_seg)]
    return {"levels": {"p10": float(lo), "p90": float(hi), "midpoint": float(mid)}, "ccd_order": order,
            "ccd_binning": bins, "seams": seams,
            "profile_1m": {"u_m": [float(prof["u0"] + i) for i in range(0, len(med), 10)],
                           "sharpness": [None if not np.isfinite(v) else float(v) for v in med[::10]]}}


def load_catalog(delivery, obs, fr):
    import pyogrio
    import shapely
    h = json.loads((delivery / "out" / obs / "handoff.json").read_text())
    layer = delivery / "out" / obs / h["layers"][0]["layer"]
    g = pyogrio.read_dataframe(layer, columns=["score"])
    geom = np.asarray(g.geometry.values, dtype=object)
    area = shapely.area(geom)
    cen = shapely.centroid(geom)
    T = fr["transform"]
    px = np.column_stack([(shapely.get_x(cen) - T.c) / T.a, (shapely.get_y(cen) - T.f) / T.e])
    d = px - fr["mu"]
    return {"t": d @ fr["along"] * fr["res"], "u": d @ fr["across"] * fr["res"],
            "diam": 2.0 * np.sqrt(area / np.pi), "area": area, "score": g["score"].to_numpy(), "layer": str(layer)}


def valid_area_by_block(prof, lo, hi):
    """km2 of valid ground per 1 km along-track block in [lo, hi) of u (1 m bins x pixel counts)."""
    i0, i1 = int(math.floor(lo - prof["u0"])), int(math.ceil(hi - prof["u0"]))
    return prof["n"][:, max(0, i0):max(0, i1)].sum(axis=1)


def band_stats(cat, prof, res, lo, hi):
    """Per 1 km block: counts by floor and size, polygon area, scores, inside u in [lo, hi)."""
    px_km2 = (res * res) / 1e6
    area_km2 = valid_area_by_block(prof, lo, hi) * px_km2
    nT = len(area_km2)
    inband = (cat["u"] >= lo) & (cat["u"] < hi)
    blk_all = np.floor((cat["t"][inband] - prof["t0"]) / SEG_M).astype(int)
    keep = (blk_all >= 0) & (blk_all < nT)
    sc_all, dm_all = cat["score"][inband][keep], cat["diam"][inband][keep]
    big = dm_all >= D0
    blk, sc, dm, ar = blk_all[keep][big], sc_all[big], dm_all[big], cat["area"][inband][keep][big]
    out = {"area_km2": area_km2, "blocks": {}}
    for f in FLOORS:
        sel = sc >= f
        out["blocks"][f"{f:.2f}|all"] = np.bincount(blk[sel], minlength=nT).astype(float)
        for (a, b) in SIZES:
            sel = (sc >= f) & (dm >= a) & (dm < b)
            out["blocks"][f"{f:.2f}|{a:g}-{b:g}m"] = np.bincount(blk[sel], minlength=nT).astype(float)
        sel = sc >= f
        out["blocks"][f"{f:.2f}|polygon_area"] = np.bincount(blk[sel], weights=ar[sel], minlength=nT)
    out["scores"], out["diam_any_size"], out["score_any_size"] = sc, dm_all, sc_all
    return out


def compare_sides(cat, prof, res, u_seam, binned_side, rng, n_boot=2000):
    """Binned over unbinned either side of one seam, with a paired 1 km block bootstrap."""
    sides = {"left": (u_seam - GUARD_M - BAND_M, u_seam - GUARD_M), "right": (u_seam + GUARD_M, u_seam + GUARD_M + BAND_M)}
    st = {k: band_stats(cat, prof, res, *v) for k, v in sides.items()}
    b, ub = (st["left"], st["right"]) if binned_side == "left" else (st["right"], st["left"])
    full = (b["area_km2"] > 0.8 * np.median(b["area_km2"][b["area_km2"] > 0])) & \
           (ub["area_km2"] > 0.8 * np.median(ub["area_km2"][ub["area_km2"] > 0]))
    idx = np.flatnonzero(full)
    pick = rng.integers(0, len(idx), (n_boot, len(idx)))
    out = {"n_blocks": int(len(idx)), "density_per_km2": {}, "binned_over_unbinned": {}, "kept_share": {},
           "median_score": {}}
    A_b, A_u = b["area_km2"][idx], ub["area_km2"][idx]
    for label in b["blocks"]:
        nb, nu = b["blocks"][label][idx], ub["blocks"][label][idx]
        db, du = nb.sum() / A_b.sum(), nu.sum() / A_u.sum()
        boot = (nb[pick].sum(1) / A_b[pick].sum(1)) / np.maximum(nu[pick].sum(1) / A_u[pick].sum(1), 1e-12)
        out["density_per_km2"][label] = {"binned": float(db), "unbinned": float(du)}
        out["binned_over_unbinned"][label] = {"point": float(db / du) if du else None,
                                              "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    for f in FLOORS[1:]:
        for side, s in (("binned", b), ("unbinned", ub)):
            all10 = s["blocks"]["0.10|all"][idx].sum()
            out["kept_share"].setdefault(f"{f:.2f}", {})[side] = (float(s["blocks"][f"{f:.2f}|all"][idx].sum() / all10)
                                                                  if all10 else None)
    for side, s in (("binned", b), ("unbinned", ub)):
        out["median_score"][side] = float(np.median(s["scores"])) if len(s["scores"]) else None
        sel = s["score_any_size"] >= 0.25
        out.setdefault("rollover_m_at_0.25", {})[side] = rollover(s["diam_any_size"][sel])
    return out


def rollover(d, lo=0.5, hi=20.0, per_decade=40):
    """Diameter of the peak of the differential size distribution dN/dD: 40 log bins per decade, 7-bin
    running mean. A simple estimator for comparing strips, not S15's."""
    d = d[(d >= lo) & (d < hi)]
    if len(d) < 200:
        return None
    edges = np.logspace(np.log10(lo), np.log10(hi), int(per_decade * np.log10(hi / lo)) + 1)
    h, _ = np.histogram(d, edges)
    dens = np.convolve(h / np.diff(edges), np.ones(7) / 7, mode="same")
    k = int(np.argmax(dens))
    return float(np.sqrt(edges[k] * edges[k + 1]))


def whole_strips(cat, seams, guard_m=50.0):
    """All detections in the binned strips against all in the unbinned strip (>= guard from a seam): the
    rollover at 0.25 and 0.40, diameter quantiles, and the kept share, over each strip's whole length."""
    lo_seam, hi_seam = sorted(s["u_m"] for s in seams)[:1] + sorted(s["u_m"] for s in seams)[-1:]
    u = cat["u"]
    sel = {"unbinned": (u > lo_seam + guard_m) & (u < hi_seam - guard_m),
           "binned": (u < lo_seam - guard_m) | (u > hi_seam + guard_m)}
    out = {}
    for name, m in sel.items():
        dm, sc = cat["diam"][m], cat["score"][m]
        big = dm >= D0
        out[name] = {"n_detections": int(m.sum()),
                     "rollover_m": {f"{f:.2f}": rollover(dm[sc >= f]) for f in (0.10, 0.25, 0.40)},
                     "diameter_quantiles_m_at_0.25": [float(q) for q in np.percentile(dm[sc >= 0.25], [10, 25, 50])],
                     "kept_share_ge_1.41m": {f"{f:.2f}": float((big & (sc >= f)).sum() / max(big.sum(), 1))
                                             for f in (0.25, 0.40)}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--delivery", default=str(DELIVERY))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-null", type=int, default=20)
    args = ap.parse_args()
    from YOLOv8BeyondEarth import presets
    delivery = Path(args.delivery)
    tif = delivery / "scenes" / OBS / f"{OBS}_RED.tif"
    ccds = label_ccds(delivery / "scenes" / OBS / f"{OBS}_RED.LBL")
    fr = swath_frame(tif)
    prof = sharpness_profile(tif, fr)
    seams = find_seams(prof, ccds)
    print(f"{OBS}: seams at {[round(s['u_m']) for s in seams['seams']]} m across track; levels {seams['levels']}", flush=True)
    cat = load_catalog(delivery, OBS, fr)
    rng = np.random.default_rng(0)
    res = fr["res"]
    out_seams = []
    for s in seams["seams"]:
        binned_side = "left" if s["sharpness_left"] < s["sharpness_right"] else "right"
        r = compare_sides(cat, prof, res, s["u_m"], binned_side, rng)
        out_seams.append({**s, "binned_side": binned_side, **r})
        x = r["binned_over_unbinned"]
        print(f"  seam {s['u_m']:.0f} m ({binned_side} binned): density ratio at 0.10 {x['0.10|all']['point']:.3f} "
              f"{x['0.10|all']['ci95']}, 0.25 {x['0.25|all']['point']:.3f}, 0.40 {x['0.40|all']['point']:.3f}; "
              f"rollover {r['rollover_m_at_0.25']}", flush=True)
    strips_whole = whole_strips(cat, seams["seams"]) if len(seams["seams"]) == 2 else None
    print(f"  whole strips: {strips_whole}", flush=True)
    # null: pseudo-seams inside one binning, >= SEP_M from the real seams and the swath edges
    valid_u = prof["u0"] + np.flatnonzero(np.nanmedian(prof["n"], axis=0) > 0)
    u_lo, u_hi = valid_u.min() + SEP_M, valid_u.max() - SEP_M
    real = [s["u_m"] for s in seams["seams"]]
    cand = [u for u in np.arange(u_lo, u_hi, 10.0) if all(abs(u - v) >= SEP_M for v in real)]
    prng = np.random.default_rng(1)
    pseudo = []
    for u in prng.permutation(cand):
        if all(abs(u - v) >= SEP_M for v in pseudo):
            pseudo.append(float(u))
        if len(pseudo) == args.n_null:
            break
    null = []
    for u in sorted(pseudo):
        r = compare_sides(cat, prof, res, u, "left", rng, n_boot=200)
        null.append({"u_m": u, "density_ratio_left_over_right": {k: v["point"] for k, v in r["binned_over_unbinned"].items()}})
    lr = {k: [abs(math.log(n["density_ratio_left_over_right"][k])) for n in null
              if n["density_ratio_left_over_right"][k]] for k in null[0]["density_ratio_left_over_right"]} if null else {}
    null_summary = {k: {"n": len(v), "p95_abs_log_ratio": float(np.percentile(v, 95)) if v else None,
                        "as_ratio_range": [float(math.exp(-np.percentile(v, 95))), float(math.exp(np.percentile(v, 95)))] if v else None}
                    for k, v in lr.items()}
    # ROM's 0.25 m/px image: its strips, and which strip each ROM tile's centroid lies in
    rom = None
    if ROM_RASTER.exists() and ROM_SHP.exists():
        import geopandas as gpd
        rlbl = ROM_RASTER.with_name(f"{ROM_IMAGE}_RED.LBL")
        rccds = label_ccds(rlbl)
        rfr = swath_frame(ROM_RASTER)
        rprof = sharpness_profile(ROM_RASTER, rfr)
        rseams = find_seams(rprof, rccds)
        tiles = gpd.read_file(ROM_SHP)
        T = rfr["transform"]
        cx, cy = tiles.geometry.centroid.x.to_numpy(), tiles.geometry.centroid.y.to_numpy()
        px = np.column_stack([(cx - T.c) / T.a, (cy - T.f) / T.e]) - rfr["mu"]
        tu = px @ rfr["across"] * rfr["res"]
        bounds = sorted(s["u_m"] for s in rseams["seams"])
        sharp_med = np.nanmedian(rprof["sharp"], axis=0)

        def strip_of(u):
            i = int(np.clip(round(u - rprof["u0"]), 0, len(sharp_med) - 1))
            v = np.nanmedian(sharp_med[max(0, i - 60):i + 60])
            return "unbinned" if v > rseams["levels"]["midpoint"] else "binned"
        ds = REPO.parent / "data_prieur" / "boulder2024"
        split = {q.name: sp for sp in ("train", "validation", "test") for q in (ds / sp / "images").glob("*.png")}
        tid = tiles["tile_id"].to_numpy() if "tile_id" in tiles.columns else [None] * len(tiles)
        rom = {"image": ROM_IMAGE, "ccds": rccds, "seams": rseams["seams"], "levels": rseams["levels"],
               "tiles": [{"rom_index": int(i), "tile_id": None if t is None else int(t), "file_name": str(fn),
                          "split": split.get(str(fn), "unknown"), "u_m": float(u), "strip": strip_of(u),
                          "distance_to_nearest_seam_m": float(min(abs(u - b) for b in bounds)) if bounds else None}
                         for i, (fn, t, u) in enumerate(zip(tiles["file_name"], tid, tu))]}
        # where configuration 2's gain over configuration 1 sits on this image, from the study's per-cell rows
        import pandas as pd
        pc = pd.read_parquet(REPO / "data" / "config_study" / "paired_cells.parquet")
        pc = pc[(pc.obs_id == ROM_IMAGE) & (pc.cell_size_m == 40.0)
                & pc.candidate.isin(["C1", "C2:256pref_engulf"])]
        strip_of_tile = {t["tile_id"]: t["strip"] for t in rom["tiles"]}
        by = {}
        for f in (0.10, 0.25, 0.40):
            x = pc[pc.floor == f].assign(strip=lambda q: q.tile_id.map(strip_of_tile))
            for name, g in x.groupby("strip"):
                a, b = g[g.candidate == "C1"], g[g.candidate == "C2:256pref_engulf"]
                by.setdefault(name, {})[f"{f:.2f}"] = {
                    "n_tiles": int(a.tile_id.nunique()), "n_cells": int(len(a)),
                    "level_C1": float(a.fa_pred.sum() / a.fa_true.sum()),
                    "level_C2": float(b.fa_pred.sum() / b.fa_true.sum()),
                    "C2_over_C1": float(b.fa_pred.sum() / a.fa_pred.sum()),
                    "count_C2_over_C1": float(b.n_pred.sum() / a.n_pred.sum())}
        rom["config2_over_config1_by_strip_40m_all_tiles"] = by
        print(f"{ROM_IMAGE}: seams {[round(s['u_m']) for s in rseams['seams']]}; tiles by strip "
              f"{ {k: sum(t['strip'] == k for t in rom['tiles']) for k in ('binned', 'unbinned')} }", flush=True)
    out = {"what": "binning inside one scene (pilot plan 8.G) and where ROM's 0.25 m/px tiles lie (read-only)",
           "code": presets.code_identity(), "delivery": str(delivery), "image": OBS, "ccds": ccds,
           "frame": {"along": fr["along"].tolist(), "across": fr["across"].tolist(), "res_m": res},
           "strips": seams, "seams": out_seams, "whole_strips": strips_whole,
           "null": {"pseudo_seams": null, "summary": null_summary},
           "catalog": {"layer": cat["layer"], "n": int(len(cat["score"])), "n_ge_1.41m": int((cat["diam"] >= D0).sum())},
           "rom_025": rom,
           "caveats": ["one scene, two seams; terrain can line up with a strip",
                       "bin 2 at TDI 64 against bin 1 at TDI 128, as flown: binning and TDI are not separated",
                       "polygon area is a sum per band, not the label's union",
                       "no human truth: this is how the detector reads the two kinds of data, not which is right"]}
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
