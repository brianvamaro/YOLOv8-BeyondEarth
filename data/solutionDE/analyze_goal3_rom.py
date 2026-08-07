# Goal-3 phase 3: recall/precision vs the Test-6 human reference + SFD readout (plan
# §Validation 4, 5). Reference = Prieur hand-mapped boulders inside the ROM (region-of-mapping):
# HiRISE = the 20 ROM-tile crops (run_goal3_ladder.py / prep_rom_crops.py), NAC = nac_crop2k
# clipped to its ROM overlap (~19% of the crop, ~1.2k human polys).
#
# Matching convention (Liu & Fa 2026, plan takeaway #2): >50% polygon overlap = correct
# detection, implemented as intersection / min(area_human, area_pred) > 0.5 with greedy
# one-to-one assignment by descending overlap; a centroid-gate match (2 x res, the project's
# standing pairing gate) is reported alongside as the sensitivity column. Precision per band is
# only interpretable at/above the human mapping floor (Prieur convention ~5x5 px) — sub-floor
# precision is the maintainer spot-check's question, not this script's.
#
# SFD (plan §Validation 5): cumulative N(>D) per catalog inside the ROM regions + a Golombek-
# style exponential fit  N(>D) = A exp(-lambda D)  over FIT_RANGE_PX (clearly-complete range);
# the roll-off point — where measured N(>D) departs below the extrapolated fit by
# ROLLOFF_TOL — is the measured completeness limit in px. Caveat (stated wherever used): a
# completeness indicator only under the assumption the true SFD follows the fitted law below
# the fit range (Golombek 2003 fits only above ~0.1 m); the spot-check breaks the degeneracy.
#
# CRS gotcha (memory: prieur-crs-override-gotcha): Prieur shapefiles share the native grid but
# carry cosmetically different CRS strings -> set_crs override, never to_crs.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import rasterio
from shapely.geometry import box as shp_box

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
from YOLOv8BeyondEarth.fusion import fuse_predictions, pool_predictions, equivalent_diameter_px
from YOLOv8BeyondEarth.benchmark import match_detections, DEFAULT_SIZE_BANDS, _band_label

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet")
DE = BASE / "YOLOv8-BeyondEarth" / "data" / "solutionDE"
PRIEUR = BASE / "data_prieur" / "raw_data"

RULES = ("scale_aware", "score", "bands")
PROVISIONAL_BANDS = {(256, 2): (0.0, 10.0), 256: (10.0, 40.0), 512: (40.0, np.inf)}
FIT_RANGE_PX = (10.0, 30.0)   # clearly-complete range for the exponential fit
ROLLOFF_TOL = 0.9             # measured/fitted below this = rolled off
ARM_KEYS = (512, 256, (256, 2))


def load_aligned(path, ref_crs):
    return gpd.read_file(path).set_crs(ref_crs, allow_override=True)


def rom_gpkg(tile_stem, ss, sr):
    tag = f"{ss}sr{sr}" if sr > 1 else f"{ss}"
    return (DE / "preds" / "rom_ladder" / tile_stem /
            f"{tile_stem}-predictions-ct-010-ss-{tag}-is-1024-ov-020-downscaled-mask-nms.gpkg")


def overlap_match(human, pred, hybrid=False):
    """Greedy 1-1 match by >50% overlap of the smaller polygon. ``hybrid=True`` additionally
    accepts centroid-in-polygon pairs (either direction) — the localization-tolerant reading
    for the <6 px regime, where a ~1-px mask offset defeats the pure overlap rule (probed
    2026-07-10: 'lost' humans all had an intersecting prediction at overlap 0.37-0.46).
    Returns (bool mask over human, bool mask over pred)."""
    h_matched = np.zeros(len(human), dtype=bool)
    p_matched = np.zeros(len(pred), dtype=bool)
    if len(human) == 0 or len(pred) == 0:
        return h_matched, p_matched
    hg = human.geometry.values
    pg = pred.geometry.values
    tree = shapely.STRtree(pg)
    hi, pi = tree.query(hg, predicate="intersects")
    if len(hi) == 0:
        return h_matched, p_matched
    inter = shapely.area(shapely.intersection(hg[hi], pg[pi]))
    smaller = np.minimum(shapely.area(hg[hi]), shapely.area(pg[pi]))
    frac = np.where(smaller > 0, inter / smaller, 0.0)
    ok = frac > 0.5
    if hybrid:
        ok = ok | shapely.contains(hg[hi], shapely.centroid(pg[pi])) \
                | shapely.contains(pg[pi], shapely.centroid(hg[hi]))
    order = np.argsort(-frac[ok])
    for k in np.flatnonzero(ok)[order]:
        if not h_matched[hi[k]] and not p_matched[pi[k]]:
            h_matched[hi[k]] = True
            p_matched[pi[k]] = True
    return h_matched, p_matched


def band_of(dpx):
    out = np.full(len(dpx), "<3", dtype=object)
    for lo, hi in DEFAULT_SIZE_BANDS:
        out[(dpx >= lo) & (dpx < hi)] = _band_label(lo, hi)
    return out


def catalogs_for(gdfs, res):
    """The catalogs compared against the human reference: each arm alone + fused per config."""
    cats = {f"arm_{k}": g for k, g in gdfs.items()}
    for rule in RULES:
        kw = dict(bands=PROVISIONAL_BANDS, native_res=res) if rule == "bands" else {}
        fused, _ = fuse_predictions(gdfs, rule=rule, **kw)
        cats[f"fused_{rule}"] = fused
    fused_e, _ = fuse_predictions(gdfs, engulf_thr=0.7)
    cats["fused_scale_aware_engulf"] = fused_e
    # 256-preferred (phase-3 ablation winner on HiRISE): 256 masks win every contested cluster
    # (production mask/size convention preserved wherever 256 saw the boulder); SR/512 members
    # survive only as rescued uniques (recall nets); fragments consolidated by the engulf pass.
    # Implemented as the bands mechanism with a trivial banding.
    fused_p, _ = fuse_predictions(gdfs, rule="bands", bands={256: (0.0, np.inf)},
                                  native_res=res, engulf_thr=0.7)
    cats["fused_256pref_engulf"] = fused_p
    return cats


def pr_rows(scene, region_label, human, cats, res):
    """Per-catalog, per-band recall (of human) and precision (of predictions) inside a region."""
    rows_r, rows_p, rows_u = [], [], []
    h_dpx = equivalent_diameter_px(human, res)
    h_band = band_of(h_dpx)
    matched_by = {}
    for name, cat in cats.items():
        hm, pm = overlap_match(human, cat)
        hh, _ = overlap_match(human, cat, hybrid=True)
        matched_by[name] = hm
        p_dpx = equivalent_diameter_px(cat, res)
        p_band = band_of(p_dpx)
        for lo, hi in DEFAULT_SIZE_BANDS:
            lab = _band_label(lo, hi)
            hsel = h_band == lab
            psel = p_band == lab
            rows_r.append(dict(scene=scene, region=region_label, catalog=name, band=lab,
                               n_human=int(hsel.sum()),
                               n_matched=int((hm & hsel).sum()),
                               recall=float((hm & hsel).sum() / hsel.sum()) if hsel.sum() else np.nan,
                               recall_hybrid=float((hh & hsel).sum() / hsel.sum()) if hsel.sum() else np.nan))
            rows_p.append(dict(scene=scene, region=region_label, catalog=name, band=lab,
                               n_pred=int(psel.sum()),
                               n_matched=int((pm & psel).sum()),
                               precision=float((pm & psel).sum() / psel.sum()) if psel.sum() else np.nan))
    # human boulders found by exactly one arm (per-arm unique recall contribution)
    arm_names = [f"arm_{k}" for k in ARM_KEYS]
    arm_mat = np.stack([matched_by[n] for n in arm_names])  # (3, n_human)
    n_arms_found = arm_mat.sum(axis=0)
    for name, hm in zip(arm_names, arm_mat):
        only = hm & (n_arms_found == 1)
        for lo, hi in DEFAULT_SIZE_BANDS:
            lab = _band_label(lo, hi)
            hsel = h_band == lab
            rows_u.append(dict(scene=scene, region=region_label, arm=name, band=lab,
                               n_human=int(hsel.sum()), n_only=int((only & hsel).sum())))
    return rows_r, rows_p, rows_u


def size_vs_human_rows(scene, human, cats, res):
    """Size-convention arbitration (validation 10): matched-pair diameter ratio arm/human by
    band. The dev-crop audit found SR masks ~0.86-0.94x and 512 masks ~1.07-1.32x the 256
    convention; the human polygons say which convention is closest to truth."""
    rows = []
    h_dpx = equivalent_diameter_px(human, res)
    h_band = band_of(h_dpx)
    for name, cat in cats.items():
        if not name.startswith("arm_"):
            continue
        m = match_detections(human, cat, max_dist=2.0 * res)
        if not len(m):
            continue
        c_dpx = equivalent_diameter_px(cat, res)
        ratio = c_dpx[m.idx_b.values] / h_dpx[m.idx_a.values]
        bands = h_band[m.idx_a.values]
        for lo, hi in DEFAULT_SIZE_BANDS:
            lab = _band_label(lo, hi)
            r = ratio[bands == lab]
            rows.append(dict(scene=scene, arm=name, band=lab, n=len(r),
                             median=float(np.median(r)) if len(r) >= 10 else np.nan,
                             q25=float(np.percentile(r, 25)) if len(r) >= 10 else np.nan,
                             q75=float(np.percentile(r, 75)) if len(r) >= 10 else np.nan))
    return rows


def sfd_rows(scene, human, cats, res, area_m2):
    """Cumulative N(>D) per km^2 on a fixed D grid + exponential fit + roll-off, per catalog."""
    grid = np.arange(2.0, 60.5, 0.5)  # px
    rows, fits = [], []
    for name, cat in {"human": human, **cats}.items():
        dpx = np.sort(equivalent_diameter_px(cat, res))
        n_gt = len(dpx) - np.searchsorted(dpx, grid, side="right")
        dens = n_gt / (area_m2 / 1e6)
        for d, n, v in zip(grid, n_gt, dens):
            rows.append(dict(scene=scene, catalog=name, d_px=float(d), n_gt=int(n),
                             per_km2=float(v)))
        sel = (grid >= FIT_RANGE_PX[0]) & (grid <= FIT_RANGE_PX[1]) & (n_gt > 5)
        if sel.sum() >= 5:
            lam, loga = np.polyfit(grid[sel], np.log(n_gt[sel]), 1)
            fit = np.exp(loga + lam * grid)
            below = grid < FIT_RANGE_PX[0]
            ratio = np.where(fit > 0, n_gt / fit, np.nan)
            rolled = below & (ratio < ROLLOFF_TOL)
            rolloff = float(grid[rolled].max()) if rolled.any() else np.nan
            fits.append(dict(scene=scene, catalog=name, lam=float(lam), logA=float(loga),
                             fit_lo=FIT_RANGE_PX[0], fit_hi=FIT_RANGE_PX[1],
                             rolloff_px=rolloff, n_total=len(dpx)))
        else:
            fits.append(dict(scene=scene, catalog=name, lam=np.nan, logA=np.nan,
                             fit_lo=FIT_RANGE_PX[0], fit_hi=FIT_RANGE_PX[1],
                             rolloff_px=np.nan, n_total=len(dpx)))
    return rows, fits


if __name__ == "__main__":
    R, P, U, S, F, SZ = [], [], [], [], [], []

    # ---------------- HiRISE: the 20 ROM-tile crops, pooled ----------------
    res = 0.5
    with rasterio.open(BASE / "YOLOv8-BeyondEarth/data/mars_hirise/raster/ESP_017355_2260_RED.tif") as src:
        crs = src.crs
    rom = (load_aligned(PRIEUR / "mars/ESP_017355_2260/shp/ESP_017355_2260-ROM.shp", crs)
           .explode(index_parts=False).reset_index(drop=True))
    hum_all = load_aligned(PRIEUR / "mars/ESP_017355_2260/shp/ESP_017355_2260-boulder-mapping.shp", crs)
    hum_cent = hum_all.geometry.centroid

    pooled_h, pooled_arms = [], {k: [] for k in ARM_KEYS}
    rom_area = 0.0
    for tile in sorted(DE.glob("hirise_rom??.tif")):
        i = int(tile.stem[-2:])
        rpoly = rom.geometry.iloc[i]
        rom_area += rpoly.area
        pooled_h.append(hum_all[hum_cent.within(rpoly)])
        for k in ARM_KEYS:
            g = gpd.read_file(rom_gpkg(tile.stem, *(k if isinstance(k, tuple) else (k, 1))))
            g = g.set_crs(crs, allow_override=True)
            pooled_arms[k].append(g[g.geometry.centroid.within(rpoly)])
    human = pd.concat(pooled_h, ignore_index=True)
    human = gpd.GeoDataFrame(human, geometry="geometry", crs=crs)
    gdfs = {k: gpd.GeoDataFrame(pd.concat(v, ignore_index=True), geometry="geometry", crs=crs)
            for k, v in pooled_arms.items()}
    print(f"[hirise] human {len(human)} | " +
          " | ".join(f"{k}: {len(g)}" for k, g in gdfs.items()), flush=True)
    cats = catalogs_for(gdfs, res)
    r, p, u = pr_rows("hirise", "rom20", human, cats, res)
    R += r; P += p; U += u
    s, f = sfd_rows("hirise", human, cats, res, rom_area)
    S += s; F += f
    SZ += size_vs_human_rows("hirise", human, cats, res)

    # ---------------- NAC: nac_crop2k clipped to its ROM overlap ----------------
    res = 0.48357597846618
    arms = {
        512: DE / "preds/nac_512/nac_crop2k-predictions-ct-010-ss-512-is-1024-ov-020-downscaled-mask-nms.gpkg",
        256: DE / "preds/nac_baseline/nac_crop2k-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg",
        (256, 2): DE / "preds/nac_e2stream/nac_crop2k-predictions-ct-010-ss-256sr2-is-1024-ov-020-downscaled-mask-nms.gpkg",
    }
    gdfs = {k: gpd.read_file(p) for k, p in arms.items()}
    crs = gdfs[256].crs
    with rasterio.open(DE / "nac_crop2k.tif") as src:
        crop_poly = shp_box(*src.bounds)
    rom = load_aligned(PRIEUR / "moon/censorinus/shp/M139694087LE-censorinus-ROM.shp", crs)
    hum_all = load_aligned(PRIEUR / "moon/censorinus/shp/M139694087LE-censorinus-boulder-mapping.shp", crs)
    rom_u = rom.union_all() if hasattr(rom, "union_all") else rom.unary_union
    region = crop_poly.intersection(rom_u)
    human = hum_all[hum_all.geometry.centroid.within(region)].reset_index(drop=True)
    gdfs = {k: g[g.geometry.centroid.within(region)].reset_index(drop=True)
            for k, g in gdfs.items()}
    print(f"[nac] human {len(human)} | " +
          " | ".join(f"{k}: {len(g)}" for k, g in gdfs.items()), flush=True)
    cats = catalogs_for(gdfs, res)
    r, p, u = pr_rows("nac", "crop_rom", human, cats, res)
    R += r; P += p; U += u
    s, f = sfd_rows("nac", human, cats, res, region.area)
    S += s; F += f
    SZ += size_vs_human_rows("nac", human, cats, res)

    pd.DataFrame(R).to_csv(DE / "goal3_rom_recall.csv", index=False)
    pd.DataFrame(P).to_csv(DE / "goal3_rom_precision.csv", index=False)
    pd.DataFrame(U).to_csv(DE / "goal3_rom_unique.csv", index=False)
    pd.DataFrame(S).to_csv(DE / "goal3_sfd.csv", index=False)
    pd.DataFrame(F).to_csv(DE / "goal3_sfd_fit.csv", index=False)
    pd.DataFrame(SZ).to_csv(DE / "goal3_size_vs_human.csv", index=False)

    rec = pd.DataFrame(R)
    print("\n=== recall vs human, key catalogs ===")
    key = rec[rec.catalog.isin(["arm_256", "fused_scale_aware"])]
    print(key.pivot_table(index=["scene", "band"], columns="catalog", values="recall")
          .to_string())
    print("\n=== SFD fits / roll-off (completeness limit, px) ===")
    print(pd.DataFrame(F).to_string(index=False))
