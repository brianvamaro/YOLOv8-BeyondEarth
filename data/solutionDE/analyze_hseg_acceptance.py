# H-seg adoption check, analysis half (GPU half = run_hseg_ladder.py). The phase-3 suite is
# the pre-registered acceptance harness for a weight swap; the swap resets every baseline, so
# every H-seg number is compared against BOTH its own 256-only baseline and the production
# (yolov8m) numbers already in the goal3_* CSVs.
#
# Axes (maintainer framing 2026-07-10: under a phase-5 G-remask future, SAM2 is prompted from
# the detector BOXES, so what a swap must not regress is which-boulders-get-found + box
# localization; the mask-mAP headline only covers the un-remasked population):
#   1. recall vs the Test-6 humans by band  (hseg 256 / hseg fused vs production 256 / fused)
#   2. box localization on human-matched detections: bbox IoU vs the human polygon's bbox +
#      centroid offset in native px, by band — the SAM2-prompt proxy
#   3. parity within H-seg, cross-scale dupe rates (engulf 0.7), orientation (well-resolved),
#      SFD roll-off, wall clocks
# Writes hseg_acceptance_{recall,box,sfd,summary}.csv.
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import rasterio
from shapely.geometry import box as shp_box

DE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionDE")
sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
sys.path.insert(0, str(DE))
from YOLOv8BeyondEarth.fusion import (fuse_predictions, matched_recall,
                                      equivalent_diameter_px, pairwise_box_metric)
from YOLOv8BeyondEarth.orientation import (attach_orientations, grid_fraction,
                                           diagonal_fraction)
from analyze_goal3_rom import (load_aligned, overlap_match, band_of, sfd_rows, PRIEUR,
                               ARM_KEYS, DEFAULT_SIZE_BANDS, _band_label)
from analyze_dupe_deepdive import near_pair_table

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet")
ENGULF = 0.7


def fuse_pref(gdfs, res):
    return fuse_predictions(gdfs, rule="bands", bands={256: (0.0, np.inf)},
                            native_res=res, engulf_thr=ENGULF)[0]


def hseg_rom_gpkg(stem, ss, sr):
    tag = f"{ss}sr{sr}" if sr > 1 else f"{ss}"
    return (DE / "preds" / "hseg_rom_ladder" / stem /
            f"{stem}-predictions-ct-010-ss-{tag}-is-1024-ov-020-downscaled-mask-nms.gpkg")


def dev_gpkg(weights, scene, stem, ss, sr):
    tag = f"{ss}sr{sr}" if sr > 1 else f"{ss}"
    if weights == "hseg":
        d = DE / "preds" / f"hseg_{scene}"
    else:
        sub = {"512": f"{scene}_512", "256": f"{scene}_baseline",
               "256sr2": f"{scene}_e2stream"}[tag]
        d = DE / "preds" / sub
    return d / f"{stem}-predictions-ct-010-ss-{tag}-is-1024-ov-020-downscaled-mask-nms.gpkg"


def box_quality(human, cat, res):
    """Human-matched pairs (>50%-of-smaller overlap, greedy): bbox IoU + centroid offset."""
    hg, pg = human.geometry.values, cat.geometry.values
    tree = shapely.STRtree(pg)
    hi, pi = tree.query(hg, predicate="intersects")
    if len(hi) == 0:
        return pd.DataFrame()
    inter = shapely.area(shapely.intersection(hg[hi], pg[pi]))
    smaller = np.minimum(shapely.area(hg[hi]), shapely.area(pg[pi]))
    frac = np.where(smaller > 0, inter / smaller, 0.0)
    ok = frac > 0.5
    hm = np.zeros(len(hg), bool); pm = np.zeros(len(pg), bool); pairs = []
    for k in np.flatnonzero(ok)[np.argsort(-frac[ok])]:
        if not hm[hi[k]] and not pm[pi[k]]:
            hm[hi[k]] = True; pm[pi[k]] = True; pairs.append((hi[k], pi[k]))
    hidx = np.array([p[0] for p in pairs]); pidx = np.array([p[1] for p in pairs])
    hb = human.geometry.bounds.to_numpy(float)[hidx]
    pb = cat.geometry.bounds.to_numpy(float)[pidx]
    biou = pairwise_box_metric(hb, pb, "iou")
    off = np.hypot(shapely.get_x(shapely.centroid(hg[hidx])) - shapely.get_x(shapely.centroid(pg[pidx])),
                   shapely.get_y(shapely.centroid(hg[hidx])) - shapely.get_y(shapely.centroid(pg[pidx]))) / res
    return pd.DataFrame(dict(band=band_of(equivalent_diameter_px(human, res)[hidx]),
                             box_iou=biou, centroid_off_px=off))


if __name__ == "__main__":
    R, B, S = [], [], []

    # ---------------- HiRISE ROM tiles: both weight sets on identical ground ----------------
    res = 0.5
    with rasterio.open(BASE / "YOLOv8-BeyondEarth/data/mars_hirise/raster/ESP_017355_2260_RED.tif") as src:
        crs = src.crs
    rom = (load_aligned(PRIEUR / "mars/ESP_017355_2260/shp/ESP_017355_2260-ROM.shp", crs)
           .explode(index_parts=False).reset_index(drop=True))
    hum_all = load_aligned(PRIEUR / "mars/ESP_017355_2260/shp/ESP_017355_2260-boulder-mapping.shp", crs)
    hum_cent = hum_all.geometry.centroid

    def pooled_rom(gpkg_fn):
        ph, parms = [], {k: [] for k in ARM_KEYS}
        area = 0.0
        for tile in sorted(DE.glob("hirise_rom??.tif")):
            i = int(tile.stem[-2:])
            rp = rom.geometry.iloc[i]
            area += rp.area
            ph.append(hum_all[hum_cent.within(rp)])
            for k in ARM_KEYS:
                ss, sr = k if isinstance(k, tuple) else (k, 1)
                g = gpd.read_file(gpkg_fn(tile.stem, ss, sr)).set_crs(crs, allow_override=True)
                parms[k].append(g[g.geometry.centroid.within(rp)])
        human = gpd.GeoDataFrame(pd.concat(ph, ignore_index=True), geometry="geometry", crs=crs)
        gdfs = {k: gpd.GeoDataFrame(pd.concat(v, ignore_index=True), geometry="geometry", crs=crs)
                for k, v in parms.items()}
        return human, gdfs, area

    def prod_rom_gpkg(stem, ss, sr):
        tag = f"{ss}sr{sr}" if sr > 1 else f"{ss}"
        return (DE / "preds" / "rom_ladder" / stem /
                f"{stem}-predictions-ct-010-ss-{tag}-is-1024-ov-020-downscaled-mask-nms.gpkg")

    human, gdfs_h, rom_area = pooled_rom(hseg_rom_gpkg)
    _, gdfs_p, _ = pooled_rom(prod_rom_gpkg)
    print(f"[hirise ROM] human {len(human)} | hseg arms " +
          " | ".join(f"{k}: {len(g)}" for k, g in gdfs_h.items()), flush=True)

    cats = {}
    for wtag, gdfs in [("prod", gdfs_p), ("hseg", gdfs_h)]:
        cats[f"{wtag}_256"] = gdfs[256]
        cats[f"{wtag}_fused"] = fuse_pref(gdfs, res)
    h_band = band_of(equivalent_diameter_px(human, res))
    for name, cat in cats.items():
        hm, _ = overlap_match(human, cat)
        hh, _ = overlap_match(human, cat, hybrid=True)
        for lo, hi in DEFAULT_SIZE_BANDS:
            lab = _band_label(lo, hi)
            sel = h_band == lab
            R.append(dict(scene="hirise", catalog=name, band=lab, n_human=int(sel.sum()),
                          recall=float(hm[sel].mean()) if sel.sum() else np.nan,
                          recall_hybrid=float(hh[sel].mean()) if sel.sum() else np.nan))
        bq = box_quality(human, cat, res)
        for lo, hi in DEFAULT_SIZE_BANDS:
            lab = _band_label(lo, hi)
            s = bq[bq.band == lab]
            B.append(dict(scene="hirise", catalog=name, band=lab, n_pairs=len(s),
                          box_iou_med=float(s.box_iou.median()) if len(s) else np.nan,
                          off_px_med=float(s.centroid_off_px.median()) if len(s) else np.nan))
        srows, frows = sfd_rows("hirise", human, {name: cat}, res, rom_area)
        S += [f for f in frows if f["catalog"] == name]

    # ---------------- summary axes on the HiRISE dev crop (hseg weights) ----------------
    gdfs_dev = {k: gpd.read_file(dev_gpkg("hseg", "hirise", "hirise_crop4k",
                                          *(k if isinstance(k, tuple) else (k, 1))))
                for k in ARM_KEYS}
    fused_dev = fuse_pref(gdfs_dev, res)
    parity, unmatched = matched_recall(gdfs_dev[256], fused_dev, engulf_thr=ENGULF)
    npt = near_pair_table(fused_dev, res, scale_col="scale_key")
    ori = attach_orientations(fused_dev, res=res, n_workers=8)
    wr = ori[(ori.aspect_ra > 1.35) & (ori.diameter_m / res >= 8)]
    ori_b = attach_orientations(gdfs_dev[256], res=res, n_workers=8)
    wr_b = ori_b[(ori_b.aspect_ra > 1.35) & (ori_b.diameter_m / res >= 8)]
    walls = json.loads((DE / "hseg_timings.json").read_text())
    dev_wall = {a: walls[f"hseg_hirise_ss{a}"]["wall_s"] for a in ("512", "256", "256sr2")
                if f"hseg_hirise_ss{a}" in walls}
    summary = dict(
        parity=parity, n_unmatched=len(unmatched),
        n_fused=len(fused_dev), n_256=len(gdfs_dev[256]),
        dupe_cross_worst=float(npt.rate_cross.max()),
        diag_frac_fused_wr=diagonal_fraction(wr.angle180.values),
        diag_frac_256_wr=diagonal_fraction(wr_b.angle180.values),
        grid_frac_fused_wr=grid_fraction(wr.angle180.values),
        wall_s=dev_wall)
    print("\n[hseg dev-crop summary]", json.dumps(summary, indent=2, default=str))

    pd.DataFrame(R).to_csv(DE / "hseg_acceptance_recall.csv", index=False)
    pd.DataFrame(B).to_csv(DE / "hseg_acceptance_box.csv", index=False)
    pd.DataFrame(S).to_csv(DE / "hseg_acceptance_sfd.csv", index=False)
    pd.Series(summary).to_json(DE / "hseg_acceptance_summary.json", indent=2, default_handler=str)

    pd.set_option("display.width", 240)
    rdf = pd.DataFrame(R)
    print("\n=== recall vs human (hirise ROM), production vs H-seg ===")
    print(rdf.pivot_table(index="band", columns="catalog", values="recall").round(3).to_string())
    bdf = pd.DataFrame(B)
    print("\n=== box axis (SAM2-prompt proxy): median bbox IoU / centroid offset px ===")
    print(bdf.pivot_table(index="band", columns="catalog",
                          values=["box_iou_med", "off_px_med"]).round(3).to_string())
    print("\n=== SFD roll-off (px) ===")
    print(pd.DataFrame(S)[["catalog", "rolloff_px", "n_total"]].to_string(index=False))
