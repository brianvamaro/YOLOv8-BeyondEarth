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
#
# 2026-08-09 — made runnable for H-seg x G. Four changes, each one blocking that experiment:
#   (a) the catalog filename is no longer hardcoded to `ct-010` + an untagged mask. It is built
#       from a per-arm (conf, mask_tag, directory) spec, because BOTH prerequisites of the H-seg x G
#       plan change that filename: retuning conf gives `ct-0XX` and remasking gives `-mask-sam2-`
#       (predict.py:880-893). Previously this raised FileNotFoundError on any such run.
#   (b) the two arms can now carry DIFFERENT conf. The plan requires H-seg at a retuned operating
#       point against production at 0.10; the old single hardcoded `ct-010` could not express it.
#   (c) precision and F1 by band, which the pre-registered read asks for and this script never
#       computed — it took `overlap_match`'s human-side mask and threw the prediction-side one away.
#   (d) `mask_src` composition per catalog: under G a re-mask that trips the background-grab guard
#       falls back to the YOLO mask, so a "sam2" catalog is not 100% SAM2, and the fallback rate can
#       differ between arms whose box localisation differs — which would confound the comparison.
import argparse
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
from YOLOv8BeyondEarth.predict import prediction_filenames
from YOLOv8BeyondEarth.orientation import (attach_orientations, grid_fraction,
                                           diagonal_fraction)
from analyze_goal3_rom import (load_aligned, overlap_match, band_of, sfd_rows, PRIEUR,
                               ARM_KEYS, DEFAULT_SIZE_BANDS, _band_label)
from analyze_dupe_deepdive import near_pair_table

BASE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet")
ENGULF = 0.7

#: Where each weight-set's catalogs live, and the two naming parameters `get_sliced_prediction`
#: bakes into the filename. Defaults reproduce the 2026-07-10 run exactly; the CLI overrides them
#: so the same harness can score an H-seg x G run without editing the file.
#: ``dev_dirs`` is keyed by arm tag because production's dev-crop layout is irregular (one
#: directory per arm) while H-seg's is flat (one directory per scene).
ARM_SPECS = {
    "prod": dict(rom_dir="rom_ladder", conf=0.10, mask_tag=None,
                 dev_dirs={"512": "{scene}_512", "256": "{scene}_baseline",
                           "256sr2": "{scene}_e2stream"},
                 timings="goal3_timings.json", timings_key="{scene}_ss{tag}"),
    "hseg": dict(rom_dir="hseg_rom_ladder", conf=0.10, mask_tag=None,
                 dev_dirs={"512": "hseg_{scene}", "256": "hseg_{scene}",
                           "256sr2": "hseg_{scene}"},
                 timings="hseg_timings.json", timings_key="hseg_{scene}_ss{tag}"),
}


def fuse_pref(gdfs, res):
    return fuse_predictions(gdfs, rule="bands", bands={256: (0.0, np.inf)},
                            native_res=res, engulf_thr=ENGULF)[0]


def arm_tag(ss, sr):
    return f"{ss}sr{sr}" if sr > 1 else f"{ss}"


def catalog_name(stem, ss, sr, conf, mask_tag):
    """The NMS catalog `run_hseg_ladder.py` / `run_goal3_ladder.py` wrote for this arm.

    Delegates to `predict.prediction_filenames` rather than re-deriving the pattern — three call
    sites here used to hardcode `ct-010` and an untagged mask, which is exactly why an H-seg x G
    run could not be scored. Both ladders run at is=1024, ov=0.2, downscale_pred=True, gpkg.
    """
    return prediction_filenames(
        stem, confidence_threshold=conf, slice_size=ss, inference_size=1024,
        overlap_height_ratio=0.2, sr_factor=sr, downscale_pred=True,
        mask_tag=mask_tag, output_format="gpkg")["nms"]


def rom_gpkg_for(arm):
    """ROM-tile catalog path builder for one arm, as `pooled_rom` expects it."""
    spec = ARM_SPECS[arm]

    def fn(stem, ss, sr):
        return (DE / "preds" / spec["rom_dir"] / stem /
                catalog_name(stem, ss, sr, spec["conf"], spec["mask_tag"]))
    return fn


def dev_gpkg(arm, scene, stem, ss, sr):
    spec = ARM_SPECS[arm]
    d = DE / "preds" / spec["dev_dirs"][arm_tag(ss, sr)].format(scene=scene)
    return d / catalog_name(stem, ss, sr, spec["conf"], spec["mask_tag"])


def _f1(precision, recall):
    """Harmonic mean, NaN-safe. NaN when either side is undefined (empty band) or both are 0 —
    an F1 of 0 would read as "measured and bad" where the truth is "nothing to measure"."""
    if not np.isfinite(precision) or not np.isfinite(recall) or (precision + recall) == 0:
        return np.nan
    return 2.0 * precision * recall / (precision + recall)


def mask_src_mix(cat):
    """Fraction of rows by geometry producer (the ``mask_src`` column, predict.py:806).

    Under Solution G a re-mask that trips the background-grab guard falls back to the YOLO mask
    rather than dropping the detection, so a catalog written with ``mask_tag="sam2"`` is **not**
    100% SAM2. The fallback rate is a property of box quality, so it can differ between two arms
    being compared — report it or a coverage difference gets read as a quality difference.
    """
    if "mask_src" not in getattr(cat, "columns", []):
        return None
    return {str(k): round(float(v), 4) for k, v in cat.mask_src.value_counts(normalize=True).items()}


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


def _cli():
    p = argparse.ArgumentParser(
        description="Score an H-seg (or H-seg x G) run against the human reference.",
        epilog="H-seg x G example: --hseg-conf 0.15 --hseg-mask-tag sam2 "
               "--hseg-rom-dir hseg_g_rom_ladder --hseg-dev-dir 'hseg_g_{scene}' "
               "--prod-mask-tag sam2 --prod-rom-dir rom_g_ladder "
               "--prod-dev-dir-512 '{scene}_g_512' ... (the matched production+G control)")
    for arm in ("hseg", "prod"):
        p.add_argument(f"--{arm}-conf", type=float,
                       help=f"confidence the {arm} catalogs were written at (default 0.10). "
                            "The two arms may differ — that is the point of the H-seg conf retune.")
        p.add_argument(f"--{arm}-mask-tag",
                       help=f"mask_tag the {arm} catalogs were written with, e.g. sam2")
        p.add_argument(f"--{arm}-rom-dir", help=f"preds/ subdirectory holding the {arm} ROM tiles")
        p.add_argument(f"--{arm}-dev-dir",
                       help=f"preds/ subdirectory for the {arm} dev crops, '{{scene}}' templated. "
                            "Sets all three arms; use --prod-dev-dir-<tag> for a per-arm override.")
    for tag in ("512", "256", "256sr2"):
        p.add_argument(f"--prod-dev-dir-{tag}",
                       help=f"per-arm override of the production dev-crop dir for the {tag} arm")
    a = p.parse_args()

    for arm in ("hseg", "prod"):
        spec = ARM_SPECS[arm]
        for key, attr in (("conf", f"{arm}_conf"), ("mask_tag", f"{arm}_mask_tag"),
                          ("rom_dir", f"{arm}_rom_dir")):
            if getattr(a, attr) is not None:
                spec[key] = getattr(a, attr)
        if getattr(a, f"{arm}_dev_dir") is not None:
            spec["dev_dirs"] = {t: getattr(a, f"{arm}_dev_dir") for t in spec["dev_dirs"]}
    for tag in ("512", "256", "256sr2"):
        v = getattr(a, f"prod_dev_dir_{tag}".replace("-", "_"))
        if v is not None:
            ARM_SPECS["prod"]["dev_dirs"][tag] = v
    return a


if __name__ == "__main__":
    _cli()
    R, B, S = [], [], []
    print("[arm specs] " + json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "timings_key"} for k, v in ARM_SPECS.items()},
        default=str))

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

    human, gdfs_h, rom_area = pooled_rom(rom_gpkg_for("hseg"))
    _, gdfs_p, _ = pooled_rom(rom_gpkg_for("prod"))
    print(f"[hirise ROM] human {len(human)} | hseg arms " +
          " | ".join(f"{k}: {len(g)}" for k, g in gdfs_h.items()), flush=True)

    cats = {}
    for wtag, gdfs in [("prod", gdfs_p), ("hseg", gdfs_h)]:
        cats[f"{wtag}_256"] = gdfs[256]
        cats[f"{wtag}_fused"] = fuse_pref(gdfs, res)
    h_band = band_of(equivalent_diameter_px(human, res))
    mask_mix = {}
    for name, cat in cats.items():
        # `overlap_match` returns (human-side, prediction-side) match masks. This script used to
        # discard the prediction side, which is exactly what precision needs — so the harness could
        # not compute its own pre-registered decision axis (F1 by band). Both are kept now.
        hm, pm = overlap_match(human, cat)
        hh, ph = overlap_match(human, cat, hybrid=True)
        # NOTE the denominators differ by construction: recall is over HUMANS in the band, precision
        # over PREDICTIONS in the band (a detection's band is its own size — there is no other
        # definition available for an unmatched false positive). F1 combines the two, so read it as
        # a summary of the pair, not as a quantity over a single population.
        p_band = band_of(equivalent_diameter_px(cat, res))
        mask_mix[name] = mask_src_mix(cat)
        for lo, hi in DEFAULT_SIZE_BANDS:
            lab = _band_label(lo, hi)
            sel, psel = h_band == lab, p_band == lab
            rec = float(hm[sel].mean()) if sel.sum() else np.nan
            prec = float(pm[psel].mean()) if psel.sum() else np.nan
            rec_h = float(hh[sel].mean()) if sel.sum() else np.nan
            prec_h = float(ph[psel].mean()) if psel.sum() else np.nan
            R.append(dict(scene="hirise", catalog=name, band=lab,
                          n_human=int(sel.sum()), n_pred=int(psel.sum()),
                          recall=rec, precision=prec, f1=_f1(prec, rec),
                          recall_hybrid=rec_h, precision_hybrid=prec_h,
                          f1_hybrid=_f1(prec_h, rec_h)))
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
    tpath = DE / ARM_SPECS["hseg"]["timings"]
    walls = json.loads(tpath.read_text()) if tpath.exists() else {}
    dev_wall = {a: walls[k]["wall_s"] for a in ("512", "256", "256sr2")
                for k in [ARM_SPECS["hseg"]["timings_key"].format(scene="hirise", tag=a)]
                if k in walls}
    summary = dict(
        parity=parity, n_unmatched=len(unmatched),
        n_fused=len(fused_dev), n_256=len(gdfs_dev[256]),
        dupe_cross_worst=float(npt.rate_cross.max()),
        diag_frac_fused_wr=diagonal_fraction(wr.angle180.values),
        diag_frac_256_wr=diagonal_fraction(wr_b.angle180.values),
        grid_frac_fused_wr=grid_fraction(wr.angle180.values),
        # Which producer each catalog's geometry actually came from. Under G this is the SAM2
        # coverage rate, and a difference between arms confounds every axis above.
        mask_src=mask_mix,
        arm_specs={k: {kk: str(vv) for kk, vv in v.items()} for k, v in ARM_SPECS.items()},
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
    print("\n=== precision (own-size bands, denominator = predictions in band) ===")
    print(rdf.pivot_table(index="band", columns="catalog", values="precision").round(3).to_string())
    print("\n=== F1 — THE PRE-REGISTERED DECISION AXIS ===")
    print(rdf.pivot_table(index="band", columns="catalog", values="f1").round(3).to_string())
    if any(mask_mix.values()):
        print("\n=== mask_src composition (SAM2 coverage; a between-arm gap confounds the rest) ===")
        print(pd.DataFrame(mask_mix).fillna(0.0).round(4).to_string())
    bdf = pd.DataFrame(B)
    print("\n=== box axis (SAM2-prompt proxy): median bbox IoU / centroid offset px ===")
    print(bdf.pivot_table(index="band", columns="catalog",
                          values=["box_iou_med", "off_px_med"]).round(3).to_string())
    print("\n=== SFD roll-off (px) ===")
    print(pd.DataFrame(S)[["catalog", "rolloff_px", "n_total"]].to_string(index=False))
