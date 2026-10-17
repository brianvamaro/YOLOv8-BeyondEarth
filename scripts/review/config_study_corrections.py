"""The configuration study's corrected numbers, recomputed in one tracked place (2026-10-16).

The audit (docs/configuration_study_audit_2026-10-16.md, 85 findings) found the registered result sound,
but the write-up's account of the score floor wrong or incomplete in places. Its corrected numbers also
lived only in untracked scratch scripts. This script recomputes every number the corrected write-up and
the hirise2ctx brief quote. It works from the study's own inputs, through the same harness
(config_study.py, and so notebook 35's label_error), and writes one JSON.

Sections, which are also the JSON keys:
- validate: the harness still reproduces notebook 35.
- v2_basis: what v2's labels were made from (the vClaire exports' scores, with each .dbf and .shp
  checked for truncation), and C1 at >= 0.25 as the v2 equivalent.
- matched: one-to-one matching of C1 to the humans at floors 0.10 / 0.25 / 0.40, giving recall, recall
  by size, and the area decomposition.
- scene: the level per image, matched-score medians, exact-zero cells, and the kept-share diagnostic.
- plateau: C1's error across floors under raw RMSE, log error and Spearman, and 0.40 against 0.50.
- concentration: where the deciding squared error comes from (image and tile shares).
- res025: the four held-out 0.25 m/px tiles.
- fine_grid: C1 and C1GS on a 0.05 floor grid, and with the floor chosen out of sample.
- selection: every candidate's floor chosen leaving one tile, or one image, out (selection optimism).
- reweighted: the registered rule with ROM's cells weighted to v2's 40 m label distribution, with v2
  deflated by 1 / 1.3 / 1.4 / 1.7 / 1.95 (audit D2).
- power: the smallest RMSE ratio the registered tile bootstrap can resolve, and an image-aware bootstrap.
- cells_160m: the 160 m comparison, with intervals; RMSE at 160 m against the 40 m cells on the same
  footprint; v2's own label cells by size (read-only from hirise2ctx's dataset_v2).
- mask_area: SAM2/YOLO area per matched detection, area-weighted over the label's own population.
- extras: the compensation test per image, size-floor reach, the ROM images' viewing geometry, the
  C1-vs-C1GS tie at each size floor, C1GS's render cost and how often it kept YOLO's mask.

Conventions:
- Held-out means validation + test.
- Cells are notebook 35's complete cells, 40 m unless stated.
- The size floor is 1.4105 m equivalent diameter on both sides.
- Matching is optimal one-to-one at IoU >= 0.3 (scipy linear_sum_assignment), over polygons whose
  centroid lies in the ROM tile. Its areas are sums of polygon areas over the tile, not the label's
  per-cell unions.

    conda run -n bouldernet python scripts/review/config_study_corrections.py --out data/review/config_study_corrections.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import config_study as cs                                                   # noqa: E402

VCLAIRE = Path(r"C:/Users/brian/Documents/PhD/HiRiseToCTXBoulders/hirise_40_vClaire")
V2_LABELS = Path(r"C:/Users/brian/Documents/PhD/HiRiseToCTXBoulders/hirise2ctx/dataset_v2/labels")
RES = {"ESP_017355_2260": 0.50, "ESP_025650_2165": 0.25, "ESP_028537_2270": 0.25,
       "ESP_037018_2170": 0.595}
RES025 = ("ESP_025650_2165", "ESP_028537_2270")
SIZE_BINS = ((1.4105, 2.5), (2.5, 4.0), (4.0, 6.0), (6.0, np.inf))
IOU_MIN = 0.3
FINE = (0.30, 0.35, 0.40, 0.45, 0.50)


# --- shared helpers --------------------------------------------------------------------------------

def eqd(area):
    return 2.0 * np.sqrt(np.asarray(area, dtype=float) / np.pi)


def held(df, le):
    return df[df.split.isin(le.HELD_OUT)]


def rmse(df):
    return float(np.sqrt(((df.fa_pred - df.fa_true) ** 2).mean()))


def tile_parts(df, index):
    """Per-tile SSE and cell count, in the order of ``index``."""
    g = df.assign(se=(df.fa_pred - df.fa_true) ** 2).groupby(["obs_id", "tile_id"])
    return g.se.sum().reindex(index).to_numpy(), g.size().reindex(index).to_numpy()


def tiles_of(df):
    return pd.MultiIndex.from_tuples(sorted(set(map(tuple, df[["obs_id", "tile_id"]]
                                                    .drop_duplicates().to_numpy()))),
                                     names=["obs_id", "tile_id"])


def draws(n, seed=cs.SEED):
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=cs.N_BOOT)


def boot_rmse(parts, w):
    sse, cnt = parts
    return np.sqrt((w @ sse) / (w @ cnt))


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def repair_aligned(geoms):
    """notebook 35's repair (make_valid, keep polygonal), but returning a keep-mask so scores stay
    aligned with geometries."""
    v = np.asarray(geoms, dtype=object)
    out, ok = np.empty(len(v), dtype=object), np.zeros(len(v), bool)
    for i, g in enumerate(v):
        if g is None or shapely.is_missing(g):
            continue
        if not shapely.is_valid(g):
            try:
                g = shapely.make_valid(g)
            except Exception:
                continue
        if g is not None and shapely.get_type_id(g) in (3, 6):
            out[i], ok[i] = g, True
    return out, ok


def match(P, H):
    """Optimal one-to-one IoU matching (maximise total IoU) over pairs with IoU >= IOU_MIN."""
    from scipy.optimize import linear_sum_assignment
    if len(P) == 0 or len(H) == 0:
        return np.empty(0, int), np.empty(0, int)
    pi, hi = shapely.STRtree(H).query(P, predicate="intersects")
    if not len(pi):
        return np.empty(0, int), np.empty(0, int)
    inter = shapely.area(shapely.intersection(P[pi], H[hi]))
    iou = inter / (shapely.area(P[pi]) + shapely.area(H[hi]) - inter)
    ok = iou >= IOU_MIN
    pi, hi, iou = pi[ok], hi[ok], iou[ok]
    if not len(pi):
        return np.empty(0, int), np.empty(0, int)
    up, ip = np.unique(pi, return_inverse=True)
    uh, ih = np.unique(hi, return_inverse=True)
    M = np.zeros((len(up), len(uh)))
    M[ip, ih] = iou
    r, c = linear_sum_assignment(-M)
    good = M[r, c] >= IOU_MIN
    return up[r[good]], uh[c[good]]


class Tiles:
    """Humans and configuration-1 predictions per registered ROM tile, loaded once."""

    def __init__(self, le, splits, keep, sub=""):
        self.rows = []
        for s in le.images().values():
            rom = le.load_rom(s, splits)
            ref = rom.crs
            hum, hok = repair_aligned(le.load_human(s, ref).geometry.values)
            hum = hum[hok]
            hcent = shapely.centroid(hum)
            spec = cs._spec(s, cs.ROM_DIR, sub)
            for _, t in rom.iterrows():
                key = (s.obs_id, int(t.rom_index), int(t.tile_id))
                if key not in keep:
                    continue
                tile = shapely.make_valid(t.geometry)
                g = le.load_predictions(spec, int(t.rom_index), int(t.tile_id), ref)
                P, pok = repair_aligned(g.geometry.values)
                sc = g.score.to_numpy()[pok]
                P = P[pok]
                inside = shapely.within(shapely.centroid(P), tile)
                self.rows.append(dict(obs=s.obs_id, tile_id=int(t.tile_id), split=t.split,
                                      H=hum[shapely.within(hcent, tile)], P=P[inside],
                                      score=sc[inside]))


# --- sections --------------------------------------------------------------------------------------

def sec_validate(le):
    want = json.loads(cs.NB35_SUMMARY.read_text())["scale_sweep"]
    splits, worst = le.split_map(), 0.0
    for cell in cs.CELLS:
        df = cs.paired(le, splits, cs.ROM_DIR, "", cs.FLOORS[0], cell)
        for tag, sel in (("all_rom", df), ("held_out", held(df, le))):
            got = le.error_metrics(sel)
            ref = next(r for r in want[tag] if r["cell_size_m"] == cell)
            for k, v in ref.items():
                if k != "cell_size_m":
                    worst = max(worst, abs(got[k] - v) / max(abs(v), 1e-12))
    return {"worst_relative_difference_vs_notebook35": worst}


def _shp_integrity(path):
    """A .shp header declares the file's length (big-endian 16-bit words at byte 24), so a partly
    copied file shows itself."""
    with open(path, "rb") as f:
        declared = 2 * struct.unpack(">i", f.read(28)[24:28])[0]
    actual = os.path.getsize(path)
    return {"declared_bytes": declared, "actual_bytes": actual, "complete": actual >= declared}


def _dbf_scores(path):
    """The score column, plus the record count the header declares: a truncated .dbf holds fewer
    rows than it declares, and the rows it lost are the low-score ones (exports are score-ordered)."""
    with open(path, "rb") as f:
        hdr = f.read(32)
        nrec, hlen, rlen = struct.unpack("<IHH", hdr[4:12])
        fields, off = [], 1
        while True:
            fd = f.read(32)
            if not fd or fd[0] == 0x0D:
                break
            fields.append((fd[:11].split(b"\0")[0].decode().lower(), fd[16], off))
            off += fd[16]
        f.seek(hlen)
        raw = f.read()
    n = len(raw) // rlen
    arr = np.frombuffer(raw[:n * rlen], dtype=np.uint8).reshape(n, rlen)
    _, fl, o = next(x for x in fields if x[0] == "score")
    col = arr[:, o:o + fl].tobytes()
    return np.array([float(col[i * fl:(i + 1) * fl].strip() or b"nan") for i in range(n)]), int(nrec)


def sec_v2_basis(le, cells, tiles):
    exports = []
    for d in sorted(p for p in VCLAIRE.iterdir() if p.is_dir()):
        fs = sorted(glob.glob(str(d / "*.dbf")))
        if not fs:
            continue
        try:
            s, declared = _dbf_scores(fs[0])
        except Exception as e:
            exports.append({"obs": d.name, "error": f"{type(e).__name__}: {e}"})
            continue
        present = int(s.size)
        s = s[np.isfinite(s)]
        lit = np.isclose(s, 0.10, atol=1e-9)
        real = s[~lit]
        shp = sorted(glob.glob(str(d / "*.shp")))
        exports.append({"obs": d.name, "dbf_records_declared": declared, "dbf_records_present": present,
                        "dbf_complete": present >= declared,
                        "shp": _shp_integrity(shp[0]) if shp else None,
                        "n": int(s.size), "n_at_exactly_0.10": int(lit.sum()),
                        "min_score_other_rows": float(real.min()) if real.size else None,
                        "n_strictly_between_0.10_and_0.25": int(((s > 0.10 + 1e-9) &
                                                                  (s < 0.25 - 1e-9)).sum()),
                        "frac_rows_below_0.40": float((s < 0.40).mean()),
                        "frac_real_rows_below_0.40": float((real < 0.40).mean()) if real.size else None})
    # only a complete .dbf can show where an export's scores stop; a truncated one lost its low rows
    ok = [e for e in exports if "n" in e and e["dbf_complete"]]
    d40 = cells[(cells.candidate == "C1") & (cells.cell_size_m == 40.0)]
    d160 = cells[(cells.candidate == "C1") & (cells.cell_size_m == 160.0)]
    levels = {}
    for f in (0.10, 0.25, 0.40):
        row = {}
        for tag, sel in (("all", lambda x: x), ("held_out", lambda x: held(x, le))):
            a = sel(d40[d40.floor == f])
            b = sel(d160[d160.floor == f])
            row[tag] = {"area_ratio_40m": le.error_metrics(a)["pooled_ratio"],
                        "count_ratio_40m": float(a.n_pred.sum() / a.n_true.sum()),
                        "area_ratio_160m": le.error_metrics(b)["pooled_ratio"],
                        "rmse_40m": rmse(a)}
        row["count_ratio_by_image_all_tiles"] = {
            o: float(g.n_pred.sum() / g.n_true.sum()) for o, g in d40[d40.floor == f].groupby("obs_id")}
        levels[f"{f:.2f}"] = row
    # share of the v2-equivalent (>= 0.25) catalog that a 0.40 floor removes, >= 1.41 m, in ROM tiles
    by_img = {}
    for r in tiles.rows:
        big = eqd(shapely.area(r["P"])) >= le.STAGE4_MIN_DIAMETER_M
        n25 = int((big & (r["score"] >= 0.25)).sum())
        n40 = int((big & (r["score"] >= 0.40)).sum())
        a, b = by_img.get(r["obs"], (0, 0))
        by_img[r["obs"]] = (a + n25, b + n40)
    removed = {o: 1 - b / a for o, (a, b) in by_img.items()}
    tot = [sum(x) for x in zip(*by_img.values())]
    return {"vclaire_exports": exports,
            "vclaire_summary": {
                "n_exports": len(exports), "n_complete_dbf": len(ok),
                "truncated_dbf": [e["obs"] for e in exports if "n" in e and not e["dbf_complete"]],
                "unreadable": [e["obs"] for e in exports if "n" not in e],
                "truncated_shp": [e["obs"] for e in exports if e.get("shp") and not e["shp"]["complete"]],
                "note": "the summary covers complete .dbf files only. A truncated .shp leaves the scores "
                        "intact but drops geometry, so that image's label sits at a higher realised "
                        "floor (hirise2ctx R23, DECISIONS 2026-08-06o)",
                "n_with_any_score_strictly_between_0.10_and_0.25":
                    sum(1 for e in ok if e["n_strictly_between_0.10_and_0.25"]),
                "rows_at_exactly_0.10_range": [min(e["n_at_exactly_0.10"] for e in ok),
                                               max(e["n_at_exactly_0.10"] for e in ok)],
                "frac_real_rows_below_0.40_range": [min(e["frac_real_rows_below_0.40"] for e in ok
                                                        if e["frac_real_rows_below_0.40"] is not None),
                                                    max(e["frac_real_rows_below_0.40"] for e in ok
                                                        if e["frac_real_rows_below_0.40"] is not None)]},
            "c1_level_by_floor": levels,
            "removed_by_0.40_from_the_0.25_catalog": {"by_image": removed,
                                                      "pooled": 1 - tot[1] / tot[0]}}


def _matched_rows(le, tiles, floors, unmatched=None):
    D0 = le.STAGE4_MIN_DIAMETER_M
    rows, pairs = [], []
    for r in tiles.rows:
        H = r["H"]
        aH = shapely.area(H)
        bigH = eqd(aH) >= D0
        for f in floors:
            P = r["P"][r["score"] >= f]
            sc = r["score"][r["score"] >= f]
            ip, ih = match(P, H)
            aP = shapely.area(P)
            bigP = eqd(aP) >= D0
            mP = np.zeros(len(P), bool)
            mP[ip] = True
            partner_big = np.zeros(len(P), bool)
            partner_big[ip] = bigH[ih]
            recalled = np.zeros(len(H), bool)
            recalled[ih] = bigP[ip]
            both = bigP[ip] & bigH[ih]
            rows.append(dict(obs=r["obs"], tile_id=r["tile_id"], split=r["split"], floor=f,
                             N_H=int(bigH.sum()), A_H=float(aH[bigH].sum()),
                             N_P=int(bigP.sum()), A_P=float(aP[bigP].sum()),
                             A_Pm=float(aP[ip][both].sum()), A_Hm=float(aH[ih][both].sum()),
                             A_cross=float(aP[bigP & mP & ~partner_big].sum()),
                             A_Pu=float(aP[bigP & ~mP].sum()), N_Pu=int((bigP & ~mP).sum()),
                             A_miss=float(aH[bigH & ~recalled].sum()),
                             N_miss=int((bigH & ~recalled).sum()),
                             **{f"N_H_bin{k}": int((bigH & (eqd(aH) >= lo) & (eqd(aH) < hi)).sum())
                                for k, (lo, hi) in enumerate(SIZE_BINS)},
                             **{f"N_rec_bin{k}": int((bigH & recalled & (eqd(aH) >= lo)
                                                      & (eqd(aH) < hi)).sum())
                                for k, (lo, hi) in enumerate(SIZE_BINS)}))
            if len(ip):
                pairs.append(pd.DataFrame(dict(obs=r["obs"], split=r["split"], floor=f,
                                               a_p=aP[ip][both], a_h=aH[ih][both],
                                               score=sc[ip][both])))
            if unmatched is not None:
                unmatched.append(pd.DataFrame(dict(obs=r["obs"], floor=f,
                                                   d_m=eqd(aP[bigP & ~mP]))))
    return pd.DataFrame(rows), (pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame())


def _agg(m):
    s = m.sum(numeric_only=True)
    AH = s.A_H
    out = {"n_human": int(s.N_H), "count_recall": float(1 - s.N_miss / s.N_H),
           "area_recall": float(1 - s.A_miss / AH),
           "area_ratio_sum_of_polygons": float(s.A_P / AH),
           "decomposition_as_fraction_of_human_area": {
               "missed": float(-s.A_miss / AH), "unmatched_detections": float(s.A_Pu / AH),
               "matched_mask_inflation": float((s.A_Pm - s.A_Hm) / AH),
               "floor_crossers": float(s.A_cross / AH)},
           "recall_by_human_size_m": {}}
    for k, (lo, hi) in enumerate(SIZE_BINS):
        n = s[f"N_H_bin{k}"]
        out["recall_by_human_size_m"][f"{lo:g}-{hi:g}"] = (float(s[f"N_rec_bin{k}"] / n) if n
                                                           else None)
    return out


def sec_matched(le, tiles):
    unmatched = []
    m, pairs = _matched_rows(le, tiles, (0.10, 0.25, 0.40), unmatched)
    um = pd.concat(unmatched, ignore_index=True)
    um = um[um.floor == 0.10].assign(d_px=lambda x: x.d_m / x.obs.map(RES))
    out = {"method": "optimal one-to-one IoU >= 0.3; both sides >= 1.4105 m; polygons with centroid "
                     "in the ROM tile; areas are sums of polygon areas", "by_floor": {}}
    for f in (0.10, 0.25, 0.40):
        mf = m[m.floor == f]
        out["by_floor"][f"{f:.2f}"] = {
            "held_out": _agg(mf[mf.split.isin(le.HELD_OUT)]), "all_tiles": _agg(mf),
            "area_recall_by_image_held_out": {o: _agg(g)["area_recall"]
                                              for o, g in mf[mf.split.isin(le.HELD_OUT)].groupby("obs")},
            "area_recall_by_image_all_tiles": {o: _agg(g)["area_recall"] for o, g in mf.groupby("obs")},
            "count_recall_by_image_all_tiles": {o: _agg(g)["count_recall"] for o, g in mf.groupby("obs")}}
    p = pairs[pairs.floor == 0.10]
    out["matched_mask_area_ratio_c1_floor_0.10"] = {
        "held_out": {"median": float(np.median((p.a_p / p.a_h)[p.split.isin(le.HELD_OUT)])),
                     "area_weighted": float(p[p.split.isin(le.HELD_OUT)].a_p.sum()
                                            / p[p.split.isin(le.HELD_OUT)].a_h.sum())},
        "all_tiles": {"median": float(np.median(p.a_p / p.a_h)),
                      "area_weighted": float(p.a_p.sum() / p.a_h.sum())},
        "median_by_image_all_tiles": {o: float(np.median(g.a_p / g.a_h)) for o, g in p.groupby("obs")}}
    out["unmatched_detections_at_0.10_all_tiles"] = {
        o: {"n": int(len(g)), "share_6_to_10_px": float(((g.d_px >= 6) & (g.d_px < 10)).mean()),
            "median_px": float(g.d_px.median())} for o, g in um.groupby("obs")}
    return out, pairs


def sec_scene(le, cells, pairs, tiles):
    d = cells[cells.cell_size_m == 40.0]

    def level_by_image(cid, f, sel=lambda x: x):
        x = sel(d[(d.candidate == cid) & (d.floor == f)])
        return {o: le.error_metrics(g)["pooled_ratio"] for o, g in x.groupby("obs_id")}

    levels = {f"C1@{f:.2f}": level_by_image("C1", f) for f in (0.10, 0.25, 0.40)}
    levels["C2:256pref_engulf@0.40"] = level_by_image("C2:256pref_engulf", 0.40)
    levels["C1GS@0.40"] = level_by_image("C1GS", 0.40)
    logsd = {k: float(np.std(np.log(list(v.values())))) for k, v in levels.items()}
    # stratified tile bootstrap (tiles resampled within each image) of the change in log-SD 0.10 -> 0.40
    rng = np.random.default_rng(cs.SEED)
    per_img = {}
    for o, g in d[(d.candidate == "C1") & d.floor.isin((0.10, 0.40))].groupby("obs_id"):
        t = g.groupby(["tile_id", "floor"])[["fa_pred", "fa_true"]].sum().unstack("floor")
        per_img[o] = t
    deltas = []
    for _ in range(cs.N_BOOT):
        lv = {0.10: [], 0.40: []}
        for o, t in per_img.items():
            pick = rng.integers(0, len(t), len(t))
            tt = t.iloc[pick]
            for f in (0.10, 0.40):
                lv[f].append(tt[("fa_pred", f)].sum() / tt[("fa_true", f)].sum())
        deltas.append(np.std(np.log(lv[0.40])) - np.std(np.log(lv[0.10])))
    # exact-zero labels: cells with mapped boulders whose label is exactly zero
    zero = {}
    for f in (0.10, 0.40):
        x = d[(d.candidate == "C1") & (d.floor == f) & (d.fa_true > 0)]
        hx = held(x, le)
        zero[f"{f:.2f}"] = {"held_out": [int((hx.fa_pred == 0).sum()), int(len(hx))],
                            "all_tiles": [int((x.fa_pred == 0).sum()), int(len(x))],
                            "held_out_by_image": {o: int((g.fa_pred == 0).sum())
                                                  for o, g in hx.groupby("obs_id")}}
    # matched boulders' score per image (C1 at 0.10, both >= 1.41 m), and the share 0.40 keeps
    p = pairs[pairs.floor == 0.10]
    scores = {o: {"median_score": float(g.score.median()), "share_kept_at_0.40":
                  float((g.score >= 0.40).mean())} for o, g in p.groupby("obs")}
    # the truth-free diagnostic: share of >= 1.41 m detections a 0.40 floor keeps relative to 0.10
    kept = {}
    for r in tiles.rows:
        big = eqd(shapely.area(r["P"])) >= le.STAGE4_MIN_DIAMETER_M
        a, b = kept.get(r["obs"], (0, 0))
        kept[r["obs"]] = (a + int((big & (r["score"] >= 0.10)).sum()),
                          b + int((big & (r["score"] >= 0.40)).sum()))
    return {"level_by_image_all_tiles": levels, "log_sd_across_images": logsd,
            "log_sd_change_C1_0.10_to_0.40": {"point": logsd["C1@0.40"] - logsd["C1@0.10"],
                                              "ci95_tiles_within_images": ci(np.array(deltas))},
            "exact_zero_label_cells_with_mapped_boulders": zero,
            "matched_score_by_image_all_tiles": scores,
            "kept_share_0.40_vs_0.10_by_image_all_tiles": {o: b / a for o, (a, b) in kept.items()}}


def sec_plateau(le, cells):
    d = held(cells[(cells.cell_size_m == 40.0) & (cells.candidate == "C1")], le)
    from scipy.stats import spearmanr
    per = {}
    for f in cs.FLOORS:
        x = d[d.floor == f]
        per[f"{f:.2f}"] = {"rmse": rmse(x),
                           "rmse_log_fa_plus_1e-4": float(np.sqrt(((np.log(x.fa_pred + 1e-4)
                                                                   - np.log(x.fa_true + 1e-4)) ** 2)
                                                                  .mean())),
                           "spearman": float(spearmanr(x.fa_true, x.fa_pred).statistic)}
    index = tiles_of(d)
    w = draws(len(index))
    boots = {f: boot_rmse(tile_parts(d[d.floor == f], index), w) for f in cs.FLOORS}
    best = np.array(cs.FLOORS)[np.argmin(np.vstack([boots[f] for f in cs.FLOORS]), axis=0)]
    return {"held_out_40m_by_floor": per,
            "rmse_0.40_over_0.50_ci95": ci(boots[0.40] / boots[0.50]),
            "bootstrap_best_floor_frequency": {f"{f:.2f}": float((best == f).mean())
                                               for f in cs.FLOORS}}


def sec_concentration(le, cells, floors):
    d = held(cells[cells.cell_size_m == 40.0], le)
    out = {"held_out_cells_by_image": {o: int(len(g)) for o, g in
                                       d[(d.candidate == "C1") & (d.floor == 0.40)].groupby("obs_id")}}
    for cid, f in floors.items():
        x = d[(d.candidate == cid) & (d.floor == f)]
        se = (x.fa_pred - x.fa_true) ** 2
        by_tile = se.groupby([x.obs_id, x.tile_id]).sum() / se.sum()
        out[cid] = {"floor": f, "sse_share_by_image": (se.groupby(x.obs_id).sum() / se.sum()).to_dict(),
                    "largest_tile_share": float(by_tile.max()),
                    "largest_tile": "_t".join(map(str, by_tile.idxmax())),
                    "effective_n_tiles": float(1.0 / (by_tile ** 2).sum())}
    return out


def sec_res025(le, cells, floors):
    d = held(cells[(cells.cell_size_m == 40.0) & cells.obs_id.isin(RES025)], le)
    out = {"n_tiles": int(d[["obs_id", "tile_id"]].drop_duplicates().shape[0]),
           "note": "4 tiles: too few for an interval (label_error's own <= 4-group flag)",
           "at_pooled_primary_floors": {}, "at_own_best_floor_on_these_tiles": {}}
    for cid, f in floors.items():
        out["at_pooled_primary_floors"][cid] = {"floor": f, "rmse": rmse(d[(d.candidate == cid)
                                                                           & (d.floor == f)])}
        own = {fl: rmse(d[(d.candidate == cid) & (d.floor == fl)]) for fl in cs.FLOORS}
        bf = min(own, key=own.get)
        out["at_own_best_floor_on_these_tiles"][cid] = {"floor": bf, "rmse": own[bf]}
    return out


def sec_fine_grid(le, cells):
    d = cells[cells.cell_size_m == 40.0]
    hd = held(d, le)
    index = tiles_of(hd)
    w = draws(len(index))
    per = {cid: {f"{f:.2f}": rmse(hd[(hd.candidate == cid) & (hd.floor == f)]) for f in FINE}
           for cid in ("C1", "C1GS")}
    best = {cid: min(FINE, key=lambda f: per[cid][f"{f:.2f}"]) for cid in per}
    pa = {cid: tile_parts(hd[(hd.candidate == cid) & (hd.floor == best[cid])], index) for cid in per}
    ratio = boot_rmse(pa["C1"], w) / boot_rmse(pa["C1GS"], w)
    wins = int(np.sum(pa["C1GS"][0] / pa["C1GS"][1] < pa["C1"][0] / pa["C1"][1]))

    def out_of_sample(cid, choose_on):
        f = min(cs.FLOORS, key=lambda fl: rmse(choose_on(d[(d.candidate == cid) & (d.floor == fl)])))
        return f, rmse(hd[(hd.candidate == cid) & (hd.floor == f)])

    train = {cid: out_of_sample(cid, lambda x: x[x.split == "train"]) for cid in per}
    loio = {}
    for cid in per:
        parts = []
        for o in sorted(hd.obs_id.unique()):
            others = hd[(hd.candidate == cid) & (hd.obs_id != o)]
            f = min(cs.FLOORS, key=lambda fl: rmse(others[others.floor == fl]))
            parts.append(hd[(hd.candidate == cid) & (hd.obs_id == o) & (hd.floor == f)].assign(fl=f))
        allp = pd.concat(parts)
        loio[cid] = {"rmse": rmse(allp), "floors_by_held_out_image":
                     {o: float(g.fl.iloc[0]) for o, g in allp.groupby("obs_id")}}
    return {"held_out_rmse_on_0.05_grid": per, "best_floor": best,
            "c1_over_c1gs_at_their_best_ci95": ci(ratio),
            "tiles_where_c1gs_lower": [wins, len(index)],
            "floor_chosen_on_training_tiles": {cid: {"floor": f, "held_out_rmse": r}
                                               for cid, (f, r) in train.items()},
            "floor_chosen_leave_one_image_out": loio}


def sec_selection(le, cells):
    """Selection optimism: each candidate's floor (registered grid) chosen with one held-out tile, or
    one image, left out, then scored on what was left out. Optimism = nested RMSE / in-sample - 1."""
    d = held(cells[(cells.cell_size_m == 40.0) & cells.floor.isin(cs.FLOORS)], le)
    out = {}
    for cid, g in d.groupby("candidate"):
        g = g.assign(se=(g.fa_pred - g.fa_true) ** 2)
        sse = g.groupby(["floor", "obs_id", "tile_id"]).se.agg(["sum", "size"])
        floors = sorted(g.floor.unique())
        tiles = sorted(set(map(tuple, g[["obs_id", "tile_id"]].drop_duplicates().to_numpy())))

        def part(f, sel):
            s = sse.loc[f]
            s = s[[t in sel for t in s.index]]
            return s["sum"].sum(), s["size"].sum()

        def pooled(f, sel):
            a, n = part(f, sel)
            return np.sqrt(a / n)
        everything = set(tiles)
        f_in = min(floors, key=lambda f: pooled(f, everything))
        row = {"in_sample_floor": f_in, "in_sample_rmse": pooled(f_in, everything)}
        images = sorted({t[0] for t in tiles})
        for name, groups in (("leave_one_tile_out", {f"{o}_t{t}": {(o, t)} for o, t in tiles}),
                             ("leave_one_image_out", {o: {t for t in tiles if t[0] == o} for o in images})):
            sse_tot, n_tot, chosen = 0.0, 0, {}
            for label, grp in groups.items():
                f = min(floors, key=lambda f: pooled(f, everything - grp))
                a, n = part(f, grp)
                sse_tot, n_tot, chosen[label] = sse_tot + a, n_tot + n, f
            nested = float(np.sqrt(sse_tot / n_tot))
            row[name] = {"rmse": nested, "optimism": nested / row["in_sample_rmse"] - 1,
                         "floors_chosen": sorted(set(chosen.values())), "by_fold": chosen}
        out[cid] = row
    return out


DEFLATIONS = (1.0, 1.3, 1.4, 1.7, 1.95)
FA_BINS = "0 | (0, 1e-3] | (1e-3, 1e-2] | (1e-2, 0.05] | > 0.05"


def _fa_bin(x):
    x = np.asarray(x, dtype=float)
    b = np.full(x.shape, 4)
    for edge, k in ((0.05, 3), (1e-2, 2), (1e-3, 1)):
        b[x <= edge] = k
    b[x == 0] = 0
    return b


def sec_reweighted(le, cells):
    """The registered rule (held-out, 40 m, each candidate at its own best floor, the same tile draws)
    with each cell weighted so that ROM's truth has v2's 40 m label distribution (audit D2). v2's
    labels are detector-made and over-state area, so they are also deflated by k before binning."""
    v2 = pd.concat([pd.read_parquet(f, columns=["tile_size_px", "fractional_area"])
                    for f in sorted(V2_LABELS.glob("*.parquet"))], ignore_index=True)
    v2 = v2[v2.tile_size_px == 8].fractional_area.dropna().to_numpy()
    d = held(cells[(cells.cell_size_m == 40.0) & cells.floor.isin(cs.FLOORS)], le)
    key = ["obs_id", "rom_index", "tile_id", "cell"]
    truth = d[(d.candidate == "C1") & (d.floor == 0.40)]
    rom_bin = _fa_bin(truth.fa_true)
    rom_share = np.bincount(rom_bin, minlength=5) / len(rom_bin)
    index = tiles_of(d)
    w_draw = draws(len(index))
    out = {"bins": FA_BINS, "rom_held_out_cells_by_bin": np.bincount(rom_bin, minlength=5).tolist(),
           "by_deflation": {}}
    for k in DEFLATIONS:
        v2_share = np.bincount(_fa_bin(v2 / k), minlength=5) / len(v2)
        w = pd.Series((v2_share / rom_share)[rom_bin],
                      index=pd.MultiIndex.from_frame(truth[key])).rename("w")
        x = d.join(w, on=key)
        x = x.assign(wse=x.w * (x.fa_pred - x.fa_true) ** 2)
        per = {}
        for cid, g in x.groupby("candidate"):
            f, gf = min(((f, gf) for f, gf in g.groupby("floor")),
                        key=lambda t: t[1].wse.sum() / t[1].w.sum())
            t = gf.groupby(["obs_id", "tile_id"])
            per[cid] = {"floor": float(f), "rmse": float(np.sqrt(gf.wse.sum() / gf.w.sum())),
                        "parts": (t.wse.sum().reindex(index).to_numpy(), t.w.sum().reindex(index).to_numpy())}
        row = {"v2_share_by_bin": v2_share.tolist(), "weight_by_bin": (v2_share / rom_share).tolist(),
               "kish_ess": float(w.sum() ** 2 / (w ** 2).sum())}
        for name, ids in (("with_C1GS", sorted(per)), ("registered", sorted(c for c in per if c != "C1GS"))):
            best = min(ids, key=lambda c: per[c]["rmse"])
            rb = {c: ci(boot_rmse(per[c]["parts"], w_draw) / boot_rmse(per[best]["parts"], w_draw))
                  for c in ids}
            row[name] = {"best": best, "rmse": {c: per[c]["rmse"] for c in ids},
                         "floor": {c: per[c]["floor"] for c in ids}, "ratio_to_best_ci95": rb,
                         "tied": [c for c in ids if rb[c][0] <= 1.0 <= rb[c][1]],
                         "C1_tied": bool(rb["C1"][0] <= 1.0 <= rb["C1"][1])}
        if "C1GS" in per:
            row["c1_over_c1gs_ci95"] = ci(boot_rmse(per["C1"]["parts"], w_draw)
                                          / boot_rmse(per["C1GS"]["parts"], w_draw))
        out["by_deflation"][f"{k:g}"] = row
    return out


def sec_power(le, cells, floors):
    d = held(cells[cells.cell_size_m == 40.0], le)
    index = tiles_of(d)
    w = draws(len(index))
    parts = {cid: tile_parts(d[(d.candidate == cid) & (d.floor == f)], index) for cid, f in floors.items()}
    out = {}
    for other in ("C3", "C1GS"):
        lr = np.log(boot_rmse(parts["C1"], w) / boot_rmse(parts[other], w))
        out[f"C1_vs_{other}"] = {"sd_log_ratio": float(lr.std()),
                                 "min_detectable_ratio_80pct_power": float(np.exp(2.8 * lr.std()))}
    # image-aware: resample images, then tiles within each drawn image
    rng = np.random.default_rng(cs.SEED)
    imgs = sorted(d.obs_id.unique())
    by_img = {o: [i for i, t in enumerate(index) if t[0] == o] for o in imgs}
    two = {k: [] for k in ("C3", "C1GS")}
    for _ in range(cs.N_BOOT):
        cnt = np.zeros(len(index))
        for o in rng.choice(imgs, len(imgs)):
            ids = by_img[o]
            for j in rng.choice(ids, len(ids)):
                cnt[j] += 1
        for k in two:
            num = np.sqrt((cnt @ parts["C1"][0]) / (cnt @ parts["C1"][1]))
            den = np.sqrt((cnt @ parts[k][0]) / (cnt @ parts[k][1]))
            two[k].append(num / den)
    for k, v in two.items():
        out[f"C1_vs_{k}"]["ci95_images_then_tiles"] = ci(np.array(v))
    return out


def _parent160(le, keep):
    """For each registered tile, the 160 m cell each 40 m cell nests in (-1 if none). Both grids are
    label_error.cells_in_rom's, anchored at the tile's north-west corner, so a 160 m cell is exactly
    4 x 4 of the 40 m cells."""
    splits, out = le.split_map(), {}
    for s in le.images().values():
        rom = le.load_rom(s, splits)
        for _, t in rom.iterrows():
            key = (s.obs_id, int(t.rom_index), int(t.tile_id))
            if key not in keep:
                continue
            tile = shapely.make_valid(t.geometry)
            c40 = np.asarray(le.cells_in_rom(tile, rom.crs, 40.0), dtype=object)
            c160 = np.asarray(le.cells_in_rom(tile, rom.crs, 160.0), dtype=object)
            par = np.full(len(c40), -1)
            if len(c40) and len(c160):
                q, p = shapely.STRtree(c160).query(shapely.centroid(c40), predicate="within")
                par[q] = p
            out[key] = par
    return out


def _same_footprint(le, cells, keep):
    """C1 at 0.40: RMSE at 160 m against RMSE of the 40 m cells inside those same 160 m cells. If the
    16 children's errors were independent, the ratio would be 1/4."""
    par = _parent160(le, keep)
    d40 = cells[(cells.cell_size_m == 40.0) & (cells.candidate == "C1") & (cells.floor == 0.40)]
    d160 = cells[(cells.cell_size_m == 160.0) & (cells.candidate == "C1") & (cells.floor == 0.40)]
    key = ["obs_id", "rom_index", "tile_id"]
    d40 = d40.assign(parent=[par[(o, r, t)][k] for o, r, t, k in
                             d40[key + ["cell"]].itertuples(index=False, name=None)])
    out = {}
    for tag, sel in (("held_out", lambda x: held(x, le)), ("all_tiles", lambda x: x)):
        a, b = sel(d40[d40.parent >= 0]), sel(d160)
        m = a.groupby(key + ["parent"])[["fa_true", "fa_pred"]].mean()
        bb = b.set_index(key + ["cell"])[["fa_true", "fa_pred"]]
        bb.index.names = m.index.names
        j = m.join(bb, lsuffix="_40", rsuffix="_160", how="inner")
        dev = np.abs(j[["fa_true_40", "fa_pred_40"]].to_numpy() - j[["fa_true_160", "fa_pred_160"]].to_numpy())
        out[tag] = {"n_160m_cells": int(len(b)), "n_child_40m_cells": int(len(a)),
                    "n_parents_matched": int(len(j)), "rmse_40m_same_footprint": rmse(a),
                    "rmse_160m": rmse(b), "ratio_160_over_40": rmse(b) / rmse(a),
                    "ratio_if_independent": 0.25,
                    "max_abs_dev_parent_vs_mean_of_children": float(dev.max()) if dev.size else None}
    return out


def _v2_label_cells():
    """hirise2ctx's v2 training labels (read-only): how much of each cell size is empty or near-empty,
    to compare with ROM's truth. S = 8 / 16 / 32 CTX px is 40 / 80 / 160 m."""
    fs = sorted(V2_LABELS.glob("*.parquet"))
    if not fs:
        return None
    d = pd.concat([pd.read_parquet(f, columns=["tile_size_px", "fractional_area"]) for f in fs],
                  ignore_index=True)
    out = {"n_images": len(fs)}
    for s, g in d.groupby("tile_size_px"):
        x = g.fractional_area.dropna()
        out[f"S{int(s)}"] = {"n_cells": int(len(x)), "frac_zero": float((x == 0).mean()),
                             "frac_in_0_to_1e-3": float(((x > 0) & (x <= 1e-3)).mean()),
                             "median_fa": float(x.median())}
    return out


def sec_160m(le, cells, floors, keep):
    d = cells[cells.cell_size_m == 160.0]
    out = {"same_footprint_C1_0.40": _same_footprint(le, cells, keep),
           "v2_label_cells_by_size": _v2_label_cells()}
    for tag, sel in (("held_out", lambda x: held(x, le)), ("all_tiles", lambda x: x)):
        x = sel(d)
        index = tiles_of(x)
        w = draws(len(index))
        parts, row = {}, {}
        for cid, f in floors.items():
            c = x[(x.candidate == cid) & (x.floor == f)]
            parts[cid] = tile_parts(c, index)
            row[cid] = {"floor": f, "rmse": rmse(c), "area_ratio": le.error_metrics(c)["pooled_ratio"],
                        "n_cells": int(len(c)), "n_tiles": int(c[["obs_id", "tile_id"]]
                                                               .drop_duplicates().shape[0])}
        for cid in floors:
            if cid != "C1":
                row[cid]["over_C1_ci95"] = ci(boot_rmse(parts[cid], w) / boot_rmse(parts["C1"], w))
        c1 = x[(x.candidate == "C1") & (x.floor == 0.40)]
        row["truth"] = {"empty_cells": int((c1.fa_true == 0).sum()), "n_cells": int(len(c1)),
                        "median_fa_true": float(c1.fa_true.median()),
                        "images": sorted(c1.obs_id.unique())}
        out[tag] = row
    e40 = held(cells[(cells.cell_size_m == 40.0) & (cells.candidate == "C1") & (cells.floor == 0.40)], le)
    out["held_out_40m_empty_cells"] = [int((e40.fa_true == 0).sum()), int(len(e40))]
    return out


def sec_extras(le, cells, tiles, keep):
    """Per-image compensation test, size-floor reach, the ROM images' viewing geometry, the
    C1-vs-C1GS tie at each size floor, and C1GS's render cost."""
    from YOLOv8BeyondEarth import pds_prep
    d = cells[(cells.cell_size_m == 40.0) & (cells.candidate == "C1") & (cells.floor == 0.40)]
    flag = {}
    for tag, x in (("held_out", held(d, le)), ("all_tiles", d)):
        rows = {}
        for o, g in x.groupby("obs_id"):
            cr = float(g.n_pred.sum() / g.n_true.sum())
            ar = le.error_metrics(g)["pooled_ratio"]
            ma = ar / cr
            rows[o] = {"area_ratio": ar, "count_ratio": cr, "mean_area_ratio": ma,
                       "flag": bool(abs(np.log(cr)) > np.log(1.2) and abs(np.log(ma)) > np.log(1.2)
                                    and np.sign(np.log(cr)) != np.sign(np.log(ma)))}
        flag[tag] = rows
    reach = {}
    for r in tiles.rows:
        dm = eqd(shapely.area(r["P"]))
        acc = reach.setdefault(r["obs"], {"n": 0, "below_1.0": 0, "below_1.4105": 0, "below_2.0": 0})
        acc["n"] += int(dm.size)
        for k, v in (("below_1.0", 1.0), ("below_1.4105", 1.4105), ("below_2.0", 2.0)):
            acc[k] += int((dm < v).sum())
    reach = {o: {k: (v / a["n"] if k != "n" else v) for k, v in a.items()} for o, a in reach.items()}
    geom = {}
    for obs in RES:
        lbl = next(iter(sorted(glob.glob(str(REPO / "data" / "**" / f"{obs}_RED.LBL"), recursive=True))),
                   None)
        if lbl is None:                     # labels are small; fetch the one not yet on disk
            lbl_url, _ = pds_prep.rdr_urls(obs)
            dest = REPO / "data" / "pds_scenes" / obs
            dest.mkdir(parents=True, exist_ok=True)
            pds_prep.fetch(lbl_url, dest / f"{obs}_RED.LBL")
            lbl = str(dest / f"{obs}_RED.LBL")
        f = pds_prep.read_label(lbl)
        s = pds_prep.label_summary(f)
        geom[obs] = {"incidence_angle": float(s["incidence_angle"]),
                     "emission_angle": float(pds_prep._lookup(f, "EMISSION_ANGLE")),
                     "map_scale": float(s["map_scale"])}
    splits = le.split_map()
    tie = {}
    for size in (1.0, 1.4105, 2.0, 3.0):
        per = {}
        for cid, sub in (("C1", ""), ("C1GS", "config1gs")):
            frames = {f: held(cs.paired(le, splits, cs.ROM_DIR, sub, f, 40.0, keep,
                                        min_diameter_m=size), le) for f in cs.FLOORS}
            bf = min(cs.FLOORS, key=lambda f: rmse(frames[f]))
            per[cid] = (bf, frames[bf])
        index = tiles_of(per["C1"][1])
        w = draws(len(index))
        r = boot_rmse(tile_parts(per["C1"][1], index), w) / boot_rmse(tile_parts(per["C1GS"][1], index), w)
        tie[f"{size:g}"] = {"best_floor": {c: v[0] for c, v in per.items()},
                            "rmse": {c: rmse(v[1]) for c, v in per.items()}, "c1_over_c1gs_ci95": ci(r)}
    cost = None
    summ = cs.ROM_DIR / "render_rom_configs_summary.json"
    if summ.exists():
        t = [x for x in json.loads(summ.read_text())["tiles"] if "seconds_1gs" in x]
        if t:
            secs, n = sum(x["seconds_1gs"] for x in t), sum(x.get("n_config1gs", 0) for x in t)
            cost = {"n_tiles": len(t), "seconds_total": secs, "rows_total": n,
                    "ms_per_output_row": 1000 * secs / n if n else None,
                    "note": "one laptop run (RTX 5070, AC); each tile in its own process, so model "
                            "loading is included: an upper bound on the SAM2 + YOLO cost"}
    import pyogrio
    src = {"all_rows": [0, 0], "label_population": [0, 0]}      # [rows, rows that kept YOLO's mask]
    for f in sorted(glob.glob(str(cs.ROM_DIR / "*" / "config1gs" / "*.gpkg"))):
        g = pyogrio.read_dataframe(f, columns=["score", "mask_src"])
        yolo = (g.mask_src == "yolo").to_numpy()
        lab = ((g.score >= 0.40).to_numpy()
               & (eqd(shapely.area(np.asarray(g.geometry.values))) >= le.STAGE4_MIN_DIAMETER_M))
        for k, sel in (("all_rows", np.ones(len(g), bool)), ("label_population", lab)):
            src[k][0] += int(sel.sum())
            src[k][1] += int((sel & yolo).sum())
    mask_source = {k: {"n_rows": n, "n_kept_yolo_mask": y, "frac": y / n if n else None}
                   for k, (n, y) in src.items()}
    mask_source["note"] = ("C1GS rows whose SAM2 re-mask failed the guard or was degenerate keep YOLO's "
                           "mask (mask_src == 'yolo'); label population = score >= 0.40, >= 1.4105 m")
    return {"compensation_test_by_image_at_0.40": flag, "size_floor_reach_by_image": reach,
            "rom_viewing_geometry": geom, "c1_vs_c1gs_by_size_floor": tie, "c1gs_render_cost": cost,
            "c1gs_mask_source": mask_source}


def sec_mask_area(le, keep):
    """SAM2/YOLO polygon area for the same detection, matched by centroid within 1 m, over the label's
    own population (YOLO polygon >= 1.4105 m, score >= 0.40, centroid in the ROM tile)."""
    import geopandas as gpd
    from YOLOv8BeyondEarth.matching import unique_centroid_match
    splits = le.split_map()
    out = {}
    for name, sub in (("C1G (window re-masker)", "config1g"), ("C1GS (in-slice, validated)", "config1gs")):
        ys, ss = [], []
        for s in le.images().values():
            rom = le.load_rom(s, splits)
            for _, t in rom.iterrows():
                if (s.obs_id, int(t.rom_index), int(t.tile_id)) not in keep:
                    continue
                y = le.load_predictions(cs._spec(s, cs.ROM_DIR, ""), int(t.rom_index), int(t.tile_id),
                                        rom.crs)
                o = le.load_predictions(cs._spec(s, cs.ROM_DIR, sub), int(t.rom_index),
                                        int(t.tile_id), rom.crs)
                tile = shapely.make_valid(t.geometry)
                y = y[shapely.within(shapely.centroid(y.geometry.values), tile)]
                ia, ib = unique_centroid_match(np.c_[y.geometry.centroid.x, y.geometry.centroid.y],
                                               np.c_[o.geometry.centroid.x, o.geometry.centroid.y], 1.0)
                ay, ao = y.geometry.area.to_numpy()[ia], o.geometry.area.to_numpy()[ib]
                sel = (eqd(ay) >= le.STAGE4_MIN_DIAMETER_M) & (y.score.to_numpy()[ia] >= 0.40)
                ys.append(ay[sel])
                ss.append(ao[sel])
        ay, ao = np.concatenate(ys), np.concatenate(ss)
        out[name] = {"n": int(ay.size), "area_weighted": float(ao.sum() / ay.sum()),
                     "median_per_detection": float(np.median(ao / ay))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    le, sha = cs.load_label_error()
    splits = le.split_map()
    reg = pd.read_parquet(cs.STUDY_DIR / "paired_cells.parquet")
    keep = set(map(tuple, reg[["obs_id", "rom_index", "tile_id"]].drop_duplicates()
                   .itertuples(index=False, name=None)))
    registered = json.loads(cs.OUT_JSON.read_text())
    floors = dict(registered["primary"]["floor"])
    floors = {k: floors[k] for k in ("C1", "C2:256pref_engulf", "C1G", "C3")}
    floors["C1GS"] = 0.40
    print("scoring C1GS and the extra floors through the harness ...", flush=True)
    extra = []
    for f in sorted(set(cs.FLOORS) | set(FINE)):
        for cell in (40.0, 160.0):
            extra.append(cs.paired(le, splits, cs.ROM_DIR, "config1gs", f, cell, keep)
                         .assign(candidate="C1GS", family="C1GS", floor=f))
    for f in (0.35, 0.45):
        extra.append(cs.paired(le, splits, cs.ROM_DIR, "", f, 40.0, keep)
                     .assign(candidate="C1", family="C1", floor=f))
    cells = pd.concat([reg] + extra, ignore_index=True)
    print("loading tiles and matching ...", flush=True)
    tiles = Tiles(le, splits, keep)
    out = {"what": "corrected numbers for the configuration study's write-up and the hirise2ctx brief "
                   "(audit: docs/configuration_study_audit_2026-10-16.md)",
           "label_error_sha256": sha, "n_tiles": len(keep), "code": cs_code_identity(),
           "validate": sec_validate(le)}
    out["matched"], pairs = sec_matched(le, tiles)
    out["v2_basis"] = sec_v2_basis(le, cells, tiles)
    out["scene"] = sec_scene(le, cells, pairs, tiles)
    out["plateau"] = sec_plateau(le, cells)
    out["concentration"] = sec_concentration(le, cells, floors)
    out["res025"] = sec_res025(le, cells, floors)
    out["fine_grid"] = sec_fine_grid(le, cells)
    out["selection"] = sec_selection(le, cells)
    out["reweighted"] = sec_reweighted(le, cells)
    out["power"] = sec_power(le, cells, floors)
    out["cells_160m"] = sec_160m(le, cells, floors, keep)
    out["mask_area"] = sec_mask_area(le, keep)
    out["extras"] = sec_extras(le, cells, tiles, keep)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {args.out}")


def cs_code_identity():
    from YOLOv8BeyondEarth import presets
    return presets.code_identity()


if __name__ == "__main__":
    main()
