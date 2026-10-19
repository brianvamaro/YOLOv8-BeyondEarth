"""Configuration 2 (its production rule) against configuration 1 at one fixed score floor, for D9.

hirise2ctx's Stage 1 note (hirise2ctx-pub/docs/bouldernet_stage1_results_2026-10-18.md, section 6,
question 4) asks for this before D9 is ruled. D36 fixed the label's floor at 0.25, and the configuration
study (docs/configuration_study_plan.md) recorded configuration 2's per-image levels only at its own best
floor, 0.40.

Everything comes from the study's per-cell rows (data/config_study/paired_cells.parquet: notebook 35's
label through hirise2ctx's label_error.py, imported read-only). The one exception is the matched
decomposition, which reads the catalogs those rows were scored from. No GPU, nothing re-rendered.

- C1 is the 256 arm (configuration 1). C2 is {512, 256} fused under the production rule,
  ``C2:256pref_engulf`` (bands with engulf 0.7; == config2/).
- The floor is applied to each delivered conf-0.10 catalog (score >= floor), as Stage 4 applies it. For
  C2 that is after fusion.
- Ground: the 48 ROM tiles every candidate has. Held-out = validation + test (10 tiles). Cells are
  notebook 35's complete cells. At 160 m only the two 0.5-0.6 m/px images have any.
- The level is label / human: sum(fa_pred) / sum(fa_true). Count = sum(n_pred) / sum(n_true), where n is
  the number of polygons touching each cell (the study's split, so a polygon on a cell edge counts in
  each cell it touches). Mean area = level / count.
- Effective tiles = 1 / sum(s_t^2), where s_t is tile t's share of what the statistic sums: human area
  (level), human count (count), cells (Spearman), squared error (RMSE), and configuration 1's label area
  (the C2 / C1 level ratio). For RMSE this is the study's measure (3.6 for C1 at 0.40).
- Intervals: the study's paired tile bootstrap (2,000 multinomial draws, seed 0, the same draws for both
  configurations). Per image, the same within the image, and only where it has at least 5 tiles. With
  fewer, the per-tile values are given instead.

    conda run -n bouldernet python scripts/review/d9_config2_at_floor.py --floors 0.25 0.40 --out data/review/d9_config2_at_025.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "review"))
sys.path.insert(0, str(REPO / "src"))
import config_study as cs                                                   # noqa: E402
import config_study_corrections as csc                                      # noqa: E402

C1, C2 = "C1", "C2:256pref_engulf"
ROOTS = {C1: (cs.ROM_DIR, ""), C2: (cs.STUDY_DIR / "c2_256pref_engulf", "")}
MIN_TILES_FOR_CI = 5
CORRECTIONS_JSON = REPO / "data" / "review" / "config_study_corrections.json"


# --- per-cell statistics --------------------------------------------------------------------------

def tile_sums(d, index):
    """Per-tile sums the pooled statistics are built from, in the order of ``index``."""
    g = d.assign(se=(d.fa_pred - d.fa_true) ** 2, one=1).groupby(["obs_id", "tile_id"])
    s = g[["fa_pred", "fa_true", "n_pred", "n_true", "se", "one"]].sum().reindex(index)
    return {k: s[k].to_numpy(dtype=float) for k in s.columns}


def eff(x):
    """1 / sum(share^2): the effective number of tiles carrying a sum."""
    x = np.asarray(x, dtype=float)
    return float(x.sum() ** 2 / (x ** 2).sum()) if (x ** 2).sum() > 0 else float("nan")


def stats(d, le, index):
    m = le.error_metrics(d)
    t = tile_sums(d, index)
    count = float(d.n_pred.sum() / d.n_true.sum()) if d.n_true.sum() else float("nan")
    return {"level": m["pooled_ratio"], "count_ratio": count,
            "mean_area_ratio": float(m["pooled_ratio"] / count) if count else float("nan"),
            "spearman": m["spearman"], "rmse": m["rmse"], "mean_fa_true": m["mean_fa_true"],
            "mean_fa_pred": m["mean_fa_pred"], "n_cells": m["n_cells"], "n_tiles": m["n_tiles"],
            "effective_tiles": {"level": eff(t["fa_true"]), "count": eff(t["n_true"]),
                                "spearman": eff(t["one"]), "rmse": eff(t["se"])}}


def ci(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if x.size else None


def boot_spearman(d, index, w):
    """Spearman of the cells of each resample (cells repeated by their tile's multinomial count)."""
    pos = pd.MultiIndex.from_frame(d[["obs_id", "tile_id"]]).map(
        {k: i for i, k in enumerate(index)}.get).to_numpy()
    a, b = d.fa_true.to_numpy(), d.fa_pred.to_numpy()
    out = np.empty(len(w))
    for k, row in enumerate(w):
        r = row[pos]
        out[k] = spearmanr(np.repeat(a, r), np.repeat(b, r)).statistic
    return out


def compare(d1, d2, le, index, rng_seed=cs.SEED, with_ci=True):
    """C2 against C1 on one set of cells: both configurations' statistics, and C2 / C1 with a paired
    tile bootstrap."""
    s1, s2 = stats(d1, le, index), stats(d2, le, index)
    t1, t2 = tile_sums(d1, index), tile_sums(d2, index)
    out = {C1: s1, C2: s2, "C2_over_C1": {
        "level": s2["level"] / s1["level"], "count": s2["count_ratio"] / s1["count_ratio"],
        "mean_area": s2["mean_area_ratio"] / s1["mean_area_ratio"], "rmse": s2["rmse"] / s1["rmse"],
        "spearman_difference": s2["spearman"] - s1["spearman"],
        "effective_tiles_of_level_ratio": eff(t1["fa_pred"])}}
    if not with_ci:
        return out
    w = np.random.default_rng(rng_seed).multinomial(len(index), np.full(len(index), 1.0 / len(index)),
                                                     size=cs.N_BOOT)
    lev = lambda t: (w @ t["fa_pred"]) / (w @ t["fa_true"])               # noqa: E731
    cnt = lambda t: (w @ t["n_pred"]) / (w @ t["n_true"])                 # noqa: E731
    rms = lambda t: np.sqrt((w @ t["se"]) / (w @ t["one"]))               # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        out["C2_over_C1_ci95"] = {
            "level": ci(lev(t2) / lev(t1)), "count": ci(cnt(t2) / cnt(t1)),
            "mean_area": ci((lev(t2) / cnt(t2)) / (lev(t1) / cnt(t1))), "rmse": ci(rms(t2) / rms(t1)),
            "spearman_difference": ci(boot_spearman(d2, index, w) - boot_spearman(d1, index, w))}
        out["level_ci95"] = {C1: ci(lev(t1)), C2: ci(lev(t2))}
    out["n_boot"], out["n_tiles_resampled"] = cs.N_BOOT, len(index)
    return out


def per_tile(d1, d2):
    rows = []
    for (o, t), g1 in d1.groupby(["obs_id", "tile_id"]):
        g2 = d2[(d2.obs_id == o) & (d2.tile_id == t)]
        rows.append({"tile_id": int(t), "split": g1.split.iloc[0], "n_cells": int(len(g1)),
                     "human_fa_sum": float(g1.fa_true.sum()),
                     "level": {C1: float(g1.fa_pred.sum() / g1.fa_true.sum()) if g1.fa_true.sum() else None,
                               C2: float(g2.fa_pred.sum() / g2.fa_true.sum()) if g2.fa_true.sum() else None},
                     "count_ratio": {C1: float(g1.n_pred.sum() / g1.n_true.sum()) if g1.n_true.sum() else None,
                                     C2: float(g2.n_pred.sum() / g2.n_true.sum()) if g2.n_true.sum() else None}})
    return rows


def log_sd(d1, d2, seed=cs.SEED):
    """The spread of the level across images (SD of log level, ddof 0, as the study), and C2 - C1 with
    tiles resampled within each image (the same picks for both)."""
    def table(d):
        return d.groupby(["obs_id", "tile_id"])[["fa_pred", "fa_true"]].sum()
    t1, t2 = table(d1), table(d2)
    imgs = sorted(t1.index.get_level_values(0).unique())
    lv = lambda t: [t.loc[o].fa_pred.sum() / t.loc[o].fa_true.sum() for o in imgs]   # noqa: E731
    point = {C1: float(np.std(np.log(lv(t1)))), C2: float(np.std(np.log(lv(t2))))}
    rng = np.random.default_rng(seed)
    diff = []
    for _ in range(cs.N_BOOT):
        a, b = [], []
        for o in imgs:
            x1, x2 = t1.loc[o], t2.loc[o].reindex(t1.loc[o].index)
            pick = rng.integers(0, len(x1), len(x1))
            a.append(x1.fa_pred.to_numpy()[pick].sum() / x1.fa_true.to_numpy()[pick].sum())
            b.append(x2.fa_pred.to_numpy()[pick].sum() / x2.fa_true.to_numpy()[pick].sum())
        diff.append(np.std(np.log(b)) - np.std(np.log(a)))
    l1, l2 = dict(zip(imgs, np.log(lv(t1)))), dict(zip(imgs, np.log(lv(t2))))
    leave_one_out = {o: {C1: float(np.std([v for k, v in l1.items() if k != o])),
                         C2: float(np.std([v for k, v in l2.items() if k != o]))} for o in imgs}
    return {"images": imgs, "point": point, "C2_minus_C1": point[C2] - point[C1],
            "C2_minus_C1_ci95_tiles_within_images": ci(diff),
            "leaving_one_image_out": leave_one_out}


def block(cells, le, floor, cell_m, subset):
    d = cells[(cells.cell_size_m == cell_m) & (cells.floor == floor)]
    if subset == "held_out":
        d = d[d.split.isin(le.HELD_OUT)]
    d1, d2 = d[d.candidate == C1], d[d.candidate == C2]
    index = csc.tiles_of(d1)
    assert index.equals(csc.tiles_of(d2)), "the two configurations must sit on the same tiles"
    assert len(d1) == len(d2), "the two configurations must have the same cells"
    out = {"subset": subset, "cell_m": cell_m, "floor": floor,
           "n": {"cells": int(len(d1)), "tiles": len(index), "images": int(d1.obs_id.nunique()),
                 "cells_by_image": {o: int(len(g)) for o, g in d1.groupby("obs_id")},
                 "tiles_by_image": {o: int(g.tile_id.nunique()) for o, g in d1.groupby("obs_id")},
                 "splits": {s: int(g.tile_id.nunique()) for s, g in
                            d1.drop_duplicates(["obs_id", "tile_id"]).groupby("split")}},
           "pooled": compare(d1, d2, le, index), "by_image": {}}
    for o in sorted(d1.obs_id.unique()):
        g1, g2 = d1[d1.obs_id == o], d2[d2.obs_id == o]
        idx = csc.tiles_of(g1)
        enough = len(idx) >= MIN_TILES_FOR_CI
        row = compare(g1, g2, le, idx, with_ci=enough)
        if not enough:
            row["note"] = f"{len(idx)} tiles: no interval; per-tile values below"
            row["per_tile"] = per_tile(g1, g2)
        out["by_image"][o] = row
    if out["n"]["images"] >= 3:
        out["log_sd_across_images"] = log_sd(d1, d2)
    else:
        out["log_sd_across_images"] = None
        out["note"] = (f"{out['n']['images']} images only: no 0.25 m/px image has a complete cell this "
                       "size (its ROM tiles are 125 m), so there is no spread across images")
    return out


# --- the matched decomposition (catalogs, not cells) -----------------------------------------------

class TilesAt:
    """``config_study_corrections.Tiles`` with the catalogs read from any root: humans and one
    configuration's predictions per registered ROM tile, centroid in the tile, scores kept."""

    def __init__(self, le, splits, keep, root, sub=""):
        self.rows = []
        for s in le.images().values():
            rom = le.load_rom(s, splits)
            ref = rom.crs
            hum, hok = csc.repair_aligned(le.load_human(s, ref).geometry.values)
            hum = hum[hok]
            hcent = shapely.centroid(hum)
            spec = cs._spec(s, root, sub)
            for _, t in rom.iterrows():
                if (s.obs_id, int(t.rom_index), int(t.tile_id)) not in keep:
                    continue
                tile = shapely.make_valid(t.geometry)
                g = le.load_predictions(spec, int(t.rom_index), int(t.tile_id), ref)
                P, pok = csc.repair_aligned(g.geometry.values)
                sc = g.score.to_numpy()[pok]
                P = P[pok]
                inside = shapely.within(shapely.centroid(P), tile)
                self.rows.append(dict(obs=s.obs_id, tile_id=int(t.tile_id), split=t.split,
                                      H=hum[shapely.within(hcent, tile)], P=P[inside],
                                      score=sc[inside]))


def matched(le, splits, keep, floor, corrections):
    out = {"method": "config_study_corrections' matching: optimal one-to-one IoU >= 0.3, both sides "
                     ">= 1.4105 m, polygons with centroid in the ROM tile; areas are sums of polygon "
                     "areas over the tile, not the label's per-cell unions; count_ratio_by_centroid "
                     "counts each polygon once (N_P / N_H)"}
    m = {}
    for cid, (root, sub) in ROOTS.items():
        rows, _ = csc._matched_rows(le, TilesAt(le, splits, keep, root, sub), (floor,))
        m[cid] = rows
    for tag, sel in (("held_out", lambda x: x[x.split.isin(le.HELD_OUT)]), ("all_tiles", lambda x: x)):
        blk = {}
        for cid, rows in m.items():
            x = sel(rows)
            a = csc._agg(x)
            s = x.sum(numeric_only=True)
            a["count_ratio_by_centroid"] = float(s.N_P / s.N_H)
            a["n_detections"] = int(s.N_P)
            a["n_unmatched_detections"] = int(s.N_Pu)
            a["by_image"] = {}
            for o, g in x.groupby("obs"):
                ag, sg = csc._agg(g), g.sum(numeric_only=True)
                a["by_image"][o] = {"count_recall": ag["count_recall"], "area_recall": ag["area_recall"],
                                    "area_ratio_sum_of_polygons": ag["area_ratio_sum_of_polygons"],
                                    "count_ratio_by_centroid": float(sg.N_P / sg.N_H),
                                    "n_human": ag["n_human"], "n_detections": int(sg.N_P),
                                    "decomposition_as_fraction_of_human_area":
                                        ag["decomposition_as_fraction_of_human_area"]}
            blk[cid] = a
        out[tag] = blk
    # configuration 1's rows must equal the corrections' own matching at this floor
    rec = corrections["matched"]["by_floor"].get(f"{floor:.2f}")
    if rec is not None:
        worst = 0.0
        for tag in ("held_out", "all_tiles"):
            for k in ("count_recall", "area_recall", "area_ratio_sum_of_polygons"):
                a, b = out[tag][C1][k], rec[tag][k]
                worst = max(worst, abs(a - b) / max(abs(b), 1e-12))
        out["c1_reproduces_corrections_matched"] = {"worst_relative_difference": worst}
        if worst > 1e-9:
            raise SystemExit(f"matching does not reproduce config_study_corrections ({worst:.2e})")
    return out


# --- validation against the study's own records --------------------------------------------------

def validation(cells, le, registered, corrections):
    worst, notes = 0.0, []

    def check(name, got, want):
        nonlocal worst
        d = abs(got - want) / max(abs(want), 1e-12)
        worst = max(worst, d)
        if d > 1e-9:
            notes.append(f"{name}: here {got!r}, recorded {want!r}")

    d = cells[cells.cell_size_m == 40.0]
    ho = d[d.split.isin(le.HELD_OUT)]
    for r in registered["held_out_40m_at_primary_floor"]:
        if r["candidate"] not in (C1, C2):
            continue
        x = ho[(ho.candidate == r["candidate"]) & (ho.floor == r["floor"])]
        m = le.error_metrics(x)
        cr = x.n_pred.sum() / x.n_true.sum()
        for k in ("pooled_ratio", "spearman", "rmse"):
            check(f"registered {r['candidate']} {k}", m[k], r[k])
        check(f"registered {r['candidate']} count_ratio", cr, r["count_ratio"])
        check(f"registered {r['candidate']} mean_area_ratio", m["pooled_ratio"] / cr, r["mean_area_ratio"])
        for o, v in r["pooled_ratio_by_image"].items():
            check(f"registered {r['candidate']} {o}", le.error_metrics(x[x.obs_id == o])["pooled_ratio"], v)
    fx = registered["checks"]["fixed floor 0.25"]
    index = csc.tiles_of(ho[ho.candidate == C1])
    w = csc.draws(len(index))
    parts = {}
    for cid in (C1, C2):
        x = ho[(ho.candidate == cid) & (ho.floor == 0.25)]
        check(f"fixed 0.25 {cid} rmse", csc.rmse(x), fx["rmse"][cid])
        parts[cid] = csc.tile_parts(x, index)
    best = fx["best"]
    got = csc.ci(csc.boot_rmse(parts[C2], w) / csc.boot_rmse(parts[best], w))
    for a, b in zip(got, fx["ratio_to_best_ci95"][C2]):
        check(f"fixed 0.25 {C2} ratio_to_best_ci95 (best {best})", a, b)
    sc = corrections["scene"]
    for key, (cid, f) in {"C1@0.25": (C1, 0.25), "C1@0.40": (C1, 0.40),
                          f"{C2}@0.40": (C2, 0.40)}.items():
        x = d[(d.candidate == cid) & (d.floor == f)]
        lv = {o: le.error_metrics(g)["pooled_ratio"] for o, g in x.groupby("obs_id")}
        for o, v in sc["level_by_image_all_tiles"][key].items():
            check(f"corrections level {key} {o}", lv[o], v)
        check(f"corrections log_sd {key}", float(np.std(np.log(list(lv.values())))),
              sc["log_sd_across_images"][key])
    for cid in (C1, C2):
        x = ho[(ho.candidate == cid) & (ho.floor == 0.40)]
        se = (x.fa_pred - x.fa_true) ** 2
        by_tile = se.groupby([x.obs_id, x.tile_id]).sum() / se.sum()
        check(f"corrections effective_n_tiles {cid}@0.40", float(1 / (by_tile ** 2).sum()),
              corrections["concentration"][cid]["effective_n_tiles"])
    return {"worst_relative_difference": worst, "mismatches": notes,
            "what": "reproduces config_study.json (the registered table at 0.40 for C1 and the production "
                    "rule; the fixed-0.25 check's RMSE and interval) and config_study_corrections.json "
                    "(levels by image and log-SD at 0.25 / 0.40; effective tiles at 0.40) from the same rows"}


def floor_sweep(cells, le):
    """Pooled C2 / C1 at every floor on the study's grid, 40 m, no intervals: how the gap moves."""
    out = {}
    d = cells[cells.cell_size_m == 40.0]
    for tag, sel in (("held_out", lambda x: x[x.split.isin(le.HELD_OUT)]), ("all_tiles", lambda x: x)):
        rows = {}
        for f in cs.FLOORS:
            x = sel(d[d.floor == f])
            a, b = x[x.candidate == C1], x[x.candidate == C2]
            ma, mb = le.error_metrics(a), le.error_metrics(b)
            ca, cb = a.n_pred.sum() / a.n_true.sum(), b.n_pred.sum() / b.n_true.sum()
            rows[f"{f:.2f}"] = {"level": {C1: ma["pooled_ratio"], C2: mb["pooled_ratio"]},
                                "count_ratio": {C1: float(ca), C2: float(cb)},
                                "rmse": {C1: ma["rmse"], C2: mb["rmse"]},
                                "spearman": {C1: ma["spearman"], C2: mb["spearman"]},
                                "C2_over_C1_level": mb["pooled_ratio"] / ma["pooled_ratio"],
                                "C2_over_C1_count": float(cb / ca),
                                "C2_over_C1_rmse": mb["rmse"] / ma["rmse"]}
        out[tag] = rows
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--floors", type=float, nargs="+", default=[0.25, 0.40],
                    help="the first is the label's floor (D36); the rest are sensitivities")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    le, sha = cs.load_label_error()
    splits = le.split_map()
    pq = cs.STUDY_DIR / "paired_cells.parquet"
    cells = pd.read_parquet(pq)
    cells = cells[cells.candidate.isin((C1, C2))]
    bad = [f for f in args.floors if f not in set(cells.floor)]
    if bad:
        raise SystemExit(f"floors {bad} are not on the study's grid {sorted(set(cells.floor))}")
    keep = set(map(tuple, cells[["obs_id", "rom_index", "tile_id"]].drop_duplicates()
                   .itertuples(index=False, name=None)))
    registered = json.loads(cs.OUT_JSON.read_text())
    corrections = json.loads(CORRECTIONS_JSON.read_text())
    val = validation(cells, le, registered, corrections)
    if val["worst_relative_difference"] > 1e-9:
        raise SystemExit(f"does not reproduce the study's records: {val['mismatches']}")
    print(f"validation: worst relative difference {val['worst_relative_difference']:.2e}", flush=True)
    out = {"what": "configuration 2 (production rule, C2:256pref_engulf) against configuration 1 (C1) at "
                   f"fixed score floors {args.floors} (the first is the label's), notebook 35's label "
                   "against Prieur's human ROM mapping, from the configuration study's per-cell rows",
           "asked_by": "hirise2ctx-pub/docs/bouldernet_stage1_results_2026-10-18.md, section 6, question 4",
           "code": csc.cs_code_identity(), "label_error_sha256": sha,
           "inputs": {"paired_cells": {"path": str(pq.relative_to(REPO)).replace("\\", "/"),
                                       "sha256": hashlib.sha256(pq.read_bytes()).hexdigest()},
                      "score_meta": json.loads((cs.STUDY_DIR / "score_meta.json").read_text())},
           "conventions": "\n\n".join(__doc__.split("\n\n")[2:4]),
           "floors": args.floors, "candidates": {C1: "the 256 arm", C2: "{512, 256}, production fusion rule"},
           "validation": val, "by_floor": {}, "floor_sweep_40m": floor_sweep(cells, le)}
    for floor in args.floors:
        fb = {"blocks": {}}
        for subset in ("held_out", "all_tiles"):
            for cell_m in (40.0, 160.0):
                key = f"{subset}_{int(cell_m)}m"
                fb["blocks"][key] = block(cells, le, floor, cell_m,
                                          "held_out" if subset == "held_out" else "all")
                p = fb["blocks"][key]["pooled"]
                print(f"{floor:.2f} {key}: level C1 {p[C1]['level']:.3f} C2 {p[C2]['level']:.3f} "
                      f"(x{p['C2_over_C1']['level']:.3f}); rmse {p[C1]['rmse']:.4f} / "
                      f"{p[C2]['rmse']:.4f}; spearman {p[C1]['spearman']:.3f} / {p[C2]['spearman']:.3f}",
                      flush=True)
        print(f"matching at {floor:.2f} ...", flush=True)
        fb["matched"] = matched(le, splits, keep, floor, corrections)
        out["by_floor"][f"{floor:.2f}"] = fb
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
