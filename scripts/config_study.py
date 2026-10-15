"""Detector configuration study for hirise2ctx v3 (D9, with the score floor D36).

Plan and pre-registered decision rule: ``docs/configuration_study_plan.md`` (top-level repo). Each
candidate is scored on **the label hirise2ctx trains on** -- per-cell boulder area fraction against
Prieur's human mapping on the ROM tiles -- through hirise2ctx's own ``src/label_error.py``, imported
read-only with its sha256 recorded, so the metric has one definition. The harness adds only a score
floor (the delivered conf-0.10 catalog filtered to ``score >= floor``, as Stage 4 applies it) and a
restriction to the tiles every candidate has.

    conda run -n bouldernet python scripts/config_study.py validate   # reproduce notebook 35, config 1
    conda run -n bouldernet python scripts/config_study.py refuse     # C2's fusion variants (CPU)
    conda run -n bouldernet python scripts/config_study.py score      # every candidate x floor x cell
    conda run -n bouldernet python scripts/config_study.py decide     # apply the registered rule

``C1G`` (the 256 arm re-masked by SAM2) is rendered by ``scripts/render_rom_configs.py --configs 1g``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ROM_DIR = REPO / "data" / "rom_label_error"
STUDY_DIR = REPO / "data" / "config_study"
OUT_JSON = REPO / "data" / "review" / "config_study.json"
LABEL_ERROR = Path(r"C:/Users/brian/Documents/PhD/HiRiseToCTXBoulders/hirise2ctx/src/label_error.py")
NB35_SUMMARY = LABEL_ERROR.parents[1] / "reports" / "figures" / "35_label_error_summary.json"

#: The delivered superset is conf 0.10, so the first floor means "no filter".
FLOORS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
CELLS = (40.0, 80.0, 160.0)
DECIDING_CELL = 40.0
N_BOOT, SEED = 2000, 0
ARMS = ((512, 1), (256, 1))

#: C2's fusion variants, simplest first (plan: score < scale_aware < bands; engulf off < on).
FUSION_VARIANTS = {
    "score": dict(rule="score", engulf_thr=None),
    "score_engulf": dict(rule="score", engulf_thr=0.7),
    "scale_aware": dict(rule="scale_aware", engulf_thr=None),
    "scale_aware_engulf": dict(rule="scale_aware", engulf_thr=0.7),
    "256pref": dict(rule="bands", engulf_thr=None),
    "256pref_engulf": dict(rule="bands", engulf_thr=0.7),          # production; == config2/
}

#: Candidates in the plan's simplicity order: (id, family, prediction root, subdirectory).
CANDIDATES = ([("C1", "C1", ROM_DIR, "")]
              + [(f"C2:{v}", "C2", STUDY_DIR / f"c2_{v}", "") for v in FUSION_VARIANTS]
              + [("C1G", "C1G", ROM_DIR, "config1g"), ("C3", "C3", ROM_DIR, "config3")])
FAMILIES = ("C1", "C2", "C1G", "C3")


def load_label_error(path=LABEL_ERROR):
    """hirise2ctx's module, imported read-only from its repo, and the sha256 of what was imported."""
    spec = importlib.util.spec_from_file_location("hirise2ctx_label_error", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod          # its @dataclass resolves annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod, hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def restricted(le, floor=None, keep=None):
    """Apply a score floor to the predictions and keep only the tiles in ``keep``, around
    notebook 35's own loaders (their names are looked up at call time, so this reaches them)."""
    orig_pred, orig_rom = le.load_predictions, le.load_rom

    def load_predictions(spec, rom_index, tile_id, ref_crs):
        g = orig_pred(spec, rom_index, tile_id, ref_crs)
        return g if floor is None else g[g["score"] >= floor]

    def load_rom(spec, splits=None):
        rom = orig_rom(spec, splits)
        if keep is None:
            return rom
        return rom[[(spec.obs_id, int(i), int(t)) in keep
                    for i, t in zip(rom["rom_index"], rom["tile_id"])]]

    le.load_predictions, le.load_rom = load_predictions, load_rom
    try:
        yield
    finally:
        le.load_predictions, le.load_rom = orig_pred, orig_rom


def _spec(base, root, sub):
    pred_dir = root / base.obs_id / sub if sub else root / base.obs_id
    return SimpleNamespace(obs_id=base.obs_id, raster=base.raster, rom_shp=base.rom_shp,
                           human_shp=base.human_shp, pred_dir=pred_dir)


def paired(le, splits, root, sub, floor, cell_m, keep=None, min_diameter_m=None):
    """notebook 35's per-cell frame for one candidate at one score floor and cell size.

    ``min_diameter_m`` is the label's size floor, applied to both sides as notebook 35 does; it
    defaults to Stage 4's live value (``STAGE4_MIN_DIAMETER_M``, 1.4105 m)."""
    size = le.STAGE4_MIN_DIAMETER_M if min_diameter_m is None else min_diameter_m
    with restricted(le, None if floor == FLOORS[0] else floor, keep):
        return pd.concat([le.paired_cells(_spec(s, root, sub), cell_m, size, splits)
                          for s in le.images().values()], ignore_index=True)


def tile_set(le, root, sub):
    """``{(obs, rom_index, tile_id)}`` this candidate has a rendered catalog for."""
    have = set()
    for s in le.images().values():
        for _, t in le.load_rom(s, {}).iterrows():
            spec = _spec(s, root, sub)
            if sorted(spec.pred_dir.glob(f"*rom{int(t.rom_index):02d}_t{int(t.tile_id)}"
                                         f"-predictions-*-nms.gpkg")):
                have.add((s.obs_id, int(t.rom_index), int(t.tile_id)))
    return have


# --- validate --------------------------------------------------------------------------------------

def validate(args):
    """Reproduce notebook 35's configuration-1 scale sweep exactly before trusting any comparison."""
    le, sha = load_label_error()
    want = json.loads(NB35_SUMMARY.read_text())["scale_sweep"]
    splits = le.split_map()
    worst = 0.0
    for cell in CELLS:
        df = paired(le, splits, ROM_DIR, "", FLOORS[0], cell)
        for tag, sel in (("all_rom", df), ("held_out", df[df.split.isin(le.HELD_OUT)])):
            got = le.error_metrics(sel)
            ref = next(r for r in want[tag] if r["cell_size_m"] == cell)
            for k, v in ref.items():
                if k == "cell_size_m":
                    continue
                d = abs(got[k] - v) / max(abs(v), 1e-12)
                worst = max(worst, d)
                if d > 1e-9:
                    print(f"MISMATCH {tag} {cell} m {k}: harness {got[k]!r} vs notebook 35 {v!r}")
    print(f"label_error.py sha256 {sha[:16]}; worst relative difference from notebook 35's "
          f"summary: {worst:.2e}")
    if worst > 1e-9:
        raise SystemExit("the harness does not reproduce notebook 35; do not compare anything")


# --- refuse ----------------------------------------------------------------------------------------

def refuse(args):
    """Re-fuse C2's arm layers under every fusion variant, into ``data/config_study/c2_<variant>``."""
    import geopandas as gpd
    import rasterio
    from YOLOv8BeyondEarth import presets
    from YOLOv8BeyondEarth.arm_ratification import arm_key, fuse_combo
    from YOLOv8BeyondEarth.polygon import outlines_to_shp

    base_cfg = presets.GOAL3_PRODUCTION
    checks = []
    for arms_dir in sorted(ROM_DIR.glob("*/config2/arms")):
        obs_dir = arms_dir.parents[1]
        for prod in sorted((obs_dir / "config2").glob("*-predictions-fused-512-256-nms.gpkg")):
            stem = prod.name.split("-predictions-")[0]
            crop = obs_dir / f"{stem}.tif"
            with rasterio.open(crop) as src:
                res = abs(src.transform.a)
            per_arm = {}
            for ss, sr in ARMS:
                hits = sorted(arms_dir.glob(f"{stem}-predictions-ct-010-ss-{ss}-*-nms.gpkg"))
                per_arm[arm_key(ss, sr)] = gpd.read_file(hits[0]) if hits else None
            nonempty = [k for k, g in per_arm.items() if g is not None and len(g)]
            for name, over in FUSION_VARIANTS.items():
                out = STUDY_DIR / f"c2_{name}" / obs_dir.name / \
                    f"{stem}-predictions-fused-512-256-{name}-nms.gpkg"
                if len(nonempty) >= 2:
                    fused, _ = fuse_combo(per_arm, ARMS, native_res=res,
                                          preset={**base_cfg, **over})
                else:          # as render_rom_configs: fewer than two arms found anything
                    fused = per_arm[nonempty[0]] if nonempty else None
                out.parent.mkdir(parents=True, exist_ok=True)
                if fused is not None and len(fused):
                    outlines_to_shp(fused, out)
                if name == "256pref_engulf" and out.exists():
                    a, b = gpd.read_file(out), gpd.read_file(prod)
                    same = len(a) == len(b) and all(
                        x.equals_exact(y, 0.0) for x, y in zip(a.geometry, b.geometry))
                    checks.append({"tile": stem, "n": len(a), "matches_config2": bool(same)})
    bad = [c for c in checks if not c["matches_config2"]]
    print(f"re-fused {len(checks)} tiles x {len(FUSION_VARIANTS)} variants; the production variant "
          f"reproduces config2/ on {len(checks) - len(bad)} of {len(checks)}")
    if bad:
        raise SystemExit(f"production re-fuse differs from config2/ on {[c['tile'] for c in bad]}")


# --- score -----------------------------------------------------------------------------------------

def score(args):
    """Every candidate x floor x cell size, on the tiles every candidate has."""
    le, sha = load_label_error()
    splits = le.split_map()
    sets = {cid: tile_set(le, root, sub) for cid, _, root, sub in CANDIDATES}
    missing = [cid for cid, s in sets.items() if not s]
    if missing:
        raise SystemExit(f"no rendered catalogs for {missing}; render or re-fuse them first")
    keep = set.intersection(*sets.values())
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for cid, fam, root, sub in CANDIDATES:
        for floor in FLOORS:
            for cell in CELLS:
                df = paired(le, splits, root, sub, floor, cell, keep)
                frames.append(df.assign(candidate=cid, family=fam, floor=floor))
            print(f"  {cid:24s} floor {floor:.2f} done", flush=True)
    cells = pd.concat(frames, ignore_index=True)
    cells.to_parquet(STUDY_DIR / "paired_cells.parquet", index=False)
    meta = {"label_error_sha256": sha, "n_tiles": len(keep),
            "tiles_excluded": sorted(set.union(*sets.values()) - keep),
            "tiles_per_candidate": {cid: len(s) for cid, s in sets.items()}}
    (STUDY_DIR / "score_meta.json").write_text(json.dumps(meta, indent=1, default=list))
    print(f"scored {len(CANDIDATES)} candidates x {len(FLOORS)} floors x {len(CELLS)} cells on "
          f"{len(keep)} tiles -> {STUDY_DIR / 'paired_cells.parquet'}")


# --- decide ----------------------------------------------------------------------------------------

def _tile_sse(df):
    """Per-tile sum of squared error and cell count, the parts RMSE is built from."""
    g = df.assign(se=(df.fa_pred - df.fa_true) ** 2).groupby(["obs_id", "tile_id"])
    return g.se.sum(), g.size()


def _boot_counts(n_tiles, rng):
    """Multinomial tile counts: one row per resample, the same draws for every candidate."""
    return rng.multinomial(n_tiles, np.full(n_tiles, 1.0 / n_tiles), size=N_BOOT)


def rule(cells, le, cell=DECIDING_CELL, subset="held_out", floor_on=None, fixed_floor=None,
         candidates=CANDIDATES, families=FAMILIES):
    """The registered decision rule on one view of the data. Returns a dict."""
    d = cells[cells.cell_size_m == cell]
    if subset == "held_out":
        d = d[d.split.isin(le.HELD_OUT)]
    elif subset == "res_025":
        d = d[d.obs_id.isin(("ESP_025650_2165", "ESP_028537_2270"))]
    sel_d = d if floor_on is None else cells[(cells.cell_size_m == cell) &
                                             (cells.split.isin(floor_on))]
    tiles = sorted(set(map(tuple, d[["obs_id", "tile_id"]].drop_duplicates().to_numpy())))
    index = pd.MultiIndex.from_tuples(tiles, names=["obs_id", "tile_id"])
    best_floor, sse, cnt = {}, {}, {}
    for cid, *_ in candidates:
        c = d[d.candidate == cid]
        if fixed_floor is not None:
            f = fixed_floor
        else:
            sc = sel_d[sel_d.candidate == cid]
            f = min(FLOORS, key=lambda fl: np.sqrt(((sc[sc.floor == fl].fa_pred -
                                                     sc[sc.floor == fl].fa_true) ** 2).mean()))
        best_floor[cid] = f
        s, n = _tile_sse(c[c.floor == f])
        sse[cid], cnt[cid] = s.reindex(index).to_numpy(), n.reindex(index).to_numpy()
    rmse = {cid: float(np.sqrt(sse[cid].sum() / cnt[cid].sum())) for cid in sse}
    best = min(rmse, key=rmse.get)
    w = _boot_counts(len(tiles), np.random.default_rng(SEED))
    boot = {cid: np.sqrt((w @ sse[cid]) / (w @ cnt[cid])) for cid in sse}
    ratio = {cid: np.percentile(boot[cid] / boot[best], [2.5, 97.5]).tolist() for cid in boot}
    tied = [cid for cid in rmse if ratio[cid][0] <= 1.0 <= ratio[cid][1]]
    family = next(f for f in families if any(c.split(":")[0] == f for c in tied))
    # the rule within C2, decided among C2's own variants against C2's best
    c2 = [cid for cid, fam, *_ in candidates if fam == "C2"]
    c2_best = min(c2, key=rmse.get)
    c2_tied = [cid for cid in c2
               if np.percentile(boot[cid] / boot[c2_best], 2.5) <= 1.0
               <= np.percentile(boot[cid] / boot[c2_best], 97.5)]
    c2_rule = c2_tied[0].split(":")[1]
    return {"cell_m": cell, "subset": subset, "floor_chosen_on": floor_on or subset,
            "fixed_floor": fixed_floor, "n_tiles": len(tiles), "best": best,
            "rmse": rmse, "floor": best_floor, "ratio_to_best_ci95": ratio, "tied": tied,
            "family": family, "c2_rule": c2_rule,
            "recommend": family if family in ("C1", "C1G", "C1GS") else f"{family}:{c2_rule}"}


def evaluate(cells, le, candidates=CANDIDATES, families=FAMILIES):
    """The registered rule, its checks, and the per-candidate table, for one candidate set."""
    kw = dict(candidates=candidates, families=families)
    primary = rule(cells, le, **kw)
    checks = {
        "80 m cells": rule(cells, le, cell=80.0, **kw),
        "all tiles": rule(cells, le, subset="all", **kw),
        "fixed floor 0.10": rule(cells, le, fixed_floor=0.10, **kw),
        "fixed floor 0.25": rule(cells, le, fixed_floor=0.25, **kw),
        "floor chosen on training tiles": rule(cells, le, floor_on=("train",), **kw),
        "0.25 m/px tiles only": rule(cells, le, subset="res_025", **kw),
    }
    rec = primary["recommend"]
    if rec.startswith("C3") and primary["c2_rule"] != "256pref_engulf":
        rec += " (re-render C3 with this rule before recommending)"
    # the label's metrics, every candidate at its own primary floor, for the write-up
    d = cells[(cells.cell_size_m == DECIDING_CELL)]
    table = []
    for cid, fam, *_ in candidates:
        f = primary["floor"][cid]
        sub = d[(d.candidate == cid) & (d.floor == f)]
        ho = sub[sub.split.isin(le.HELD_OUT)]
        m = le.error_metrics(ho)
        count_ratio = ho.n_pred.sum() / ho.n_true.sum() if ho.n_true.sum() else float("nan")
        per_image = {o: le.error_metrics(g)["pooled_ratio"] for o, g in ho.groupby("obs_id")}
        table.append({"candidate": cid, "floor": f, **m, "count_ratio": float(count_ratio),
                      "mean_area_ratio": float(m["pooled_ratio"] / count_ratio),
                      "pooled_ratio_by_image": per_image})
    # the candidate actually recommended (C3's rule is a C2 rule, not a separate candidate id)
    rec_cid = rec.split(" ")[0]
    rec_cid = "C3" if rec_cid.startswith("C3") else rec_cid
    win = next(r for r in table if r["candidate"] == rec_cid)
    # ties are not transitive: the C2 rule is chosen against C2's best, so say whether the exact
    # candidate recommended is itself tied with the overall best
    lo, hi = primary["ratio_to_best_ci95"][rec_cid]
    compensating = (abs(np.log(win["count_ratio"])) > np.log(1.2) and
                    abs(np.log(win["mean_area_ratio"])) > np.log(1.2) and
                    np.sign(np.log(win["count_ratio"])) != np.sign(np.log(win["mean_area_ratio"])))
    return {"primary": primary, "checks": checks,
            "checks_agree": {k: v["recommend"] == primary["recommend"] for k, v in checks.items()},
            "recommendation": rec, "recommended_candidate": rec_cid,
            "recommended_is_tied_with_best": bool(lo <= 1.0 <= hi),
            "compensating_errors_flag": bool(compensating),
            "held_out_40m_at_primary_floor": table}


def _print(out):
    print(json.dumps({k: out[k] for k in ("recommendation", "recommended_is_tied_with_best",
                                          "checks_agree", "compensating_errors_flag")}, indent=1))
    for r in out["held_out_40m_at_primary_floor"]:
        print(f"  {r['candidate']:24s} floor {r['floor']:.2f}  RMSE {r['rmse']:.4f}  pooled ratio "
              f"{r['pooled_ratio']:.3f}  spearman {r['spearman']:.3f}  ratio-to-best CI "
              f"{[round(x, 3) for x in out['primary']['ratio_to_best_ci95'][r['candidate']]]}")


def decide(args):
    le, sha = load_label_error()
    cells = pd.read_parquet(STUDY_DIR / "paired_cells.parquet")
    out = {"what": "detector configuration study (docs/configuration_study_plan.md); label = "
                   "per-cell area fraction vs Prieur's human ROM mapping, notebook 35's metric",
           "label_error_sha256": sha, **evaluate(cells, le),
           "meta": json.loads((STUDY_DIR / "score_meta.json").read_text())}
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float))
    _print(out)


# --- post hoc --------------------------------------------------------------------------------------

#: Added 2026-10-15 AFTER the registered decision ran: the validated in-slice SAM2 path. The
#: registered C1G and C3 re-masked with 5a's post-fusion ``remask_windows``, whose masks came out
#: 0.97x YOLO's area per row, against 0.70-0.76x for the in-slice path, so neither had tested tight
#: SAM2 masks. Reported beside the registered result, never in place of it.
POSTHOC_CANDIDATE = ("C1GS", "C1GS", ROM_DIR, "config1gs")
POSTHOC_JSON = REPO / "data" / "review" / "config_study_posthoc.json"


def _mask_area_vs_yolo(le):
    """Per-boulder mask area of each SAM2 path over configuration 1's YOLO mask for the same
    detection (centroids matched within 1 m), overall and per image."""
    import geopandas as gpd
    from YOLOv8BeyondEarth.matching import unique_centroid_match
    ratios = {"C1G (window re-masker)": {}, "C1GS (in-slice, validated)": {}}
    for s in le.images().values():
        obs_dir = ROM_DIR / s.obs_id
        for y_path in sorted(obs_dir.glob(f"{s.obs_id}_rom??_t*-predictions-*-nms.gpkg")):
            stem = y_path.name.split("-predictions-")[0]
            y = gpd.read_file(y_path)
            for name, sub in (("C1G (window re-masker)", "config1g"),
                              ("C1GS (in-slice, validated)", "config1gs")):
                hits = sorted((obs_dir / sub).glob(f"{stem}-predictions-*-nms.gpkg"))
                if not hits or not len(y):
                    continue
                o = gpd.read_file(hits[0])
                ia, ib = unique_centroid_match(
                    np.c_[y.geometry.centroid.x, y.geometry.centroid.y],
                    np.c_[o.geometry.centroid.x, o.geometry.centroid.y], 1.0)
                r = o.geometry.area.to_numpy()[ib] / y.geometry.area.to_numpy()[ia]
                ratios[name].setdefault(s.obs_id, []).append(r)
    out = {}
    for name, by_obs in ratios.items():
        allr = np.concatenate([np.concatenate(v) for v in by_obs.values()])
        out[name] = {"n_matched": int(allr.size), "median": float(np.median(allr)),
                     "iqr": [float(np.percentile(allr, 25)), float(np.percentile(allr, 75))],
                     "median_by_image": {o: float(np.median(np.concatenate(v)))
                                         for o, v in by_obs.items()}}
    return out


def posthoc(args):
    """Score the post-hoc candidate on the registered tiles and re-apply the same rule with it
    slotted in after C1G (same family size: the 256 arm plus SAM2)."""
    le, sha = load_label_error()
    splits = le.split_map()
    reg = pd.read_parquet(STUDY_DIR / "paired_cells.parquet")
    reg_tiles = set(map(tuple, reg[["obs_id", "rom_index", "tile_id"]].drop_duplicates()
                        .itertuples(index=False, name=None)))
    cid, fam, root, sub = POSTHOC_CANDIDATE
    keep = reg_tiles & tile_set(le, root, sub)
    frames = []
    for floor in FLOORS:
        for cell in CELLS:
            frames.append(paired(le, splits, root, sub, floor, cell, keep)
                          .assign(candidate=cid, family=fam, floor=floor))
        print(f"  {cid} floor {floor:.2f} done", flush=True)
    new = pd.concat(frames, ignore_index=True)
    reg = reg[[(o, int(i), int(t)) in keep
               for o, i, t in zip(reg.obs_id, reg.rom_index, reg.tile_id)]]
    cells = pd.concat([reg, new], ignore_index=True)
    i = next(k for k, c in enumerate(CANDIDATES) if c[0] == "C1G") + 1
    candidates = CANDIDATES[:i] + [POSTHOC_CANDIDATE] + CANDIDATES[i:]
    families = ("C1", "C2", "C1G", "C1GS", "C3")
    out = {"what": "POST HOC addition to the configuration study: the validated in-slice SAM2 path "
                   "(C1GS), scored beside the registered candidates by the same rule",
           "why": ("the registered C1G/C3 window re-masker did not tighten "
                   "masks (0.97x YOLO per row), so SAM2 had not been tested"),
           "label_error_sha256": sha, "n_tiles": len(keep),
           "tiles_dropped_vs_registered": sorted(reg_tiles - keep),
           **evaluate(cells, le, candidates, families),
           "mask_area_vs_yolo": _mask_area_vs_yolo(le)}
    POSTHOC_JSON.write_text(json.dumps(out, indent=1, default=float))
    _print(out)
    print(json.dumps(out["mask_area_vs_yolo"], indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("validate", "refuse", "score", "decide", "posthoc"))
    args = ap.parse_args(argv)
    {"validate": validate, "refuse": refuse, "score": score, "decide": decide,
     "posthoc": posthoc}[args.cmd](args)


if __name__ == "__main__":
    main()
