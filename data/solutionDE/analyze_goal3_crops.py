# Goal-3 phase 3: dev-crop validation suite (plan §Validation 2, 3, 6, 7, 10 + composition).
# Consumes the per-scale ladder outputs on the two dev crops (256 + SRx2 cached from phases 1-2,
# 512 from run_goal3_ladder.py) and writes the CSVs the Goal3 notebook renders:
#   goal3_fusion_summary.csv   - per scene x rule: n, composition, parity gate, consolidations
#   goal3_dupe_rates.csv       - per scene x rule x band: near-pair rate (< 2% gate)
#   goal3_unique_contrib.csv   - per scene x arm x band: detections only that arm found
#   goal3_size_ratio.csv       - cross-scale matched-pair size ratios by band (audit 10)
#   goal3_orientation.csv      - baseline vs inputs vs fused: grid/diag frac + 135/45 (+CIs)
# CPU-only; idempotent overwrite. Bands rule boundaries are PROVISIONAL (plan §Open risks):
# SRx2 <10 px, 256 10-40 px, 512 >=40 px; the rule ablation reads these results to re-derive.
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\src")
from YOLOv8BeyondEarth.fusion import (DEFAULT_CLUSTER_THR, fuse_predictions, pool_predictions,
                                      matched_recall, duplicate_rate, equivalent_diameter_px)
from YOLOv8BeyondEarth.benchmark import match_detections, DEFAULT_SIZE_BANDS, _band_label
from YOLOv8BeyondEarth.orientation import (attach_orientations, grid_fraction,
                                           diagonal_fraction, asymmetry_ratio)

DE = Path(r"c:\Users\brian\Documents\PhD\BoulderNet\YOLOv8-BeyondEarth\data\solutionDE")
PROVISIONAL_BANDS = {(256, 2): (0.0, 10.0), 256: (10.0, 40.0), 512: (40.0, np.inf)}
ENGULF = 0.7   # poly-containment threshold; evidence: deep-dive containment gap (p10 0.44 vs
#                mass at 0.93+ for fragment pairs; distinct neighbours << 0.5)
# ablation configs: name -> fuse_predictions kwargs (bands filled per scene)
CONFIGS = {
    "scale_aware": {},
    "score": dict(rule="score"),
    "bands": dict(rule="bands"),
    "scale_aware_engulf": dict(engulf_thr=ENGULF),
    # phase-3 ablation winner vs human truth (see analyze_goal3_rom): 256 masks win contested
    # clusters (production convention preserved); SR/512 = recall nets; fragments engulfed
    "256pref_engulf": dict(rule="bands", bands="PREF", engulf_thr=ENGULF),
}

SCENES = {
    "nac": dict(res=0.48357597846618, arms={
        512: DE / "preds/nac_512/nac_crop2k-predictions-ct-010-ss-512-is-1024-ov-020-downscaled-mask-nms.gpkg",
        256: DE / "preds/nac_baseline/nac_crop2k-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg",
        (256, 2): DE / "preds/nac_e2stream/nac_crop2k-predictions-ct-010-ss-256sr2-is-1024-ov-020-downscaled-mask-nms.gpkg"}),
    "hirise": dict(res=0.5, arms={
        512: DE / "preds/hirise_512/hirise_crop4k-predictions-ct-010-ss-512-is-1024-ov-020-downscaled-mask-nms.gpkg",
        256: DE / "preds/hirise_baseline/hirise_crop4k-predictions-ct-010-ss-256-is-1024-ov-020-downscaled-mask-nms.gpkg",
        (256, 2): DE / "preds/hirise_e2stream/hirise_crop4k-predictions-ct-010-ss-256sr2-is-1024-ov-020-downscaled-mask-nms.gpkg"}),
}


def band_of(dpx):
    out = np.full(len(dpx), "<3", dtype=object)
    for lo, hi in DEFAULT_SIZE_BANDS:
        out[(dpx >= lo) & (dpx < hi)] = _band_label(lo, hi)
    return out


def boot_ci(vals, fn, nb=1000, seed=0):
    vals = np.asarray(vals)
    if len(vals) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats = [fn(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(nb)]
    return tuple(np.percentile(stats, [2.5, 97.5]))


def orientation_rows(scene, source, ori, res):
    """Metrics on the all-elongated and well-resolved subsets of an orientation frame."""
    rows = []
    for subset, sel in [("elongated", ori[ori.aspect_ra > 1.35]),
                        ("well_resolved", ori[(ori.aspect_ra > 1.35) &
                                              (ori.diameter_m / res >= 8)])]:
        a = sel.angle180.values
        gf_lo, gf_hi = boot_ci(a, grid_fraction)
        df_lo, df_hi = boot_ci(a, diagonal_fraction)
        rows.append(dict(scene=scene, source=source, subset=subset, n=len(a),
                         grid_frac=grid_fraction(a) if len(a) else np.nan,
                         grid_lo=gf_lo, grid_hi=gf_hi,
                         diag_frac=diagonal_fraction(a) if len(a) else np.nan,
                         diag_lo=df_lo, diag_hi=df_hi,
                         r135=asymmetry_ratio(a) if len(a) else np.nan))
    return rows


if __name__ == "__main__":
    fusion_rows, dupe_rows, uniq_rows, size_rows, ori_rows = [], [], [], [], []

    for scene, cfg in SCENES.items():
        res = cfg["res"]
        gdfs = {k: gpd.read_file(p) for k, p in cfg["arms"].items()}
        print(f"[{scene}] arms: " + ", ".join(f"{k}: {len(g)}" for k, g in gdfs.items()),
              flush=True)

        # one pooled orientation pass reused by every rule (pool order is deterministic)
        pool = pool_predictions(gdfs)
        pool["pool_index"] = np.arange(len(pool))
        t0 = time.time()
        pool_ori = attach_orientations(pool, res=res, n_workers=8)
        print(f"[{scene}] orientations on pool ({len(pool)}) in {time.time() - t0:.0f}s",
              flush=True)
        pool_ori["dpx"] = pool_ori.diameter_m / res

        base = gdfs[256]
        base_ori = pool_ori[pool_ori.scale_key == "256"]
        ori_rows += orientation_rows(scene, "input_256_baseline", base_ori, res)
        ori_rows += orientation_rows(scene, "input_512", pool_ori[pool_ori.scale_key == "512"], res)
        ori_rows += orientation_rows(scene, "input_sr2", pool_ori[pool_ori.scale_key == "(256, 2)"], res)

        # --- per-arm unique contributions (validation 4b: only-this-arm detections, by band)
        for key in gdfs:
            others = {k: g for k, g in gdfs.items() if k != key}
            _, unmatched = matched_recall(gdfs[key], pool_predictions(others))
            dpx = equivalent_diameter_px(gdfs[key], res)
            ub = band_of(dpx[unmatched])
            ab = band_of(dpx)
            for lo, hi in DEFAULT_SIZE_BANDS:
                lab = _band_label(lo, hi)
                uniq_rows.append(dict(scene=scene, arm=str(key), band=lab,
                                      n_arm=int((ab == lab).sum()),
                                      n_unique=int((ub == lab).sum())))

        # --- cross-scale size-consistency audit (validation 10), vs the 256 baseline
        dpx_base = equivalent_diameter_px(base, res)
        for key in (512, (256, 2)):
            m = match_detections(base, gdfs[key], max_dist=2.0 * res)
            if not len(m):
                continue
            dpx_arm = equivalent_diameter_px(gdfs[key], res)
            ratio = dpx_arm[m.idx_b.values] / dpx_base[m.idx_a.values]
            bands = band_of(dpx_base[m.idx_a.values])
            for lo, hi in DEFAULT_SIZE_BANDS:
                lab = _band_label(lo, hi)
                r = ratio[bands == lab]
                if len(r) < 10:
                    size_rows.append(dict(scene=scene, arm=str(key), band=lab, n=len(r),
                                          median=np.nan, q25=np.nan, q75=np.nan))
                    continue
                size_rows.append(dict(scene=scene, arm=str(key), band=lab, n=len(r),
                                      median=float(np.median(r)),
                                      q25=float(np.percentile(r, 25)),
                                      q75=float(np.percentile(r, 75))))

        # --- fuse under each config: parity, dupes, composition, consolidations, orientation
        for rule, kw in CONFIGS.items():
            kw = dict(kw)
            if kw.get("rule") == "bands":
                pref = kw.get("bands") == "PREF"
                kw.update(bands={256: (0.0, np.inf)} if pref else PROVISIONAL_BANDS,
                          native_res=res)
            t0 = time.time()
            fused, audit = fuse_predictions(gdfs, **kw)
            dt = time.time() - t0
            parity, unmatched = matched_recall(base, fused,
                                               engulf_thr=kw.get("engulf_thr"))
            comp = fused.scale_key.value_counts()
            # consolidations: a 512 keeper absorbing >=2 same-cluster 256 members
            a256 = audit[(audit.scale_key == "256") & (audit.keeper_scale_key == "512")]
            consol = int((a256.groupby("keeper").size() >= 2).sum())
            fusion_rows.append(dict(
                scene=scene, rule=rule, n_pool=len(pool), n_fused=len(fused),
                from_512=int(comp.get("512", 0)), from_256=int(comp.get("256", 0)),
                from_sr2=int(comp.get("(256, 2)", 0)),
                parity=parity, n_unmatched=len(unmatched),
                n_suppressed=len(audit), n_cross_scale=int(audit.cross_scale.sum()),
                n_512_consolidations=consol,
                n_merged_max=int(fused.n_merged.max()) if len(fused) else 0,
                fuse_seconds=round(dt, 2)))

            dr = duplicate_rate(fused, native_res=res)
            dr.insert(0, "rule", rule); dr.insert(0, "scene", scene)
            dupe_rows.append(dr)

            fused_ori = pool_ori[pool_ori.pool_index.isin(fused.pool_index)]
            ori_rows += orientation_rows(scene, f"fused_{rule}", fused_ori, res)

            # matched-pair (replaced-representative) orientation delta, scale_aware only
            if rule == "scale_aware":
                m = match_detections(base, fused, max_dist=2.0 * res)
                fk = fused.scale_key.to_numpy()[m.idx_b.values]
                repl = m[fk != "256"]
                ang_b = base_ori.angle180.to_numpy()[repl.idx_a.values]
                ang_f = pool_ori.set_index("pool_index").angle180.reindex(
                    fused.pool_index.to_numpy()[repl.idx_b.values]).to_numpy()
                d = np.abs(ang_b - ang_f) % 180.0
                d = np.minimum(d, 180.0 - d)
                print(f"[{scene}] scale_aware replaced {len(repl)}/{len(m)} baseline reps "
                      f"cross-scale; |angle delta| median={np.nanmedian(d):.1f} deg", flush=True)

    pd.DataFrame(fusion_rows).to_csv(DE / "goal3_fusion_summary.csv", index=False)
    pd.concat(dupe_rows, ignore_index=True).to_csv(DE / "goal3_dupe_rates.csv", index=False)
    pd.DataFrame(uniq_rows).to_csv(DE / "goal3_unique_contrib.csv", index=False)
    pd.DataFrame(size_rows).to_csv(DE / "goal3_size_ratio.csv", index=False)
    pd.DataFrame(ori_rows).to_csv(DE / "goal3_orientation.csv", index=False)

    print("\n=== fusion summary ===")
    print(pd.DataFrame(fusion_rows).to_string(index=False))
    print("\n=== duplicate rates (near_pair_rate by band) ===")
    print(pd.concat(dupe_rows, ignore_index=True)
          .pivot_table(index=["scene", "band"], columns="rule", values="near_pair_rate")
          .to_string())
    print("\nCSV outputs written to", DE)
