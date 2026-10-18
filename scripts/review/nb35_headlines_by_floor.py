"""Notebook 35's two remaining superset headlines, re-read at score floors 0.25 and 0.40 (2026-10-17).

hirise2ctx's pilot answers (§5, question 1) ask for two numbers that their CLAUDE.md still quotes from
the conf-0.10 superset:
- **range compression**, "~1.9x": at 160 m, over cells with human truth > 0, the truth's Q4/Q1 span
  of quartile means divided by the label's (notebook 35 §4, `compression_quartile`);
- **rank fidelity at 160 m**, "0.98": Spearman of label against truth over all 160 m cells
  (notebook 35 §3, `error_metrics`; 0.99 held out).

Both use notebook 35's own tile set: configuration 1's renders on all 50 ROM tiles. At 160 m that is
60 cells in 30 tiles, all from the two 0.5-0.6 m/px images, because the 0.25 m/px images' 125 m
tiles hold no complete 160 m cell. The quartiles are of truth, so the same cells fall in each quartile
at every floor. The harness is the configuration study's (`config_study.paired`, notebook 35's
`label_error` imported read-only). At 0.10 it must reproduce notebook 35's summary, and the script
stops if it does not. Intervals are a cluster bootstrap over ROM tiles. The held-out set has 6 tiles at
160 m, too few for an interval to mean much, so its intervals are recorded and not quoted.

    conda run -n bouldernet python -I scripts/review/nb35_headlines_by_floor.py --out data/review/nb35_headlines_by_floor.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import config_study as cs                                                   # noqa: E402
from YOLOv8BeyondEarth import presets                                       # noqa: E402

FLOORS = (0.10, 0.25, 0.40)
N_BOOT, SEED = 2000, 0


def compression(d, by_rank=False):
    """notebook 35 §4: truth's Q4/Q1 span of quartile means over the label's, on cells with truth > 0.

    ``by_rank`` cuts the quartiles on ranks (ties broken by order) rather than on values. The bootstrap
    needs it, because a resample that draws a tile twice duplicates its cells and can tie a quartile
    edge. The point estimate is notebook 35's value cut."""
    pos = d[d.fa_true > 0].copy()
    pos["q"] = pd.qcut(pos.fa_true.rank(method="first") if by_rank else pos.fa_true, 4, labels=False)
    q = pos.groupby("q").agg(fa_true=("fa_true", "mean"), fa_pred=("fa_pred", "mean"))
    span_true, span_pred = q.fa_true.iloc[-1] / q.fa_true.iloc[0], q.fa_pred.iloc[-1] / q.fa_pred.iloc[0]
    return float(span_true / span_pred) if span_pred > 0 else float("nan"), q


def rho(d):
    return float(spearmanr(d.fa_true, d.fa_pred).statistic)


def boot(d, stat):
    groups = [g for _, g in d.groupby(["obs_id", "tile_id"], sort=True)]
    rng = np.random.default_rng(SEED)
    draws = [stat(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
             for _ in range(N_BOOT)]
    draws = np.asarray([x for x in draws if np.isfinite(x)])
    return {"n_tiles": len(groups), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data" / "review" / "nb35_headlines_by_floor.json"))
    args = ap.parse_args()
    le, sha = cs.load_label_error()
    nb35 = json.loads(cs.NB35_SUMMARY.read_text())
    splits = le.split_map()
    out = {"what": "notebook 35's range compression and 160 m Spearman, configuration 1, by score floor",
           "code": presets.code_identity(), "label_error_sha256": sha, "floors": {}}
    for floor in FLOORS:
        rec = {}
        for cell in (40.0, 160.0):
            df = cs.paired(le, splits, cs.ROM_DIR, "", floor, cell)
            for tag, sel in (("all_rom", df), ("held_out", df[df.split.isin(le.HELD_OUT)])):
                m = le.error_metrics(sel)
                r = {"n_cells": m["n_cells"], "n_tiles": m["n_tiles"], "spearman": m["spearman"],
                     "pooled_ratio": m["pooled_ratio"], "rmse": m["rmse"],
                     "spearman_boot": boot(sel, rho)}
                if cell == 160.0:
                    comp, q = compression(sel)
                    r.update(compression_quartile=comp,
                             compression_boot=boot(sel, lambda x: compression(x, by_rank=True)[0]),
                             compression_boot_quartiles="rank cut, ties broken by order",
                             quartile_means=q.reset_index().to_dict("records"))
                rec[f"{tag}_{int(cell)}m"] = r
        out["floors"][f"{floor:.2f}"] = rec
        a = rec["all_rom_160m"]
        print(f"floor {floor:.2f}: 160 m all ROM n={a['n_cells']} ({a['n_tiles']} tiles) Spearman "
              f"{a['spearman']:.4f} {[round(x, 3) for x in a['spearman_boot']['ci95']]}, compression "
              f"{a['compression_quartile']:.3f} {[round(x, 2) for x in a['compression_boot']['ci95']]}, "
              f"pooled {a['pooled_ratio']:.3f}; held out Spearman {rec['held_out_160m']['spearman']:.4f} "
              f"(n={rec['held_out_160m']['n_cells']})")
    base = out["floors"]["0.10"]["all_rom_160m"]
    want_rho = next(r for r in nb35["scale_sweep"]["all_rom"] if r["cell_size_m"] == 160.0)["spearman"]
    want_comp = nb35["compression"]["compression_quartile"]
    out["reproduces_notebook_35_at_0.10"] = {
        "spearman_160m": [base["spearman"], want_rho], "compression_quartile": [base["compression_quartile"], want_comp],
        "ok": bool(abs(base["spearman"] - want_rho) < 1e-9 and abs(base["compression_quartile"] - want_comp) < 1e-9)}
    Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    print("reproduces notebook 35 at 0.10:", out["reproduces_notebook_35_at_0.10"])
    if not out["reproduces_notebook_35_at_0.10"]["ok"]:
        raise SystemExit("does not reproduce notebook 35 at the superset; do not quote the other floors")


if __name__ == "__main__":
    main()
