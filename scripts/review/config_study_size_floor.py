"""How the configuration study's answer moves with the label's size floor (2026-10-15).

hirise2ctx's v3 re-sets its size floor from the v3 detections' size-distribution rollover (D13),
so the 1.4105 m the study used is provisional. Two questions:
1. Does the score floor remove small boulders preferentially? (If so, the rollover a
   score-filtered catalog shows moves up, coupling D36 and D13.)
2. At other size floors, which score floor minimises the label error, and what does it do?

Same harness and tiles as the study. Held-out tiles, 40 m cells.

    conda run -n bouldernet python scripts/review/config_study_size_floor.py --out data/review/config_study_size_floor.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import config_study as cs                                                   # noqa: E402

SIZE_FLOORS = (1.0, 1.4105, 2.0, 3.0)
CANDS = {"C1": (cs.ROM_DIR, ""), "C1GS": (cs.ROM_DIR, "config1gs")}


def score_vs_size(le):
    """Fraction of configuration 1's detections each score floor keeps, by equivalent diameter."""
    import geopandas as gpd
    rows = []
    for s in le.images().values():
        res = None
        for p in sorted((cs.ROM_DIR / s.obs_id).glob(f"{s.obs_id}_rom??_t*-predictions-*-nms.gpkg")):
            g = gpd.read_file(p)
            if res is None:
                import rasterio
                with rasterio.open(p.parent / (p.name.split("-predictions-")[0] + ".tif")) as src:
                    res = abs(src.transform.a)
            d_m = 2 * np.sqrt(g.geometry.area.to_numpy() / np.pi)
            rows.append(pd.DataFrame({"obs": s.obs_id, "d_m": d_m, "d_px": d_m / res,
                                      "score": g.score.to_numpy()}))
    df = pd.concat(rows, ignore_index=True)
    out = []
    for lo, hi in ((0, 4), (4, 6), (6, 8), (8, 12), (12, 1e9)):
        b = df[(df.d_px >= lo) & (df.d_px < hi)]
        out.append({"d_px": [lo, hi], "n": int(len(b)),
                    **{f"kept_at_{f:.2f}": float((b.score >= f).mean()) for f in (0.25, 0.40, 0.50)}})
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
    sweep = []
    for size in SIZE_FLOORS:
        for cid, (root, sub) in CANDS.items():
            per_floor = []
            for f in cs.FLOORS:
                d = cs.paired(le, splits, root, sub, f, 40.0, keep, min_diameter_m=size)
                ho = d[d.split.isin(le.HELD_OUT)]
                m = le.error_metrics(ho)
                per_floor.append({"score_floor": f, "rmse": m["rmse"], "area_ratio": m["pooled_ratio"],
                                  "count_ratio": float(ho.n_pred.sum() / ho.n_true.sum()),
                                  "spearman": m["spearman"]})
            best = min(per_floor, key=lambda r: r["rmse"])
            sweep.append({"size_floor_m": size, "candidate": cid, "best": best, "per_floor": per_floor})
            print(f"size floor {size:.2f} m  {cid:5s} best score floor {best['score_floor']:.2f}  RMSE "
                  f"{best['rmse']:.4f}  area {best['area_ratio']:.3f}  count {best['count_ratio']:.3f}  "
                  f"(at 0.10: RMSE {per_floor[0]['rmse']:.4f}, area {per_floor[0]['area_ratio']:.3f})",
                  flush=True)
    svs = score_vs_size(le)
    for r in svs:
        print(f"  d {r['d_px'][0]:>3}-{r['d_px'][1]:<4} px  n {r['n']:6d}  kept at 0.25 "
              f"{r['kept_at_0.25']:.2f}  0.40 {r['kept_at_0.40']:.2f}  0.50 {r['kept_at_0.50']:.2f}")
    Path(args.out).write_text(json.dumps({
        "what": "configuration study vs the label's size floor; held-out, 40 m cells",
        "label_error_sha256": sha, "sweep": sweep, "score_kept_by_size_C1_all_tiles": svs},
        indent=1, default=float))


if __name__ == "__main__":
    main()
