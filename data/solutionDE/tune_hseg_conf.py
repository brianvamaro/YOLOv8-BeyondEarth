"""Step 0 of the H-seg x G plan: find the H-seg confidence that matches production's operating point.

**Why this is a prerequisite and not a refinement.** A retrained model's score distribution shifts,
so comparing H-seg at a nominal 0.10 against production at 0.10 compares two different operating
points and calls the difference a model difference. That is the unfair comparison that produced the
original H-seg verdict, and running H-seg x G without fixing it would reproduce it exactly.

**Method: re-run, do not filter.** The obvious cheap trick is one pass at a low floor and then
`score >= t` for each t. This script does not do that. Fusion promotes by *scale* rather than score
(B-3: catalog `score` is not threshold-stable, reversibility 87.9% not 99.5%), and even single-arm
NMS sees a different candidate set at a different floor, so a filtered catalog is an approximation
of a re-run. A 20-tile 256-arm pass costs ~90 s, which is cheap enough that the assumption is not
worth making. Each sweep point is a real run. `--filter-check` additionally measures how wrong the
cheap trick would have been, since that is worth knowing for the full-scene case where it is not.

**Population.** The 20 HiRISE ROM tiles against the Prieur human mapping — the same ground the
acceptance harness scores on. The **256 arm** only: production's fusion rule is
`bands={256: (0, inf)}`, which hands the 256 arm every contested cluster, so that is where the
operating point effectively lives, and it is the arm the harness reports as `*_256`.

**PRE-REGISTERED READ.** Two candidate criteria, both reported, and the maintainer freezes one:
  (a) **equal precision** — the conf at which H-seg's precision equals production's at 0.10;
  (b) **equal detections/km2** — the conf at which H-seg's density equals production's.
Headline population is detections with equivalent diameter **>= 4 native px**; pooled-all and
>= 8 px are reported as sensitivity, because pooled-all counts sub-floor detections the human
reference never mapped (Prieur maps at 5x5 px) as false positives and so is not a like-for-like
precision. If (a) and (b) disagree by more than one grid step, say so and pick deliberately —
they are different definitions of "same operating point", not an inconsistency to average away.

    python data/solutionDE/tune_hseg_conf.py --preset goal1_production
    python data/solutionDE/tune_hseg_conf.py --preset goal1_production --score-only   # re-score
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[2]
DE = ROOT / "data" / "solutionDE"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(DE))

from _ladder import add_ladder_args, run_ladder  # noqa: E402
from YOLOv8BeyondEarth.fusion import equivalent_diameter_px  # noqa: E402
from YOLOv8BeyondEarth.predict import prediction_filenames  # noqa: E402
from analyze_goal3_rom import PRIEUR, load_aligned, overlap_match  # noqa: E402

HSEG_W = ROOT / "data" / "weights" / "yolo26m-h-seg.pt"
PROD_W = ROOT / "data" / "weights" / "yolov8-m-boulder-detection-tmp.pt"
RES = 0.5                                     # HiRISE m/px
SIZE_FLOORS = {"all": 0.0, "ge4px": 4.0, "ge8px": 8.0}
HEADLINE = "ge4px"


def _tag(conf):
    return f"_tune{int(round(conf * 100)):03d}"


def _sub(kind, conf, stem):
    return f"{kind}{_tag(conf)}_rom_ladder/{stem}"


def _catalog(kind, conf, stem, ss=256, sr=1):
    return DE / "preds" / _sub(kind, conf, stem) / prediction_filenames(
        stem, confidence_threshold=conf, slice_size=ss, inference_size=1024,
        overlap_height_ratio=0.2, sr_factor=sr, downscale_pred=True, mask_tag=None,
        output_format="gpkg")["nms"]


def run_one(kind, conf, args, ss=256, sr=1):
    """One sweep point: the whole ROM tile set for one weight-set at one confidence."""
    tiles = sorted(DE.glob("hirise_rom??.tif"))
    arms = [(_sub(kind, conf, t.stem), t, ss, sr) for t in tiles]
    sweep_args = SimpleNamespace(conf=conf, remask=False, dir_tag=_tag(conf), preset=args.preset,
                                 device=args.device, trust_existing=args.trust_existing)
    run_ladder(weights=HSEG_W if kind == "hseg" else PROD_W, arms=arms,
               timings_path=DE / f"conf_sweep_timings_{kind}.json", args=sweep_args,
               key_fn=lambda sub, s, r: f"{kind}_{Path(sub).name}_ss{s}_c{int(conf * 100):03d}")


def score_one(kind, conf, human, rom_polys, area_km2, ss=256, sr=1, floor_filter=None):
    """Precision / recall / F1 / density for one catalog against the pooled human reference.

    Pools per ROM polygon the same way `analyze_hseg_acceptance.pooled_rom` does — kept local
    because that one is a nested function over a 3-arm dict and this is one arm over many confs.
    """
    parts = []
    for tile, poly in zip(sorted(DE.glob("hirise_rom??.tif")), rom_polys):
        g = gpd.read_file(_catalog(kind, conf, tile.stem, ss, sr))
        g = g.set_crs(human.crs, allow_override=True)
        parts.append(g[g.geometry.centroid.within(poly)])
    cat = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry",
                           crs=human.crs)
    if floor_filter is not None and len(cat):
        cat = cat[cat.score >= floor_filter].reset_index(drop=True)

    out = {}
    h_dpx = equivalent_diameter_px(human, RES)
    p_dpx = equivalent_diameter_px(cat, RES) if len(cat) else np.array([])
    hm, pm = overlap_match(human, cat)
    for name, floor in SIZE_FLOORS.items():
        hsel, psel = h_dpx >= floor, (p_dpx >= floor if len(cat) else np.zeros(0, bool))
        rec = float(hm[hsel].mean()) if hsel.sum() else np.nan
        prec = float(pm[psel].mean()) if psel.sum() else np.nan
        f1 = (2 * prec * rec / (prec + rec)
              if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else np.nan)
        out[name] = dict(n_pred=int(psel.sum()), n_human=int(hsel.sum()), precision=prec,
                         recall=rec, f1=f1, per_km2=float(psel.sum()) / area_km2)
    return out, cat


def crossing(df, col, target):
    """Linear interpolation of the conf at which ``col`` crosses ``target``. None if it never does."""
    d = df.dropna(subset=[col]).sort_values("conf")
    x, y = d.conf.to_numpy(float), d[col].to_numpy(float)
    for i in range(len(x) - 1):
        if (y[i] - target) * (y[i + 1] - target) <= 0 and y[i] != y[i + 1]:
            return float(x[i] + (target - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_ladder_args(p)
    p.add_argument("--grid", type=float, nargs="+",
                   default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
                   help="H-seg confidences to sweep (each one a real run)")
    p.add_argument("--prod-conf", type=float, default=0.10,
                   help="production's shipped operating point, the thing being matched")
    p.add_argument("--score-only", action="store_true", help="skip inference, re-score what exists")
    p.add_argument("--filter-check", action="store_true",
                   help="also measure how far `score >= t` on the lowest run diverges from the "
                        "real run at t — the shortcut this script deliberately does not take")
    args = p.parse_args()
    args.remask, args.dir_tag = False, ""       # the sweep owns these

    with rasterio.open(ROOT / "data" / "mars_hirise" / "raster"
                       / "ESP_017355_2260_RED.tif") as src:
        crs = src.crs
    rom = (load_aligned(PRIEUR / "mars/ESP_017355_2260/shp/ESP_017355_2260-ROM.shp", crs)
           .explode(index_parts=False).reset_index(drop=True))
    hum_all = load_aligned(PRIEUR / "mars/ESP_017355_2260/shp/ESP_017355_2260-boulder-mapping.shp",
                           crs)
    hcent = hum_all.geometry.centroid
    tiles = sorted(DE.glob("hirise_rom??.tif"))
    polys = [rom.geometry.iloc[int(t.stem[-2:])] for t in tiles]
    human = gpd.GeoDataFrame(pd.concat([hum_all[hcent.within(q)] for q in polys],
                                       ignore_index=True), geometry="geometry", crs=crs)
    area_km2 = sum(q.area for q in polys) / 1e6
    print(f"[pop] {len(tiles)} ROM tiles, {area_km2:.3f} km2, {len(human)} human boulders, "
          f"preset={args.preset or 'legacy_ladder'}", flush=True)

    if not args.score_only:
        run_one("prod", args.prod_conf, args)
        for c in args.grid:
            run_one("hseg", c, args)

    rows = []
    prod, _ = score_one("prod", args.prod_conf, human, polys, area_km2)
    for band, m in prod.items():
        rows.append(dict(arm="prod", conf=args.prod_conf, band=band, **m))
    for c in args.grid:
        got, _ = score_one("hseg", c, human, polys, area_km2)
        for band, m in got.items():
            rows.append(dict(arm="hseg", conf=c, band=band, **m))
    df = pd.DataFrame(rows)
    df.to_csv(DE / "hseg_conf_sweep.csv", index=False)

    print("\n=== sweep (headline population: "
          f"{HEADLINE}, equivalent diameter >= {SIZE_FLOORS[HEADLINE]:.0f} native px) ===")
    show = df[df.band == HEADLINE].drop(columns="band")
    print(show.round(4).to_string(index=False))

    answer = {"preset": args.preset, "prod_conf": args.prod_conf, "headline_band": HEADLINE}
    for band in SIZE_FLOORS:
        h = df[(df.arm == "hseg") & (df.band == band)]
        ref = df[(df.arm == "prod") & (df.band == band)].iloc[0]
        eq_p = crossing(h, "precision", ref.precision)
        eq_d = crossing(h, "per_km2", ref.per_km2)
        answer[band] = {"prod_precision": float(ref.precision), "prod_per_km2": float(ref.per_km2),
                        "equal_precision_conf": eq_p, "equal_density_conf": eq_d}
        tag = "  <-- HEADLINE" if band == HEADLINE else ""
        print(f"\n[{band}] production precision {ref.precision:.4f}, {ref.per_km2:.1f}/km2{tag}")
        print(f"   (a) equal-precision  H-seg conf = "
              f"{'not reached in grid' if eq_p is None else f'{eq_p:.4f}'}")
        print(f"   (b) equal-density    H-seg conf = "
              f"{'not reached in grid' if eq_d is None else f'{eq_d:.4f}'}")
        if eq_p is not None and eq_d is not None and abs(eq_p - eq_d) > 0.05:
            print(f"   ⚠ the two criteria differ by {abs(eq_p - eq_d):.3f} — they are different "
                  "definitions of the same operating point. Freeze one deliberately.")

    if args.filter_check:
        lo = min(args.grid)
        print(f"\n=== filter-check: `score >= t` on the conf-{lo} run vs the real run at t ===")
        for c in [g for g in args.grid if g > lo]:
            real, _ = score_one("hseg", c, human, polys, area_km2)
            filt, _ = score_one("hseg", lo, human, polys, area_km2, floor_filter=c)
            r, f = real[HEADLINE], filt[HEADLINE]
            print(f"  t={c:.2f}  n_pred real {r['n_pred']:5d} vs filtered {f['n_pred']:5d} "
                  f"({100 * (f['n_pred'] - r['n_pred']) / max(r['n_pred'], 1):+.2f}%)   "
                  f"precision {r['precision']:.4f} vs {f['precision']:.4f}")
            answer.setdefault("filter_check", {})[f"{c}"] = {
                "n_pred_real": r["n_pred"], "n_pred_filtered": f["n_pred"],
                "precision_real": r["precision"], "precision_filtered": f["precision"]}

    (DE / "hseg_conf_sweep.json").write_text(json.dumps(answer, indent=2))
    print(f"\nwrote hseg_conf_sweep.csv and hseg_conf_sweep.json to {DE}")
    print("NOTE this reports the curve and both crossings; freezing the number is a maintainer "
          "decision, and whichever is chosen must be stated with the criterion that produced it.")


if __name__ == "__main__":
    main()
