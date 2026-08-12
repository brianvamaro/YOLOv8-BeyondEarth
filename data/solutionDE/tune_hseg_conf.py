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

RESULT SO FAR (2026-08-09, `--preset goal1_production`, 20 ROM tiles, >= 4 px):

    arm   conf  precision  recall     F1   det/km2
    prod  0.10     0.4226  0.8842  0.5719    8733.6
    hseg  0.10     0.5176  0.8838  0.6529    7121.6
    hseg  0.20     0.6268  0.8143  0.7083    5414.4
    hseg  0.40     0.7655  0.6675  0.7132    3623.2

**H-seg at nominal 0.10 is already stricter than production at 0.10** — more precise and sparser —
so both crossings lie BELOW the swept range and the grid has to be extended downward, not upward.
At equal nominal conf, pooled recall is a dead heat (0.8838 vs 0.8842) while precision differs by
0.095: the two score distributions are calibrated very differently, which is exactly why step 0
exists, and the mismatch ran AGAINST H-seg in the original comparison.

Linear extrapolation from the 0.10/0.15 points puts **both** criteria at **conf ~0.017-0.021** —
they agree, so there is probably no criterion conflict to adjudicate. That is an extrapolation over
4x the grid step on a convex curve; it needs measuring, and 0.02 is a low enough operating point
that whether it is a fair match or a pathological one is a judgement call worth making explicitly
before the H-seg x G run inherits it.

TO FINISH IT (not yet run):

    python data/solutionDE/tune_hseg_conf.py --preset goal1_production --filter-check \
        --batch-size 2 --grid 0.01 0.02 0.03 0.05 0.07 0.10

⚠ **`--batch-size 2` is required below ~0.05.** This is an 8151 MiB card and `max_det=600` at
batch 8 already peaks near 7.7 GiB; a conf-0.02 point OOM'd after 3 of 20 tiles. Batch-2 points get
their own directories, so they cannot silently mix with the batch-8 points above.
⚠ **The mixed-batch caveat, and it differs by criterion.** G1-02 measured batch 1 vs 8 at *exact
count parity* with *0.70% differing geometry*. So the **density crossing is unaffected**, while the
**precision crossing carries a small geometry caveat** because it depends on overlap matching. If
that matters, re-run the whole curve at `--batch-size 2` (~25 min) for uniformity rather than
splicing. Also re-run the production reference at the same batch — the command above does.

FILTER-CHECK, measured: `score >= t` on the conf-0.10 run reproduces a real run at t to within
-0.05% on count and 0.0004 on precision, across t = 0.15..0.40. So on a SINGLE ARM the shortcut this
script declines to take would in fact have been fine — worth knowing for the full-scene case where
re-running is expensive. It does **not** license the same shortcut after fusion, where B-3 measured
87.9%.
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


#: Set from the CLI. Batch size is result-defining (G1-02: exact count parity, 0.70% differing
#: geometry) and absent from the filename, so a non-default batch gets its own directory rather
#: than colliding with the batch-8 points — the ladder's manifest guard would otherwise, correctly,
#: refuse to continue. Low-confidence points need it: max_det=600 at batch 8 OOMs an 8 GiB card.
BATCH = None


def _tag(conf):
    b = f"b{BATCH}" if BATCH else ""
    return f"_tune{int(round(conf * 100)):03d}{b}"


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
    p.add_argument("--prod-grid", type=float, nargs="*", default=None,
                   help="ALSO sweep production upward and report the reverse match: the production "
                        "conf that meets H-seg at --match-hseg-conf. Matching is symmetric, and "
                        "this is the executable direction — driving H-seg DOWN to production's "
                        "0.10 needs conf ~0.02, where mask memory OOMs an 8 GiB card (measured: "
                        "2.8x the detections per tile at 0.01, ~1.1 GiB of masks per slice), while "
                        "production going UP has fewer detections and is cheaper, not dearer.")
    p.add_argument("--match-hseg-conf", type=float, default=0.10,
                   help="the H-seg operating point that --prod-grid is matched against")
    p.add_argument("--score-only", action="store_true", help="skip inference, re-score what exists")
    p.add_argument("--filter-check", action="store_true",
                   help="also measure how far `score >= t` on the lowest run diverges from the "
                        "real run at t — the shortcut this script deliberately does not take")
    args = p.parse_args()
    args.remask, args.dir_tag = False, ""       # the sweep owns these
    global BATCH
    BATCH = args.batch_size

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

    prod_grid = sorted({args.prod_conf, *(args.prod_grid or [])})
    if not args.score_only:
        for c in prod_grid:
            run_one("prod", c, args)
        for c in args.grid:
            run_one("hseg", c, args)

    rows = []
    for c in prod_grid:
        got, _ = score_one("prod", c, human, polys, area_km2)
        for band, m in got.items():
            rows.append(dict(arm="prod", conf=c, band=band, **m))
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
        pr = df[(df.arm == "prod") & (df.band == band)]
        ref = pr[np.isclose(pr.conf, args.prod_conf)].iloc[0]
        eq_p = crossing(h, "precision", ref.precision)
        eq_d = crossing(h, "per_km2", ref.per_km2)
        answer[band] = {"prod_precision": float(ref.precision), "prod_per_km2": float(ref.per_km2),
                        "equal_precision_conf": eq_p, "equal_density_conf": eq_d}
        tag = "  <-- HEADLINE" if band == HEADLINE else ""
        print(f"\n[{band}] production at {args.prod_conf}: precision {ref.precision:.4f}, "
              f"{ref.per_km2:.1f}/km2{tag}")
        print(f"   FORWARD (move H-seg down to production):")
        print(f"     (a) equal-precision  H-seg conf = "
              f"{'not reached in grid' if eq_p is None else f'{eq_p:.4f}'}")
        print(f"     (b) equal-density    H-seg conf = "
              f"{'not reached in grid' if eq_d is None else f'{eq_d:.4f}'}")
        if eq_p is not None and eq_d is not None and abs(eq_p - eq_d) > 0.05:
            print(f"     ⚠ the two criteria differ by {abs(eq_p - eq_d):.3f} — different "
                  "definitions of the same operating point. Freeze one deliberately.")

        # Reverse: matching is symmetric, and moving production UP is the direction that runs.
        if len(pr) > 1:
            hm = h[np.isclose(h.conf, args.match_hseg_conf)]
            if len(hm):
                hr = hm.iloc[0]
                rp = crossing(pr, "precision", hr.precision)
                rd = crossing(pr, "per_km2", hr.per_km2)
                answer[band].update(hseg_ref_conf=args.match_hseg_conf,
                                    hseg_precision=float(hr.precision),
                                    hseg_per_km2=float(hr.per_km2),
                                    reverse_equal_precision_prod_conf=rp,
                                    reverse_equal_density_prod_conf=rd)
                print(f"   REVERSE (move production up to H-seg at {args.match_hseg_conf}: "
                      f"precision {hr.precision:.4f}, {hr.per_km2:.1f}/km2):")
                print(f"     (a) equal-precision  production conf = "
                      f"{'not reached in grid' if rp is None else f'{rp:.4f}'}")
                print(f"     (b) equal-density    production conf = "
                      f"{'not reached in grid' if rd is None else f'{rd:.4f}'}")
                if rp is not None and rd is not None and abs(rp - rd) > 0.05:
                    print(f"     ⚠ the two criteria differ by {abs(rp - rd):.3f} — freeze one "
                          "deliberately.")

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
