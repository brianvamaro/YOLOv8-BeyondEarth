"""X-07 re-test of the production fusion-rule choice, on current catalogs (2026-10-14).

Phase 3 chose `256pref_engulf` over `scale_aware` because scale_aware collapsed Mars small-boulder
recall against the humans (4-6 px 0.69 -> 0.36). Those tables were made on crops of data/mars_hirise,
which sits (+0.75, -0.75) m from the humans' label frame (X-07), and before X-01 (effective conf 0.25).
This fuses the August Sherlock arms (contract v3, conf 0.10) under each rule, before and after moving
the predictions into the humans' frame, on ROM tiles eroded 1.5 m (the arms were pre-clipped).

    conda run -n bouldernet python scripts/review/x07_fusion_rule_retest.py . data/review/x07_fusion_rule_retest.csv
"""
import sys, json, importlib.util
from pathlib import Path
import pandas as pd
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo / "src"))
from YOLOv8BeyondEarth import arm_ratification as ar, presets
spec = importlib.util.spec_from_file_location("ra", repo / "scripts/ratify_arms.py")
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
P = presets.GOAL3_PRODUCTION
rules = {"256pref_engulf (production)": P,
         "scale_aware": {**P, "rule": "scale_aware", "engulf_thr": None},
         "scale_aware_engulf": {**P, "rule": "scale_aware"}}
combos = {"512+256+SR": [(512, 1), (256, 1), (256, 2)], "512+256": [(512, 1), (256, 1)]}
rows = []
for shift in (None, (-0.75, 0.75)):
    data = ra.load_body("mars", pred_shift_m=shift, erode_tiles_m=1.5)
    for cname, arms in combos.items():
        for rname, cfg in rules.items():
            fused, _ = ar.fuse_combo(data["arms"], arms, data["res"], preset=cfg)
            fused = ar.clip_to_tiles(fused, data["tiles"])
            sc = ar.score_bands(data["humans"], fused, data["res"], hybrid=False, resolve="optimal")
            sc = sc[sc.split == "all"] if "split" in sc.columns else sc
            for _, r in sc.iterrows():
                rows.append(dict(frame="corrected" if shift else "old", combo=cname, rule=rname, band=r.band,
                                 n_pred=int(r.n_pred), recall=float(r.recall), precision=float(r.precision), f1=float(r.f1)))
df = pd.DataFrame(rows)
df.to_csv(sys.argv[2], index=False)
for cname in combos:
    print(f"\n=== {cname}  recall / precision by band, corrected frame (old frame in brackets)")
    for b in ["4-6", "6-8", "8-10", "10-14", "14-inf", "all"]:
        line = f"  {b:7s}"
        for rname in rules:
            c = df[(df.frame == "corrected") & (df.combo == cname) & (df.rule == rname) & (df.band == b)]
            o = df[(df.frame == "old") & (df.combo == cname) & (df.rule == rname) & (df.band == b)]
            if len(c): line += f" | {rname[:14]:14s} R {c.iloc[0].recall:.3f} [{o.iloc[0].recall:.3f}] P {c.iloc[0].precision:.3f} [{o.iloc[0].precision:.3f}]"
        print(line)
