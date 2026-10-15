"""X-07: the Mars arm ratification before vs after moving the predictions into the humans' frame (2026-10-15).

Reads the two gitignored `ratify_arms.py` CSVs (`--erode-tiles-m 1.5`, without and with
`--pred-shift-m -0.75 0.75`) and writes the tracked JSON the fusion plan and backlog cite, including the
hybrid (centroid-in-polygon) gate's effect, which the per-run JSONs do not carry (they hold strict
marginals only).

    conda run -n bouldernet python scripts/review/x07_ratification_compare.py . data/review/x07_ratification_compare.json
"""
import sys, json
from pathlib import Path
import pandas as pd
repo = Path(sys.argv[1])
d = repo / "data/review/ratification"
base = pd.read_csv(d / "arm_ratification_mars-erode1.5m.csv")
shf = pd.read_csv(d / "arm_ratification_mars-predshift-erode1.5m.csv")
key = ["combo", "band", "split", "resolve", "hybrid"]
m = base.merge(shf, on=key, suffixes=("_old", "_corrected"))
m = m[(m.split == "all") & (m.resolve == "optimal")]
bands = ["3-4", "4-6", "6-8", "8-10", "10-14", "14-inf", "all"]
out = {"what": "Mars ROM arm ratification (current Sherlock arms, ROM tiles eroded 1.5 m, optimal matching): "
               "old frame vs predictions shifted (-0.75, +0.75) m into the humans' frame (X-07)",
       "sources": [str(p.relative_to(repo)) for p in (d / "arm_ratification_mars-erode1.5m.csv",
                                                      d / "arm_ratification_mars-predshift-erode1.5m.csv")],
       "strict": [], "hybrid_gate_effect": []}
for combo in sorted(m.combo.unique()):
    for b in bands:
        s = m[(m.combo == combo) & (m.band == b) & (~m.hybrid)]
        h = m[(m.combo == combo) & (m.band == b) & (m.hybrid)]
        if not len(s):
            continue
        s = s.iloc[0]
        out["strict"].append(dict(combo=combo, band=b, n_human=int(s.n_human_old),
                                  recall=[float(s.recall_old), float(s.recall_corrected)],
                                  precision=[float(s.precision_old), float(s.precision_corrected)]))
        if len(h):
            h = h.iloc[0]
            out["hybrid_gate_effect"].append(dict(combo=combo, band=b,
                                                  recall_gain=[float(h.recall_old - s.recall_old),
                                                               float(h.recall_corrected - s.recall_corrected)]))
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
for g in out["hybrid_gate_effect"]:
    if g["band"] in ("4-6", "6-8", "all"):
        print(f'{g["combo"]:12s} {g["band"]:6s} hybrid gate effect {g["recall_gain"][0]:+.3f} -> {g["recall_gain"][1]:+.3f}')
