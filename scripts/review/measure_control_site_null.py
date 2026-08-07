"""Recompute the Goal-4 **control-site null** on the equal-area sample. CPU only, ~1 min.

Why this exists
---------------
The live Goal-4 fabric gate (`docs/goal4_global_maps_plan.md` gate 4) says a candidate fabric must
beat a control-site null of **R ~ 0.35 with peaks at 43-76 deg**. Every number in that bar comes
from the T1 control-site probe, whose sampler (raster-order tiles from a central strip, stopped at
a detection quota) `docs/results_summary.md:519-536` declares unrecoverable — "only resampling".
At the comparable >= 4 m floor that sample held **8/22/103/41/12** boulders per site.

The equal-area re-run (`data/summary/fixed_area_scenes.csv`, 10 km2 of valid ground per scene,
shuffled tiles) replaced the peaks and `f_lower`, but **R was never recomputed**, so the one
statistic the gate is actually written in had no replacement on disk. This produces it.

It also produces the thing the old bar was missing. R is **not scale-free in n**: under isotropy
E[R] ~ 0.886/sqrt(n), which at n = 8 is 0.313 — so "R ~ 0.35" on an 8-boulder site was very close
to pure noise, and a bar stated as an absolute R silently gets *harder* as a candidate's sample
grows. The replacement is therefore stated as a **Rayleigh criterion**, `r_crit = sqrt(-ln(a)/n)`,
which is a function of the candidate's own n, with the measured control-site R values reported
beside it as the empirical floor.

Two scope notes, both deliberate:

* **Densities from this sample are lower bounds** (X-06): the run that produced the CSV did not
  pass `max_det`, so ultralytics' 300 applied. It does not plausibly move these statistics — the
  cap keeps the top N by confidence and this cut is the high-confidence >= 8 px / >= 4 m tail — but
  no boulders/km2 is quoted here, and `orientation_validation` now passes and records the cap.
* **The cut is fig09's own** (`aspect > 1.35`, `dpx_sam >= 8`, `dm_sam >= 4 m`), so the R here is
  directly comparable to the peaks and `f_lower` already published from the same figure. The
  producer-conditioning defect fig09 has does **not** apply to the SAM row read on its own terms,
  which is what a SAM-produced Goal-4 catalog would be; a producer-common (`no aspect cut`) row is
  reported alongside so the sensitivity is visible rather than assumed.

Usage:
    ...\\conda.exe run -n bouldernet python scripts/review/measure_control_site_null.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from YOLOv8BeyondEarth.orientation import (                            # noqa: E402
    axial_resultant, axial_resultant_uniform, rayleigh_p, refine_peak, half_plane_fraction,
    grid_fraction, diagonal_fraction, axial_distance)

CSV = ROOT / "data" / "summary" / "fixed_area_scenes.csv"
META = ROOT / "data" / "summary" / "fixed_area_scenes_meta.json"
OUT = ROOT / "data" / "review" / "control_site_null.json"

ASPECT_MIN = 1.35     # project standard
DPX_MIN = 8           # project standard well-resolved cut
DM_MIN = 4.0          # metres: fig09's common physical floor, so scenes count the same rocks
N_BOOT = 2000
SEED = 20260806


def _bootstrap_R_ci(a, groups, rng, n_boot=N_BOOT):
    """Percentile CI for R under a **tile-clustered** resample (G2-06).

    Boulders inside one 512-px tile are not independent draws of the scene's fabric, so resampling
    rows understates the width. Resampling whole tiles is the project's rule wherever a pooled
    statistic crosses tiles.
    """
    keys, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(keys))]
    if len(keys) < 5:
        return None, None
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[p] for p in pick])
        out[b] = axial_resultant(a[idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _verify_rayleigh(rng, n_trials=20000):
    """Do not assert the null — check it. Monte-Carlo the isotropic R distribution at three n and
    confirm both the mean (`axial_resultant_uniform`) and the tail (`rayleigh_p`)."""
    checks = {}
    for n in (8, 100, 1000):
        draws = rng.uniform(0.0, 180.0, size=(n_trials, n))
        r = np.abs(np.mean(np.exp(2j * np.deg2rad(draws)), axis=1))
        crit = float(np.sqrt(-np.log(0.05) / n))
        checks[n] = dict(mc_mean_R=round(float(r.mean()), 4),
                         formula_mean_R=round(axial_resultant_uniform(n), 4),
                         mc_p95_R=round(float(np.percentile(r, 95)), 4),
                         formula_p95_R=round(crit, 4),
                         mc_frac_above_formula_p95=round(float((r > crit).mean()), 4))
    return checks


def main():
    meta = {m["scene"]: m for m in json.loads(META.read_text())["scenes"] if "error" not in m}
    controls = {s: m for s, m in meta.items() if m.get("group") == "controls"}
    if not controls:
        raise SystemExit(f"no control-group scenes in {META}")

    print(f"reading {CSV.name} ({CSV.stat().st_size / 1e6:.0f} MB)...", flush=True)
    d = pd.read_csv(CSV, usecols=["scene", "res", "tile_x", "tile_y", "angle_yolo", "aspect_yolo",
                                  "dpx_yolo", "angle_sam", "aspect_sam", "dpx_sam", "dm_sam"])
    d = d[d.scene.isin(controls)]
    print(f"{len(d):,} control-site detections before the cut", flush=True)

    rng = np.random.default_rng(SEED)
    rows = []
    for sc in sorted(controls, key=lambda s: controls[s].get("incidence") or 0.0):
        g = d[(d.scene == sc) & (d.dpx_sam >= DPX_MIN) & (d.dm_sam >= DM_MIN)]
        # Site-level tile count only; each population re-derives its own keys for the bootstrap.
        tiles = (g.tile_x.astype(str) + "_" + g.tile_y.astype(str)).to_numpy()
        rec = dict(scene=sc, res_m=controls[sc].get("res_m"),
                   incidence=controls[sc].get("incidence"),
                   sun_axis=controls[sc].get("shadow_axis"),
                   km2_valid=controls[sc].get("km2_valid"), n_tiles=int(len(set(tiles))))
        # Two populations per producer-of-record. `own_aspect` is what fig09 publishes and what a
        # SAM-produced Goal-4 catalog would gate on; `no_aspect_cut` is producer-common and is the
        # neutral read pass 7 settled on. Reporting both is the point: the gate must not be quietly
        # conditional on a selection.
        for tag, sub in (("own_aspect", g[g.aspect_sam > ASPECT_MIN]),
                         ("no_aspect_cut", g)):
            sub = sub[sub.angle_sam.notna()]
            a = sub.angle_sam.to_numpy() % 180.0
            n = int(a.size)
            R = axial_resultant(a)
            sub_tiles = (sub.tile_x.astype(str) + "_" + sub.tile_y.astype(str)).to_numpy()
            lo, hi = _bootstrap_R_ci(a, sub_tiles, rng)
            rec[tag] = dict(
                n=n, R=round(R, 4), R_null_mean=round(axial_resultant_uniform(n), 4),
                R_null_p95=round(float(np.sqrt(-np.log(0.05) / n)), 4) if n else None,
                R_over_null=round(R / axial_resultant_uniform(n), 2) if n else None,
                rayleigh_p=rayleigh_p(R, n),
                R_ci=None if lo is None else [round(lo, 4), round(hi, 4)],
                peak=round(float(refine_peak(a)), 1) if n >= 5 else None,
                f_lower=round(half_plane_fraction(a), 3) if n else None,
                grid=round(grid_fraction(a), 3) if n else None,
                diag=round(diagonal_fraction(a), 3) if n else None)
        rows.append(rec)

    own = [r["own_aspect"] for r in rows]
    peaks = [r["peak"] for r in own if r["peak"] is not None]
    summary = dict(
        n_sites=len(rows),
        cut=f"aspect_sam > {ASPECT_MIN} (own_aspect row only), dpx_sam >= {DPX_MIN}, "
            f"dm_sam >= {DM_MIN} m",
        producer="sam256 (angle_sam) — the Goal-4 production masker",
        n_range=[min(r["n"] for r in own), max(r["n"] for r in own)],
        R_range=[min(r["R"] for r in own), max(r["R"] for r in own)],
        R_median=round(float(np.median([r["R"] for r in own])), 4),
        R_null_mean_range=[min(r["R_null_mean"] for r in own),
                           max(r["R_null_mean"] for r in own)],
        R_over_null_range=[min(r["R_over_null"] for r in own),
                           max(r["R_over_null"] for r in own)],
        max_rayleigh_p=max(r["rayleigh_p"] for r in own),
        peak_range=[min(peaks), max(peaks)] if peaks else None,
        f_lower_range=[min(r["f_lower"] for r in own), max(r["f_lower"] for r in own)],
        # The bar to write into the plan. Set at the largest bootstrap UPPER limit, not the largest
        # point estimate: a bar at max-of-five point estimates is a bar set at its own sampling
        # noise, and 0.443 sits below one control site's own R, so that site would essentially tie
        # its own gate. Both are recorded so the difference stays visible.
        gate_form="R_candidate > max(R_control_max_upper95, sqrt(-ln(0.05)/n_candidate))",
        R_control_max_upper95=round(max(r["R_ci"][1] for r in own if r["R_ci"]), 4),
        R_control_max_point_estimate=max(r["R"] for r in own),
        gate_form_caveats=[
            "sqrt(-ln(0.05)/n) exceeds the absolute term only for n < 16, so for any realistic "
            "candidate this is a FLAT bar with a small-n guard, not a bar that scales with n.",
            "R_control_max is strongly conditional on the aspect cut (0.330/0.443/0.509/0.539/0.642 "
            "at aspect 1.0/1.35/1.5/1.7/2.0), so candidate and null MUST be read at one stated cut.",
            "A map is thousands of cells. Applied per cell at alpha=0.05 this gate manufactures "
            "fabric; it needs FDR control, a minimum effective n, and a tiling-preserving "
            "permutation null before any map cell may be claimed. See goal4_global_maps_plan.md "
            "gate 4 and backlog item G4-07.",
            "A scalar R cannot separate a fabric aligned with the artifact axis from one orthogonal "
            "to it, and this null is Mars-only at incidence 44.8-63.4 deg.",
        ],
        note=("Densities from this sample are X-06 lower bounds (max_det unpassed on the "
              "2026-07-27 run); no density is quoted here. Angle statistics on this >=8 px / "
              ">=4 m tail are the high-confidence end and are not plausibly moved by the cap."))

    out = dict(source=str(CSV.relative_to(ROOT)), sample="equal-area, 10 km2/scene, shuffled tiles",
               seed=SEED, n_boot=N_BOOT, sites=rows, summary=summary,
               null_verification=_verify_rayleigh(rng),
               supersedes=dict(bar="R ~ 0.35, peaks 43-76 deg",
                               source="T1 control-site probe (raster-order prefix + detection "
                                      "quota); declared unrecoverable at results_summary.md:519-536",
                               old_n_at_4m=[8, 22, 103, 41, 12]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))

    print(f"\n{'scene':22s} {'n':>7s} {'R':>7s} {'E[R]|iso':>9s} {'R/null':>7s} "
          f"{'p':>10s} {'peak':>6s} {'f_low':>6s}")
    for r in rows:
        o = r["own_aspect"]
        print(f"{r['scene']:22s} {o['n']:7,d} {o['R']:7.4f} {o['R_null_mean']:9.4f} "
              f"{o['R_over_null']:7.2f} {o['rayleigh_p']:10.2e} {o['peak']:6.1f} "
              f"{o['f_lower']:6.3f}")
    print(f"\nsummary: {json.dumps(summary, indent=1)}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
