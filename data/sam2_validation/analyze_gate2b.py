"""Gate 2b follow-up (CPU, no GPU) — is SAM's crop-position jitter RANDOM or SYSTEMATIC?

Gate 2b showed window-vs-slice disagreement is matched by the shifted-window control, i.e. it is
generic crop-translation sensitivity in SAM, not a defect of the phase-5a window grid. But |Δ| says
nothing about *direction*, and the consequence differs completely:

  * random / zero-mean  -> per-boulder angles are noisier than believed; AGGREGATE roses survive
                           (the jitter averages out), so Goal-2 fabric conclusions stand.
  * systematic          -> the jitter can move a rose, and every aggregate orientation statistic
                           computed from one crop grid inherits a bias nobody has measured.

Reads gate2b_pairs.csv (written by run_gate2b_identical_boxes.py). Reports, per crop:
  1. signed axial deviation win-vs-slice: mean, and a bootstrap CI on the mean;
  2. whether the per-arm ROSE statistics differ (grid_fraction, diagonal_fraction, f_lower,
     resultant R, peak) — the thing that actually matters for Goal 2/4;
  3. how the jitter scales with boulder size.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from YOLOv8BeyondEarth.orientation import (signed_axial_deviation, grid_fraction,
                                           diagonal_fraction, half_plane_fraction, refine_peak,
                                           grid_fraction_uniform)

HERE = Path(__file__).resolve().parent
ASPECT_MIN = 1.35


def _resultant(a):
    th = 2.0 * np.deg2rad(np.asarray(a, float) % 180.0)
    return float(np.abs(np.mean(np.exp(1j * th))))


def _boot_mean_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 10:
        return None, None
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    df = pd.read_csv(HERE / "gate2b_pairs.csv")
    out = []
    for crop, d in df.groupby("crop"):
        el = d[(d.aspect_win > ASPECT_MIN) & (d.aspect_sli > ASPECT_MIN)].dropna(
            subset=["angle_win", "angle_sli", "angle_shx"])

        sd = signed_axial_deviation(el.angle_win.values, el.angle_sli.values)
        lo, hi = _boot_mean_ci(sd)

        rose = {}
        for arm, col in (("window", "angle_win"), ("slice", "angle_sli"), ("shifted", "angle_shx")):
            a = el[col].dropna().values
            rose[arm] = dict(n=int(len(a)), grid=float(grid_fraction(a, 10)),
                             diag=float(diagonal_fraction(a, 10)),
                             f_lower=float(half_plane_fraction(a)),
                             R=_resultant(a), peak=float(refine_peak(a)))

        # size dependence of the |delta|
        bands = [(0, 6), (6, 8), (8, 12), (12, 1e9)]
        by_size = []
        for lo_b, hi_b in bands:
            m = (d.dpx_win >= lo_b) & (d.dpx_win < hi_b)
            sub = d[m & (d.aspect_win > ASPECT_MIN) & (d.aspect_sli > ASPECT_MIN)]
            by_size.append(dict(band=f"{lo_b}-{hi_b if hi_b < 1e9 else 'inf'}px", n=int(len(sub)),
                                median_dangle=float(sub.dangle_win_sli.median()) if len(sub) else None,
                                median_darea=float(sub.darea_win_sli.median()) if len(sub) else None))

        out.append(dict(
            crop=crop, n_elong=int(len(el)),
            signed_mean_dev_deg=float(np.nanmean(sd)),
            signed_mean_ci95=[lo, hi],
            zero_in_ci=bool(lo is not None and lo <= 0 <= hi),
            median_abs_dev_deg=float(np.nanmedian(np.abs(sd))),
            uniform_ref=float(grid_fraction_uniform(10)),
            rose=rose,
            rose_spread=dict(
                grid=float(max(r["grid"] for r in rose.values()) - min(r["grid"] for r in rose.values())),
                diag=float(max(r["diag"] for r in rose.values()) - min(r["diag"] for r in rose.values())),
                f_lower=float(max(r["f_lower"] for r in rose.values()) - min(r["f_lower"] for r in rose.values())),
                R=float(max(r["R"] for r in rose.values()) - min(r["R"] for r in rose.values())),
                peak_deg=float(max(r["peak"] for r in rose.values()) - min(r["peak"] for r in rose.values())),
            ),
            by_size=by_size,
        ))

    print(json.dumps(out, indent=2))
    (HERE / "gate2b_jitter_analysis.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
