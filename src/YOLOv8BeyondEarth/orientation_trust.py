"""At which **aspect ratio × size** do we trust a SAM orientation?

The working answer — ``aspect >= 1.7, dpx >= 8`` — came from a single ladder inside the SAM-signature
characterization, was never a dedicated study, and the archive flags it PROVISIONAL *"from one coarse
ladder never crossed with size"*. Meanwhile the code default is ``aspect >= 1.35``. The two disagree,
and the 1.35–1.7 band is exactly where SAM's ΔR against human tracing is significant (Mars +0.134
[+0.05, +0.20]) — so the looser default admits the boulders most likely to carry the artifact.

This matters because the cut is not a detail. The Goal-4 fabric bar ``R_control_max`` moves
**0.330 → 0.642** across aspect 1.0 → 2.0, which is larger than most fabric signals anyone would
want to claim, and :func:`YOLOv8BeyondEarth.fabric_gate.fabric_gate` *enforces* that a candidate and
its null share a cut without being able to say which cut is right. Goal-3 phase 5a's re-mask floor
waits on the same number.

**Three surfaces over the same (aspect, size) grid**, because a threshold is a trade-off and one
number cannot show it:

* **accuracy** — median ``|Δangle|`` between SAM and a human tracer on matched boulders. Lower is
  better. This is the axis that says "is the angle right".
* **artifact** — grid fraction above the isotropic 0.2222 on fabric-free control ground. Lower is
  better. This is the axis that says "is what's left contaminated", and it is the one that feeds the
  fabric gate.
* **yield** — how many boulders survive. Strict cuts buy accuracy with statistical power, and a
  gate with no power manufactures nothing but shrugs.

**On cutting by the model's own aspect/size.** That is deliberate and is *not* the differential-
selection error the review retired. That error was comparing two producers each conditioned on its
**own** aspect column — two different sets of rocks. Here a single producer is characterised against
truth, and the cut has to be one an operator can actually apply at map time, when no human labels
exist. The model's own ``aspect_sam``/``dpx_sam`` is the only thing available then, so it is the
right thing to threshold on. Human-side columns are used for the *comparison*, never for selection.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .orientation import grid_fraction, grid_fraction_uniform

#: Isotropic reference for the grid fraction (±10° of a cardinal), i.e. 4/18.
ISOTROPIC_GRID = grid_fraction_uniform()

#: Default grid. Aspect spans the disputed 1.35-1.7 band with room either side; size spans the
#: Prieur 5x5 px mapping floor up to where masks are comfortably resolved.
DEFAULT_ASPECT_CUTS = (1.0, 1.2, 1.35, 1.5, 1.7, 2.0)
DEFAULT_DPX_CUTS = (0.0, 4.0, 6.0, 8.0, 12.0, 16.0)

#: Below this many surviving boulders a cell's statistics are noise, and are reported as NaN rather
#: than as a small number that looks precise.
MIN_CELL_N = 50


def axial_abs_delta(a_deg, b_deg):
    """Absolute difference between two **axial** angles, in [0, 90].

    Orientations are mod 180, so 179° and 1° differ by 2°, not 178°.
    """
    d = np.abs(np.asarray(a_deg, dtype=float) - np.asarray(b_deg, dtype=float)) % 180.0
    return np.minimum(d, 180.0 - d)


@dataclass
class TrustSurface:
    """The three surfaces plus the grid they were computed on."""
    accuracy: pd.DataFrame          # median |Δangle| vs human, degrees
    accuracy_n: pd.DataFrame
    artifact: pd.DataFrame          # grid fraction − isotropic, on control ground
    artifact_n: pd.DataFrame
    aspect_cuts: tuple
    dpx_cuts: tuple
    R: pd.DataFrame = None          # background resultant length — what the fabric gate consumes
    peak: pd.DataFrame = None       # its direction, doubled-angle circular mean, degrees
    accuracy_source: str = ""
    artifact_source: str = ""

    def as_long(self):
        """Tidy form: one row per (aspect, dpx) cell with every statistic."""
        rows = []
        for a in self.aspect_cuts:
            for s in self.dpx_cuts:
                rows.append(dict(
                    aspect_min=a, dpx_min=s,
                    median_abs_dangle=self.accuracy.loc[s, a],
                    n_accuracy=self.accuracy_n.loc[s, a],
                    grid_excess=self.artifact.loc[s, a],
                    n_artifact=self.artifact_n.loc[s, a],
                    background_R=(np.nan if self.R is None else self.R.loc[s, a]),
                    background_peak_deg=(np.nan if self.peak is None else self.peak.loc[s, a])))
        return pd.DataFrame(rows)


def _empty(dpx_cuts, aspect_cuts):
    return pd.DataFrame(np.nan, index=list(dpx_cuts), columns=list(aspect_cuts))


def accuracy_surface(paired, *, aspect_cuts=DEFAULT_ASPECT_CUTS, dpx_cuts=DEFAULT_DPX_CUTS,
                     model_angle="angle_sam", human_angle="h_angle",
                     aspect_col="aspect_sam", dpx_col="dpx_sam", min_n=MIN_CELL_N,
                     max_size_ratio=None, human_dpx_col="h_dpx"):
    """Median ``|Δangle|`` between model and human, over the (aspect, size) grid.

    ``paired`` needs one row per boulder matched between the producer and a human tracer.
    Selection uses the **model's** aspect/size (see module docstring); the human columns are only
    ever the comparison.

    ⚠ **What "matched" means upstream, because it is looser than it sounds.**
    ``data/sam2_validation/run_t35_val.py:107`` pairs a detection to a human polygon by **centroid
    proximity alone** — greedy nearest within ``max(0.75 * h_dpx, 6.0)`` px, one-to-one, with **no
    overlap, size or shape test**. So nothing structurally prevents a fragment pairing with a large
    human polygon.

    Measured, it mostly holds: **99.6%** of pairs agree in diameter within 2x (80.5% within 25%), and
    the median ratio is **1.119** — SAM reads ~12% larger, rising to **1.21** below 5 px and falling
    to **1.00** above 20 px. But the 0.4% that fail are **catastrophic per pair**: median |Δangle|
    **40.8°** against 5.92° overall. Medians absorb that, so filtering moves the surface by only
    ~0.3° and leaves the aspect trend identical — which is why ``max_size_ratio`` defaults to ``None``
    (comparable with the archive) rather than being imposed.

    ``max_size_ratio=2.0`` drops pairs whose ``model/human`` diameter ratio falls outside
    ``[1/2, 2]``. Use it when a *per-boulder* statistic is wanted rather than a median, where a 0.4%
    tail at 40° does matter.
    """
    need = [model_angle, human_angle, aspect_col, dpx_col]
    d = paired.dropna(subset=need)
    if max_size_ratio is not None and human_dpx_col in d:
        r = d[dpx_col] / d[human_dpx_col]
        d = d[(r >= 1.0 / max_size_ratio) & (r <= max_size_ratio)]
    med = _empty(dpx_cuts, aspect_cuts)
    cnt = _empty(dpx_cuts, aspect_cuts)
    dang = axial_abs_delta(d[model_angle].to_numpy(), d[human_angle].to_numpy())
    asp, dpx = d[aspect_col].to_numpy(), d[dpx_col].to_numpy()
    for a in aspect_cuts:
        for s in dpx_cuts:
            sel = (asp > a) & (dpx >= s)
            n = int(sel.sum())
            cnt.loc[s, a] = n
            med.loc[s, a] = float(np.median(dang[sel])) if n >= min_n else np.nan
    return med, cnt


def artifact_surface(control, *, aspect_cuts=DEFAULT_ASPECT_CUTS, dpx_cuts=DEFAULT_DPX_CUTS,
                     angle_col="angle_sam", aspect_col="aspect_sam", dpx_col="dpx_sam",
                     min_n=MIN_CELL_N):
    """Grid-fraction **excess over isotropic** on fabric-free control ground.

    This is the axis the fabric gate actually cares about: not whether individual angles are right,
    but whether the surviving population still carries the cardinal signature that a fabric claim
    would be confused with.
    """
    need = [angle_col, aspect_col, dpx_col]
    d = control.dropna(subset=need)
    out = _empty(dpx_cuts, aspect_cuts)
    cnt = _empty(dpx_cuts, aspect_cuts)
    ang, asp, dpx = d[angle_col].to_numpy(), d[aspect_col].to_numpy(), d[dpx_col].to_numpy()
    for a in aspect_cuts:
        for s in dpx_cuts:
            sel = (asp > a) & (dpx >= s)
            n = int(sel.sum())
            cnt.loc[s, a] = n
            out.loc[s, a] = (float(grid_fraction(pd.Series(ang[sel]))) - ISOTROPIC_GRID
                             if n >= min_n else np.nan)
    return out, cnt


def resultant_surface(control, *, aspect_cuts=DEFAULT_ASPECT_CUTS, dpx_cuts=DEFAULT_DPX_CUTS,
                      angle_col="angle_sam", aspect_col="aspect_sam", dpx_col="dpx_sam",
                      min_n=MIN_CELL_N):
    """Background **resultant length R** and its peak, over the grid.

    This is the statistic the fabric gate actually consumes — it tests a candidate's resultant
    against the control background's, projected onto direction — so it is the surface that decides
    how high the bar sits. Grid fraction is reported alongside it for continuity with the archive,
    but it is the *worse* statistic for this population: measured on the control sites the excess is
    a **broad lean centred 13-20° off E-W** (per-site circular-mean peaks 69.2 / 75.1 / 74.4 / 51.4 /
    77.2°), not a pixel-grid snap — only **5.9%** of the ±10° excess lies within 0.5° of a cardinal,
    and the observed/uniform ratio is near-constant across the 0.5-10° bands. So grid fraction counts
    the *shoulder* of something that is not cardinal, and a rising grid fraction under a tightening
    aspect cut mostly means "the broad lean got stronger", not "the grid artifact got worse".
    """
    from .fabric_gate import axial_resultant

    need = [angle_col, aspect_col, dpx_col]
    d = control.dropna(subset=need)
    R = _empty(dpx_cuts, aspect_cuts)
    peak = _empty(dpx_cuts, aspect_cuts)
    ang, asp, dpx = d[angle_col].to_numpy(), d[aspect_col].to_numpy(), d[dpx_col].to_numpy()
    for a in aspect_cuts:
        for s in dpx_cuts:
            sel = (asp > a) & (dpx >= s)
            if int(sel.sum()) < min_n:
                continue
            z = axial_resultant(ang[sel])
            R.loc[s, a] = float(abs(z))
            peak.loc[s, a] = float(np.rad2deg(np.angle(z)) / 2 % 180)
    return R, peak


def cardinal_concentration_profile(angles_deg, bands=(0.5, 1.0, 2.0, 5.0, 10.0)):
    """Observed-vs-uniform ratio at a ladder of distances from the nearest cardinal.

    Separates a **pixel-grid snap** from a **broad directional lean**: a snap piles the excess into
    the smallest band, so the ratio falls steeply as the band widens; a broad lean gives a
    near-constant ratio. Used to show that the control-site excess is the latter.
    """
    a = np.asarray(angles_deg, dtype=float)
    a = a[np.isfinite(a)] % 180.0
    dc = np.minimum(np.minimum(a, np.abs(a - 90.0)), np.abs(a - 180.0))
    rows = []
    for b in bands:
        obs, exp = float((dc <= b).mean()), b / 90.0
        rows.append(dict(band_deg=b, observed=obs, uniform=exp,
                         ratio=obs / exp if exp else np.nan, excess=obs - exp))
    return pd.DataFrame(rows)


def rose_agreement(paired, *, group=None, model_angle="angle_sam", human_angle="h_angle",
                   select="model", aspect_min=1.35, dpx_min=8.0, min_n=100):
    """Compare a producer's aggregate rose against the human reference **on the same rocks**.

    Splits an aggregate lean into the part that is *real* and the part the producer *manufactures*:

    * **peak direction** — if the producer and the human agree, the direction is a property of the
      ground, not of the masker.
    * **resultant length R** — if the producer's R exceeds the human's, it is exaggerating the
      strength of a real preference.

    That decomposition is what makes a control-site "null" interpretable. Measured on the labelled
    scenes: peaks agree to **0.9-7.8° per scene**, while SAM's R runs above the human's everywhere and
    by **2.1x on Mars** (0.335 vs 0.157, i.e. +0.178) — consistent with the archive's Mars ΔR of
    +0.134 [+0.05, +0.20]. So a control-site R is an **over-estimate** of the real background, which
    makes a gate calibrated on it **conservative rather than permissive** — the safe direction.

    ``select`` picks the selection column: ``"model"`` cuts on the producer's own aspect/size (what an
    operator can do at map time), ``"human"`` on the reference's (the sensitivity check). Selection
    happens **once**, and both roses are then read on that one set — this is a matched comparison, not
    the differential-selection error, which conditioned each producer on its own aspect separately.
    """
    from .fabric_gate import axial_resultant

    a_col, d_col = ("aspect_sam", "dpx_sam") if select == "model" else ("h_aspect", "h_dpx")
    d = paired.dropna(subset=[model_angle, human_angle, a_col, d_col])
    d = d[(d[a_col] > aspect_min) & (d[d_col] >= dpx_min)]
    keys = [group] if isinstance(group, str) else (list(group) if group else [])
    groups = d.groupby(keys) if keys else [((), d)]
    rows = []
    for key, g in groups:
        if len(g) < min_n:
            continue
        zm, zh = axial_resultant(g[model_angle].to_numpy()), axial_resultant(g[human_angle].to_numpy())
        pm = float(np.rad2deg(np.angle(zm)) / 2 % 180)
        ph = float(np.rad2deg(np.angle(zh)) / 2 % 180)
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,))) if keys else {}
        rec.update(n=len(g), R_model=abs(zm), R_human=abs(zh),
                   R_inflation=(abs(zm) / abs(zh) if abs(zh) > 0 else np.nan),
                   dR=abs(zm) - abs(zh), peak_model=pm, peak_human=ph,
                   abs_dpeak=abs((pm - ph + 90) % 180 - 90))
        rows.append(rec)
    return pd.DataFrame(rows)


def trust_surface(paired, control, *, aspect_cuts=DEFAULT_ASPECT_CUTS, dpx_cuts=DEFAULT_DPX_CUTS,
                  min_n=MIN_CELL_N, accuracy_source="", artifact_source="", **kw):
    """All surfaces on one grid. See :class:`TrustSurface`."""
    acc, acc_n = accuracy_surface(paired, aspect_cuts=aspect_cuts, dpx_cuts=dpx_cuts,
                                  min_n=min_n, **kw)
    art, art_n = artifact_surface(control, aspect_cuts=aspect_cuts, dpx_cuts=dpx_cuts, min_n=min_n)
    R, peak = resultant_surface(control, aspect_cuts=aspect_cuts, dpx_cuts=dpx_cuts, min_n=min_n)
    return TrustSurface(accuracy=acc, accuracy_n=acc_n, artifact=art, artifact_n=art_n,
                        R=R, peak=peak,
                        aspect_cuts=tuple(aspect_cuts), dpx_cuts=tuple(dpx_cuts),
                        accuracy_source=accuracy_source, artifact_source=artifact_source)


def recommend_cut(surface, *, max_abs_dangle, max_grid_excess, min_n=200):
    """Cheapest admissible cut: the one keeping most boulders while meeting both tolerances.

    "Cheapest" is deliberate. Tightening a cut always improves both statistics and always costs
    yield, so without a yield objective the answer is trivially "the strictest cell" — which is how
    a threshold ends up set at its own sampling noise. This returns the **loosest** cut that still
    satisfies the stated tolerances, and reports what was rejected so the choice is inspectable.
    """
    long = surface.as_long()
    ok = long[(long.median_abs_dangle <= max_abs_dangle)
              & (long.grid_excess <= max_grid_excess)
              & (long.n_artifact >= min_n)
              & long.median_abs_dangle.notna()
              & long.grid_excess.notna()]
    if ok.empty:
        return {"found": False, "reason": "no cell meets both tolerances",
                "tolerances": {"max_abs_dangle": max_abs_dangle,
                               "max_grid_excess": max_grid_excess, "min_n": min_n},
                "best_accuracy": float(long.median_abs_dangle.min(skipna=True)),
                "best_grid_excess": float(long.grid_excess.min(skipna=True))}
    best = ok.sort_values(["n_artifact"], ascending=False).iloc[0]
    return {"found": True, "aspect_min": float(best.aspect_min), "dpx_min": float(best.dpx_min),
            "median_abs_dangle": float(best.median_abs_dangle),
            "grid_excess": float(best.grid_excess), "n_artifact": int(best.n_artifact),
            "n_accuracy": int(best.n_accuracy), "n_admissible_cells": int(len(ok)),
            "tolerances": {"max_abs_dangle": max_abs_dangle,
                           "max_grid_excess": max_grid_excess, "min_n": min_n}}
