"""The Goal-4 fabric gate: may this map cell's boulder rose be claimed as geological fabric?

Until now the gate existed only as a string in an artifact —
``"R_candidate > max(R_control_max_upper95, sqrt(-ln(0.05)/n_candidate))"`` — with its own defects
recorded beside it (`data/review/control_site_null.json`, `gate_form_caveats`). This module makes it
executable and fixes the four (review item G4-07):

1. **A scalar R cannot tell a fabric aligned with the artifact axis from one orthogonal to it.**
   SAM imprints a confirmed *cardinal* signature — grid fraction above isotropic at 5/5 control
   sites — while the diagonal fraction sits at or below uniform. So a rose peaking at 45° is far
   better evidence than the same R peaking at 0°/90°, and a single number scores them identically.
   Fixed by testing the candidate's resultant **projected onto its own direction** against a null
   built the same way: the null's own cardinal lean then raises the bar for cardinal candidates and
   lowers it for diagonal ones, automatically and without anyone choosing a per-direction threshold.

2. **Applied per cell at alpha = 0.05, the gate manufactures fabric.** A map is thousands of cells,
   so ~5% clear by construction. Fixed with Benjamini-Hochberg FDR across the whole map.

3. **No minimum effective n.** The ``sqrt(-ln(0.05)/n)`` term only exceeds the absolute bar for
   n < 16, so in practice the old form was a flat bar with no small-sample protection — and
   small-sample noise is exactly what discredited the retracted R ~ 0.35 bar (8/22/103/41/12
   boulders per site). Fixed with an explicit ``min_n_eff`` on the **clustering-corrected** count.

4. **An i.i.d. null understates variance because boulders cluster by tile.** Fixed by resampling
   whole tiles: the null inherits the candidate's clustering instead of assuming it away.

**Axial convention.** Orientations are axial (mod 180°), so everything is done in doubled-angle
space, ``z = mean(exp(2i*theta))``. There, ``Re(z)`` carries the **cardinal** content (theta ~ 0 or
90 map to doubled directions 0 and 180, i.e. the real axis) and ``Im(z)`` the **diagonal** content
(45 and 135 map to 90 and 270). ``R = |z|`` is the usual resultant length. This is the same
doubled-angle mean the review settled on as the estimator for peaks — *not* ``refine_peak``, whose
mode-of-smoothed-density is bandwidth-dependent and was found to drive a headline on its own.

⚠ **The cut travels with the number.** ``R_control_max`` is strongly conditional on the aspect cut
(0.330/0.443/0.509/0.539/0.642 at aspect 1.0/1.35/1.5/1.7/2.0), so a candidate and its null must be
read at one stated cut. :func:`fabric_gate` records the cut it was given and refuses to compare
against a null built at a different one.

⚠ **Scope of the shipped null:** Mars, incidence 44.8-63.4°, producer ``sam256``. It is not
established for the Moon or for other incidence ranges.
"""
from dataclasses import dataclass, field

import numpy as np

#: Default floor on the clustering-corrected sample size. Below this a rose is not interpretable:
#: the isotropic expectation of R is itself ~sqrt(pi)/(2*sqrt(n)) ~ 0.16 at n = 30, which is already
#: a third of the control-site maximum.
DEFAULT_MIN_N_EFF = 30

#: Bootstrap draws for the tile-resampled null. 2000 gives a p-value resolution of 5e-4, enough for
#: BH at the alpha values a map uses.
DEFAULT_N_BOOT = 2000


def axial_resultant(angles_deg):
    """Doubled-angle mean resultant ``z`` for axial data. ``|z|`` is R; see module docstring."""
    a = np.asarray(angles_deg, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return complex(np.nan, np.nan)
    return complex(np.mean(np.exp(2j * np.deg2rad(a))))


def resultant_length(angles_deg):
    return float(abs(axial_resultant(angles_deg)))


def cardinal_diagonal_split(z):
    """``(cardinal, diagonal)`` magnitudes of a doubled-angle resultant.

    Reported for interpretation, not used by the test: the test projects onto the candidate's own
    direction, which handles every intermediate angle rather than just the two special axes.
    """
    return abs(z.real), abs(z.imag)


def effective_n(n, design_effect):
    """Clustering-corrected count. ``design_effect`` > 1 means tiles are positively correlated."""
    if not np.isfinite(design_effect) or design_effect <= 0:
        return float(n)
    return float(n) / float(design_effect)


def tile_design_effect(angles_deg, tile_ids):
    """Variance inflation of the resultant from tile clustering, measured on the data itself.

    Ratio of the tile-resampled variance of R to the i.i.d.-resampled variance. The control-site
    measurement put this at 0.75-1.35, i.e. clustering is real but modest; values below 1 are kept
    rather than clamped, because pretending a benign design effect is 1.0 would be a free penalty.
    """
    a = np.asarray(angles_deg, dtype=float)
    t = np.asarray(tile_ids)
    ok = np.isfinite(a)
    a, t = a[ok], t[ok]
    if a.size < 2 or len(np.unique(t)) < 2:
        return 1.0
    rng = np.random.default_rng(0)
    n_boot = 400
    clustered = np.array([abs(axial_resultant(_resample_tiles(a, t, rng)[0])) for _ in range(n_boot)])
    iid = np.array([abs(axial_resultant(rng.choice(a, a.size, replace=True))) for _ in range(n_boot)])
    v_iid = float(np.var(iid))
    if v_iid <= 0:
        return 1.0
    return float(np.var(clustered) / v_iid)


def _resample_tiles(angles, tile_ids, rng, n_tiles=None):
    """Draw whole tiles with replacement — the tiling-preserving resample (defect 4)."""
    uniq, inv = np.unique(tile_ids, return_inverse=True)
    k = len(uniq) if n_tiles is None else int(n_tiles)
    picks = rng.integers(0, len(uniq), size=k)
    idx = [np.flatnonzero(inv == p) for p in picks]
    if not idx:
        return np.empty(0), 0
    out = np.concatenate(idx)
    return angles[out], k


@dataclass
class ControlNull:
    """Fabric-free background, as tile-resampled draws of the doubled-angle resultant.

    ``draws`` is complex: keeping the direction (not just R) is what makes the gate direction-aware,
    because a cardinal candidate is then compared against a background that also leans cardinal.
    """
    draws: np.ndarray
    cut: dict
    n_per_draw: np.ndarray = field(default_factory=lambda: np.empty(0))
    sites: tuple = ()

    def __len__(self):
        return len(self.draws)


@dataclass
class ControlPool:
    """Raw fabric-free control ground, from which per-cell nulls are built on demand.

    Prefer this over a single pre-built :class:`ControlNull`: the null has to be matched to the
    candidate on **n as well as tile count**. A null carrying more boulders than the cell is a
    tighter distribution than the cell's own sampling variability, and the gate then reads ordinary
    noise as fabric — measured, that mismatch alone passed ~19% of fabric-free cells.
    """
    angles_deg: np.ndarray
    tile_ids: np.ndarray
    cut: dict
    sites: tuple = ()


def build_control_null(angles_deg, tile_ids, *, cut, n_tiles, n_target=None,
                       n_boot=DEFAULT_N_BOOT, sites=(), seed=0):
    """Tile-resampled null from pooled fabric-free control ground.

    ``n_tiles`` matches the candidate's tile count so the null inherits comparable clustering, and
    ``n_target`` matches its boulder count — control tiles are not necessarily as dense as the
    candidate's, so resampling the right *number of tiles* does not by itself give the right
    *number of boulders*. Both must match or the comparison is not like for like.
    """
    a = np.asarray(angles_deg, dtype=float)
    t = np.asarray(tile_ids)
    ok = np.isfinite(a)
    a, t = a[ok], t[ok]
    rng = np.random.default_rng(seed)
    draws, ns = np.empty(n_boot, dtype=complex), np.empty(n_boot, dtype=int)
    for i in range(n_boot):
        samp, _ = _resample_tiles(a, t, rng, n_tiles=n_tiles)
        if n_target is not None and samp.size > n_target:
            samp = rng.choice(samp, int(n_target), replace=False)
        draws[i] = axial_resultant(samp)
        ns[i] = samp.size
    return ControlNull(draws=draws, cut=dict(cut), n_per_draw=ns, sites=tuple(sites))


def _bucket(n, n_tiles):
    """Coarse (n, n_tiles) key so a map of thousands of cells reuses a handful of nulls."""
    return (int(round(n / max(1, 10 ** int(np.log10(max(n, 1))) / 10.0))), int(n_tiles))


def directional_pvalue(z, null_draws):
    """P(background resultant reaches this far **in the candidate's own direction**).

    This is the direction-awareness (defect 1). Projecting the null onto the candidate's direction
    means a candidate pointing at a cardinal is judged against the background's cardinal lean —
    which is large — while one pointing at 45° is judged against the background's diagonal content,
    which is at or below uniform. No per-direction threshold has to be chosen by hand.
    """
    if not np.isfinite(z.real) or not np.isfinite(z.imag) or abs(z) == 0:
        return 1.0
    u = z / abs(z)
    proj = (null_draws * np.conj(u)).real
    return float((np.sum(proj >= abs(z)) + 1) / (len(null_draws) + 1))   # add-one: never reports 0


def benjamini_hochberg(pvalues, alpha=0.05):
    """BH step-up. Returns a boolean array of rejections (defect 2)."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passing = np.flatnonzero(ranked <= thresh)
    if passing.size:
        out[order[: passing[-1] + 1]] = True
    return out


def fabric_gate(cells, null, *, cut, alpha=0.05, min_n_eff=DEFAULT_MIN_N_EFF, fdr=True,
                design_effect=None):
    """Decide, for every cell on a map, whether a fabric claim is admissible.

    ``cells``: iterable of ``(cell_id, angles_deg, tile_ids)``.
    ``null``: a :class:`ControlPool` (preferred — per-cell nulls matched on n and tile count) or a
    pre-built :class:`ControlNull`, at the **same cut** — enforced, because R_control_max is
    strongly cut-conditional and comparing across cuts is how the retracted bar happened.

    ``design_effect``: pass a measured scalar to apply one clustering correction across the map.
    Leave ``None`` to estimate it per cell, which is more faithful but costs ~800 resamples per
    cell — on a map of thousands, measure it once on representative ground and pass it.

    Returns a list of dicts, one per cell. ``claimable`` is the answer; ``reason`` says why not.
    """
    if dict(cut) != dict(null.cut):
        raise ValueError(
            f"candidate cut {dict(cut)} != null cut {dict(null.cut)}. R_control_max moves "
            "0.330->0.642 across aspect 1.0->2.0, so a candidate and its null must be read at one "
            "stated cut (control_site_null.json, gate_form_caveats).")

    pool = null if isinstance(null, ControlPool) else None
    cache = {}

    def null_for(n, n_tiles):
        """Per-cell null, matched on n and tile count, cached by coarse bucket."""
        if pool is None:
            return null.draws
        key = _bucket(n, n_tiles)
        if key not in cache:
            cache[key] = build_control_null(
                pool.angles_deg, pool.tile_ids, cut=pool.cut, n_tiles=max(1, n_tiles),
                n_target=n, n_boot=DEFAULT_N_BOOT, sites=pool.sites, seed=abs(hash(key)) % 2**31
            ).draws
        return cache[key]

    rows = []
    for cell_id, angles, tiles in cells:
        a = np.asarray(angles, dtype=float)
        finite = np.isfinite(a)
        a = a[finite]
        t = np.asarray(tiles)[finite]
        z = axial_resultant(a)
        if design_effect is not None:
            deff = float(design_effect)
        else:
            deff = tile_design_effect(a, t) if a.size else 1.0
        n_eff = effective_n(a.size, deff)
        card, diag = cardinal_diagonal_split(z) if a.size else (np.nan, np.nan)
        n_tiles = int(len(np.unique(t))) if a.size else 0
        p = directional_pvalue(z, null_for(a.size, n_tiles)) if a.size else 1.0
        rows.append(dict(cell_id=cell_id, n=int(a.size), n_tiles=n_tiles,
                         design_effect=round(float(deff), 4), n_eff=round(n_eff, 1),
                         R=round(float(abs(z)), 4),
                         peak_deg=round(float(np.rad2deg(np.angle(z)) / 2 % 180), 2) if a.size else np.nan,
                         cardinal_component=round(float(card), 4),
                         diagonal_component=round(float(diag), 4),
                         p_directional=round(p, 5),
                         underpowered=bool(n_eff < min_n_eff)))

    # FDR over the cells that are powered enough to be tested at all. Including underpowered cells
    # in the correction would penalise the rest for tests that were never going to be informative.
    testable = [i for i, r in enumerate(rows) if not r["underpowered"]]
    rejected = np.zeros(len(rows), dtype=bool)
    if testable:
        pv = np.array([rows[i]["p_directional"] for i in testable])
        rej = benjamini_hochberg(pv, alpha) if fdr else (pv <= alpha)
        rejected[testable] = rej
    for i, r in enumerate(rows):
        r["claimable"] = bool(rejected[i])
        r["reason"] = ("" if r["claimable"] else
                       f"n_eff {r['n_eff']} < {min_n_eff}" if r["underpowered"] else
                       "does not clear the control-site background in its own direction"
                       + (" after FDR" if fdr else ""))
    return rows
