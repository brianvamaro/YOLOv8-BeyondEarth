"""House style for the presentation-grade orientation figures (``notebooks/Results_summary``).

Why this module exists
----------------------
The Goal-2 figures accumulated across a dozen notebooks and analysis scripts, each with its own
inlined rose code, its own colours, and its own auto-scaled radial axis. That is fine for an
investigation log and fatal for a figure someone reads cold: two roses side by side with different
radial limits are **not** comparable, and a rose with no stated size/aspect cut is not
interpretable at all (the artifact is size-dependent, so the subset *is* half the claim).

Two rules are therefore enforced **in code**, not by discipline:

1. **Every rose/histogram states its minimum aspect ratio and minimum size.** ``aspect_min`` and
   ``dpx_min`` are keyword-only arguments *without defaults* on :func:`orientation_panel` and
   :func:`orientation_grid` — omitting one is a ``TypeError``, and both are rendered into the
   panel/figure captions automatically. There is no code path that draws an unlabelled rose.
2. **Multi-panel figures share their axes.** :func:`orientation_grid` computes one radial limit
   across every rose and one y-limit across every count histogram, so bar lengths are directly
   comparable, and prints ``n`` in each panel title so unequal sample sizes are visible rather than
   hidden by the shared scale.

Colour
------
Colours are not hand-picked. The categorical slots below are the validated default instance of the
Anthropic data-viz palette, and the assignment was checked with the **all-pairs** gate (roses are
small multiples — any two panels can sit side by side, so the harder pairlist applies) under
Machado-Oliveira-Fernandes 2009 protanopia/deuteranopia at severity 1.0, ΔE = Euclidean in OKLab
×100:

- producers ``{yolo, reference, sam}`` → worst-pair CVD ΔE **9.2** (target 8), normal-vision **24.0**
  (floor 15);
- fusion arms ``{arm_256, arm_512, arm_sr2}`` → same triple, same numbers;
- OBB appendix ``{reference, obb_d0, obb_ra}`` → CVD ΔE **9.2**, normal-vision **27.6**.

Two consequences worth knowing before editing:

- ``sam``/``arm_sr2``/``obb_ra`` (aqua) sits at **2.74:1** against a white surface, below the 3:1
  mark minimum. That is a WARN, not a pass, and it obligates a relief channel — which is why every
  panel here carries a text title naming its producer and printing ``n``. Do not remove those labels.
- The **sun axis is drawn in ink, not in a series hue**, deliberately. Orange would have been the
  obvious choice and it collapses against aqua under deuteranopia (CVD ΔE 5.6); ink is ΔE 32 from
  every series colour. It is further distinguished by its dash pattern and a direct label, so it can
  never be misread as a fourth producer.
- Series colours are **1.1–2.2:1** against mid-grey imagery, i.e. invisible. Outlines drawn on a
  raster therefore *must* go through :func:`outline_effects`, which adds a contrasting casing.

Statistics are never re-implemented here — :mod:`YOLOv8BeyondEarth.orientation` owns them
(:func:`~YOLOv8BeyondEarth.orientation.grid_fraction`,
:func:`~YOLOv8BeyondEarth.orientation.diagonal_fraction`,
:func:`~YOLOv8BeyondEarth.orientation.asymmetry_ratio`,
:func:`~YOLOv8BeyondEarth.orientation.half_plane_fraction`,
:func:`~YOLOv8BeyondEarth.orientation.refine_peak`, …). This module only draws.

Azimuth convention throughout, matching :func:`~YOLOv8BeyondEarth.orientation.ellipse_angle180`:
axial (mod 180°), North = 0°, East = 90°, clockwise.
"""
from __future__ import annotations

import numpy as np

from .orientation import (asymmetry_ratio, axial_distance, bootstrap_peak_ci, diagonal_fraction,
                          grid_fraction, grid_fraction_uniform, half_plane_fraction, refine_peak)

# --- palette ---------------------------------------------------------------------------------
# Validated default categorical instance (light mode). Slot numbers are the palette's own order;
# the *assignment* of entity -> slot is what was validated all-pairs (see the module docstring).
PALETTE = {
    # producers / maskers — the §1-§4 figures
    "yolo": "#2a78d6",        # slot 1 blue   — the baseline producer (YOLOv8-seg mask)
    "sam": "#1baf7a",         # slot 3 aqua   — the fix (SAM2 re-mask)
    "human": "#eb6834",       # slot 2 orange — the reference (hand-drawn polygons)
    "truth": "#eb6834",       # same slot as `human`: on synthetic data, known truth PLAYS the
                              # reference role, so the two never appear in one panel set and the
                              # 3-slot all-pairs cap is never exceeded.
    "proxy": "#4a3aa7",       # slot 7 violet — rock-tracing proxy (illumination appendix only)
    # multi-scale fusion arms — §5. `arm_256` shares slot 1 with `yolo` on purpose: the 256 arm
    # *is* the production YOLO pass, so blue keeps meaning "the production detection". `arm_512` takes
    # violet rather than the reference orange because the fusion gallery draws the arms *and* the
    # human mapping in one figure, and two entities must never share a hue.
    # {blue, violet, aqua} + orange validates all-pairs (worst CVD dE 9.2, normal-vision 16.3).
    "arm_256": "#2a78d6",
    "arm_512": "#4a3aa7",
    "arm_sr2": "#1baf7a",
    # OBB angle head — appendix A. `obb_d0` takes violet rather than the intuitive red: red against
    # the orange reference fails both CVD (ΔE 5.6) and normal-vision (7.1) gates.
    "obb_d0": "#4a3aa7",
    "obb_ra": "#1baf7a",
}

INK = "#0b0b0b"          # text-primary; also the sun-axis annotation
INK_SOFT = "#52514e"     # text-secondary: captions, stat lines
SURFACE = "#ffffff"
GRID_GREY = "#c9c8c4"    # recessive grid / axes
NULL_GREY = "#6f6e6a"    # the uniform-null reference line, an annotation and never a series

# Uniform (isotropic) null for a +/-10 deg two-axis fraction: 2*10/90 = 0.222. Re-exported from
# `orientation` so callers state the null next to the measurement without importing both modules.
UNIFORM_FRACTION = grid_fraction_uniform(10.0)

# Sun-axis annotation: ink + a dash pattern + a direct label = three channels, no hue.
_SUN_DASH = (0, (7, 2, 1.5, 2))


def color_of(color):
    """Resolve a :data:`PALETTE` key (``"yolo"``) or pass a literal matplotlib colour through.

    Every drawing helper in this module goes through this, so callers can name the *entity* rather
    than the hex everywhere — which is the point of having a validated assignment at all.
    """
    return PALETTE.get(color, color)


def outline_effects(linewidth=1.6, casing="white", casing_extra=1.8):
    """Path effects that make a coloured outline legible **on imagery**.

    Every series colour in :data:`PALETTE` sits at 1.1-2.2:1 contrast against mid-grey regolith
    (measured), i.e. effectively invisible where it matters most — so a boulder outline drawn
    straight onto a raster is not readable. This wraps the stroke in a contrasting casing, the
    standard cartographic fix, and is **required** for any outline plotted over an image.

    Use as ``ax.plot(x, y, color=PALETTE["sam"], lw=1.6, path_effects=outline_effects())``.
    """
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=linewidth + casing_extra, foreground=casing)]


# --- captions ---------------------------------------------------------------------------------

def subset_caption(*, aspect_min, dpx_min, extra=None, res_m=None, dm_min=None):
    """The mandatory subset label: ``"aspect > 1.35 · d_eq >= 8 px · ..."``.

    ``aspect_min`` / ``dpx_min`` are required because an orientation distribution is meaningless
    without them: the grid artifact is carried by small, marginally-elongated boulders, so a rose
    quoted without its cuts cannot be compared to anything. ``dpx_min`` may be ``None`` for the
    deliberate *no-size-floor* variants, which then say so explicitly rather than staying silent.
    ``res_m`` (m/px), when given, adds the metric equivalent of the pixel floor.

    ``dm_min`` is a **metric** floor applied in addition to the pixel one. Cross-scene figures need it:
    a pixel floor selects a different physical population at each ground resolution, so comparing
    scenes on pixels alone compares different rocks. It is captioned rather than silently applied.
    """
    parts = [f"aspect > {aspect_min:g}"]
    if dpx_min is None:
        parts.append("no size floor")
    else:
        px = f"d$_{{eq}}$ ≥ {dpx_min:g} px"
        if res_m and not dm_min:
            px += f" (≥ {dpx_min * res_m:.1f} m)"
        parts.append(px)
    if dm_min:
        parts.append(f"and ≥ {dm_min:g} m")
    if extra:
        parts.append(str(extra))
    return "  ·  ".join(parts)


def _stat_line(a, keys, sun_axis=None):
    """Compact per-panel statistics, computed by :mod:`orientation` (never re-derived here)."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)] % 180.0
    if not a.size:
        return ""
    out = []
    for k in keys:
        if k == "diag":
            out.append(f"diag {diagonal_fraction(a):.2f}")
        elif k == "grid":
            out.append(f"grid {grid_fraction(a):.2f}")
        elif k == "chir":
            out.append(f"135/45 {asymmetry_ratio(a):.2f}")
        elif k == "flower":
            out.append(f"$f_{{low}}$ {half_plane_fraction(a):.2f}")
        elif k == "peak":
            out.append(f"peak {refine_peak(a):.0f}°")
        elif k == "peakci":
            # A per-scene peak is only meaningful with its uncertainty: on a small, broad rose the
            # bootstrap interval can span most of the circle, and a bare peak number hides that.
            _, lo, hi = bootstrap_peak_ci(a, n_boot=300)
            w = (hi - lo)
            out.append(f"peak {refine_peak(a):.0f}° [{lo:.0f}–{hi:.0f}]"
                       + ("  ⚠ no power" if w > 90 else ""))
        elif k == "dsun" and sun_axis is not None:
            out.append(f"|peak−sun| {axial_distance(refine_peak(a), sun_axis):.0f}°")
        else:
            raise ValueError(f"unknown stat key {k!r}")
    return "  ".join(out)


# --- primitives -------------------------------------------------------------------------------

def rose(ax, angles, *, color, bins=36, rmax=None, uniform_ref=True, show_rticks=False):
    """Axial rose (density) on a polar Axes. Returns the max bin density, for shared scaling.

    Mirrors each azimuth to ``theta`` and ``theta + 180`` for the conventional symmetric look, so
    the plotted density sums to 1 over the full circle. North up, clockwise. ``rmax`` forces the
    radial limit (this is how :func:`orientation_grid` makes panels comparable); pass ``None`` on a
    first pass to discover it. ``uniform_ref`` draws the isotropic null as a dashed circle — the
    reference the whole artifact argument is made against. ``show_rticks`` prints the radial density
    values; :func:`orientation_grid` sets it on the first panel only, so the shared scale is legible
    exactly once instead of cluttering every rose.
    """
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)] % 180.0
    edges = np.linspace(0.0, 2 * np.pi, 2 * bins + 1)
    counts, _ = np.histogram(np.deg2rad(np.concatenate([a, a + 180.0])), bins=edges)
    dens = counts / counts.sum() if counts.sum() else counts.astype(float)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(edges[:-1], dens, width=np.diff(edges), align="edge",
           color=color_of(color), edgecolor=INK, linewidth=0.25, alpha=0.85, zorder=2)
    if uniform_ref:
        th = np.linspace(0, 2 * np.pi, 361)
        ax.plot(th, np.full_like(th, 1.0 / (2 * bins)), ls=(0, (4, 3)), lw=0.9,
                color=NULL_GREY, zorder=3)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["N", "", "E", "", "S", "", "W", ""], fontsize=7, color=INK_SOFT)
    ax.grid(color=GRID_GREY, lw=0.5)
    ax.set_axisbelow(True)
    if rmax:
        ax.set_ylim(0, rmax)
    if show_rticks:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(3, prune="lower"))
        ax.tick_params(axis="y", labelsize=5.5, colors=INK_SOFT, pad=-1)
        ax.set_rlabel_position(247.5)
    else:
        ax.set_yticklabels([])
    return float(dens.max()) if dens.size else 0.0


def count_hist(ax, angles, *, color, bins=36, ymax=None, uniform_ref=True):
    """Linear histogram of azimuths in **counts** over [0, 180). Returns the max bin count.

    Counts, not density, because the audience asked to see the number of boulders directly: a rose
    hides sample size behind normalisation, and "SAM's rose is flatter" means something different at
    n = 96 than at n = 26,000. The dashed line is the count a uniform population would put in each
    bin (``n / bins``), which makes "above/below isotropic" readable per bin.
    """
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)] % 180.0
    edges = np.linspace(0.0, 180.0, bins + 1)
    counts, _ = np.histogram(a, bins=edges)
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
           color=color_of(color), edgecolor=INK, linewidth=0.25, alpha=0.85, zorder=2)
    if uniform_ref and a.size:
        ax.axhline(a.size / bins, ls=(0, (4, 3)), lw=0.9, color=NULL_GREY, zorder=3)
    ax.set_xlim(0, 180)
    ax.set_xticks([0, 45, 90, 135, 180])
    for v in (45, 90, 135):
        ax.axvline(v, color=GRID_GREY, lw=0.5, zorder=1)
    ax.tick_params(labelsize=7, colors=INK_SOFT, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_GREY)
    if ymax:
        ax.set_ylim(0, ymax)
    return int(counts.max()) if counts.size else 0


def draw_sun_axis(ax_rose=None, ax_hist=None, *, sun_axis, label="sun", rmax=None):
    """Mark the illumination (sun) axis on a rose and/or a histogram.

    Drawn in **ink with a dash pattern and a direct label**, never in a series hue (see the module
    docstring). Read it as a *reference axis* the viewer can check the distribution against — it is
    not an attribution: the illumination shaping this project measured is sub-degree per boulder,
    and the sun is *not* established as the cause of the residual aggregate lean.
    """
    if ax_rose is not None:
        r = rmax or ax_rose.get_ylim()[1]
        t = np.deg2rad(sun_axis)
        ax_rose.plot([t, t + np.pi], [r, r], color=INK, lw=1.4, ls=_SUN_DASH,
                     solid_capstyle="butt", zorder=5)
        ax_rose.text(t, r * 1.13, label, fontsize=6.5, color=INK,
                     ha="center", va="center", zorder=6)
    if ax_hist is not None:
        ax_hist.axvline(sun_axis % 180.0, color=INK, lw=1.4, ls=_SUN_DASH, zorder=5)
        ax_hist.annotate(label, xy=(sun_axis % 180.0, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(2, -8), textcoords="offset points", fontsize=6.5, color=INK)


# --- panels & grids ----------------------------------------------------------------------------

def orientation_panel(ax_rose, ax_hist, angles, *, color, label, aspect_min, dpx_min,
                      sun_axis=None, sun_label="sun", rmax=None, ymax=None, bins=36,
                      stats=("diag", "grid"), note=None, res_m=None, show_rticks=False,
                      show_xlabel=True, rescale_hist=False, dm_min=None):
    """One producer: a density rose over a count histogram, fully self-labelling.

    ``aspect_min`` and ``dpx_min`` are **required** — see the module docstring. They are rendered
    into the panel caption, together with ``n``, so the panel remains interpretable when it is
    lifted out of the figure and pasted into a slide.

    ``color`` may be a :data:`PALETTE` key or any matplotlib colour. ``rmax``/``ymax`` force the
    shared limits; :func:`orientation_grid` supplies them.

    ``rescale_hist=True`` opts this panel *out* of the shared count axis and stamps a visible
    "y-axis rescaled ×N" badge on it. Use it only for the genuine case it exists for: a producer
    with far fewer detections than its neighbours over the same ground (human labels vs the model on
    a same-area comparison), where a shared axis would squash the panel to an unreadable strip. The
    badge is not optional — an axis that differs silently is the failure mode this module exists to
    prevent.

    Returns ``(max_rose_density, max_hist_count)`` so a caller doing its own two-pass scaling can
    collect them.
    """
    c = color_of(color)
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    d = rose(ax_rose, a, color=c, bins=bins, rmax=rmax, show_rticks=show_rticks)
    m = count_hist(ax_hist, a, color=c, bins=bins, ymax=None if rescale_hist else ymax)
    if rescale_hist:
        # A rescaled panel is a break in the figure's own comparability rule, so it declares
        # itself: the reader is told the axis differs and by how much, rather than being left to
        # infer "flat" from a squashed bar. Used where a producer legitimately has far fewer
        # detections than its neighbours over the same ground (e.g. human labels vs the model).
        factor = (ymax / (ax_hist.get_ylim()[1] or 1)) if ymax else None
        ax_hist.text(0.98, 0.94, "y-axis rescaled" + (f" ×{factor:.0f}" if factor and factor >= 1.5
                                                      else ""),
                     transform=ax_hist.transAxes, ha="right", va="top", fontsize=6.2, color=INK,
                     bbox=dict(fc="#fff6ec", ec=PALETTE["human"], lw=0.6, pad=1.6))
    if sun_axis is not None:
        draw_sun_axis(ax_rose, ax_hist, sun_axis=sun_axis, label=sun_label, rmax=rmax)
    # The statistics ride in the rose title, not under the histogram: below the histogram they
    # collided with its x-label (seen on a rendered draft). Title order is identity -> sample size
    # -> measurement, so a panel lifted into a slide still says what it is and how big it is.
    sl = _stat_line(a, stats, sun_axis)
    ax_rose.set_title("\n".join(x for x in (label, f"n = {a.size:,}", sl) if x),
                      fontsize=9, color=INK, pad=9)
    if show_xlabel:
        ax_hist.set_xlabel("azimuth from North (°)", fontsize=7.5, color=INK_SOFT, labelpad=1)
    cap = subset_caption(aspect_min=aspect_min, dpx_min=dpx_min, res_m=res_m, dm_min=dm_min)
    ax_hist.text(0.5, -0.44 if show_xlabel else -0.18,
                 "\n".join(x for x in (cap, note) if x), fontsize=6.6,
                 color=INK_SOFT, ha="center", va="top", transform=ax_hist.transAxes)
    return d, m


def _wrap(text, width):
    """Wrap a title/footer, preserving any newlines the caller put in deliberately.

    Long single-line titles are the reason early drafts rendered squashed: ``bbox_inches="tight"``
    widens the canvas to fit an over-wide title, so every panel shrinks. Wrapping keeps the figure's
    own geometry authoritative.
    """
    import textwrap
    if not text:
        return text
    return "\n".join("\n".join(textwrap.wrap(line, width)) if line.strip() else line
                     for line in text.split("\n"))


def _rose_max(angles, bins):
    """Max axial-rose bin density, without drawing (the discovery pass for shared radial limits)."""
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)] % 180.0
    if not a.size:
        return 0.0
    c, _ = np.histogram(np.concatenate([a, a + 180.0]), bins=np.linspace(0, 360, 2 * bins + 1))
    return float(c.max() / c.sum())


def _limits(rows, bins, share):
    """Discover the shared rose/histogram limits without drawing: ``share`` in {figure, row, None}.

    Returns ``(rmax_per_row, ymax_per_row)``, each a list aligned with ``rows`` (entries may be
    ``None`` meaning "autoscale"). Densities and counts are computed with plain numpy here rather
    than by rendering throwaway axes — cheaper, and it keeps the discovery pass free of side effects.
    """
    def stats_of(p):
        if p.get("rescale_hist"):
            return _rose_max(p["angles"], bins), 0   # excluded from the shared count limit
        a = np.asarray(p["angles"], float)
        a = a[np.isfinite(a)] % 180.0
        if not a.size:
            return 0.0, 0
        c2, _ = np.histogram(np.concatenate([a, a + 180.0]), bins=np.linspace(0, 360, 2 * bins + 1))
        c1, _ = np.histogram(a, bins=np.linspace(0, 180, bins + 1))
        return float(c2.max() / c2.sum()), int(c1.max())

    per_row = [[stats_of(p) for p in row] for row in rows]
    if share == "figure":
        rd = max(d for row in per_row for d, _ in row)
        yc = max(c for row in per_row for _, c in row)
        return [rd * 1.08 or None] * len(rows), [yc * 1.12 or None] * len(rows)
    if share == "row":
        return ([max(d for d, _ in row) * 1.08 or None for row in per_row],
                [max(c for _, c in row) * 1.12 or None for row in per_row])
    return [None] * len(rows), [None] * len(rows)


def orientation_grid(rows, *, aspect_min, dpx_min, row_labels=None, suptitle=None, bins=36,
                     stats=("diag", "grid"), share_rose="figure", share_hist="row",
                     panel_w=2.8, panel_h=4.6, res_m=None, footer_extra=None):
    """Build a rows x cols grid of :func:`orientation_panel`\\ s with **shared axes**.

    ``rows`` is a list of rows; each row is a list of panel dicts with keys ``angles``, ``label``,
    ``color`` and optionally ``sun_axis``, ``sun_label``, ``note``, ``stats``, ``res_m``,
    ``rescale_hist``, and per-panel ``aspect_min`` / ``dpx_min`` overrides (used when one figure
    deliberately shows two different cuts — the caption then follows the panel). Rows are typically
    datasets (real scene / synthetic uniform truth / another body) and columns producers.

    **Sharing is split by axis type, on purpose.** The rose radial axis is a *density* — dimensionless
    and therefore comparable no matter how many boulders a dataset holds — so it defaults to
    ``share_rose="figure"``: one radial limit everywhere, which is what makes "YOLO's spokes are
    taller than SAM's" a legitimate read across rows. The histogram is in *counts* (the audience
    asked to see boulder numbers directly), and counts are only comparable within a dataset: forcing
    one count axis across a 15,000-boulder row and an 1,800-boulder row flattens the small row into
    an unreadable strip. So ``share_hist`` defaults to ``"row"`` — comparable exactly where
    comparison is meaningful — and the footer states which sharing was used. Pass
    ``share_hist="figure"`` when rows really do have comparable n, or ``None`` to autoscale.

    ``aspect_min`` / ``dpx_min`` are required and captioned on every panel.
    Returns ``(fig, axes)`` where ``axes[i][j] == (ax_rose, ax_hist)``.
    """
    import matplotlib.pyplot as plt
    if not rows or not all(rows):
        raise ValueError("`rows` must be a non-empty list of non-empty panel lists")
    for s, nm in ((share_rose, "share_rose"), (share_hist, "share_hist")):
        if s not in ("figure", "row", None):
            raise ValueError(f"{nm} must be 'figure', 'row' or None (got {s!r})")
    ncols = max(len(r) for r in rows)
    nrows = len(rows)
    rmax, _ = _limits(rows, bins, share_rose)
    _, ymax = _limits(rows, bins, share_hist)

    # Vertical budget in INCHES, not figure fractions: a fraction that works for a 3-row figure
    # collides for a 1-row one (measured on rendered drafts). Reserve fixed space for the suptitle
    # above and for the last row's caption + figure footer below, then convert to fractions.
    width = panel_w * ncols + 1.0
    sup = _wrap(suptitle, max(58, int(11 * width))) if suptitle else None
    head = (0.55 + 0.20 * (sup.count("\n") + 1) if sup else 0.0) + 0.50
    # The last row's per-panel caption hangs ~0.65" below its histogram, and the figure footer can
    # run to three wrapped lines, so the bottom reserve has to cover both (measured on drafts).
    foot = 1.30
    height = panel_h * nrows + head + foot
    fig = plt.figure(figsize=(width, height))
    outer = fig.add_gridspec(nrows, 1, hspace=0.66, top=1 - head / height, bottom=foot / height,
                             left=0.095, right=0.98)
    axes = []
    for i, row in enumerate(rows):
        inner = outer[i].subgridspec(2, ncols, height_ratios=[2.15, 1.0], hspace=0.55, wspace=0.28)
        axrow = []
        for j, p in enumerate(row):
            axr = fig.add_subplot(inner[0, j], projection="polar")
            axh = fig.add_subplot(inner[1, j])
            # A panel may carry its own cuts (e.g. a no-floor / floored pair in one figure). The
            # caption then reports the PANEL's cuts, not the figure's — otherwise the caption lies,
            # which is precisely what this module exists to prevent.
            orientation_panel(axr, axh, p["angles"], color=p["color"], label=p["label"],
                              aspect_min=p.get("aspect_min", aspect_min),
                              dpx_min=p.get("dpx_min", dpx_min),
                              sun_axis=p.get("sun_axis"), sun_label=p.get("sun_label", "sun"),
                              rmax=rmax[i], ymax=ymax[i], bins=bins,
                              stats=p.get("stats", stats), note=p.get("note"),
                              res_m=p.get("res_m", res_m), show_rticks=(i == 0 and j == 0),
                              rescale_hist=p.get("rescale_hist", False))
            if j == 0:
                axh.set_ylabel("boulders", fontsize=7.5, color=INK_SOFT)
            elif not p.get("rescale_hist"):
                axh.set_yticklabels([])       # a rescaled panel keeps its ticks — it must be legible
            axrow.append((axr, axh))
        if row_labels:
            axrow[0][0].text(-0.34, 0.5, row_labels[i], transform=axrow[0][0].transAxes,
                             rotation=90, va="center", ha="center", fontsize=9.5, color=INK)
        axes.append(axrow)
    if sup:
        fig.suptitle(sup, fontsize=11.5, color=INK, y=1 - 0.12 / height, va="top")
    share_txt = {"figure": "shared across the whole figure", "row": "shared within each row",
                 None: "autoscaled per panel"}
    fig.text(0.5, 0.10 / height,
             _wrap(f"radial = density, {share_txt[share_rose]}  ·  bars = boulder counts, "
                   f"{share_txt[share_hist]}  ·  dashed = isotropic null  ·  "
                   f"{subset_caption(aspect_min=aspect_min, dpx_min=dpx_min)}"
                   + (f"  ·  {footer_extra}" if footer_extra else ""),
                   max(70, int(16 * width))),
             ha="center", va="bottom", fontsize=7.2, color=INK_SOFT)
    return fig, axes


def rose_grid(panels, *, aspect_min, dpx_min, dm_min=None, ncols=None, suptitle=None, bins=36,
              stats=("peak",), share=True, panel_w=1.95, panel_h=2.5, row_labels=None,
              footer_extra=None, share_min_n=200):
    """A compact roses-only grid (no histograms) for many-panel comparisons — same rules apply.

    Used where the point is the *shape across many scenes or sites* and a histogram per panel would
    make the figure unreadable (11 HiRISE scenes, 5 control sites x 2 producers). ``aspect_min`` and
    ``dpx_min`` are required exactly as in :func:`orientation_grid`, and the shared radial limit is
    computed across every panel so spoke lengths remain comparable — a grid of individually
    autoscaled roses invites precisely the wrong conclusion.

    ``panels`` is a flat list of panel dicts (``angles``, ``label``, ``color``, optional ``sun_axis``,
    ``note``, ``stats``); ``ncols`` defaults to a near-square arrangement. Returns ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt
    if not panels:
        raise ValueError("`panels` must be non-empty")
    n = len(panels)
    ncols = ncols or int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    # A rose's peak *density* scales like 1/n at small n: a 20-boulder panel puts 2 rocks in one bin
    # and reads 0.05, while a 4,000-boulder panel peaks near 0.02. Letting the smallest panel set the
    # shared radial limit therefore squashes every well-populated rose into a dot at the centre (seen
    # on a rendered draft). So panels below `share_min_n` are excluded from the shared limit and are
    # autoscaled with a visible badge instead — the same "declare it, never do it silently" rule as
    # the rescaled histograms.
    def _n_of(p):
        a = np.asarray(p["angles"], float)
        return int(np.isfinite(a).sum())

    big = [p for p in panels if _n_of(p) >= share_min_n] if share_min_n else panels
    rmax = ((max(_rose_max(p["angles"], bins) for p in (big or panels)) * 1.08) or None) \
        if share else None

    width = panel_w * ncols + 0.7
    sup = _wrap(suptitle, max(58, int(11 * width))) if suptitle else None
    head = (0.45 + 0.20 * (sup.count("\n") + 1) if sup else 0.0) + 0.45
    foot = 0.62
    height = panel_h * nrows + head + foot
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(nrows, ncols, top=1 - head / height, bottom=foot / height,
                          left=0.05, right=0.985, hspace=0.62, wspace=0.24)
    axes = []
    for i, p in enumerate(panels):
        ax = fig.add_subplot(gs[i // ncols, i % ncols], projection="polar")
        a = np.asarray(p["angles"], float)
        a = a[np.isfinite(a)]
        own = _rose_max(a, bins)
        off_scale = bool(rmax and a.size < (share_min_n or 0) and own * 1.08 > rmax)
        rose(ax, a, color=p["color"], bins=bins, rmax=None if off_scale else rmax,
             show_rticks=(i == 0 or off_scale))
        if off_scale:
            ax.text(0.5, -0.03, f"own radial scale (n<{share_min_n})", transform=ax.transAxes,
                    ha="center", va="top", fontsize=5.8, color=INK,
                    bbox=dict(fc="#fff6ec", ec=PALETTE["human"], lw=0.5, pad=1.2))
        # Only N and E are labelled: on a tightly-packed grid the W of one panel collides with the
        # E of its neighbour, and for an AXIAL rose S/W are redundant with N/E anyway.
        ax.set_xticklabels(["N", "", "E", "", "", "", "", ""], fontsize=7, color=INK_SOFT)
        if p.get("sun_axis") is not None:
            draw_sun_axis(ax, sun_axis=p["sun_axis"], label=p.get("sun_label", "sun"),
                          rmax=None if off_scale else rmax)
        sl = _stat_line(a, p.get("stats", stats), p.get("sun_axis"))
        ax.set_title("\n".join(x for x in (p["label"], f"n = {a.size:,}", sl, p.get("note")) if x),
                     fontsize=7.4, color=INK, pad=6)
        ax._subset_cut = subset_caption(aspect_min=p.get("aspect_min", aspect_min),
                                        dpx_min=p.get("dpx_min", dpx_min),
                                        dm_min=p.get("dm_min", dm_min))
        axes.append(ax)
    for i in range(n, nrows * ncols):
        fig.add_subplot(gs[i // ncols, i % ncols]).axis("off")
    if row_labels:
        for r, lab in enumerate(row_labels):
            axes[r * ncols].text(-0.34, 0.5, lab, transform=axes[r * ncols].transAxes, rotation=90,
                                 va="center", ha="center", fontsize=9, color=INK)
    if sup:
        fig.suptitle(sup, fontsize=11, color=INK, y=1 - 0.10 / height, va="top")
    fig.text(0.5, 0.07 / height,
             _wrap(f"radial = density, {'shared across all panels' if share else 'per panel'}  ·  "
                   f"dashed = isotropic null  ·  "
                   f"{subset_caption(aspect_min=aspect_min, dpx_min=dpx_min, dm_min=dm_min)}"
                   + (f"  ·  {footer_extra}" if footer_extra else ""), max(70, int(16 * width))),
             ha="center", va="bottom", fontsize=7.2, color=INK_SOFT)
    return fig, axes


def apply_rcparams():
    """Project-wide matplotlib defaults for the summary figures (recessive chrome, readable text)."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_SOFT, "axes.edgecolor": GRID_GREY,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "axes.titlesize": 10, "axes.labelsize": 8, "legend.fontsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.grid": False, "grid.color": GRID_GREY, "grid.linewidth": 0.5,
        "legend.frameon": False,
    })
