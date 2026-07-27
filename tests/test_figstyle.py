"""Contract tests for `figstyle` — the two figure rules the summary document promises.

These are not cosmetic tests. The presentation doc claims (a) every rose/histogram states its
minimum aspect ratio and minimum size, and (b) multi-panel orientation figures share their axes so
bar lengths are comparable. Both are properties of `figstyle`, so both are asserted here rather than
left to whoever edits a figure next.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from YOLOv8BeyondEarth import figstyle as fs


@pytest.fixture
def angles():
    rng = np.random.default_rng(0)
    return {
        "uniform": rng.uniform(0, 180, 500),
        "locked": np.r_[rng.normal(135, 4, 400) % 180, rng.uniform(0, 180, 100)],
        "small": rng.uniform(0, 180, 40),
    }


def _panel(**kw):
    fig = plt.figure()
    axr = fig.add_subplot(2, 1, 1, projection="polar")
    axh = fig.add_subplot(2, 1, 2)
    try:
        return fs.orientation_panel(axr, axh, **kw)
    finally:
        plt.close(fig)


# --- rule 1: the subset cuts can never be omitted ----------------------------------------------

def test_panel_requires_aspect_and_size(angles):
    """Omitting either cut must be a hard error, not a silently unlabelled figure."""
    base = dict(angles=angles["uniform"], color="yolo", label="YOLO")
    with pytest.raises(TypeError):
        _panel(**base, dpx_min=8)              # aspect_min missing
    with pytest.raises(TypeError):
        _panel(**base, aspect_min=1.35)        # dpx_min missing
    _panel(**base, aspect_min=1.35, dpx_min=8)  # both present -> fine


def test_grid_requires_aspect_and_size(angles):
    rows = [[dict(angles=angles["uniform"], color="yolo", label="YOLO")]]
    with pytest.raises(TypeError):
        fs.orientation_grid(rows, dpx_min=8)
    with pytest.raises(TypeError):
        fs.orientation_grid(rows, aspect_min=1.35)


def test_caption_states_both_cuts_and_n(angles):
    """The rendered text must actually contain the thresholds and the sample size."""
    fig = plt.figure()
    axr = fig.add_subplot(2, 1, 1, projection="polar")
    axh = fig.add_subplot(2, 1, 2)
    fs.orientation_panel(axr, axh, angles["uniform"], color="sam", label="SAM",
                         aspect_min=1.7, dpx_min=8)
    text = " ".join(t.get_text() for t in fig.findobj(plt.Text))
    assert "1.7" in text, "minimum aspect ratio missing from the panel"
    assert "8 px" in text, "minimum size missing from the panel"
    assert f"{angles['uniform'].size:,}" in text, "n missing from the panel title"
    plt.close(fig)


def test_caption_says_so_when_there_is_no_size_floor():
    """A no-floor variant must declare it explicitly rather than just leaving the cut unstated."""
    cap = fs.subset_caption(aspect_min=1.35, dpx_min=None)
    assert "no size floor" in cap
    assert "1.35" in cap


def test_caption_can_report_metres():
    cap = fs.subset_caption(aspect_min=1.35, dpx_min=8, res_m=0.5)
    assert "4.0 m" in cap


# --- rule 2: shared axes across every panel ----------------------------------------------------

def _rows(angles):
    """Two rows with deliberately mismatched n (500 vs 40) — the case sharing must handle."""
    return [[dict(angles=angles["locked"], color="yolo", label="YOLO"),
             dict(angles=angles["uniform"], color="sam", label="SAM")],
            [dict(angles=angles["small"], color="human", label="human"),
             dict(angles=angles["small"][:20], color="sam", label="SAM (few)")]]


def test_grid_default_shares_rose_across_figure_and_counts_within_row(angles):
    """The default: density comparable everywhere, counts comparable where counts mean something.

    A density is dimensionless so one radial limit is legitimate across rows; counts are not, and
    forcing one count axis over a 500- and a 40-boulder row would flatten the small row into an
    unreadable strip. So rose = figure-wide, histogram = per-row, and each is asserted here.
    """
    fig, axes = fs.orientation_grid(_rows(angles), aspect_min=1.35, dpx_min=8)
    flat = [p for row in axes for p in row]
    assert len(flat) == 4
    rlims = {round(axr.get_ylim()[1], 9) for axr, _ in flat}
    assert len(rlims) == 1, f"rose radial limits not shared figure-wide: {rlims}"
    assert rlims.pop() > 0
    for i, row in enumerate(axes):                       # within a row: identical
        ylims = {round(axh.get_ylim()[1], 9) for _, axh in row}
        assert len(ylims) == 1, f"row {i} histogram y-limits not shared: {ylims}"
    across = {round(row[0][1].get_ylim()[1], 9) for row in axes}
    assert len(across) == 2, "rows with 500 vs 40 boulders should not share a count axis"
    plt.close(fig)


def test_grid_can_share_counts_figure_wide(angles):
    fig, axes = fs.orientation_grid(_rows(angles), aspect_min=1.35, dpx_min=8,
                                    share_hist="figure")
    ylims = {round(axh.get_ylim()[1], 9) for row in axes for _, axh in row}
    assert len(ylims) == 1
    plt.close(fig)


def test_grid_share_none_leaves_panels_autoscaled(angles):
    """The escape hatch exists and is visibly different — so sharing is doing real work."""
    rows = [[dict(angles=angles["locked"], color="yolo", label="YOLO"),
             dict(angles=angles["uniform"], color="sam", label="SAM")]]
    fig, axes = fs.orientation_grid(rows, aspect_min=1.35, dpx_min=8,
                                    share_rose=None, share_hist=None)
    ylims = {round(axh.get_ylim()[1], 6) for _, axh in axes[0]}
    assert len(ylims) == 2, "a locked and a uniform panel should autoscale differently"
    plt.close(fig)


def test_grid_rejects_unknown_share_mode(angles):
    with pytest.raises(ValueError, match="share_hist"):
        fs.orientation_grid(_rows(angles), aspect_min=1.35, dpx_min=8, share_hist="all")
    with pytest.raises(ValueError, match="share_rose"):
        fs.orientation_grid(_rows(angles), aspect_min=1.35, dpx_min=8, share_rose=True)


def test_per_panel_cuts_are_captioned_per_panel(angles):
    """A figure showing two different size cuts must caption each panel with ITS cut.

    Regression: an early draft put the cut in the panel title and left the grid-level `dpx_min` in
    the caption, so a `>= 8 px` panel was captioned "no size floor" — a caption that lies is worse
    than no caption.
    """
    rows = [[dict(angles=angles["uniform"], color="yolo", label="no floor", dpx_min=None),
             dict(angles=angles["uniform"][:200], color="yolo", label="floored", dpx_min=8)]]
    fig, axes = fs.orientation_grid(rows, aspect_min=1.35, dpx_min=None)
    caps = [" ".join(t.get_text() for t in axh.texts) for _, axh in axes[0]]
    assert "no size floor" in caps[0]
    assert "8 px" in caps[1] and "no size floor" not in caps[1]
    plt.close(fig)


def test_rescaled_panel_declares_itself(angles):
    """Opting a low-n panel out of the shared count axis must be visible, never silent.

    This is the same-area case: over one footprint the model reports many more boulders than the
    humans mapped, so a shared count axis squashes the human panel flat. Rescaling it is allowed —
    rescaling it *quietly* is exactly the failure this module exists to prevent.
    """
    rows = [[dict(angles=angles["uniform"], color="yolo", label="YOLO"),
             dict(angles=angles["small"], color="human", label="human", rescale_hist=True)]]
    fig, axes = fs.orientation_grid(rows, aspect_min=1.35, dpx_min=8)
    (_, big), (_, small) = axes[0]
    assert small.get_ylim()[1] < big.get_ylim()[1], "the rescaled panel should use its own scale"
    badge = [t.get_text() for t in small.texts if "rescaled" in t.get_text()]
    assert badge, "a rescaled panel must carry a visible badge"
    assert "×" in badge[0], "the badge should state the factor"
    assert small.get_yticklabels()[0].get_text(), "a rescaled panel must keep readable y ticks"
    plt.close(fig)


def test_rose_grid_enforces_the_same_rules(angles):
    """The compact roses-only grid must not be a loophole around the labelling/sharing rules."""
    panels = [dict(angles=angles["locked"], label="scene A", color="sam", sun_axis=75.0),
              dict(angles=angles["uniform"], label="scene B", color="sam", sun_axis=120.0),
              dict(angles=angles["small"], label="scene C", color="sam")]
    with pytest.raises(TypeError):
        fs.rose_grid(panels, dpx_min=8)
    with pytest.raises(TypeError):
        fs.rose_grid(panels, aspect_min=1.35)
    with pytest.raises(ValueError):
        fs.rose_grid([], aspect_min=1.35, dpx_min=8)
    fig, axes = fs.rose_grid(panels, aspect_min=1.7, dpx_min=8)
    assert len({round(a.get_ylim()[1], 9) for a in axes}) == 1, "radial limits must be shared"
    text = " ".join(t.get_text() for t in fig.findobj(plt.Text))
    assert "1.7" in text and "8 px" in text
    assert all(f"n = {np.asarray(p['angles']).size:,}" in " ".join(
        t.get_text() for t in ax.texts + [ax.title]) for p, ax in zip(panels, axes))
    plt.close(fig)


def test_grid_footer_states_the_sharing_used(angles):
    """A reader must never have to guess whether two panels are comparable."""
    fig, _ = fs.orientation_grid(_rows(angles), aspect_min=1.35, dpx_min=8)
    text = " ".join(t.get_text() for t in fig.findobj(plt.Text))
    assert "shared across the whole figure" in text
    assert "shared within each row" in text
    plt.close(fig)


def test_grid_rejects_empty_input():
    with pytest.raises(ValueError):
        fs.orientation_grid([], aspect_min=1.35, dpx_min=8)
    with pytest.raises(ValueError):
        fs.orientation_grid([[]], aspect_min=1.35, dpx_min=8)


# --- primitives behave as the figures assume ---------------------------------------------------

def test_rose_is_axial_and_density_normalised(angles):
    """theta and theta+180 are one axis, and the plotted density sums to 1 over the full circle."""
    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    a = np.array([10.0, 190.0, 10.0])            # three azimuths, all the same axis
    d = fs.rose(ax, a, color="yolo", bins=36)
    heights = [p.get_height() for p in ax.patches]
    assert np.isclose(sum(heights), 1.0)
    assert np.isclose(d, max(heights))
    assert sum(h > 0 for h in heights) == 2, "an axial rose puts mass in exactly two mirrored bins"
    plt.close(fig)


def test_count_hist_is_counts_not_density(angles):
    """The audience asked for boulder numbers directly: bar heights must sum to n."""
    fig, ax = plt.subplots()
    m = fs.count_hist(ax, angles["uniform"], color="sam", bins=36)
    heights = [p.get_height() for p in ax.patches]
    assert sum(heights) == angles["uniform"].size
    assert m == max(heights)
    plt.close(fig)


def test_count_hist_handles_empty_input():
    fig, ax = plt.subplots()
    assert fs.count_hist(ax, np.array([]), color="yolo") == 0
    plt.close(fig)


def test_sun_axis_is_ink_not_a_series_colour(angles):
    """The sun line must never be mistakable for a fourth producer."""
    fig = plt.figure()
    axr = fig.add_subplot(2, 1, 1, projection="polar")
    axh = fig.add_subplot(2, 1, 2)
    fs.orientation_panel(axr, axh, angles["uniform"], color="sam", label="SAM",
                         aspect_min=1.35, dpx_min=8, sun_axis=75.3)
    series = set(fs.PALETTE.values())
    sun_lines = [ln for ln in axh.get_lines() + axr.get_lines()
                 if ln.get_linestyle() not in ("-", "solid") or ln.get_color() == fs.INK]
    assert sun_lines, "no sun-axis annotation was drawn"
    assert all(ln.get_color() not in series for ln in sun_lines)
    assert any(ln.get_color() == fs.INK for ln in sun_lines)
    plt.close(fig)


def test_palette_assignments_are_the_validated_triples():
    """Guard the CVD-validated entity->slot assignment against a well-meant colour 'tidy-up'.

    These triples were checked all-pairs under Machado-Oliveira-Fernandes protan/deutan at severity
    1.0 (worst-pair OKLab dE 9.2, normal-vision >= 24). Changing a hex here without re-running that
    check silently breaks colourblind readability, so the assignment is pinned.
    """
    assert fs.PALETTE["yolo"] == "#2a78d6"
    assert fs.PALETTE["sam"] == "#1baf7a"
    assert fs.PALETTE["human"] == "#eb6834"
    # truth plays the reference role on synthetic data, so it shares the reference slot and the
    # 3-slot all-pairs cap is never exceeded.
    assert fs.PALETTE["truth"] == fs.PALETTE["human"]
    # the production 256 arm *is* the YOLO detection pass, so blue keeps its meaning
    assert fs.PALETTE["arm_256"] == fs.PALETTE["yolo"]
    # the fusion gallery draws all three arms AND the human reference in one figure, so no arm may
    # share the reference hue
    arm_hues = {fs.PALETTE[k] for k in ("arm_256", "arm_512", "arm_sr2")}
    assert len(arm_hues) == 3, "the three fusion arms must be distinguishable"
    assert fs.PALETTE["human"] not in arm_hues
    # red against the orange reference fails both CVD and normal-vision gates -> violet
    assert fs.PALETTE["obb_d0"] == "#4a3aa7"
    assert fs.INK not in fs.PALETTE.values(), "the annotation ink must not double as a series hue"


def test_outline_effects_add_a_casing():
    """Series colours are ~1.1-2.2:1 on mid-grey imagery, so raster overlays need a casing."""
    eff = fs.outline_effects(linewidth=1.6)
    assert eff and len(eff) == 1
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color=fs.PALETTE["sam"], lw=1.6, path_effects=eff)
    fig.canvas.draw()          # a broken path-effect spec raises here, not silently
    plt.close(fig)


def test_uniform_fraction_matches_orientation_module():
    from YOLOv8BeyondEarth.orientation import grid_fraction_uniform
    assert fs.UNIFORM_FRACTION == grid_fraction_uniform(10.0)
    assert abs(fs.UNIFORM_FRACTION - 0.2222) < 1e-3
