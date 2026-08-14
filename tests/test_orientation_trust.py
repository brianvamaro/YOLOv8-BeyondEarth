"""The aspect x size trust surface (queue item 9)."""
import numpy as np
import pandas as pd
import pytest

from YOLOv8BeyondEarth import orientation_trust as ot


def test_axial_delta_wraps_at_180_not_360():
    """179 deg and 1 deg are 2 deg apart for an axis, not 178."""
    assert ot.axial_abs_delta([179.0], [1.0])[0] == pytest.approx(2.0)
    assert ot.axial_abs_delta([0.0], [90.0])[0] == pytest.approx(90.0)
    assert ot.axial_abs_delta([10.0], [170.0])[0] == pytest.approx(20.0)
    assert (ot.axial_abs_delta(np.linspace(0, 360, 50), np.linspace(0, 360, 50)) == 0).all()


def _rose(peak_deg, kappa, n, seed=0):
    """n axial angles concentrated on `peak_deg` (von Mises in doubled-angle space)."""
    rng = np.random.default_rng(seed)
    return np.rad2deg(rng.vonmises(np.deg2rad(peak_deg) * 2, kappa, n)) / 2 % 180


def _paired(n, err_deg, aspect, dpx, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.uniform(0, 180, n)
    return pd.DataFrame({"h_angle": h,
                         "angle_sam": (h + rng.normal(0, err_deg, n)) % 180,
                         "aspect_sam": np.full(n, aspect), "dpx_sam": np.full(n, dpx)})


def test_accuracy_improves_as_the_cut_tightens_when_small_boulders_are_worse():
    """The surface must actually see a size-dependent error, which is its whole premise."""
    bad = _paired(400, 25.0, aspect=1.5, dpx=5.0, seed=1)     # small: noisy angles
    good = _paired(400, 3.0, aspect=1.5, dpx=12.0, seed=2)    # large: clean angles
    med, cnt = ot.accuracy_surface(pd.concat([bad, good], ignore_index=True),
                                   aspect_cuts=(1.0,), dpx_cuts=(0.0, 8.0), min_n=10)
    assert med.loc[0.0, 1.0] > med.loc[8.0, 1.0], "cutting out small boulders must reduce |dangle|"
    assert cnt.loc[0.0, 1.0] == 800 and cnt.loc[8.0, 1.0] == 400


def test_max_size_ratio_drops_the_catastrophically_mismatched_pairs():
    """Upstream matching is centroid-only, so a fragment can pair with a big human polygon.

    Rare (0.4% of real pairs) but ~40 deg wrong when it happens, so the guard has to exist even
    though the median barely moves.
    """
    good = _paired(500, 4.0, aspect=1.5, dpx=10.0, seed=60)
    good["h_dpx"] = 10.0
    junk = _paired(20, 45.0, aspect=1.5, dpx=10.0, seed=61)      # angle unrelated to the human
    junk["h_dpx"] = 40.0                                          # 4x size disagreement
    d = pd.concat([good, junk], ignore_index=True)

    med_all, cnt_all = ot.accuracy_surface(d, aspect_cuts=(1.0,), dpx_cuts=(0.0,), min_n=10)
    med_flt, cnt_flt = ot.accuracy_surface(d, aspect_cuts=(1.0,), dpx_cuts=(0.0,), min_n=10,
                                           max_size_ratio=2.0)
    assert cnt_all.loc[0.0, 1.0] == 520 and cnt_flt.loc[0.0, 1.0] == 500
    assert med_flt.loc[0.0, 1.0] <= med_all.loc[0.0, 1.0]


def test_no_size_filter_by_default_so_the_archive_stays_comparable():
    d = _paired(300, 5.0, aspect=1.5, dpx=10.0, seed=62)
    d["h_dpx"] = 100.0                                            # wildly mismatched on purpose
    _, cnt = ot.accuracy_surface(d, aspect_cuts=(1.0,), dpx_cuts=(0.0,), min_n=10)
    assert cnt.loc[0.0, 1.0] == 300, "default must not silently filter"


def test_thin_cells_are_nan_not_a_precise_looking_small_number():
    d = _paired(5, 3.0, aspect=2.5, dpx=20.0)
    med, cnt = ot.accuracy_surface(d, aspect_cuts=(2.0,), dpx_cuts=(16.0,), min_n=50)
    assert cnt.loc[16.0, 2.0] == 5 and np.isnan(med.loc[16.0, 2.0])


def _control(n, grid_frac_target, aspect, dpx, seed=0):
    """Angles with a controllable share inside +-10 deg of a cardinal."""
    rng = np.random.default_rng(seed)
    n_grid = int(round(n * grid_frac_target))
    cardinal = np.concatenate([rng.uniform(-10, 10, n_grid // 2) % 180,
                               rng.uniform(80, 100, n_grid - n_grid // 2)])
    rest = rng.uniform(10, 80, n - n_grid)          # strictly off-cardinal
    return pd.DataFrame({"angle_sam": np.concatenate([cardinal, rest]),
                         "aspect_sam": np.full(n, aspect), "dpx_sam": np.full(n, dpx)})


def test_artifact_surface_is_excess_over_isotropic_not_raw_grid_fraction():
    contaminated = _control(2000, 0.50, aspect=1.5, dpx=10.0, seed=3)
    art, _ = ot.artifact_surface(contaminated, aspect_cuts=(1.0,), dpx_cuts=(0.0,), min_n=10)
    assert art.loc[0.0, 1.0] == pytest.approx(0.50 - ot.ISOTROPIC_GRID, abs=0.03)


def test_an_isotropic_control_reads_about_zero_excess():
    rng = np.random.default_rng(4)
    iso = pd.DataFrame({"angle_sam": rng.uniform(0, 180, 20000),
                        "aspect_sam": 1.5, "dpx_sam": 10.0})
    art, _ = ot.artifact_surface(iso, aspect_cuts=(1.0,), dpx_cuts=(0.0,), min_n=10)
    assert abs(art.loc[0.0, 1.0]) < 0.015


def test_recommend_cut_returns_the_loosest_admissible_cell_not_the_strictest():
    """Tightening always improves both statistics, so without a yield objective the answer is
    trivially the strictest cell — which is how a threshold gets set at its own sampling noise."""
    paired = pd.concat([_paired(600, 4.0, 1.35, 6.0, 5), _paired(600, 4.0, 2.0, 16.0, 6)],
                       ignore_index=True)
    control = pd.concat([_control(4000, 0.24, 1.35, 6.0, 7), _control(4000, 0.24, 2.0, 16.0, 8)],
                        ignore_index=True)
    s = ot.trust_surface(paired, control, aspect_cuts=(1.0, 1.35, 2.0), dpx_cuts=(0.0, 6.0, 16.0),
                         min_n=50)
    rec = ot.recommend_cut(s, max_abs_dangle=10.0, max_grid_excess=0.05, min_n=100)
    assert rec["found"]
    # Every cell qualifies here, so it must pick the one that keeps the most boulders.
    assert (rec["aspect_min"], rec["dpx_min"]) == (1.0, 0.0), rec


def test_recommend_cut_says_so_when_nothing_qualifies():
    paired = _paired(400, 30.0, 1.5, 8.0, 9)
    control = _control(4000, 0.45, 1.5, 8.0, 10)
    s = ot.trust_surface(paired, control, aspect_cuts=(1.0,), dpx_cuts=(0.0,), min_n=50)
    rec = ot.recommend_cut(s, max_abs_dangle=5.0, max_grid_excess=0.01)
    assert not rec["found"] and "best_accuracy" in rec


def test_profile_tells_a_snap_from_a_broad_lean():
    """The distinction that reinterpreted the control-site excess.

    A snap piles the excess into the smallest band, so the observed/uniform ratio falls steeply as
    the band widens. A broad lean gives a near-constant ratio.
    """
    rng = np.random.default_rng(20)
    n = 40000
    # snap: 5% of rows pinned within 0.2 deg of a cardinal, rest uniform
    snap = np.concatenate([rng.uniform(-0.2, 0.2, n // 20) % 180, rng.uniform(0, 180, n)])
    # broad lean: von Mises about 90 deg, wide
    lean = _rose(90.0, 0.6, n, 21)

    ps = ot.cardinal_concentration_profile(snap)
    pl = ot.cardinal_concentration_profile(lean)
    # snap: ratio at 0.5 deg is far above the ratio at 10 deg
    assert ps.ratio.iloc[0] / ps.ratio.iloc[-1] > 4
    # broad lean: roughly flat
    assert 0.5 < pl.ratio.iloc[0] / pl.ratio.iloc[-1] < 2.0


def test_resultant_surface_reports_R_and_a_direction():
    """R is what the fabric gate consumes, so the surface has to expose it."""
    control = pd.concat([_control(3000, 0.25, 1.5, 10.0, 30)], ignore_index=True)
    control["angle_sam"] = _rose(70.0, 0.8, len(control), 31)
    R, peak = ot.resultant_surface(control, aspect_cuts=(1.0,), dpx_cuts=(0.0,), min_n=50)
    assert 0.2 < R.loc[0.0, 1.0] < 0.6
    assert abs(peak.loc[0.0, 1.0] - 70.0) < 6.0


def _paired_rose(n, peak, kappa_model, kappa_human, seed=0):
    """Both producers leaning the same way, the model more strongly than the human."""
    return pd.DataFrame({
        "angle_sam": _rose(peak, kappa_model, n, seed),
        "h_angle": _rose(peak, kappa_human, n, seed + 1),
        "aspect_sam": 1.5, "dpx_sam": 10.0, "h_aspect": 1.5, "h_dpx": 10.0})


def test_rose_agreement_separates_a_shared_direction_from_an_inflated_strength():
    """The decomposition that made the control-site lean interpretable: same peak, bigger R."""
    d = _paired_rose(6000, peak=70.0, kappa_model=0.9, kappa_human=0.4, seed=40)
    r = ot.rose_agreement(d, min_n=100).iloc[0]
    assert r.abs_dpeak < 6.0, "a shared direction must show as agreeing peaks"
    assert r.R_model > r.R_human and r.R_inflation > 1.3, "and the model must read stronger"
    assert abs(r.peak_human - 70.0) < 8.0


def test_rose_agreement_reports_no_inflation_when_there_is_none():
    d = _paired_rose(6000, peak=70.0, kappa_model=0.6, kappa_human=0.6, seed=42)
    r = ot.rose_agreement(d, min_n=100).iloc[0]
    assert 0.8 < r.R_inflation < 1.25 and r.abs_dpeak < 6.0


def test_rose_agreement_groups_and_honours_the_selection_column():
    d = pd.concat([_paired_rose(400, 30.0, 0.9, 0.4, 50).assign(body="a"),
                   _paired_rose(400, 120.0, 0.9, 0.4, 60).assign(body="b")], ignore_index=True)
    out = ot.rose_agreement(d, group="body", min_n=100)
    assert set(out.body) == {"a", "b"} and len(out) == 2
    # Selecting on the human columns must still work and still find both groups.
    d.loc[d.body == "a", "aspect_sam"] = 1.0          # model would now reject group a
    assert len(ot.rose_agreement(d, group="body", select="model", min_n=100)) == 1
    assert len(ot.rose_agreement(d, group="body", select="human", min_n=100)) == 2


def test_as_long_covers_every_grid_cell():
    s = ot.trust_surface(_paired(300, 5.0, 1.5, 8.0), _control(3000, 0.25, 1.5, 8.0),
                         aspect_cuts=(1.0, 1.5), dpx_cuts=(0.0, 4.0, 8.0), min_n=50)
    assert len(s.as_long()) == 2 * 3
    assert set(s.as_long().columns) >= {"aspect_min", "dpx_min", "median_abs_dangle",
                                       "grid_excess", "n_accuracy", "n_artifact"}


def test_selection_uses_the_model_columns_and_never_the_human_ones():
    """At map time there are no human labels, so the cut has to be applied to what the model
    reports. A surface that secretly selected on `h_dpx` would not be usable on a real map."""
    d = _paired(400, 5.0, aspect=1.5, dpx=12.0, seed=11)
    d["h_dpx"] = 1.0            # human says everything is tiny
    d["h_aspect"] = 1.0         # and round
    med, cnt = ot.accuracy_surface(d, aspect_cuts=(1.2,), dpx_cuts=(8.0,), min_n=50)
    assert cnt.loc[8.0, 1.2] == 400, "selection must follow aspect_sam/dpx_sam, not h_*"
