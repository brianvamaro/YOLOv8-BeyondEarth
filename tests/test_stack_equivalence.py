"""Science-equivalence comparison between two compute stacks.

The bit-level parity check (`measure_stack_parity.py`) asks whether two catalogs are identical, which
across GPU architectures can only ever answer "no". This module asks whether the *published* numbers
move. These tests pin the two things that would make it useless: a comparison that reports agreement
when the science really has shifted, and a gate-crossing count that misses population churn.
"""
import numpy as np
import pytest

pytest.importorskip("geopandas")
import geopandas as gpd                                          # noqa: E402
from shapely.affinity import rotate, scale, translate            # noqa: E402
from shapely.geometry import Point                               # noqa: E402

from YOLOv8BeyondEarth import stack_equivalence as se            # noqa: E402

RES = 0.5


def _ellipse(x, y, a, b, angle_deg):
    """An ellipse polygon centred at (x, y) with semi-axes a, b, rotated CCW by angle_deg."""
    return rotate(scale(Point(x, y).buffer(1.0, quad_segs=64), a, b), angle_deg)


def _catalog(specs, score=0.5):
    geoms = [_ellipse(*s) for s in specs]
    return gpd.GeoDataFrame({"score": [score] * len(geoms)}, geometry=geoms, crs="EPSG:32631")


# --- paired deltas ----------------------------------------------------------------------------------

def test_identical_catalogs_report_zero_delta():
    specs = [(0, 0, 6, 3, 20), (50, 50, 8, 4, 70), (100, 0, 5, 4, 120)]
    a, b = _catalog(specs), _catalog(specs)
    r = se.paired_deltas(a, b, RES, n_workers=1)
    assert r["n_matched"] == 3
    assert r["unmatched_a"]["n"] == 0 and r["unmatched_b"]["n"] == 0
    assert r["dangle_median_deg"] == pytest.approx(0.0, abs=1e-6)
    assert r["ddiam_rel_median"] == pytest.approx(0.0, abs=1e-9)
    assert r["within_accepted_dangle"] is True


def test_a_real_rotation_is_detected_and_fails_the_inherited_bar():
    """The test must be able to FAIL. A 10 degree rotation is far outside the 2.69 deg tolerance."""
    specs = [(0, 0, 6, 3, 20), (50, 50, 8, 4, 70), (100, 0, 7, 3, 120)]
    rotated = [(x, y, p, q, ang + 10.0) for x, y, p, q, ang in specs]
    r = se.paired_deltas(_catalog(specs), _catalog(rotated), RES, n_workers=1)
    assert r["n_matched"] == 3                                  # same centroids, still matched
    assert r["dangle_median_deg"] == pytest.approx(10.0, abs=0.5)
    assert r["within_accepted_dangle"] is False


def test_a_missing_detection_is_reported_with_its_score():
    """Membership differences are the headline, and where they sit in score is the interpretation."""
    specs = [(0, 0, 6, 3, 20), (50, 50, 8, 4, 70)]
    a = _catalog(specs)
    a.loc[len(a)] = {"score": 0.1001, "geometry": _ellipse(200, 200, 5, 3, 10)}
    r = se.paired_deltas(a, _catalog(specs), RES, n_workers=1)
    assert r["unmatched_a"]["n"] == 1
    assert r["unmatched_a"]["score_max"] == pytest.approx(0.1001)
    assert r["unmatched_b"]["n"] == 0


def test_size_change_shows_up_in_the_diameter_delta():
    specs = [(0, 0, 6, 3, 20), (50, 50, 8, 4, 70)]
    bigger = [(x, y, p * 1.10, q * 1.10, ang) for x, y, p, q, ang in specs]
    r = se.paired_deltas(_catalog(specs), _catalog(bigger), RES, n_workers=1)
    assert r["ddiam_rel_median"] == pytest.approx(0.10, abs=0.01)


# --- the gate-crossing mechanism -------------------------------------------------------------------

def test_gate_crossing_catches_population_churn_a_summary_would_miss():
    """A boulder can sit either side of the aspect cut while every per-boulder delta looks tiny.

    That matters because `R_control_max` is strongly cut-conditional, so churn in the gated population
    is exactly what would move a fabric claim.
    """
    from YOLOv8BeyondEarth.orientation import attach_orientations
    # aspect just under vs just over 1.35, at a diameter comfortably past the 8 px floor
    a = attach_orientations(_catalog([(0, 0, 6.0, 4.55, 30)]), RES, n_workers=1)
    b = attach_orientations(_catalog([(0, 0, 6.0, 4.20, 30)]), RES, n_workers=1)
    assert a["aspect_ra"].iloc[0] < 1.35 < b["aspect_ra"].iloc[0]
    g = se.gate_crossing(a, b, RES, aspect_min=1.35, dpx_min=8.0)
    assert g["n_crossed"] == 1 and g["frac_crossed"] == pytest.approx(1.0)


def test_gate_crossing_is_zero_when_both_sides_agree():
    from YOLOv8BeyondEarth.orientation import attach_orientations
    specs = [(0, 0, 8, 4, 30), (40, 0, 9, 5, 60)]
    a = attach_orientations(_catalog(specs), RES, n_workers=1)
    g = se.gate_crossing(a, a.copy(), RES, aspect_min=1.35, dpx_min=8.0)
    assert g["n_crossed"] == 0 and g["n_in_a"] == g["n_in_b"]


# --- aggregates and the CI-based comparison --------------------------------------------------------

def test_aggregate_stats_reports_the_gate_population_and_isotropic_references():
    from YOLOv8BeyondEarth.orientation import attach_orientations
    rng = np.random.default_rng(0)
    specs = [(float(x) * 40, 0.0, 8.0, 4.0, float(ang))
             for x, ang in enumerate(rng.uniform(0, 180, 60))]
    df = attach_orientations(_catalog(specs), RES, n_workers=1)
    s = se.aggregate_stats(df, RES, already_oriented=True, n_boot=40)
    assert s["n"] == 60                                          # all elongated and well-resolved
    assert s["grid_fraction_isotropic"] == pytest.approx(2 * 20.0 / 180.0, abs=1e-9)
    assert 0.0 <= s["grid_fraction"] <= 1.0 and 0.0 <= s["R"] <= 1.0
    assert s["peak_ci"][0] <= s["peak_deg"] <= s["peak_ci"][1] or np.isnan(s["peak_deg"])


def test_indistinguishable_is_none_without_an_interval_and_false_when_far_apart():
    a = {"peak_deg": 40.0, "peak_ci": [38.0, 42.0]}
    b = {"peak_deg": 41.0, "peak_ci": [39.0, 43.0]}
    assert se.indistinguishable(a, b, "peak_deg") is True
    far = {"peak_deg": 120.0, "peak_ci": [118.0, 122.0]}
    assert se.indistinguishable(a, far, "peak_deg") is False
    assert se.indistinguishable({"peak_deg": 1.0}, {"peak_deg": 2.0}, "peak_deg") is None


# --- SFD --------------------------------------------------------------------------------------------

def test_sfd_bands_partition_the_population_exactly():
    from YOLOv8BeyondEarth.orientation import attach_orientations
    specs = [(float(i) * 40, 0.0, r, r * 0.6, 10.0) for i, r in enumerate([1.0, 2.0, 3.5, 6.0, 12.0])]
    df = attach_orientations(_catalog(specs), RES, n_workers=1)
    counts = se.sfd_counts(df, RES)
    below = int((df["diameter_m"] / RES < se.SFD_EDGES_PX[0]).sum())
    assert sum(counts.values()) + below == len(df)               # nothing lost, nothing double-counted


def test_compare_sfd_reports_a_shift():
    from YOLOv8BeyondEarth.orientation import attach_orientations
    small = attach_orientations(_catalog([(0, 0, 2.0, 1.5, 0)]), RES, n_workers=1)
    big = attach_orientations(_catalog([(0, 0, 8.0, 6.0, 0)]), RES, n_workers=1)
    df = se.compare_sfd(small, big, RES)
    assert df["delta"].sum() == 0                                # one moved out, one moved in
    assert (df["delta"] != 0).sum() == 2


def test_the_accepted_tolerance_is_the_documented_crop_jitter_figure():
    """The bar must stay traceable to the SAM crop-position jitter it was inherited from."""
    assert se.ACCEPTED_DANGLE_MEDIAN_DEG == pytest.approx(2.69)
