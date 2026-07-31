"""One-to-one matching and cluster-aware uncertainty (review items G2-05 and G2-06).

G2-05: comparisons against human truth used a nearest-centroid query, which is many-to-one.
Several references could all claim the same prediction, so a model that merged two neighbouring
boulders into one still scored full recall and its merged mask was counted twice in the area-ratio
distribution — the gate could not see the failure it exists to catch.

G2-06: `bootstrap_peak_ci` resampled individual boulders even when thousands of detections come
from a handful of tiles or scenes, treating every detection as independent evidence.
"""
import geopandas as gpd
import numpy as np
import pytest
import shapely

from YOLOv8BeyondEarth.matching import (
    max_cardinality_matching, min_cost_matching, unique_centroid_match)
from YOLOv8BeyondEarth.orientation import bootstrap_peak_ci, peak_ci_report, refine_peak
from YOLOv8BeyondEarth.orientation_validation import matched_area_ratio


# --- the matching primitives ------------------------------------------------------------------------

def test_unique_centroid_match_is_one_to_one():
    """Two references, one model detection between them. Only one may claim it."""
    refs = np.array([[0.0, 0.0], [1.0, 0.0]])
    model = np.array([[0.5, 0.0]])
    ri, mi = unique_centroid_match(refs, model, max_dist=2.0)
    assert len(ri) == 1 and len(mi) == 1
    assert len(set(ri.tolist())) == len(ri) and len(set(mi.tolist())) == len(mi)


def test_unique_centroid_match_prefers_the_closer_reference():
    refs = np.array([[0.0, 0.0], [10.0, 0.0]])
    model = np.array([[9.8, 0.0]])
    ri, _ = unique_centroid_match(refs, model, max_dist=20.0)
    assert ri.tolist() == [1]


def test_matching_is_maximum_cardinality_not_greedy():
    """The greedy counterexample. Greedy by distance pairs R0 with its nearest (M0) and leaves R1
    unmatched; the optimal pairing matches both."""
    refs = np.array([[0.0, 0.0], [0.4, 0.0]])
    model = np.array([[0.1, 0.0], [1.2, 0.0]])   # M1 is only reachable from R1
    ri, mi = unique_centroid_match(refs, model, max_dist=0.9)
    assert sorted(ri.tolist()) == [0, 1], "an achievable match was missed"
    assert sorted(mi.tolist()) == [0, 1]


def test_radius_gate_is_per_reference():
    refs = np.array([[0.0, 0.0], [10.0, 0.0]])
    model = np.array([[0.0, 3.0], [10.0, 3.0]])
    ri, _ = unique_centroid_match(refs, model, max_dist=np.array([5.0, 1.0]))
    assert ri.tolist() == [0]                    # the second reference's radius is too tight


def test_matching_handles_empty_inputs():
    empty = np.empty((0, 2))
    for a, b in ((empty, np.array([[0.0, 0.0]])), (np.array([[0.0, 0.0]]), empty), (empty, empty)):
        ri, mi = unique_centroid_match(a, b, max_dist=1.0)
        assert len(ri) == 0 and len(mi) == 0


def test_min_cost_matching_picks_the_cheaper_assignment():
    #   r0 -> c0 (10) or c1 (1);  r1 -> c1 (1) only.  Optimal: r0->c0, r1->c1  (total 11)
    ri, mi = min_cost_matching([0, 0, 1], [0, 1, 1], [10.0, 1.0, 1.0], 2, 2)
    assert dict(zip(ri.tolist(), mi.tolist())) == {0: 0, 1: 1}


def test_max_cardinality_matching_counts_rows():
    covered = max_cardinality_matching([0, 1, 1], [0, 0, 1], 3, 2)
    assert sorted(covered.tolist()) == [0, 1]     # row 2 has no edges


def test_component_decomposition_is_optimal_against_a_dense_solver():
    """The component split is the risky part of the implementation, so check it against a full
    dense assignment on many random problems: same cardinality, same total cost."""
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(0)
    for _ in range(120):
        n_r, n_c = int(rng.integers(1, 9)), int(rng.integers(1, 9))
        dense = rng.uniform(0.1, 1.0, size=(n_r, n_c))
        admissible = rng.random((n_r, n_c)) < 0.35
        if not admissible.any():
            continue
        rr, cc = np.nonzero(admissible)
        ri, ci = min_cost_matching(rr, cc, dense[rr, cc], n_r, n_c)
        assert len(set(ri.tolist())) == len(ri) and len(set(ci.tolist())) == len(ci)

        # reference: dense Hungarian with non-edges priced above any complete real alternative
        big = dense.max() * (min(n_r, n_c) + 1) + 1.0
        padded = np.where(admissible, dense, big)
        dr, dc = linear_sum_assignment(padded)
        real = padded[dr, dc] < big
        assert len(ri) == int(real.sum())
        assert float(dense[ri, ci].sum()) == pytest.approx(float(dense[dr[real], dc[real]].sum()),
                                                           abs=1e-9)


# --- the adversarial fixture the review names --------------------------------------------------------

def _gdf(boxes):
    return gpd.GeoDataFrame(geometry=[shapely.box(*b) for b in boxes], crs="EPSG:32631")


def test_two_references_cannot_reuse_one_prediction():
    """The G2-05 acceptance fixture: two adjacent human boulders, one merged model mask covering
    both. Recall must be 0.5, and the merged area must be counted once, not twice."""
    refs = _gdf([(0, 0, 2, 2), (2.2, 0, 4.2, 2)])
    merged = _gdf([(0, 0, 4.2, 2)])
    df, stats = matched_area_ratio(merged, refs, min_radius_m=5.0)

    assert stats["n_matched"] == 1
    assert stats["recall"] == pytest.approx(0.5)
    assert stats["n_ref_unmatched"] == 1
    assert df.model_index.is_unique and df.ref_index.is_unique
    assert len(df) == 1, "the merged mask was counted once per reference"


def test_matched_area_ratio_publishes_the_full_flow():
    refs = _gdf([(0, 0, 2, 2), (20, 20, 22, 22), (40, 40, 42, 42)])
    model = _gdf([(0.1, 0.1, 2.1, 2.1), (20, 20, 22, 22), (90, 90, 92, 92)])
    df, stats = matched_area_ratio(model, refs, min_radius_m=2.0)
    assert stats["n_ref"] == 3 and stats["n_model"] == 3
    assert stats["n_matched"] == 2
    assert stats["n_ref_unmatched"] == 1 and stats["n_model_unmatched"] == 1
    assert stats["recall"] == pytest.approx(2 / 3)
    assert stats["precision"] == pytest.approx(2 / 3)
    assert len(df) == 2


def test_perfect_correspondence_is_unchanged():
    """The common case must behave exactly as before: one-to-one already, ratio 1."""
    boxes = [(0, 0, 2, 2), (20, 20, 23, 23), (40, 40, 41, 41)]
    df, stats = matched_area_ratio(_gdf(boxes), _gdf(boxes), min_radius_m=1.0)
    assert stats["n_matched"] == 3 and stats["recall"] == pytest.approx(1.0)
    np.testing.assert_allclose(df.ratio.values, 1.0)


# --- cluster-aware uncertainty (G2-06) ----------------------------------------------------------------

def _clustered_angles(n_clusters=6, per_cluster=16, spread=3.0, offset=12.0, seed=0):
    """Angles that are tight within a tile and scattered between tiles — the real structure, and
    the one an object-level bootstrap is blind to."""
    rng = np.random.default_rng(seed)
    angles, groups = [], []
    for c in range(n_clusters):
        centre = 120.0 + offset * rng.normal()
        angles.append(centre + spread * rng.normal(size=per_cluster))
        groups.append(np.full(per_cluster, f"tile{c}"))
    return np.concatenate(angles) % 180.0, np.concatenate(groups)


def test_cluster_bootstrap_is_wider_than_the_object_bootstrap():
    """The point of G2-06. When boulders cluster by tile, resampling boulders reports an interval
    far narrower than the six independent tiles support."""
    angles, groups = _clustered_angles()
    _, lo_obj, hi_obj = bootstrap_peak_ci(angles, n_boot=300, seed=1)
    _, lo_clu, hi_clu = bootstrap_peak_ci(angles, n_boot=300, seed=1, groups=groups)
    assert (hi_clu - lo_clu) > 2.0 * (hi_obj - lo_obj)


def test_cluster_bootstrap_matches_object_bootstrap_when_every_boulder_is_its_own_cluster():
    angles, _ = _clustered_angles()
    groups = np.arange(len(angles))
    _, lo_obj, hi_obj = bootstrap_peak_ci(angles, n_boot=300, seed=3)
    _, lo_clu, hi_clu = bootstrap_peak_ci(angles, n_boot=300, seed=3, groups=groups)
    assert lo_clu == pytest.approx(lo_obj, abs=1.0) and hi_clu == pytest.approx(hi_obj, abs=1.0)


def test_group_length_is_validated():
    with pytest.raises(ValueError, match="groups"):
        bootstrap_peak_ci(np.arange(10.0), groups=np.arange(3))


def test_peak_ci_report_publishes_cluster_count_and_leave_one_out():
    angles, groups = _clustered_angles()
    rep = peak_ci_report(angles, groups=groups, n_boot=200)
    assert rep["n"] == len(angles) and rep["n_clusters"] == 6
    assert rep["cluster_bootstrap"] is True
    assert len(rep["leave_one_out"]) == 6
    assert rep["lo"] <= rep["peak"] <= rep["hi"]
    assert np.isfinite(rep["loo_spread"])
    # the peak with all clusters is close to the peak of the full sample
    assert rep["peak"] == pytest.approx(refine_peak(angles), abs=1e-9)


def test_peak_ci_report_flags_the_ungrouped_case():
    angles, _ = _clustered_angles()
    rep = peak_ci_report(angles, n_boot=100)
    assert rep["cluster_bootstrap"] is False and rep["n_clusters"] is None
