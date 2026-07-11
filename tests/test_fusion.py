"""Unit tests for cross-scale fusion (Goal 3 phase 1) on synthetic GeoDataFrames.

Pins the calibrated DIoU cluster threshold (0.30, from
data/solutionDE/calibrate_fusion_threshold.py, 2026-07-10) and the tiny-box behaviour at that
threshold; exercises the three rules, truncation guard, provenance, audit table, and the
validation helpers. No GPU, no raster, no CRS needed (fusion is pure map-coordinate geometry).
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
import shapely

from YOLOv8BeyondEarth.fusion import (DEFAULT_CLUSTER_THR, fuse_predictions, pool_predictions,
                                      pairwise_box_metric, matched_recall, duplicate_rate,
                                      equivalent_diameter_px)


def make_gdf(boxes, scores=None, within=None, **cols):
    """Synthetic detection set: boxes = [(xmin, ymin, xmax, ymax), ...] in map units."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    n = len(boxes)
    g = gpd.GeoDataFrame({
        "score": np.full(n, 0.5) if scores is None else np.asarray(scores, dtype=float),
        "category_id": np.zeros(n, dtype=int),
        "category_name": ["boulder"] * n,
        "is_within_slice": np.ones(n, dtype=bool) if within is None else np.asarray(within),
        **cols,
    }, geometry=list(shapely.box(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])))
    return g


# --- metric + pinned threshold -------------------------------------------------------------------

def test_default_threshold_is_the_calibrated_one():
    # 0.30 from the three-arm calibration: true-pair DIoU p01 >= ~0.42 in every band,
    # distinct-neighbour p99 <= ~0.03 in the small bands. Changing this needs a recalibration.
    assert DEFAULT_CLUSTER_THR == 0.30


def test_metric_identical_and_disjoint_boxes():
    a = [[0, 0, 4, 4]]
    assert pairwise_box_metric(a, a, "diou")[0] == pytest.approx(1.0)
    assert pairwise_box_metric(a, a, "iou")[0] == pytest.approx(1.0)
    b = [[10, 10, 14, 14]]
    assert pairwise_box_metric(a, b, "iou")[0] == 0.0
    assert pairwise_box_metric(a, b, "diou")[0] < 0.0  # center-distance penalty


def test_diou_never_exceeds_iou():
    rng = np.random.default_rng(0)
    lo = rng.uniform(0, 100, size=(200, 2))
    wh = rng.uniform(1, 20, size=(200, 2))
    a = np.hstack([lo, lo + wh])
    lo2 = lo + rng.uniform(-5, 5, size=(200, 2))
    b = np.hstack([lo2, lo2 + wh])
    assert np.all(pairwise_box_metric(a, b, "diou") <= pairwise_box_metric(a, b, "iou") + 1e-12)


def test_tiny_box_regime_at_pinned_threshold():
    # The SFD-critical mode: 4-px boxes with small localization shifts (map res 1 unit/px).
    base = [0, 0, 4, 4]
    shift1 = [1, 0, 5, 4]   # 1-px shift: IoU 0.6 -> clusters at 0.30 (plain IoU 0.5 would too)
    shift2 = [2, 0, 6, 4]   # 2-px shift: IoU 1/3, DIoU ~0.256 -> just below 0.30 (documented
    #                         boundary; such pairs land in the duplicate-rate audit window)
    d1 = pairwise_box_metric([base], [shift1], "diou")[0]
    d2 = pairwise_box_metric([base], [shift2], "diou")[0]
    assert d1 == pytest.approx(0.6 - 1.0 / 41.0, abs=1e-9)
    assert d1 >= DEFAULT_CLUSTER_THR
    assert d2 == pytest.approx(1.0 / 3.0 - 4.0 / 52.0, abs=1e-9)
    assert d2 < DEFAULT_CLUSTER_THR


# --- pooling + provenance ------------------------------------------------------------------------

def test_pool_provenance_and_sr_key():
    gdfs = {512: make_gdf([[0, 0, 8, 8]]),
            256: make_gdf([[20, 0, 24, 4]]),
            (256, 2): make_gdf([[40, 0, 44, 4]])}
    pool = pool_predictions(gdfs)
    assert list(pool.columns).count("geometry") == 1
    got = pool.set_index("scale_key")[["slice_size", "sr_factor", "footprint"]]
    assert got.loc["512"].tolist() == [512, 1.0, 512.0]
    assert got.loc["(256, 2)"].tolist() == [256, 2.0, 128.0]  # SR arm = finest footprint
    assert (pool["src_index"] == 0).all()


def test_file_schema_columns_are_normalized():
    g = make_gdf([[0, 0, 4, 4]]).rename(columns={"category_id": "cat_id",
                                                 "category_name": "cat_name",
                                                 "is_within_slice": "isin_slice"})
    pool = pool_predictions({256: g})
    assert "is_within_slice" in pool.columns and "category_id" in pool.columns


def test_missing_truncation_flag_raises():
    g = make_gdf([[0, 0, 4, 4]]).drop(columns=["is_within_slice"])
    with pytest.raises(ValueError, match="is_within_slice"):
        pool_predictions({256: g})


# --- fusion rules --------------------------------------------------------------------------------

def test_scale_aware_keeps_finest_footprint():
    # same boulder at three scales; coarser scales score higher (the known score confound)
    gdfs = {512: make_gdf([[0, 0, 8, 8]], scores=[0.9]),
            256: make_gdf([[0.2, 0, 8.2, 8]], scores=[0.8]),
            (256, 2): make_gdf([[0, 0.2, 8, 8.2]], scores=[0.3])}
    fused, audit = fuse_predictions(gdfs, rule="scale_aware")
    assert len(fused) == 1
    assert fused.iloc[0]["scale_key"] == "(256, 2)"  # footprint 128 wins despite lowest score
    assert fused.iloc[0]["n_merged"] == 2
    assert len(audit) == 2
    assert audit["cross_scale"].all()
    assert (audit["keeper_scale_key"] == "(256, 2)").all()
    assert (audit["metric"] >= DEFAULT_CLUSTER_THR).all()


def test_score_rule_is_pooled_nms():
    gdfs = {512: make_gdf([[0, 0, 8, 8]], scores=[0.9]),
            256: make_gdf([[0.2, 0, 8.2, 8]], scores=[0.8])}
    fused, _ = fuse_predictions(gdfs, rule="score")
    assert len(fused) == 1 and fused.iloc[0]["scale_key"] == "512"  # highest score, scale-blind


def test_truncation_guard_demotes_clipped_members():
    # finest member is slice-truncated (is_within_slice=False, a footprint-edge keep) ->
    # scale_aware must keep the coarser intact mask instead
    gdfs = {256: make_gdf([[0, 0, 8, 8]], scores=[0.4]),
            (256, 2): make_gdf([[0.2, 0, 8.2, 8]], scores=[0.9], within=[False])}
    fused, audit = fuse_predictions(gdfs, rule="scale_aware")
    assert len(fused) == 1 and fused.iloc[0]["scale_key"] == "256"


def test_distinct_neighbours_both_survive():
    # abutting but distinct boulders (DIoU << 0.30) must NOT merge
    gdfs = {256: make_gdf([[0, 0, 4, 4]]),
            (256, 2): make_gdf([[4.5, 0, 8.5, 4]])}
    fused, audit = fuse_predictions(gdfs)
    assert len(fused) == 2 and len(audit) == 0


def test_no_transitive_chain_merging():
    # A~B and B~C exceed the threshold, A~C does not (DIoU(A,C) ~ 0.004): once B is suppressed
    # it must not drag C into A's cluster — connected components would; greedy must not
    a, c = [0, 0, 4, 4], [3, 0, 7, 4]
    b = [1.5, 0, 5.5, 4]  # DIoU vs both A and C ~ 0.406
    gdfs = {(256, 2): make_gdf([a, c]),   # A, C at the finest scale = kept first
            256: make_gdf([b])}           # B straddles both
    fused, audit = fuse_predictions(gdfs)
    assert len(fused) == 2 and sorted(fused["scale_key"]) == ["(256, 2)", "(256, 2)"]
    assert len(audit) == 1 and audit.iloc[0]["scale_key"] == "256"


def test_empty_scale_and_all_empty():
    gdfs = {256: make_gdf([[0, 0, 4, 4]]), 512: make_gdf(np.empty((0, 4)))}
    fused, audit = fuse_predictions(gdfs)
    assert len(fused) == 1 and len(audit) == 0
    fused0, audit0 = fuse_predictions({256: make_gdf(np.empty((0, 4)))})
    assert len(fused0) == 0 and len(audit0) == 0


def test_iou_metric_option():
    gdfs = {256: make_gdf([[0, 0, 4, 4]]), (256, 2): make_gdf([[1, 0, 5, 4]])}
    fused, _ = fuse_predictions(gdfs, metric="iou", cluster_thr=0.5)
    assert len(fused) == 1  # IoU 0.6 >= 0.5


def test_nonpositive_threshold_rejected():
    with pytest.raises(ValueError, match="cluster_thr"):
        fuse_predictions({256: make_gdf([[0, 0, 4, 4]])}, cluster_thr=0.0)


# --- bands rule ----------------------------------------------------------------------------------

def test_bands_rule_in_band_members_win_their_clusters():
    # native_res=1: 4x4 box -> dpx ~4.51; 12x12 -> ~13.5
    small_256 = [0, 0, 4, 4]
    small_sr = [0.2, 0, 4.2, 4]
    big_256 = [20, 20, 32, 32]
    big_sr = [20.2, 20, 32.2, 32]
    gdfs = {256: make_gdf([small_256, big_256]), (256, 2): make_gdf([small_sr, big_sr])}
    bands = {256: (6.0, np.inf), (256, 2): (0.0, 6.0)}  # large from 256, small from SR
    fused, audit = fuse_predictions(gdfs, rule="bands", bands=bands, native_res=1.0)
    assert len(fused) == 2
    by_scale = fused.set_index("scale_key")
    assert equivalent_diameter_px(by_scale.loc[["(256, 2)"]], 1.0)[0] < 6.0   # small from SR
    assert equivalent_diameter_px(by_scale.loc[["256"]], 1.0)[0] >= 6.0       # big from 256
    assert fused["in_band"].all() and len(audit) == 2


def test_bands_rescues_detection_with_no_in_band_partner():
    # a boulder found ONLY by the out-of-band arm must survive (hard exclusion deleted these
    # and failed the parity gate at 0.79/0.47 on the real treatment pools)
    lonely_small_256 = [50, 50, 54, 54]   # dpx ~4.5, assigned band says SR — but SR missed it
    gdfs = {256: make_gdf([lonely_small_256, [20, 20, 32, 32]]),
            (256, 2): make_gdf([[20.2, 20, 32.2, 32]])}
    bands = {256: (6.0, np.inf), (256, 2): (0.0, 6.0)}
    fused, _ = fuse_predictions(gdfs, rule="bands", bands=bands, native_res=1.0)
    rec, unmatched = matched_recall(gdfs[256], fused)
    assert rec == 1.0 and len(unmatched) == 0
    rescued = fused[~fused["in_band"]]
    assert len(rescued) == 1 and rescued.iloc[0]["scale_key"] == "256"


def test_bands_full_range_degenerates_to_scale_aware():
    gdfs = {256: make_gdf([[0, 0, 8, 8]], scores=[0.9]),
            (256, 2): make_gdf([[0.2, 0, 8.2, 8]], scores=[0.3])}
    bands = {256: (0.0, np.inf), (256, 2): (0.0, np.inf)}
    fused_b, _ = fuse_predictions(gdfs, rule="bands", bands=bands, native_res=1.0)
    fused_sa, _ = fuse_predictions(gdfs, rule="scale_aware")
    assert fused_b.iloc[0]["scale_key"] == fused_sa.iloc[0]["scale_key"] == "(256, 2)"


def test_bands_requires_bands_and_res():
    with pytest.raises(ValueError, match="bands"):
        fuse_predictions({256: make_gdf([[0, 0, 4, 4]])}, rule="bands")


# --- engulf pass (fragment consolidation, phase-3 finding) ----------------------------------------

# The measured mode (HiRISE dev crop, 2026-07-10): a fine-arm partial mask of a big boulder sits
# almost wholly inside the whole-boulder mask (poly containment median 0.99) but at DIoU ~0.14 —
# below any sane cluster threshold — so DIoU clustering keeps both and the boulder is counted
# twice. WHOLE = the coarse-arm complete mask; FRAG = the fine-arm partial, offset inside it.
WHOLE = [0, 0, 20, 20]
FRAG = [1, 1, 9, 9]      # fully contained, DIoU(WHOLE, FRAG) ~ -0.03


def test_engulf_default_off_keeps_fragment_pairs():
    gdfs = {256: make_gdf([WHOLE]), (256, 2): make_gdf([FRAG])}
    fused, audit = fuse_predictions(gdfs)
    assert len(fused) == 2 and len(audit) == 0  # pinned phase-1 behaviour


def test_engulf_consolidates_fragment_into_container():
    gdfs = {256: make_gdf([WHOLE]), (256, 2): make_gdf([FRAG])}
    fused, audit = fuse_predictions(gdfs, engulf_thr=0.7)
    assert len(fused) == 1
    assert fused.iloc[0]["scale_key"] == "256"       # container wins, regardless of footprint
    assert fused.iloc[0]["n_merged"] == 1
    assert len(audit) == 1 and bool(audit.iloc[0]["engulfed"])
    assert audit.iloc[0]["metric"] >= 0.7            # containment recorded, not DIoU
    # parity: the suppressed fragment counts as represented under the engulf-aware match
    rec_plain, _ = matched_recall(gdfs[(256, 2)], fused)
    rec_engulf, unmatched = matched_recall(gdfs[(256, 2)], fused, engulf_thr=0.7)
    assert rec_plain < 1.0 and rec_engulf == 1.0 and len(unmatched) == 0


def test_engulf_chain_resolves_to_outermost_container():
    inner = [2, 2, 6, 6]  # inside FRAG, which is inside WHOLE
    gdfs = {512: make_gdf([WHOLE]), 256: make_gdf([FRAG]), (256, 2): make_gdf([inner])}
    fused, audit = fuse_predictions(gdfs, engulf_thr=0.7)
    assert len(fused) == 1 and fused.iloc[0]["scale_key"] == "512"
    assert fused.iloc[0]["n_merged"] == 2
    assert (audit["keeper_scale_key"] == "512").all()


def test_engulf_leaves_side_by_side_boulders_alone():
    # partial box overlap without mask containment must not consolidate
    gdfs = {256: make_gdf([[0, 0, 10, 10]]), (256, 2): make_gdf([[6, 0, 14, 8]])}
    fused, _ = fuse_predictions(gdfs, engulf_thr=0.7)
    assert len(fused) == 2


def test_engulf_thr_validated():
    with pytest.raises(ValueError, match="engulf_thr"):
        fuse_predictions({256: make_gdf([[0, 0, 4, 4]])}, engulf_thr=1.5)


# --- validation helpers --------------------------------------------------------------------------

def test_parity_gate_holds_by_construction():
    rng = np.random.default_rng(1)
    lo = rng.uniform(0, 500, size=(60, 2))
    wh = rng.uniform(3, 15, size=(60, 2))
    base_boxes = np.hstack([lo, lo + wh])
    jitter = rng.uniform(-0.5, 0.5, size=(60, 2))
    sr_boxes = np.hstack([lo + jitter, lo + wh + jitter])[:40]  # partial overlap population
    gdfs = {256: make_gdf(base_boxes), (256, 2): make_gdf(sr_boxes)}
    for rule in ("scale_aware", "score"):
        fused, _ = fuse_predictions(gdfs, rule=rule)
        rec, unmatched = matched_recall(gdfs[256], fused)
        assert rec == 1.0 and len(unmatched) == 0


def test_matched_recall_reports_unrepresented():
    base = make_gdf([[0, 0, 4, 4], [100, 100, 104, 104]])
    fused = make_gdf([[0, 0, 4, 4]])
    rec, unmatched = matched_recall(base, fused)
    assert rec == pytest.approx(0.5) and unmatched.tolist() == [1]


def test_duplicate_rate_flags_escaped_near_pair():
    # 2-px shift on a 4-px box: DIoU ~0.256 -> survives fusion, must show as a near-pair
    gdfs = {256: make_gdf([[0, 0, 4, 4]]), (256, 2): make_gdf([[2, 0, 6, 4]])}
    fused, audit = fuse_predictions(gdfs)
    assert len(fused) == 2 and len(audit) == 0  # escaped the cluster threshold
    dup = duplicate_rate(fused, native_res=1.0)
    assert dup["n_near_pairs"].sum() == 1 and dup["n_bug_pairs"].sum() == 0


def test_duplicate_rate_flags_bug_pairs():
    # a fused output containing a >=thr pair can only come from a clustering bug
    fake_fused = make_gdf([[0, 0, 4, 4], [0.2, 0, 4.2, 4]])
    dup = duplicate_rate(fake_fused, native_res=1.0)
    assert dup["n_bug_pairs"].sum() == 1
