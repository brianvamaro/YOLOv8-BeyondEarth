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
                                      _parse_scale_key,
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


# --- cross-arm-only suppression (G3-02) ------------------------------------------------------------

def test_same_arm_overlaps_are_not_re_suppressed():
    """Two detections from ONE arm that survived that arm's own calibrated NMS are its considered
    answer. Fusion's candidate generation was arm-agnostic and re-suppressed them at a different,
    uncalibrated threshold — 5,043 such suppressions for HiRISE in the saved audits."""
    pair = make_gdf([[0, 0, 4, 4], [1, 0, 5, 4]], scores=[0.9, 0.5])   # DIoU ~0.58, same arm
    fused, audit = fuse_predictions({256: pair})
    assert len(fused) == 2 and len(audit) == 0

    fused_off, audit_off = fuse_predictions({256: pair}, cross_arm_only=False)
    assert len(fused_off) == 1 and len(audit_off) == 1        # available, but only on request
    assert not bool(audit_off.iloc[0].cross_scale)


def test_cross_arm_suppression_still_happens():
    gdfs = {256: make_gdf([[0, 0, 4, 4]]), (256, 2): make_gdf([[1, 0, 5, 4]])}
    fused, audit = fuse_predictions(gdfs)
    assert len(fused) == 1 and len(audit) == 1 and bool(audit.iloc[0].cross_scale)


def test_production_audit_has_no_same_arm_suppressions():
    """The G3-02 acceptance invariant, over a dense random population."""
    rng = np.random.default_rng(7)
    lo = rng.uniform(0, 120, size=(150, 2))          # dense enough for same-arm overlaps
    wh = rng.uniform(3, 14, size=(150, 2))
    boxes = np.hstack([lo, lo + wh])
    gdfs = {256: make_gdf(boxes[:80]), 512: make_gdf(boxes[80:]),
            (256, 2): make_gdf(boxes[:40] + 0.3)}
    fused, audit = fuse_predictions(gdfs, rule="bands", bands={256: (0.0, np.inf)},
                                    native_res=1.0, engulf_thr=0.7)
    assert len(audit) > 0, "fixture produced no suppressions at all"
    assert int((~audit.cross_scale).sum()) == 0
    assert len(fused) + len(audit) == sum(len(g) for g in gdfs.values())    # parity


def test_cross_arm_only_keys_on_the_full_scale_key_not_the_parsed_scale():
    """The trap: neighbouring tiles are the SAME (slice_size, sr_factor) but DIFFERENT arms.
    Keying same-arm on the parsed scale would make every cross-tile pair look same-arm and turn
    seam dedup into a silent no-op — no error, duplicates just flow into the regional catalog."""
    tile_a = make_gdf([[50, 5, 54, 9]], scores=[0.6])
    tile_b = make_gdf([[50.2, 5.1, 54.1, 9.0]], scores=[0.6])
    fused, audit = fuse_predictions({(256, 1, "A"): tile_a, (256, 1, "B"): tile_b})
    assert len(fused) == 1 and len(audit) == 1, "cross-tile dedup became a no-op"
    assert bool(audit.iloc[0].cross_scale)      # different arm, same scale


# --- engulf chain provenance (G3-03) ---------------------------------------------------------------

def test_engulf_chain_metric_describes_the_pair_it_names():
    """Two mutually overlapping fine-arm masks inside one coarse container. The DIoU pass keeps one
    of them and suppresses the other into it; the engulf pass then swallows the survivor into the
    container, and the suppressed one is redirected there too. Its recorded metric used to stay the
    old DIoU against the *fragment*, so the audit named one pair and measured another."""
    frag = [1, 1, 9, 9]                       # inside WHOLE
    twin = [1.4, 1, 9.4, 9]                   # DIoU ~0.6 vs frag -> one suppresses the other
    boxes = {"256": frag, "(256, 2)": twin}
    gdfs = {512: make_gdf([WHOLE]), 256: make_gdf([frag]), (256, 2): make_gdf([twin])}
    fused, audit = fuse_predictions(gdfs, engulf_thr=0.7)

    assert len(fused) == 1 and fused.iloc[0]["scale_key"] == "512"
    assert (audit["keeper_scale_key"] == "512").all()
    assert set(audit["relation"]) == {"engulf", "engulf_chain"}

    chained = audit[audit.relation == "engulf_chain"].iloc[0]
    # the metric must be the redirected member's containment in the CONTAINER, recomputed --
    # not the DIoU against the fragment it was originally measured against
    own = shapely.box(*boxes[chained["scale_key"]])
    truth = own.intersection(shapely.box(*WHOLE)).area / own.area
    assert chained["metric"] == pytest.approx(truth, abs=1e-9)
    assert bool(chained["engulfed"])           # its final relationship is containment
    stale_diou = pairwise_box_metric([shapely.box(*frag).bounds],
                                     [shapely.box(*twin).bounds], "diou")[0]
    assert chained["metric"] != pytest.approx(stale_diou, abs=1e-6)


def test_every_audit_metric_matches_its_named_relation():
    """No audit row may carry a metric measured against a different pair."""
    frag = [1, 1, 9, 9]
    child = [1.4, 1, 9.4, 9]
    gdfs = {512: make_gdf([WHOLE, [200, 200, 208, 208]]), 256: make_gdf([frag]),
            (256, 2): make_gdf([child, [200.2, 200, 208.2, 208]])}
    fused, audit = fuse_predictions(gdfs, engulf_thr=0.7)
    geoms = {}
    for key, g in gdfs.items():
        for i, geom in enumerate(g.geometry):
            geoms[(str(key), i)] = geom
    keeper_geom = dict(zip(fused["pool_index"], fused.geometry))

    for _, row in audit.iterrows():
        dropped = geoms[(row["scale_key"], row["src_index"])]
        keeper = keeper_geom[row["keeper"]]
        if row["relation"] == "cluster":
            expected = pairwise_box_metric([dropped.bounds], [keeper.bounds], "diou")[0]
        else:
            expected = dropped.intersection(keeper).area / dropped.area
        assert row["metric"] == pytest.approx(expected, abs=1e-9), row["relation"]


def test_relation_is_cluster_when_engulf_is_off():
    gdfs = {256: make_gdf([[0, 0, 4, 4]]), (256, 2): make_gdf([[1, 0, 5, 4]])}
    _, audit = fuse_predictions(gdfs)
    assert list(audit["relation"]) == ["cluster"]


# --- validation helpers --------------------------------------------------------------------------

def test_matched_recall_is_one_to_one():
    """G3-02 adversarial fixture: two distinct neighbouring truths and ONE fused detection that
    overlaps both. Many-to-one matching scored parity 1.0 and hid the collapse."""
    two_truths = make_gdf([[0, 0, 4, 4], [1, 0, 5, 4]])
    one_keeper = make_gdf([[0.5, 0, 4.5, 4]])
    rec_unique, unmatched = matched_recall(two_truths, one_keeper)
    assert rec_unique == pytest.approx(0.5) and len(unmatched) == 1
    rec_many, _ = matched_recall(two_truths, one_keeper, unique=False)
    assert rec_many == pytest.approx(1.0)      # the old behaviour, kept only for comparison


def test_matched_recall_is_maximum_cardinality_not_greedy():
    """A greedy pass ordered by metric can consume a keeper that another reference needed, scoring
    below the true optimum. R0 overlaps both keepers; R1 only overlaps K0."""
    refs = make_gdf([[0, 0, 4, 4],          # R0: matches K0 strongly and K1 as well
                     [0.2, 0, 4.2, 4]])     # R1: matches K0 only
    keepers = make_gdf([[0.1, 0, 4.1, 4],   # K0
                        [0, 0, 4, 4]])      # K1: identical to R0
    rec, unmatched = matched_recall(refs, keepers)
    assert rec == pytest.approx(1.0), "maximum matching should pair both references"
    assert len(unmatched) == 0

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


# --- phase 5c: seam fusion across neighbouring tiles ---------------------------------------------
# Cross-*tile* dedup at scene/tile seams is the same fuse_predictions call with keys = neighbouring
# tiles at the SAME scale. That last part is why the key contract needed a third element: a dict
# cannot hold 256 twice, so scale alone cannot key two tiles apart (found 2026-07-30, plan §5c).


def test_scale_key_accepts_a_tile_tag():
    assert _parse_scale_key((256, 1, "tile_a")) == (256, 1.0)
    assert _parse_scale_key((256, 2, "tile_a")) == (256, 2.0)
    with pytest.raises(ValueError, match="scale key"):
        _parse_scale_key((256, 1, "tile_a", "extra"))


def test_tile_tag_does_not_change_footprint_or_priority():
    """The tag is provenance only — two tiles at one scale must tie on footprint."""
    pool = pool_predictions({(256, 1, "a"): make_gdf([[0, 0, 4, 4]]),
                             (256, 1, "b"): make_gdf([[40, 40, 44, 44]])})
    assert pool["footprint"].nunique() == 1
    assert set(pool["scale_key"]) == {"(256, 1, 'a')", "(256, 1, 'b')"}


def test_seam_overlap_fuses_to_parity_with_no_duplicates():
    """Two overlapping tiles of one scene: the shared boulder must survive exactly once."""
    # tile A covers x in [0, 60], tile B covers x in [40, 100]; overlap strip is x in [40, 60]
    tile_a = make_gdf([[5, 5, 9, 9],        # A-only interior
                       [50, 5, 54, 9]])     # in the overlap strip - both tiles see it
    tile_b = make_gdf([[50.2, 5.1, 54.1, 9.0],   # same boulder, tile B's view
                       [90, 5, 94, 9]])          # B-only interior
    fused, audit = fuse_predictions({(256, 1, "A"): tile_a, (256, 1, "B"): tile_b})

    assert len(fused) == 3                      # 4 pooled, 1 duplicate suppressed
    assert len(audit) == 1
    assert len(fused) + len(audit) == len(tile_a) + len(tile_b)     # parity by construction

    strip = fused[fused.geometry.centroid.x.between(40, 60)]
    assert len(strip) == 1                      # no duplicate in the overlap strip
    dup = duplicate_rate(fused, native_res=1.0)
    assert dup["n_bug_pairs"].sum() == 0


def test_seam_truncation_priority_prefers_the_whole_mask():
    """The tile that clipped a boulder at its edge must lose to the tile that saw it whole."""
    clipped = make_gdf([[50, 5, 53, 9]], scores=[0.95], within=[False])   # tile A's edge
    whole = make_gdf([[50, 5, 54, 9]], scores=[0.40], within=[True])      # tile B, fully inside
    fused, audit = fuse_predictions({(256, 1, "A"): clipped, (256, 1, "B"): whole})

    assert len(fused) == 1
    assert bool(fused["is_within_slice"].iloc[0]) is True     # despite the lower score
    assert fused["scale_key"].iloc[0] == "(256, 1, 'B')"


def test_seam_interior_only_boulders_are_untouched():
    """No detection deficit in the strip: non-overlapping content passes straight through."""
    tile_a = make_gdf([[5, 5, 9, 9], [12, 5, 16, 9]])
    tile_b = make_gdf([[90, 5, 94, 9]])
    fused, audit = fuse_predictions({(256, 1, "A"): tile_a, (256, 1, "B"): tile_b})
    assert len(fused) == 3 and len(audit) == 0


def test_seam_fusion_is_order_independent():
    tile_a = make_gdf([[50, 5, 54, 9]], scores=[0.6])
    tile_b = make_gdf([[50.2, 5.1, 54.1, 9.0]], scores=[0.6])
    ab, _ = fuse_predictions({(256, 1, "A"): tile_a, (256, 1, "B"): tile_b})
    ba, _ = fuse_predictions({(256, 1, "B"): tile_b, (256, 1, "A"): tile_a})
    assert len(ab) == len(ba) == 1


def test_tile_edge_flag_survives_pooling():
    """``is_at_edge`` (raster-footprint edge, set by get_sliced_prediction) IS the tile-edge flag
    once the raster is a tile — it must reach the fused catalog for seam provenance."""
    a = make_gdf([[50, 5, 53, 9]], within=[False], is_at_edge=[True])
    b = make_gdf([[90, 5, 94, 9]], within=[True], is_at_edge=[False])
    fused, _ = fuse_predictions({(256, 1, "A"): a, (256, 1, "B"): b})
    assert "is_at_edge" in fused.columns
    assert fused.sort_values("scale_key")["is_at_edge"].tolist() == [True, False]


# --- the bands rule must not depend on how a caller spells its scale keys ------------------------
# Found 2026-08-05 by code review. `pool_predictions` stored `str(key)` and band matching compared
# against `str(band_key)`, so `rule="bands"` silently matched nothing whenever the caller keyed its
# arms differently from the bands dict. The two production callers do exactly that:
# `get_multiscale_prediction` iterates `scales=(512, 256, (256, 2))` and keys the 256 arm as the int
# `256`; `scripts/run_multiscale_arms.py` keys it via `parse_arm()` as `(256, 1)`. So the production
# rule `256pref_engulf` was ACTIVE on crops and INERT on every full scene — and no test saw it,
# because the in-memory wrapper refuses scene-scale rasters (G3-09) and only the wrapper was tested.

def _three_arm_ladder(key_512, key_256, key_sr):
    """The production ladder, with the caller's choice of key spelling."""
    return {key_512: make_gdf([(0, 0, 8, 8)], scores=[0.90]),
            key_256: make_gdf([(1, 0, 9, 8)], scores=[0.50]),
            key_sr: make_gdf([(2, 0, 10, 8)], scores=[0.40])}


@pytest.mark.parametrize("keys", [
    (512, 256, (256, 2)),                    # get_multiscale_prediction
    ((512, 1), (256, 1), (256, 2)),          # scripts/run_multiscale_arms.py (parse_arm)
])
def test_bands_rule_is_invariant_to_scale_key_spelling(keys):
    """Both production callers must get the same fusion. The 256 arm is in-band and so must win the
    contested cluster despite having the *lowest* score of the three — that is the entire content of
    `256pref`, and it is what silently stopped happening at scene scale."""
    fused, _ = fuse_predictions(_three_arm_ladder(*keys), rule="bands",
                                bands={256: (0.0, np.inf)}, native_res=1.0, cross_arm_only=True)
    assert int(fused["in_band"].sum()) == 1, "the 256 arm was not matched by the band"
    assert len(fused) == 1, "the three overlapping detections did not cluster"
    kept = fused.iloc[0]
    assert kept["arm_key"] == "256", f"256pref did not win; kept {kept['arm_key']!r}"
    assert kept["score"] == pytest.approx(0.50), "a higher-scoring off-band arm won the cluster"


def test_arm_key_collapses_spellings_but_keeps_seam_tiles_distinct():
    """`arm_key` is the *scale* identity a band names; `scale_key` is the *full* identity cross-tile
    dedup needs. Collapsing the second would turn seam fusion into a no-op (interaction note 1)."""
    from YOLOv8BeyondEarth.fusion import arm_key
    assert arm_key(256) == arm_key((256, 1)) == arm_key((256, 1, "tile_03")) == "256"
    assert arm_key((256, 2)) == "256sr2"
    assert arm_key(512) != arm_key(256)

    tiles = {(256, 1, "tile_a"): make_gdf([(0, 0, 8, 8)]),
             (256, 1, "tile_b"): make_gdf([(1, 0, 9, 8)])}
    pooled = pool_predictions(tiles)
    assert set(pooled["arm_key"]) == {"256"}, "both tiles are the same arm"
    assert pooled["scale_key"].nunique() == 2, "seam tiles must stay distinguishable"


def test_a_bands_key_that_matches_no_arm_is_refused():
    """The failure mode this defect had: `in_band` all-False, `bands` quietly degrading to
    `scale_aware`, and a plausible catalog out the other end. Silence is the bug, so it now raises."""
    with pytest.raises(ValueError, match="matches no arm"):
        fuse_predictions({512: make_gdf([(0, 0, 8, 8)])}, rule="bands",
                         bands={256: (0.0, np.inf)}, native_res=1.0)
