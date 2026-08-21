"""Characterization tests for the 2x2 arm-ratification harness.

The properties pinned here are the ones the conclusion rests on, not the plumbing: that the
one-arm cell is a genuine no-op through fusion (so the 2x2 is an ablation and not a code-path
comparison), that the interaction is well defined whichever arm you compute it from, that the
optimal resolution never matches fewer humans than the greedy one, and that the reference digest
actually moves when the reference does.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

geopandas = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")
from shapely.geometry import Polygon, box  # noqa: E402

from YOLOv8BeyondEarth import arm_ratification as ar  # noqa: E402
from YOLOv8BeyondEarth import presets  # noqa: E402

RES = 0.5


def squares(centres, side):
    """A GeoDataFrame of axis-aligned squares, one per centre."""
    geoms = [box(x - side / 2, y - side / 2, x + side / 2, y + side / 2) for x, y in centres]
    # ``isin_slice`` is not decoration: ``fusion.pool_predictions`` refuses a frame without it,
    # because the truncation guard cannot tell a real boulder from a slice-edge cut without it.
    return geopandas.GeoDataFrame({"score": np.linspace(0.9, 0.5, len(geoms)),
                                   "isin_slice": True}, geometry=geoms, crs=None)


# --- the arm keys and the 2x2 --------------------------------------------------------------------

def test_arm_key_and_tag_spelling():
    assert ar.arm_key(256, 1) == 256
    assert ar.arm_key(256, 2) == (256, 2)
    assert ar.arm_tag(256, 1) == "256"
    assert ar.arm_tag(256, 2) == "256sr2"


def test_2x2_is_a_full_factorial():
    """Four cells, and the factor levels are each present exactly twice."""
    assert len(ar.ARM_2X2) == 4
    has_512 = [any(ss == 512 for ss, _ in arms) for arms in ar.ARM_2X2.values()]
    has_sr = [any(sr > 1 for _, sr in arms) for arms in ar.ARM_2X2.values()]
    assert sum(has_512) == 2 and sum(has_sr) == 2
    assert len(set(zip(has_512, has_sr))) == 4       # all four combinations, no duplicates
    # every cell contains the 256 arm, because production's rule hands it every contested cluster
    assert all(any(ss == 256 and sr == 1 for ss, sr in arms) for arms in ar.ARM_2X2.values())


def test_production_scales_are_the_full_cell():
    """The ``512_256_sr`` cell must be exactly what production ships, or the 2x2 ratifies nothing."""
    shipped = {s if isinstance(s, int) else tuple(s)
               for s in presets.GOAL3_PRODUCTION["scales"]}
    cell = {ar.arm_key(*a) for a in ar.ARM_2X2["512_256_sr"]}
    assert cell == {512, 256, (256, 2)} == shipped


# --- fusion through the one-arm cell -------------------------------------------------------------

def test_one_arm_cell_suppresses_nothing():
    """The reference cell goes through fusion and comes back whole.

    Two overlapping detections from the *same* arm must both survive, because ``cross_arm_only``
    makes every within-arm pair ineligible. If this ever fails, the ``{256}`` cell stops being a
    baseline and starts being a different fusion.
    """
    arm = squares([(0, 0), (0.6, 0), (20, 20)], side=4.0)      # first two overlap heavily
    fused, audit = ar.fuse_combo({256: arm}, ((256, 1),), RES)
    assert len(fused) == len(arm) == 3
    assert len(audit) == 0


def test_multi_arm_cell_does_suppress():
    """The same overlap across two arms is suppressed -- the contrast that makes the test above
    meaningful rather than a statement that fusion never suppresses."""
    a = squares([(0, 0)], side=4.0)
    b = squares([(0.3, 0.0)], side=4.0)
    fused, audit = ar.fuse_combo({256: a, 512: b}, ((512, 1), (256, 1)), RES)
    assert len(fused) == 1
    assert len(audit) == 1


# --- matching ------------------------------------------------------------------------------------

def test_overlap_admissibility_is_half_of_the_smaller():
    """A small polygon 60% inside a large one is admissible; 40% is not. Under IoU neither would be,
    which is exactly why the gate is half-of-smaller."""
    human = squares([(0, 0)], side=10.0)
    inside = geopandas.GeoDataFrame(geometry=[box(3.8, -1.0, 5.8, 1.0)])     # 60% of the smaller
    outside = geopandas.GeoDataFrame(geometry=[box(4.4, -1.0, 6.4, 1.0)])    # 30% of the smaller
    assert len(ar.overlap_candidates(human, inside)[0]) == 1
    assert len(ar.overlap_candidates(human, outside)[0]) == 0


def test_hybrid_gate_is_strictly_wider():
    """A centroid-inside pair below the overlap gate is admissible only under ``hybrid``."""
    human = squares([(0, 0)], side=10.0)
    pred = geopandas.GeoDataFrame(geometry=[box(4.4, -1.0, 20.0, 1.0)])   # centroid outside human
    pred_in = geopandas.GeoDataFrame(geometry=[box(-20.0, -1.0, 4.6, 1.0)])  # centroid inside
    assert len(ar.overlap_candidates(human, pred, hybrid=False)[0]) == 0
    assert len(ar.overlap_candidates(human, pred_in, hybrid=True)[0]) == 1


def test_matching_is_one_to_one():
    """One prediction cannot credit two humans -- the merged-boulder failure G2-05 exists for."""
    humans = squares([(0, 0), (3.0, 0)], side=2.0)
    merged = geopandas.GeoDataFrame(geometry=[box(-1.2, -1.2, 4.2, 1.2)])
    for resolve in ar.RESOLUTIONS:
        h_mask, p_mask = ar.match_overlap(humans, merged, resolve=resolve)
        assert h_mask.sum() == 1, resolve
        assert p_mask.sum() == 1, resolve


def test_optimal_resolution_never_matches_fewer_than_greedy():
    """The bracket the harness reports only ever opens one way.

    Constructed so greedy takes the wrong pair first: the highest-overlap edge consumes the only
    prediction that human A could have used, leaving A unmatched, while the optimal resolution
    pairs both.
    """
    humans = geopandas.GeoDataFrame(geometry=[box(0, 0, 2, 2), box(2, 0, 6, 2)])
    preds = geopandas.GeoDataFrame(geometry=[box(0.1, 0.1, 2.1, 1.9), box(0.0, 0.0, 6.0, 2.0)])
    greedy = ar.match_overlap(humans, preds, resolve="greedy")[0].sum()
    optimal = ar.match_overlap(humans, preds, resolve="optimal")[0].sum()
    assert optimal >= greedy
    assert (optimal, greedy) == (2, 1)


def test_match_overlap_rejects_unknown_resolution():
    humans = squares([(0, 0)], side=2.0)
    with pytest.raises(ValueError, match="resolve must be"):
        ar.match_overlap(humans, humans, resolve="whatever")


def test_empty_inputs_are_not_a_crash():
    empty = geopandas.GeoDataFrame(geometry=[])
    humans = squares([(0, 0)], side=2.0)
    for resolve in ar.RESOLUTIONS:
        assert ar.match_overlap(humans, empty, resolve=resolve)[0].sum() == 0
        assert ar.match_overlap(empty, humans, resolve=resolve)[1].sum() == 0


# --- scoring -------------------------------------------------------------------------------------

def test_score_bands_denominators_and_perfect_case():
    """Identical human and prediction sets: recall and precision are 1 in every populated band."""
    centres = [(0, 0), (30, 0), (60, 0)]
    humans = squares(centres, side=6.0)          # 6 m across at 0.5 m/px -> ~13.5 px equivalent
    scores = ar.score_bands(humans, humans.copy(), RES)
    allrow = scores[(scores.band == "all") & (scores.split == "all")].iloc[0]
    assert allrow.n_human == 3 and allrow.n_pred == 3
    assert allrow.recall == 1.0 and allrow.precision == 1.0 and allrow.f1 == 1.0
    populated = scores[(scores.split == "all") & (scores.n_human > 0) & (scores.band != "all")]
    assert (populated.recall == 1.0).all()


def test_score_bands_undefined_band_is_nan_not_zero():
    """An empty band reports NaN, so "nothing to measure" cannot be read as "measured and bad"."""
    humans = squares([(0, 0)], side=6.0)
    scores = ar.score_bands(humans, humans.copy(), RES)
    empty = scores[(scores.split == "all") & (scores.n_human == 0) & (scores.n_pred == 0)]
    assert len(empty) > 0
    assert empty.recall.isna().all() and empty.f1.isna().all()


def test_score_bands_matches_once_then_splits():
    """Splitting must partition the matched set, not re-match inside each split.

    The same prediction must not be credited in two splits, so the per-split matched counts sum to
    the pooled one.
    """
    humans = squares([(0, 0), (30, 0)], side=6.0)
    humans["split"] = ["train", "test"]
    preds = humans.copy()
    scores = ar.score_bands(humans, preds, RES)
    pooled = scores[(scores.split == "all") & (scores.band == "all")].n_matched.iloc[0]
    per_split = scores[(scores.split.isin(["train", "test"]))
                       & (scores.band == "all")].n_matched.sum()
    assert pooled == per_split == 2


# --- the 2x2 readout -----------------------------------------------------------------------------

def _three_arm_fixture():
    """A 256 arm that finds two of three humans, a 512 arm that adds the large one, an SR arm that
    adds the small one. Every polygon is far enough from the others to match unambiguously."""
    humans = squares([(0, 0), (100, 0), (200, 0)], side=7.0)
    small = box(200 - 1.0, -1.0, 200 + 1.0, 1.0)
    humans = geopandas.GeoDataFrame(
        geometry=[humans.geometry[0], humans.geometry[1], small], crs=None)
    a256 = geopandas.GeoDataFrame(geometry=[humans.geometry[0]], crs=None)
    a512 = geopandas.GeoDataFrame(geometry=[humans.geometry[1]], crs=None)
    asr = geopandas.GeoDataFrame(geometry=[humans.geometry[2]], crs=None)
    for frame in (a256, a512, asr):
        frame["score"] = 0.8
        frame["isin_slice"] = True
    return humans, {256: a256, 512: a512, (256, 2): asr}


def test_ratify_arms_shape_and_factor_columns():
    humans, per_arm = _three_arm_fixture()
    scores = ar.ratify_arms(per_arm, humans, RES, body="unit")
    assert set(scores.combo) == set(ar.ARM_2X2)
    assert set(scores.resolve) == set(ar.RESOLUTIONS)
    assert set(scores.hybrid) == {False, True}
    assert (scores.body == "unit").all()
    # the factor columns must agree with the combo definitions
    for combo, arms in ar.ARM_2X2.items():
        part = scores[scores.combo == combo]
        assert part.has_512.unique().tolist() == [any(ss == 512 for ss, _ in arms)]
        assert part.has_sr.unique().tolist() == [any(sr > 1 for _, sr in arms)]


def test_ratify_arms_recovers_the_planted_contributions():
    """Each arm's marginal contribution is what was planted, and the arms do not interact here."""
    humans, per_arm = _three_arm_fixture()
    scores = ar.ratify_arms(per_arm, humans, RES, hybrids=(False,), resolutions=("optimal",))
    marg = ar.marginal_table(scores, band="all", split="all", resolve="optimal")
    assert marg["base"] == pytest.approx(1 / 3)
    assert marg["d512_without_sr"] == pytest.approx(1 / 3)
    assert marg["d512_with_sr"] == pytest.approx(1 / 3)
    assert marg["dsr_without_512"] == pytest.approx(1 / 3)
    assert marg["interaction"] == pytest.approx(0.0)


def test_interaction_is_symmetric_in_the_two_arms():
    """The interaction is a property of the 2x2, not of which arm you call the treatment.

    This identity is the reason the handoff asked for one 2x2 rather than two ablations, so it is
    pinned rather than assumed.
    """
    humans, per_arm = _three_arm_fixture()
    scores = ar.ratify_arms(per_arm, humans, RES, hybrids=(False,), resolutions=("optimal",))
    marg = ar.marginal_table(scores, band="all", split="all", resolve="optimal")
    from_512 = marg["d512_with_sr"] - marg["d512_without_sr"]
    from_sr = marg["dsr_with_512"] - marg["dsr_without_512"]
    assert from_512 == pytest.approx(from_sr)
    assert marg["interaction"] == pytest.approx(from_512)


def test_marginal_table_refuses_an_incomplete_2x2():
    humans, per_arm = _three_arm_fixture()
    combos = {k: v for k, v in ar.ARM_2X2.items() if k != "256_sr"}
    scores = ar.ratify_arms(per_arm, humans, RES, combos=combos, hybrids=(False,),
                            resolutions=("optimal",))
    with pytest.raises(KeyError, match="need all four cells"):
        ar.marginal_table(scores, band="all", split="all", resolve="optimal")


def test_fuse_combo_raises_on_a_missing_arm():
    humans, per_arm = _three_arm_fixture()
    del per_arm[(256, 2)]
    with pytest.raises(KeyError, match="missing arm layer"):
        ar.fuse_combo(per_arm, ar.ARM_2X2["512_256_sr"], RES)


# --- the reference: splits, clipping, provenance -------------------------------------------------

def test_dataset_splits_reads_filenames_only(tmp_path):
    for split, names in [("train", ["a_0001_image"]), ("validation", ["b_0002_image"]),
                         ("test", ["c_0003_image"])]:
        d = tmp_path / split / "images"
        d.mkdir(parents=True)
        for name in names:
            (d / f"{name}.png").write_bytes(b"")
    mapping = ar.dataset_splits(tmp_path)
    assert mapping == {"a_0001_image": "train", "b_0002_image": "validation",
                       "c_0003_image": "test"}


def test_rom_tiles_labels_absent_tiles_absent(tmp_path):
    """A reference tile that is not in the training dataset is the useful case; it must not be
    silently labelled training."""
    rom = geopandas.GeoDataFrame(
        {"tile_id": [1, 2], "file_name": ["s_0001_image.png", "s_0002_image.png"]},
        geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)], crs="EPSG:4326")
    path = tmp_path / "rom.shp"
    rom.to_file(path)
    tiles = ar.rom_tiles(path, "EPSG:4326", split_map={"s_0001_image": "train"})
    assert sorted(tiles["split"]) == ["absent", "train"]
    unknown = ar.rom_tiles(path, "EPSG:4326", split_map=None)
    assert set(unknown["split"]) == {"unknown"}


def test_clip_to_tiles_is_centroid_in_tile(tmp_path):
    """A boulder straddling a tile boundary belongs to exactly one tile."""
    tiles = geopandas.GeoDataFrame({"tile_id": [1, 2], "split": ["train", "test"]},
                                   geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10)], crs=None)
    objs = geopandas.GeoDataFrame(geometry=[box(8, 4, 11, 6),       # straddles, centroid in tile 1
                                            box(1, 1, 3, 3),        # tile 1
                                            box(15, 1, 17, 3),      # tile 2
                                            box(30, 1, 32, 3)],     # outside every tile
                                  crs=None)
    kept = ar.clip_to_tiles(objs, tiles)
    assert len(kept) == 3                       # the outside one is dropped
    assert sorted(kept.tile_id) == [1, 1, 2]
    assert len(kept) == len(kept.drop_duplicates(subset="geometry"))


def test_provenance_hashes_every_shapefile_sidecar(tmp_path):
    rom = geopandas.GeoDataFrame({"tile_id": [1], "file_name": ["s_0001_image.png"]},
                                 geometry=[box(0, 0, 10, 10)], crs="EPSG:4326")
    path = tmp_path / "rom.shp"
    rom.to_file(path)
    ident = ar.shapefile_identity(path)
    assert {".shp", ".dbf", ".shx", ".prj"} <= set(ident)
    assert all("sha256" in v for v in ident.values())


def test_provenance_digest_moves_with_the_reference(tmp_path):
    """A different reference must not produce the same provenance block."""
    def write(name, tile_id):
        frame = geopandas.GeoDataFrame({"tile_id": [tile_id], "file_name": ["s.png"]},
                                       geometry=[box(0, 0, 10, 10)], crs="EPSG:4326")
        path = tmp_path / name
        frame.to_file(path)
        return path

    a, b = write("a.shp", 1), write("b.shp", 2)
    pa = ar.ratification_provenance({"rom": a}, {"256": a}, 0.5)
    pb = ar.ratification_provenance({"rom": b}, {"256": b}, 0.5)
    assert pa["reference"]["rom"][".dbf"]["sha256"] != pb["reference"]["rom"][".dbf"]["sha256"]
    # the config hash covers the *rule*, not the inputs, so it must be stable across references
    assert pa["config_hash"] == pb["config_hash"]


def test_provenance_records_the_matching_rule_and_survives_json(tmp_path):
    """A recall number is not pinned by its inputs alone (X-01, G3-01), so the rule is recorded --
    and the block must round-trip through JSON or it cannot be written next to the table."""
    frame = geopandas.GeoDataFrame({"tile_id": [1], "file_name": ["s.png"]},
                                   geometry=[box(0, 0, 10, 10)], crs="EPSG:4326")
    path = tmp_path / "r.shp"
    frame.to_file(path)
    prov = ar.ratification_provenance({"rom": path}, {"256": path}, 0.5)
    assert prov["matching_rule"]["admissibility"].startswith("intersection > 0.5")
    assert prov["matching_rule"]["bands"][-1] == [14, None]      # inf survives as null
    assert prov["fusion"]["cross_arm_only"] is True
    text = json.dumps(prov)
    assert json.loads(text)["config_hash"] == prov["config_hash"]


def test_band_of_puts_sub_floor_objects_in_their_own_bucket():
    """Below the smallest band edge is its own label, never pooled into the first band."""
    labels = ar._band_of(np.array([1.0, 3.5, 100.0]))
    assert labels[0] == "<3"
    assert labels[1] == "3-4"
    assert labels[2] == "14-inf"


def test_post_fuse_runs_after_fusion_not_before():
    """The clip must see the fused catalog, and the raw/scored counts must both be reported.

    Production fuses over the whole scene and clips afterwards, because an arm's detection just
    outside the reference footprint can still suppress one just inside. A ``post_fuse`` that ran on
    the arms instead would silently change the catalog being scored.
    """
    humans, per_arm = _three_arm_fixture()
    seen = {}

    def keep_first_two(fused):
        seen["n_raw"] = len(fused)
        return fused.iloc[:2].copy()

    scores = ar.ratify_arms(per_arm, humans, RES, hybrids=(False,), resolutions=("optimal",),
                            post_fuse=keep_first_two)
    full = scores[scores.combo == "512_256_sr"]
    assert seen["n_raw"] == 3                      # post_fuse saw the fused catalog, not one arm
    assert full.n_fused.unique().tolist() == [3]
    assert full.n_scored.unique().tolist() == [2]
    allrow = full[(full.band == "all") & (full.split == "all")].iloc[0]
    assert allrow.n_pred == 2                      # scoring used the post_fuse output
