"""Fusion validation gaps: the orientation claim and the duplicate heuristic (G3-05, G3-06).

G3-05: phase 4 explained a non-regressed orientation distribution as "256 masks untouched, fusion
only adds". The engulf pass can suppress 256 rows, and HiRISE ended with 6,307 fewer 256-sourced
rows than it started with — so the additions-only argument was being made about a set that also
lost members. Separately, the standing content-free null was never run against `256pref_engulf`,
the rule that actually ships, and its bootstrap intervals were computed and then unused.

G3-06: surviving pairs with containment >= 0.5 were called "true duplicates" without anyone
looking at the imagery, and production already removes everything at >= 0.7 — so the label was
being applied only in the ambiguous 0.5-0.7 band.
"""
import numpy as np
import pandas as pd
import pytest
import shapely

from YOLOv8BeyondEarth.fusion import (
    decompose_against_baseline, duplicate_audit_sample, duplicate_rate, fuse_predictions,
    fusion_orientation_null, heuristic_precision, isotropic_arms, orientation_decomposition)
from tests.test_fusion import FRAG, WHOLE, make_gdf


# --- G3-05: the decomposition -------------------------------------------------------------------

def test_decomposition_shows_the_baseline_is_not_untouched():
    """The claim under test. A 256 fragment inside a 512 boulder is *removed* by the engulf pass,
    so "256 masks untouched" is false and the decomposition has to say so."""
    gdfs = {512: make_gdf([WHOLE]), 256: make_gdf([FRAG, [200, 200, 208, 208]])}
    fused, audit = fuse_predictions(gdfs, engulf_thr=0.7)

    parts = decompose_against_baseline(fused, audit, baseline_key=256)
    assert parts["n_baseline"] == 2
    assert parts["n_removed"] == 1, "the engulfed 256 fragment was not counted as removed"
    assert parts["n_kept"] == 1
    assert parts["n_added"] == 1                       # the 512 whole boulder
    assert parts["removed_fraction"] == pytest.approx(0.5)


def test_decomposition_reports_untouched_when_it_really_is():
    gdfs = {512: make_gdf([[500, 500, 520, 520]]), 256: make_gdf([[0, 0, 8, 8]])}
    fused, audit = fuse_predictions(gdfs, engulf_thr=0.7)
    parts = decompose_against_baseline(fused, audit, baseline_key=256)
    assert parts["n_removed"] == 0 and parts["n_kept"] == 1 and parts["n_added"] == 1


def test_orientation_decomposition_separates_the_groups():
    """Kept and added are summarised apart, and the removed count is on the record — pooling the
    three is how an orientation change hides."""
    rng = np.random.default_rng(0)
    base = make_gdf([[40 * i, 0, 40 * i + 10, 6] for i in range(12)])          # E-W elongated
    other = make_gdf([[40 * i, 200, 40 * i + 6, 210] for i in range(12)])      # N-S elongated
    fused, audit = fuse_predictions({256: base, 512: other})

    out = orientation_decomposition(fused, audit, baseline_key=256, res=1.0,
                                    aspect_min=1.2, dpx_min=4.0)
    assert list(out.group) == ["kept", "added"]
    assert out.attrs["baseline_untouched"] is True
    assert out.attrs["n_removed"] == 0
    kept = out[out.group == "kept"].iloc[0]
    added = out[out.group == "added"].iloc[0]
    assert kept.n == 12 and added.n == 12
    # the two groups genuinely differ in orientation, which is exactly why pooling them is wrong
    d = abs(kept.median_angle - added.median_angle) % 180.0
    assert min(d, 180.0 - d) > 45.0
    assert np.isfinite(out.attrs["uniform_grid_fraction"])


# --- G3-05: the production null ------------------------------------------------------------------

def test_isotropic_arms_are_content_free():
    from YOLOv8BeyondEarth.orientation import (
        boulder_orientations, grid_fraction, grid_fraction_uniform)
    gdfs = isotropic_arms((512, 256), n_per_arm=800, seed=3)
    assert set(gdfs) == {512, 256}
    angles = boulder_orientations(gdfs[256], res=1.0, n_workers=1).angle180.to_numpy()
    assert abs(grid_fraction(angles) - grid_fraction_uniform(10.0)) < 0.04


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_production_rule_null_is_consistent_with_isotropy(seed):
    """The saved regression G3-05 asks for: `256pref_engulf` -- the rule that ships -- run on
    content-free input, gated on an interval rather than a point comparison.

    Measured 2026-07-31: grid fraction 0.2011-0.2289 across seeds, intervals all bracketing the
    isotropic 0.2222. Fast enough to live in the default lane, so it is a regression that actually
    runs."""
    out = fusion_orientation_null(n_per_arm=900, n_boot=200, seed=seed)
    assert out["rule"] == "bands" and out["engulf_thr"] == 0.7
    assert out["n_fused"] > 0 and out["n_suppressed"] > 0    # fusion actually did something
    assert out["lo"] < out["hi"]
    assert out["passes"], (f"fused rose is not consistent with isotropy: "
                           f"{out['grid_fraction']:.4f} [{out['lo']:.4f}, {out['hi']:.4f}] "
                           f"vs uniform {out['uniform']:.4f}")


def test_null_gate_is_an_interval_not_a_point():
    """A point comparison cannot distinguish a real anisotropy from sampling noise; the returned
    verdict must be the interval containing the isotropic baseline."""
    out = fusion_orientation_null(scales=(256,), rule="scale_aware", engulf_thr=None,
                                  n_per_arm=400, n_boot=120, seed=5)
    assert out["passes"] == (out["lo"] <= out["uniform"] <= out["hi"])


# --- G3-06: the duplicate heuristic ----------------------------------------------------------------

def _near_pair_catalog():
    """Surviving near-pairs at a range of containments, plus isolated boulders."""
    boxes = []
    for i in range(12):                      # near-pairs: 2 px shift on a 4 px box -> DIoU ~0.26
        boxes += [[100 * i, 0, 100 * i + 4, 4], [100 * i + 2, 0, 100 * i + 6, 4]]
    for i in range(6):                       # isolated
        boxes.append([100 * i, 500, 100 * i + 8, 508])
    return make_gdf(boxes)


def test_duplicate_audit_sample_is_blinded_and_stratified():
    fused = _near_pair_catalog()
    sample = duplicate_audit_sample(fused, native_res=1.0, n_per_band=5, seed=0)
    assert len(sample) > 0
    assert sample["is_duplicate"].isna().all(), "the label column must start empty"
    assert "heuristic_is_duplicate" in sample                 # present, but in its own column
    assert sample.band.nunique() >= 1
    assert sample.pair_id.is_unique
    # shuffled: the rows must not arrive grouped by band
    if sample.band.nunique() > 1:
        assert list(sample.band) != sorted(sample.band)


def test_audit_sample_is_empty_when_there_are_no_near_pairs():
    isolated = make_gdf([[100 * i, 0, 100 * i + 4, 4] for i in range(5)])
    sample = duplicate_audit_sample(isolated, native_res=1.0)
    assert len(sample) == 0 and "is_duplicate" in sample.columns


def test_heuristic_precision_reports_intervals_and_bands():
    labelled = pd.DataFrame({
        "band": ["3-4"] * 10 + ["4-6"] * 10,
        "heuristic_is_duplicate": [True] * 8 + [False] * 2 + [True] * 5 + [False] * 5,
        "is_duplicate": [True] * 6 + [False] * 2 + [True, False]
                        + [True] * 5 + [False] * 5,
    })
    rep = heuristic_precision(labelled)
    overall = rep[rep.band == "all"].iloc[0]
    assert overall.n == 20
    assert overall.precision == pytest.approx(11 / 13, abs=1e-9)
    assert overall.precision_lo < overall.precision < overall.precision_hi
    assert set(rep.band) == {"all", "3-4", "4-6"}


def test_heuristic_precision_requires_labels():
    unlabelled = duplicate_audit_sample(_near_pair_catalog(), native_res=1.0, n_per_band=3)
    with pytest.raises(ValueError, match="is_duplicate"):
        heuristic_precision(unlabelled)


def test_near_pair_rate_is_still_reported_but_named_as_a_bound():
    """The rate itself is unchanged — what changed is that it no longer claims to be validated."""
    fused = _near_pair_catalog()
    dup = duplicate_rate(fused, native_res=1.0)
    assert dup["n_near_pairs"].sum() > 0
    assert "near_pair_rate" in dup.columns
    assert "upper bound" in duplicate_rate.__doc__
