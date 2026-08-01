"""Development versus confirmation (review item G3-11).

The same 20 HiRISE ROMs chose `256pref_engulf` and supplied the recall gains reported for it. A
number selected on a set and quoted on the same set carries the selection. The fix is to reserve
ground the selection never saw, make the reservation checkable, and make the label travel with the
number.
"""
import numpy as np
import pandas as pd
import pytest

from YOLOv8BeyondEarth.splits import (
    SPLITS, assign_split, evaluate_by_split, group_split, labelled_result, threshold_sensitivity)


def test_split_is_deterministic():
    keys = [f"rom{i:02d}" for i in range(20)]
    a, ma = group_split(keys, seed=7)
    b, mb = group_split(keys, seed=7)
    assert a == b and ma["assignment_sha256"] == mb["assignment_sha256"]


def test_split_is_stable_when_groups_are_added():
    """A held-out set that reshuffles when the dataset grows is not a held-out set. Hashing the
    key rather than shuffling an index is what buys this."""
    keys = [f"rom{i:02d}" for i in range(20)]
    before, _ = group_split(keys, seed=1)
    after, _ = group_split(keys + [f"rom{i:02d}" for i in range(20, 30)], seed=1)
    assert all(after[k] == before[k] for k in keys)


def test_split_respects_the_requested_fraction_roughly():
    keys = [f"rom{i:03d}" for i in range(400)]
    mapping, manifest = group_split(keys, holdout_fraction=0.3, seed=0)
    assert manifest["n_groups"] == 400
    assert 0.24 < manifest["n_held_out"] / 400 < 0.36
    assert set(mapping.values()) <= set(SPLITS)


def test_different_seeds_give_different_splits():
    keys = [f"rom{i:02d}" for i in range(40)]
    a, _ = group_split(keys, seed=1)
    b, _ = group_split(keys, seed=2)
    assert a != b


def test_duplicate_and_degenerate_inputs_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        group_split(["a", "b", "a"])
    with pytest.raises(ValueError, match="holdout_fraction"):
        group_split(["a", "b"], holdout_fraction=0.0)
    with pytest.raises(ValueError, match="holdout_fraction"):
        group_split(["a", "b"], holdout_fraction=1.0)


def test_manifest_digest_changes_with_the_assignment():
    _, a = group_split([f"r{i}" for i in range(30)], seed=1)
    _, b = group_split([f"r{i}" for i in range(30)], seed=2)
    assert a["assignment_sha256"] != b["assignment_sha256"]


# --- labels travel with the number ------------------------------------------------------------------

def test_assign_split_refuses_to_guess():
    mapping, _ = group_split(["rom00", "rom01"], seed=0)
    frame = pd.DataFrame({"rom": ["rom00", "rom01", "rom99"], "recall": [0.8, 0.9, 0.7]})
    with pytest.raises(ValueError, match="rom99"):
        assign_split(frame, "rom", mapping)
    ok = assign_split(frame[frame.rom != "rom99"], "rom", mapping)
    assert set(ok["split"]) <= set(SPLITS)


def test_labelled_result_marks_confirmatory_status():
    dev = labelled_result(0.42, "development", note="selected on these ROMs")
    held = labelled_result(0.39, "held_out", lo=0.35, hi=0.43, n_groups=6)
    assert dev["is_confirmatory"] is False
    assert held["is_confirmatory"] is True and held["lo"] == 0.35
    with pytest.raises(ValueError, match="split"):
        labelled_result(0.5, "test")


def test_evaluate_by_split_never_pools():
    mapping, _ = group_split([f"rom{i:02d}" for i in range(10)], holdout_fraction=0.4, seed=3)
    frame = pd.DataFrame({"rom": list(mapping) * 4})
    frame["recall"] = np.where(frame.rom.map(mapping) == "held_out", 0.5, 0.9)
    frame = assign_split(frame, "rom", mapping)

    out = evaluate_by_split(frame, lambda s: s.recall.mean(), group_col="rom")
    assert set(out.split) == set(SPLITS)
    dev = out[out.split == "development"].iloc[0]
    held = out[out.split == "held_out"].iloc[0]
    assert dev.value == pytest.approx(0.9) and held.value == pytest.approx(0.5)
    assert held.is_confirmatory and not dev.is_confirmatory
    assert dev.n_groups + held.n_groups == 10


# --- completeness threshold sensitivity ---------------------------------------------------------------

def test_threshold_sensitivity_reports_a_range_of_limits():
    """The completeness limit is read off a chosen threshold; quoting one hides the choice."""
    rng = np.random.default_rng(0)
    # a power-law population that rolls off below ~6 px, as a detector-limited SFD does
    d = rng.pareto(1.8, 40000) * 4.0 + 3.0
    d = d[rng.random(len(d)) < np.clip((d - 3.0) / 4.0, 0, 1)]
    sens = threshold_sensitivity(d)
    assert list(sens.threshold) == [0.5, 0.6, 0.7, 0.8, 0.9]
    assert sens.limit_px.is_monotonic_increasing or sens.limit_px.nunique() > 1
    assert (sens.n_above_limit > 0).all()
    assert np.isfinite(sens.attrs["limit_spread_px"])
    assert sens.attrs["peak_bin_px"] > 0


def test_threshold_sensitivity_handles_an_empty_population():
    out = threshold_sensitivity([])
    assert len(out) == 0 and "limit_px" in out.columns


def test_threshold_sensitivity_ignores_non_finite_sizes():
    d = np.array([5.0, 6.0, np.nan, np.inf, -1.0, 7.0, 8.0, 9.0])
    out = threshold_sensitivity(d, thresholds=(0.5,))
    assert len(out) == 1 and np.isfinite(out.limit_px.iloc[0])
