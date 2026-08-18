"""The stack-parity check's decision logic (scripts/review/measure_stack_parity.py).

The tool answers whether the pipeline reproduces itself on another machine, which gates both the
512-arm Sherlock run and Goal 4. Its numbers come from `compare_catalogs`, which is already tested
by way of the batch-neutrality and device-downscale work; what is new and untested is the *verdict*
rule, and a verdict rule that quietly passes a real difference is worse than no check at all.

The load-bearing test is `test_the_precedent_passes_its_own_bar`: the bar is not invented here, it is
G1-02's shipped batch_size=1-vs-8 result, so that result must classify as acceptable under it.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REVIEW = Path(__file__).resolve().parents[1] / "scripts" / "review"


def _tool():
    sys.path.insert(0, str(_REVIEW))
    spec = importlib.util.spec_from_file_location(
        "measure_stack_parity", _REVIEW / "measure_stack_parity.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["measure_stack_parity"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cmp(frac_identical=1.0, max_dscore=0.0, unmatched_a=0, unmatched_b=0):
    return {"frac_geometry_identical": frac_identical, "max_abs_score_delta": max_dscore,
            "n_unmatched_a": unmatched_a, "n_unmatched_b": unmatched_b}


# --- arm naming -------------------------------------------------------------------------------------

def test_arm_parsing_and_tags_match_the_runner():
    t = _tool()
    assert t.parse_arm("256") == (256, 1)
    assert t.parse_arm("256:2") == (256, 2)
    assert t.arm_tag("512") == "ss512"
    assert t.arm_tag("256:2") == "ss256sr2"        # the layer-name convention run_multiscale_arms uses


def test_the_ladder_under_test_is_the_production_ladder():
    """Parity measured on a subset of arms would not license moving the production path."""
    t = _tool()
    from YOLOv8BeyondEarth import presets
    want = set()
    for s in presets.GOAL3_PRODUCTION["scales"]:
        want.add((int(s), 1) if isinstance(s, int) else (int(s[0]), int(s[1])))
    assert {t.parse_arm(a) for a in t.DEFAULT_ARMS} == want


# --- the verdict rule -------------------------------------------------------------------------------

def test_bit_identical_is_exact():
    t = _tool()
    assert t._verdict(_cmp()) == "EXACT"


def test_the_precedent_passes_its_own_bar():
    """G1-02's batch_size=1-vs-8 is where the bar comes from, so it must clear it.

    A bar the shipping configuration fails would condemn production, not the new stack.
    """
    t = _tool()
    assert t._verdict(_cmp(frac_identical=0.9930, max_dscore=3.6e-4)) == "WITHIN_ACCEPTED_BAR"


def test_worse_than_the_precedent_is_flagged():
    t = _tool()
    assert t._verdict(_cmp(frac_identical=0.95, max_dscore=3.6e-4)) == "EXCEEDS_ACCEPTED_BAR"
    assert t._verdict(_cmp(frac_identical=0.9930, max_dscore=1e-2)) == "EXCEEDS_ACCEPTED_BAR"


def test_differing_detections_outrank_perfect_geometry():
    """The failure that matters most: the catalogs disagree about which boulders exist.

    Every matched pair can be byte-identical while one stack found rocks the other did not, and a rule
    that looked only at matched geometry would call that EXACT.
    """
    t = _tool()
    assert t._verdict(_cmp(unmatched_b=1)) == "DIFFERENT_DETECTIONS"
    assert t._verdict(_cmp(unmatched_a=3)) == "DIFFERENT_DETECTIONS"


# --- the guards on what may be compared -------------------------------------------------------------

def _identity_dir(tmp_path, label, raster_sha, weights_sha="w", preset_version=2):
    d = tmp_path / label
    d.mkdir(parents=True, exist_ok=True)
    (d / "identity.json").write_text(json.dumps({
        "label": label, "code": {"torch": "2.11.0"},
        "raster": {"sha256": raster_sha}, "weights": {"sha256": weights_sha},
        "preset": "goal1_production", "preset_version": preset_version,
        "arms": ["256"], "repeats": 2, "counts": {}}))
    return d


def test_comparing_across_different_rasters_is_refused(tmp_path):
    """Two stacks run on two different crops would produce a difference that means nothing."""
    t = _tool()
    a = _identity_dir(tmp_path, "a", "sha_one")
    b = _identity_dir(tmp_path, "b", "sha_two")
    args = type("A", (), {"a": str(a), "b": str(b), "raster": "r.tif", "out": str(tmp_path)})()
    with pytest.raises(SystemExit, match="raster sha256 differs"):
        t.compare(args)


def test_comparing_across_different_weights_is_refused(tmp_path):
    t = _tool()
    a = _identity_dir(tmp_path, "a", "sha", weights_sha="w1")
    b = _identity_dir(tmp_path, "b", "sha", weights_sha="w2")
    args = type("A", (), {"a": str(a), "b": str(b), "raster": "r.tif", "out": str(tmp_path)})()
    with pytest.raises(SystemExit, match="weights sha256 differs"):
        t.compare(args)


def test_comparing_across_preset_versions_is_refused(tmp_path):
    """PRESET_VERSION is bumped exactly when the preset's values change (its documented contract)."""
    t = _tool()
    a = _identity_dir(tmp_path, "a", "sha", preset_version=2)
    b = _identity_dir(tmp_path, "b", "sha", preset_version=3)
    args = type("A", (), {"a": str(a), "b": str(b), "raster": "r.tif", "out": str(tmp_path)})()
    with pytest.raises(SystemExit, match="preset_version"):
        t.compare(args)
