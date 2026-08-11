"""`_ladder.ladder_config` must not return keys the ladder passes itself.

This exists because the first version of the `--preset goal1_production` path had never been
executed and could not have worked. Two ways:

1. It routed through `predict_production`, whose `presets.resolve` rejects any override key the
   preset does not define — and `sr_factor` is not a preset key, so the SR arm could never run.
2. Rewritten to forward the preset dict to `get_sliced_prediction`, it then passed `save_bbox`,
   `save_prenms` and `output_format` *both* explicitly and inside the preset dict:
   `TypeError: got multiple values for keyword argument 'save_bbox'`.

Both are "the config dict and the explicit kwargs disagree about who owns a key", and both are
caught by checking the returned dict against what `run_ladder` supplies. Data-free: `_ladder.py` is
tracked even though `data/` is not.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

LADDER = Path(__file__).resolve().parents[1] / "data" / "solutionDE" / "_ladder.py"

#: What `run_ladder` passes to `get_sliced_prediction` alongside `**cfg`. Any overlap is a TypeError
#: at runtime, on a path that only executes after the model has loaded.
SUPPLIED_BY_RUN_LADDER = {"in_raster", "detection_model", "confidence_threshold", "slice_size",
                          "sr_factor", "inference_size", "output_dir", "remask_fn", "mask_tag"}


@pytest.fixture(scope="module")
def ladder():
    spec = importlib.util.spec_from_file_location("_ladder_undertest", LADDER)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("preset", [None, "goal1_production"])
def test_config_never_collides_with_what_run_ladder_passes(ladder, preset):
    cfg = ladder.ladder_config(preset)
    assert not (set(cfg) & SUPPLIED_BY_RUN_LADDER), (
        f"{preset!r} config would be passed twice: {sorted(set(cfg) & SUPPLIED_BY_RUN_LADDER)}")


@pytest.mark.parametrize("preset", [None, "goal1_production"])
def test_config_carries_no_preset_bookkeeping(ladder, preset):
    """`resolve` adds `_preset`/`_preset_version`; forwarding those to the writer is a TypeError."""
    assert not [k for k in ladder.ladder_config(preset) if k.startswith("_")]


def test_production_differs_from_legacy_in_exactly_the_documented_keys(ladder):
    """The three deltas behind review finding 4. If this list changes, the finding text must too."""
    legacy, prod = ladder.ladder_config(None), ladder.ladder_config("goal1_production")
    differing = {k for k in set(legacy) | set(prod)
                 if legacy.get(k, "<absent>") != prod.get(k, "<absent>")}
    # Keys legacy leaves to the function defaults, whose defaults happen to equal production's.
    absent_but_equal = {"batch_size", "contour_method", "postprocess_class_agnostic"}
    assert differing - absent_but_equal == {"min_area_threshold", "postprocess_match_threshold",
                                            "max_det"}
    assert (legacy.get("min_area_threshold"), prod["min_area_threshold"]) == (None, 6)
    assert (legacy.get("postprocess_match_threshold", 0.5), prod["postprocess_match_threshold"]) \
        == (0.5, 0.2)
    assert prod["max_det"] == 600


def test_a_preset_that_moved_off_1024_is_a_loud_failure(ladder, monkeypatch):
    """Catalog names hardcode `-is-1024-`, on both the writing and the reading side."""
    from YOLOv8BeyondEarth import presets
    monkeypatch.setitem(presets.PRESETS, "bad", {**presets.GOAL1_PRODUCTION, "inference_size": 640})
    with pytest.raises(ValueError, match="inference_size"):
        ladder.ladder_config("bad")
