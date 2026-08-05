"""Arithmetic and cache-identity guards for the `scripts/review/` probes (items B1, B2).

The 2026-08-05 review found that `grep -rn "run_multiscale_arms\\|benchmark_driver\\|
predict_production" tests/` returned nothing: the probes that produce the numbers in the backlog and
the paper had no tests at all, and that is how B1 reached HEAD *inside the very probe that found the
same error class elsewhere*. These cover the two pure-logic parts, which need no GPU and no raster:

- **B1** — `fraction_discarded_at_cap`: truncation is `min(c, cap)` per slice, so the aggregate must
  be summed per slice. Pushing the *mean* through it, then clamping at 0, published `0.0` beside
  `"cap_300_engages": true`.
- **B2** — the operating-point catalog cache, which was keyed on confidence alone while the ground
  truth was rebuilt from a fresh sample.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REVIEW = Path(__file__).resolve().parents[1] / "scripts" / "review"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _REVIEW / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- B1: the truncated fraction ------------------------------------------------------------------

probe_max_det = _load("probe_max_det")
frac = probe_max_det.fraction_discarded_at_cap


def _old_formula(counts, cap):
    """What this replaced, kept so the tests can show where the two disagree."""
    mean = sum(counts) / len(counts)
    return max(0.0, 1 - cap / mean)


def test_fraction_discarded_is_hand_computable():
    # 3 slices, cap 300: lose 37 + 0 + 0. The denominator is what the model *produced* (787), not
    # what survived the cap (750) — this is the fraction of detections the cap costs you.
    assert frac([337, 250, 200], 300) == pytest.approx(37 / 787)
    # every slice over the cap: lose 100+200+300 of 400+500+600
    assert frac([400, 500, 600], 300) == pytest.approx(600 / 1500)
    assert frac([100, 200], 300) == 0.0            # nothing truncated
    assert frac([300, 300], 300) == 0.0            # exactly at the cap loses nothing
    assert frac([], 300) == 0.0                    # no slices
    assert frac([0, 0], 300) == 0.0                # occupied-but-empty, no ZeroDivisionError


def test_partial_truncation_is_not_reported_as_zero():
    """The published defect. `max_det_probe-hirise-ss256.json` recorded 3 of 16 slices at the cap,
    mean 266.8 capped against 271.1 uncapped — and published `fraction_discarded_at_300: 0.0`
    alongside `cap_300_engages: true`, because the mean (271.1) sits *below* the cap even though
    individual slices are above it."""
    counts = [337, 320, 305] + [260] * 13          # 3 of 16 over the cap, mean 271.1 < 300
    assert sum(counts) / len(counts) < 300
    assert _old_formula(counts, 300) == 0.0        # what was published
    assert frac(counts, 300) > 0.0                 # what is true
    assert frac(counts, 300) == pytest.approx((37 + 20 + 5) / sum(counts))


def test_the_two_forms_agree_only_when_every_slice_is_saturated():
    """Why the 512 cells were right and the 256 cells were wrong: with 16/16 slices at the cap the
    capped mean *is* the cap, so `1 - cap/mean_uncapped` is exact. That regime is the whole domain
    the old formula was valid on, and nothing said so."""
    saturated = [756, 800, 1013, 900]              # every slice above 300
    assert frac(saturated, 300) == pytest.approx(_old_formula(saturated, 300), abs=1e-12)

    partial = [310, 200, 150, 100]                 # one slice above 300, mean well below
    assert _old_formula(partial, 300) == 0.0
    assert frac(partial, 300) > 0.0


@pytest.mark.parametrize("counts", [[301], [500, 1], [301] + [10] * 99, [1000] + [1] * 999])
def test_any_slice_over_the_cap_implies_a_positive_fraction(counts):
    """The invariant the old formula violated, and the one that makes the published contradiction
    impossible: if `cap_300_engages` is true then the discarded fraction cannot be zero."""
    assert max(counts) > 300
    assert frac(counts, 300) > 0.0


def test_recorded_probe_artifacts_reproduce_the_corrected_figures():
    """Against the real numbers on disk. Both cells' means come from the same 16 slices, so
    `1 - mean_at_cap/mean_uncapped` is the exact fraction for those runs."""
    d = Path(__file__).resolve().parents[1] / "data" / "review"
    cases = {                     # artifact -> (mean_at_cap, mean_uncapped, published, corrected)
        "max_det_probe-hirise-ss256.json": (266.8, 271.1, 0.0, 0.016),
        "max_det_probe-hirise-ss256-topslices.json": (293.2, 317.4, 0.055, 0.076),
    }
    for name, (m_cap, m_free, published, corrected) in cases.items():
        path = d / name
        if not path.exists():
            pytest.skip(f"{name} not present (data/ is gitignored)")
        cells = [c for c in json.loads(path.read_text())["cells"]
                 if not c.get("failed") and c["conf"] == 0.1]
        capped = [c for c in cells if c["max_det"] == 300]
        free = [c for c in cells if not c["cap_bound"]]
        assert capped and free, name
        assert max(c["detections_per_slice_mean"] for c in capped) == m_cap
        assert max(c["detections_per_slice_mean"] for c in free) == m_free
        assert 1 - m_cap / m_free == pytest.approx(corrected, abs=5e-4)
        assert _old_formula([m_free], 300) == pytest.approx(published, abs=5e-4)


# --- B2: the operating-point catalog cache -------------------------------------------------------

@pytest.fixture
def op():
    return _load("measure_operating_point")


def _identity(op, tmp_path, stems, conf=0.10, seed=0, n_tiles=200):
    weights = tmp_path / "w.pt"
    if not weights.exists():
        weights.write_bytes(b"weights")
    return op.cache_identity([tmp_path / f"{s}.png" for s in stems], weights, conf,
                             "validation", seed, n_tiles)


def test_a_different_sample_is_a_different_identity(op, tmp_path):
    a = _identity(op, tmp_path, ["t1", "t2", "t3"])
    assert _identity(op, tmp_path, ["t3", "t2", "t1"]) == a, "order must not matter"
    assert _identity(op, tmp_path, ["t1", "t2", "t4"]) != a, "a different draw must differ"
    assert _identity(op, tmp_path, ["t1", "t2"]) != a, "a different size must differ"
    assert _identity(op, tmp_path, ["t1", "t2", "t3"], conf=0.25) != a


def test_reuse_is_refused_when_the_sample_changed(op, tmp_path):
    """The B2 failure: a cached 200-tile catalog rescored against a freshly drawn 250-tile label set.
    `sample_tiles` draws are not nested, so 62 of 200 overlapped — 138 truths with no predictions and
    138 predictions counted as false positives, and `ap_eval` raises nothing for a missing image."""
    cached = tmp_path / "catalog-conf010.gpkg"
    cached.write_bytes(b"not really a gpkg")
    want = _identity(op, tmp_path, ["t1", "t2", "t3"])
    cached.with_suffix(".identity.json").write_text(json.dumps(want))
    assert op.check_cached_identity(cached, want, refresh=False) is True

    other = _identity(op, tmp_path, ["t1", "t2", "t9"])
    with pytest.raises(FileExistsError, match="image_ids_sha256"):
        op.check_cached_identity(cached, other, refresh=False)


def test_reuse_is_refused_when_the_arms_changed(op, tmp_path):
    """The silent variant, and the one that mattered: re-running to measure X-06's effect returned
    the pre-X-06 catalog and reported AP50 0.4992 unchanged. `max_det` lives in the arm config, so a
    stale catalog now fails instead of being rescored."""
    cached = tmp_path / "catalog-conf010.gpkg"
    cached.write_bytes(b"x")
    want = _identity(op, tmp_path, ["t1"])
    stale = json.loads(json.dumps(want))
    stale["arm_config_hash"] = "deadbeef"
    stale["preset_version"] = 1
    cached.with_suffix(".identity.json").write_text(json.dumps(stale))
    with pytest.raises(FileExistsError, match="arm_config_hash|preset_version"):
        op.check_cached_identity(cached, want, refresh=False)


def test_an_unidentified_cached_catalog_is_refused_not_trusted(op, tmp_path):
    """A catalog written before the sidecar existed cannot be verified, so it must not be reused —
    the on-disk 200-tile catalog is exactly this case."""
    cached = tmp_path / "catalog-conf010.gpkg"
    cached.write_bytes(b"x")
    with pytest.raises(FileExistsError, match="cannot be verified"):
        op.check_cached_identity(cached, _identity(op, tmp_path, ["t1"]), refresh=False)


def test_refresh_overrides_everything(op, tmp_path):
    cached = tmp_path / "catalog-conf010.gpkg"
    cached.write_bytes(b"x")
    assert op.check_cached_identity(cached, _identity(op, tmp_path, ["t1"]), refresh=True) is False


def test_the_documented_command_reproduces_the_published_sample_size(op):
    """The published table is 200 tiles; the driver defaulted to 250, so its own documented
    no-argument command measured a different sample than the one in the backlog."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tiles", type=int, default=200)
    assert parser.parse_args([]).n_tiles == 200
    src = (_REVIEW / "measure_operating_point.py").read_text()
    assert '"--n-tiles", type=int, default=200' in src
