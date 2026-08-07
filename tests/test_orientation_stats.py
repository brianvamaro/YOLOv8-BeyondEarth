"""Tests for the diagonal-fraction / 45-135 asymmetry orientation stats (added for Test 3)."""
import numpy as np
import pytest

from YOLOv8BeyondEarth.orientation import (
    diagonal_fraction,
    grid_fraction,
    grid_fraction_uniform,
    asymmetry_ratio,
    axial_resultant,
    axial_resultant_uniform,
    rayleigh_p,
)


def test_diagonal_fraction_uniform_baseline():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 180, 200_000)
    # a uniform distribution hits the diagonal band at the same rate as the cardinal band
    assert abs(diagonal_fraction(a, 10) - grid_fraction_uniform(10)) < 0.01
    assert abs(asymmetry_ratio(a, 10) - 1.0) < 0.05


def test_diagonal_vs_cardinal_are_complementary_axes():
    # all mass exactly on 135 -> full diagonal fraction, zero cardinal fraction
    a = np.full(1000, 135.0)
    assert diagonal_fraction(a, 10) == 1.0
    assert grid_fraction(a, 10) == 0.0


def test_asymmetry_ratio_direction():
    # 3x as many near 135 as near 45 -> ratio ~3
    a = np.concatenate([np.full(300, 135.0), np.full(100, 45.0)])
    assert abs(asymmetry_ratio(a, 5) - 3.0) < 1e-6
    # wraps correctly: 179 deg is axially ~1 deg from 0/180 (cardinal), not near a diagonal
    assert diagonal_fraction(np.full(100, 179.0), 10) == 0.0


def test_asymmetry_ratio_nan_when_no_45():
    a = np.full(100, 135.0)
    assert np.isnan(asymmetry_ratio(a, 5))


# --- mean resultant length R, and the null that has to travel with it -----------------------------
# The Goal-4 fabric gate was for a time set at "R ~ 0.35" from control sites holding 8-103 boulders,
# where the isotropic expectation of R is itself 0.09-0.31. These pin the small-n behaviour that
# made that bar meaningless, so it cannot be reintroduced by accident.

def test_axial_resultant_endpoints():
    assert axial_resultant(np.full(500, 37.0)) == pytest.approx(1.0)
    # axial: 37 and 217 are the same long axis, so R must stay at 1
    assert axial_resultant(np.concatenate([np.full(250, 37.0),
                                           np.full(250, 217.0)])) == pytest.approx(1.0)
    # two axes 90 deg apart cancel exactly in the doubled-angle frame
    assert axial_resultant(np.concatenate([np.full(250, 0.0),
                                           np.full(250, 90.0)])) == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(axial_resultant([]))


def test_axial_resultant_matches_the_private_copies_it_replaced():
    """Agreement with ``analyze_sam_chirality.axial_mean`` and ``analyze_gate2b._resultant``, the
    two implementations this consolidates. They differ only in whether ``% 180`` is applied before
    doubling — mathematically a no-op, so the agreement is to floating point (~1 ULP) rather than
    bit-exact."""
    rng = np.random.default_rng(7)
    a = rng.uniform(0, 180, 5_000)
    chirality = float(abs(np.mean(np.exp(2j * np.deg2rad(a)))))            # no % 180
    gate2b = float(np.abs(np.mean(np.exp(1j * 2.0 * np.deg2rad(a % 180.0)))))
    assert axial_resultant(a) == pytest.approx(chirality, rel=1e-12)
    assert axial_resultant(a) == pytest.approx(gate2b, rel=1e-12)


def test_isotropic_expectation_formula_matches_monte_carlo():
    """``E[R] = sqrt(pi)/(2*sqrt(n))`` — the reason a small-n R is large for free."""
    rng = np.random.default_rng(11)
    for n, tol in ((8, 0.02), (100, 0.005), (1000, 0.002)):
        draws = rng.uniform(0.0, 180.0, size=(4000, n))
        mc = float(np.abs(np.mean(np.exp(2j * np.deg2rad(draws)), axis=1)).mean())
        assert abs(mc - axial_resultant_uniform(n)) < tol
    # the retracted bar, made explicit: at n=8 an isotropic site is expected to reach R ~ 0.31
    assert axial_resultant_uniform(8) == pytest.approx(0.3133, abs=1e-4)
    assert axial_resultant_uniform(2388) < 0.02


def test_rayleigh_p_is_calibrated():
    """The tail, not just the mean: the criterion `sqrt(-ln(a)/n)` must reject at rate `a`."""
    rng = np.random.default_rng(13)
    for n in (100, 1000):
        draws = rng.uniform(0.0, 180.0, size=(20000, n))
        r = np.abs(np.mean(np.exp(2j * np.deg2rad(draws)), axis=1))
        crit = np.sqrt(-np.log(0.05) / n)
        assert abs(float((r > crit).mean()) - 0.05) < 0.01
        # and the p-value itself is uniform under the null
        assert abs(float(np.mean([rayleigh_p(x, n) < 0.05 for x in r])) - 0.05) < 0.01


def test_rayleigh_p_small_n_R_is_not_a_result():
    """R = 0.35 on 8 boulders is nothing; the same R on 2,000 is overwhelming."""
    assert rayleigh_p(0.35, 8) > 0.3
    assert rayleigh_p(0.35, 2000) < 1e-100
