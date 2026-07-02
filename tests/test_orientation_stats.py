"""Tests for the diagonal-fraction / 45-135 asymmetry orientation stats (added for Test 3)."""
import numpy as np

from YOLOv8BeyondEarth.orientation import (
    diagonal_fraction,
    grid_fraction,
    grid_fraction_uniform,
    asymmetry_ratio,
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
