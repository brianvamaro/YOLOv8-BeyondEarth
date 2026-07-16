"""Tests for the half-plane fraction (f_lower) — the SAM2 half-plane-imbalance metric added for the
SAM2 orientation-artifact validation plan (docs/sam2_orientation_validation_plan.md)."""
import numpy as np
import pandas as pd

from YOLOv8BeyondEarth.orientation import (
    half_plane_fraction,
    half_plane_fraction_uniform,
)
from YOLOv8BeyondEarth.benchmark import half_plane_by_band


def test_half_plane_uniform_baseline():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 180, 200_000)
    # reflection-symmetric input splits [0,90) vs [90,180) at the 0.5 null
    assert abs(half_plane_fraction(a) - half_plane_fraction_uniform()) < 0.01
    assert half_plane_fraction_uniform() == 0.5


def test_half_plane_edge_cases():
    # all mass in the lower half-plane -> 1.0; all in the upper -> 0.0
    assert half_plane_fraction(np.full(1000, 30.0)) == 1.0
    assert half_plane_fraction(np.full(1000, 120.0)) == 0.0
    # 90 is the exclusive upper edge of [0,90); 0 is included
    assert half_plane_fraction(np.array([90.0])) == 0.0
    assert half_plane_fraction(np.array([0.0])) == 1.0
    # wrap: 180 % 180 == 0 -> lower half
    assert half_plane_fraction(np.array([180.0])) == 1.0


def test_half_plane_drops_nan_and_handles_empty():
    a = np.array([10.0, np.nan, 100.0, np.nan])
    assert half_plane_fraction(a) == 0.5           # one of the two finite values is < 90
    assert np.isnan(half_plane_fraction(np.array([])))
    assert np.isnan(half_plane_fraction(np.array([np.nan, np.nan])))


def test_half_plane_by_band_elongated_and_bands():
    rng = np.random.default_rng(1)
    n = 5000
    df = pd.DataFrame({
        "angle180": rng.uniform(0, 180, n),
        "aspect": rng.uniform(1.0, 3.0, n),
        "dpx": rng.uniform(3, 20, n),
    })
    tab = half_plane_by_band(df["angle180"], df["aspect"], df["dpx"])
    assert tab.attrs["uniform"] == 0.5
    assert tab.iloc[0]["band"] == "all"
    # elongated subset only (aspect > 1.35)
    assert tab.iloc[0]["n"] == int((df["aspect"] > 1.35).sum())
    # uniform truth -> every band near 0.5
    assert (tab["f_lower"].sub(0.5).abs() < 0.06).all()
