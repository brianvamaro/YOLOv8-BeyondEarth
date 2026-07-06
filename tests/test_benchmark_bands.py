"""Tests for the size-band orientation readouts + the re-mask gating sweep (Stage-2 / G cards)."""
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from YOLOv8BeyondEarth.benchmark import (
    orientation_by_band, remask_gating_sweep, match_detections, DEFAULT_SIZE_BANDS,
)


def test_orientation_by_band_isolates_a_locked_band():
    # 5-px boulders all locked on the 135 diagonal; 12-px boulders uniform. Elongated (aspect 2).
    rng = np.random.default_rng(0)
    n = 4000
    dpx = np.concatenate([np.full(n, 5.0), np.full(n, 12.0)])
    angle = np.concatenate([np.full(n, 135.0), rng.uniform(0, 180, n)])
    aspect = np.full(2 * n, 2.0)
    tab = orientation_by_band(angle, aspect, dpx, grid_axis="diagonal")
    row = {r["band"]: r for _, r in tab.iterrows()}
    assert row["4-6"]["frac"] == pytest.approx(1.0)          # locked band caught
    assert row["10-14"]["frac"] == pytest.approx(tab.attrs["uniform"], abs=0.03)  # tail uniform


def test_orientation_by_band_drops_round_boulders_and_small_bands():
    # round boulders (aspect < 1.35) are excluded; bands under min_n do not appear.
    angle = np.full(100, 135.0)
    aspect = np.full(100, 1.1)          # all round -> elongated subset empty
    dpx = np.full(100, 7.0)
    tab = orientation_by_band(angle, aspect, dpx, grid_axis="diagonal", min_n=30)
    assert tab.loc[tab.band == "all", "n"].item() == 0
    assert "6-8" not in set(tab.band)  # no band clears min_n


def test_remask_gating_sweep_monotone_and_endpoints():
    # baseline all locked on 135; re-mask is uniform. Gating in more (larger tau) must lower frac.
    rng = np.random.default_rng(1)
    n = 6000
    dpx = rng.uniform(3, 16, n)
    angle_base = np.full(n, 135.0)
    angle_remask = rng.uniform(0, 180, n)
    aspect = np.full(n, 2.0)
    sweep = remask_gating_sweep(dpx, aspect, angle_base, angle_remask,
                                taus=[0, 6, 10, np.inf], grid_axis="diagonal")
    by_tau = {r["tau"]: r for _, r in sweep.iterrows()}
    assert by_tau[0]["remask_frac"] == 0.0 and by_tau[0]["frac"] == pytest.approx(1.0)
    assert by_tau["inf"]["remask_frac"] == 1.0
    assert by_tau["inf"]["frac"] == pytest.approx(sweep.attrs["uniform"], abs=0.03)
    # more re-masking -> lower diagonal fraction (monotone non-increasing)
    fracs = [by_tau[t]["frac"] for t in (0, 6, 10, "inf")]
    assert all(fracs[i] >= fracs[i + 1] - 1e-9 for i in range(len(fracs) - 1))


def test_match_detections_greedy_one_to_one():
    # two points in A, one in B nearby -> exactly one pair, the closer A wins.
    import geopandas as gpd
    a = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(0.5, 0)])
    b = gpd.GeoDataFrame(geometry=[Point(0.1, 0)])
    m = match_detections(a, b, max_dist=1.0)
    assert len(m) == 1
    assert m.iloc[0]["idx_a"] == 0 and m.iloc[0]["idx_b"] == 0  # nearest wins
