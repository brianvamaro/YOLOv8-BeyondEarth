"""Area calibration on one convention, one-to-one (review item G2-04).

The published SAM/human (1.06-1.27x) and YOLO/human (1.58-2.13x) factors compared a model
**integer pixel count** against a human **continuous polygon area**. Part of every one of them is
therefore the boundary-convention gap rather than a real difference in mask size, which makes them
unusable as calibration factors for Goal 4.
"""
import geopandas as gpd
import numpy as np
import pytest
import shapely

from YOLOv8BeyondEarth.orientation_validation import (
    AREA_CONVENTIONS, area_calibration, convention_sensitivity, mask_areas)


def _block(n, pad=3):
    m = np.zeros((n + 2 * pad, n + 2 * pad), dtype=np.float32)
    m[pad:pad + n, pad:pad + n] = 1.0
    return m


def test_the_three_conventions_measure_a_solid_block_differently():
    """The gap, in one assertion. A 5x5 block is 24.5 / 16.0 / 25 depending on who is asking."""
    m = [_block(5)]
    assert mask_areas(m, "polygon_skimage")[0] == pytest.approx(24.5, abs=1e-6)
    assert mask_areas(m, "polygon_cv2")[0] == pytest.approx(16.0, abs=1e-6)
    assert mask_areas(m, "pixel_count")[0] == pytest.approx(25.0, abs=1e-6)


def test_pixel_count_overstates_the_production_polygon():
    """The specific mix-up in T35: model pixel sums against human polygon areas. On small boulders
    the count is the larger number, so it inflates a model/human ratio."""
    masks = [_block(n) for n in (4, 5, 6, 8, 12)]
    counts = mask_areas(masks, "pixel_count")
    polys = mask_areas(masks, "polygon_skimage")
    assert np.all(counts > polys)
    assert np.median(counts / polys) > 1.0


def test_area_scales_with_resolution_squared():
    m = [_block(10)]
    assert mask_areas(m, "polygon_skimage", res=0.5)[0] == pytest.approx(
        mask_areas(m, "polygon_skimage", res=1.0)[0] * 0.25)


def test_unknown_convention_is_rejected():
    with pytest.raises(ValueError, match="convention"):
        mask_areas([_block(5)], convention="pixels")


def test_convention_sensitivity_reports_the_ratio_to_production():
    masks = [_block(n) for n in (5, 7, 9, 11)]
    sens = convention_sensitivity(masks)
    assert list(sens.convention) == list(AREA_CONVENTIONS)
    prod = sens[sens.convention == "polygon_skimage"].iloc[0]
    assert prod.ratio_to_production == pytest.approx(1.0)
    cv2_row = sens[sens.convention == "polygon_cv2"].iloc[0]
    count_row = sens[sens.convention == "pixel_count"].iloc[0]
    assert cv2_row.ratio_to_production < 1.0        # inscribed
    assert count_row.ratio_to_production > 1.0      # circumscribed
    assert (sens.n == len(masks)).all()


def test_convention_gap_is_largest_for_the_smallest_boulders():
    """Why it matters for an SFD: the bias is size-dependent, so a single median multiplier
    cannot repair it."""
    small = convention_sensitivity([_block(4)])
    large = convention_sensitivity([_block(40)])
    gap = lambda s: abs(s[s.convention == "polygon_cv2"].iloc[0].ratio_to_production - 1.0)
    assert gap(small) > 3 * gap(large)


# --- the calibration itself ------------------------------------------------------------------------

def _gdf(boxes, **cols):
    return gpd.GeoDataFrame(dict(cols), geometry=[shapely.box(*b) for b in boxes],
                            crs="EPSG:32631")


def test_calibration_is_one_to_one_and_reports_an_interval():
    human = _gdf([(0, 0, 2, 2), (20, 20, 22, 22), (40, 40, 42, 42), (60, 60, 62, 62)])
    model = _gdf([(0, 0, 2.4, 2.4), (20, 20, 22.4, 22.4),
                  (40, 40, 42.4, 42.4), (60, 60, 62.4, 62.4)])
    df, stats = area_calibration(model, human, min_radius_m=2.0, n_boot=300)
    assert stats["n_matched"] == 4
    assert df.ref_index.is_unique and df.model_index.is_unique
    assert stats["median_ratio"] == pytest.approx(1.44, abs=0.01)      # (2.4/2)^2
    assert stats["median_ratio_lo"] <= stats["median_ratio"] <= stats["median_ratio_hi"]
    assert stats["convention"].startswith("polygon_skimage")


def test_calibration_cluster_bootstrap_widens_the_interval():
    """Boulders in one tile are not independent evidence; the interval has to say so."""
    rng = np.random.default_rng(0)
    boxes_h, boxes_m, groups = [], [], []
    for tile in range(5):
        scale = 1.2 + 0.25 * rng.normal()          # each tile has its own systematic offset
        for j in range(12):
            x, y = 100 * tile + 6 * j, 0
            boxes_h.append((x, y, x + 2, y + 2))
            s = 2 * np.sqrt(scale)
            boxes_m.append((x, y, x + s, y + s))
            groups.append(f"tile{tile}")
    human, model = _gdf(boxes_h), _gdf(boxes_m)

    _, plain = area_calibration(model, human, min_radius_m=2.0, n_boot=400)
    _, clustered = area_calibration(model, human, min_radius_m=2.0, n_boot=400,
                                    groups=np.array(groups))
    assert clustered["n_clusters"] == 5
    width = lambda s: s["median_ratio_hi"] - s["median_ratio_lo"]
    assert width(clustered) > 2 * width(plain)


def test_calibration_refuses_to_hide_unmatched_boulders():
    human = _gdf([(0, 0, 2, 2), (500, 500, 502, 502)])
    model = _gdf([(0, 0, 2, 2)])
    _, stats = area_calibration(model, human, min_radius_m=2.0, n_boot=100)
    assert stats["n_matched"] == 1 and stats["n_ref_unmatched"] == 1
    assert stats["recall"] == pytest.approx(0.5)


def test_empty_cohort_returns_nan_not_a_crash():
    human = _gdf([(0, 0, 2, 2)])
    model = _gdf([(900, 900, 902, 902)])
    df, stats = area_calibration(model, human, min_radius_m=1.0, n_boot=50)
    assert len(df) == 0 and np.isnan(stats["median_ratio_lo"])
