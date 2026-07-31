"""Valid-data coverage and density denominators (review item G3-07).

The HiRISE test scene is a skewed swath inside a north-up bounding box: 45.6% of the rectangle is
acquisition fill, so a density normalised by the rectangle is understated 1.84x. Nothing in the
file declares that — `nodata` is unset and the mask reports all-valid — so validity has to be
inferred from the pixel values.
"""
import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_origin

from YOLOv8BeyondEarth.coverage import (
    coverage_report, density_per_area, valid_area, valid_footprint, valid_pixel_count)


def _swath(tmp_path, width=200, height=200, res=2.0, name="swath.tif"):
    """A skewed swath in a north-up box, like a real HiRISE product: the corners are fill."""
    path = tmp_path / name
    data = np.zeros((height, width), dtype=np.uint8)
    for r in range(height):
        start = int(0.3 * r)
        data[r, start:start + 100] = 200
    with rio.open(path, "w", driver="GTiff", height=height, width=width, count=1,
                  dtype="uint8", crs="EPSG:32631",
                  transform=from_origin(0.0, height * res, res, res)) as dst:
        dst.write(data, 1)
    return path, data


def test_valid_pixel_count_is_exact(tmp_path):
    path, data = _swath(tmp_path)
    n_valid, n_total = valid_pixel_count(path)
    assert n_valid == int((data != 0).sum())
    assert n_total == data.size


def test_decimated_count_approximates_the_exact_one(tmp_path):
    path, data = _swath(tmp_path, width=480, height=480)
    exact = valid_pixel_count(path)
    approx = valid_pixel_count(path, decimate=6)
    assert approx[0] / approx[1] == pytest.approx(exact[0] / exact[1], abs=0.02)


def test_valid_area_uses_pixel_size_and_reports_the_rectangle_too(tmp_path):
    path, data = _swath(tmp_path, res=2.0)
    info = valid_area(path)
    px = 4.0                                          # 2 m x 2 m
    assert info["pixel_area"] == pytest.approx(px)
    assert info["valid_area"] == pytest.approx(int((data != 0).sum()) * px, rel=1e-9)
    assert info["rect_area"] == pytest.approx(data.size * px)
    assert info["valid_fraction"] < 1.0


def test_coverage_report_states_the_overstatement_and_that_nodata_was_inferred(tmp_path):
    path, data = _swath(tmp_path)
    rep = coverage_report(path)
    expected = data.size / int((data != 0).sum())
    assert rep["overstatement_if_rectangle_used"] == pytest.approx(expected, rel=1e-9)
    assert rep["declared_nodata"] is None and rep["nodata_inferred"] is True
    assert rep["res"] == pytest.approx((2.0, 2.0))
    assert rep["crs"]


def test_fully_valid_raster_reports_no_overstatement(tmp_path):
    path = tmp_path / "full.tif"
    with rio.open(path, "w", driver="GTiff", height=64, width=64, count=1, dtype="uint8",
                  crs="EPSG:32631", transform=from_origin(0.0, 64.0, 1.0, 1.0)) as dst:
        dst.write(np.full((1, 64, 64), 200, dtype=np.uint8))
    rep = coverage_report(path)
    assert rep["valid_fraction"] == 1.0
    assert rep["overstatement_if_rectangle_used"] == pytest.approx(1.0)


# --- footprint geometry ------------------------------------------------------------------------

def test_footprint_area_matches_the_pixel_count(tmp_path):
    path, data = _swath(tmp_path)
    gdf = valid_footprint(path, decimate=1)
    total = float(gdf.geometry.area.sum())
    assert total == pytest.approx(int((data != 0).sum()) * 4.0, rel=1e-9)


def test_footprint_preserves_interior_holes(tmp_path):
    """The reason `true_footprint`'s convex hull cannot serve as observed area: the real scene has
    526 distinct zero regions and only half are the outer wedges."""
    path = tmp_path / "holed.tif"
    data = np.full((100, 100), 200, dtype=np.uint8)
    data[40:60, 40:60] = 0                              # an interior gap
    with rio.open(path, "w", driver="GTiff", height=100, width=100, count=1, dtype="uint8",
                  crs="EPSG:32631", transform=from_origin(0.0, 100.0, 1.0, 1.0)) as dst:
        dst.write(data, 1)

    gdf = valid_footprint(path, decimate=1)
    assert len(gdf) == 1
    poly = gdf.geometry.iloc[0]
    assert len(poly.interiors) == 1, "the interior gap was filled"
    assert poly.area == pytest.approx(100 * 100 - 20 * 20)
    assert poly.convex_hull.area == pytest.approx(100 * 100)   # what a hull would have claimed


def test_footprint_min_area_drops_specks(tmp_path):
    path = tmp_path / "specks.tif"
    data = np.zeros((100, 100), dtype=np.uint8)
    data[10:60, 10:60] = 200                            # the real swath
    data[90, 90] = 200                                  # a single-pixel speck
    with rio.open(path, "w", driver="GTiff", height=100, width=100, count=1, dtype="uint8",
                  crs="EPSG:32631", transform=from_origin(0.0, 100.0, 1.0, 1.0)) as dst:
        dst.write(data, 1)
    assert len(valid_footprint(path, decimate=1)) == 2
    assert len(valid_footprint(path, decimate=1, min_area_px=4)) == 1


# --- densities carry their denominator -------------------------------------------------------------

def test_density_reports_which_area_it_used():
    valid = density_per_area(1000, 5.0e5, area_units="m2", per="km2")
    rect = density_per_area(1000, 9.2e5, area_units="m2", per="km2")
    assert valid["density"] == pytest.approx(2000.0)
    assert rect["density"] == pytest.approx(1000 / 0.92)
    assert valid["density"] > rect["density"]           # the rectangle understates
    assert valid["area_m2"] == 5.0e5 and valid["per"] == "km2"


def test_density_of_zero_area_is_nan_not_infinite():
    assert np.isnan(density_per_area(10, 0.0)["density"])
