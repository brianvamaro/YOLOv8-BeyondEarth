"""Characterization tests for the geometry-neutral Goal-1 refactor.

These pin the behavior of the vectorized polygon helpers against the original per-element
implementations, so future perf work can't silently change the geometry that feeds the
orientation pipeline (Goal 2). All tests are fast and GPU-free.
"""
import numpy as np
import rasterio as rio
from rasterio.transform import from_origin
from shapely.geometry import Polygon
import pandas as pd
import pytest

from YOLOv8BeyondEarth.polygon import (
    add_geometries, binary_mask_to_polygon, is_within_slice, bboxes_to_shp,
)


@pytest.fixture
def small_raster(tmp_path):
    """A tiny north-up GeoTIFF with a non-trivial transform (pixel size 2.0, origin offset)."""
    path = tmp_path / "tiny.tif"
    transform = from_origin(1000.0, 5000.0, 2.0, 2.0)  # west, north, xres, yres
    data = np.zeros((1, 64, 64), dtype=np.uint8)
    with rio.open(path, "w", driver="GTiff", height=64, width=64, count=1,
                  dtype="uint8", crs="EPSG:32631", transform=transform) as dst:
        dst.write(data)
    return path


def _make_df(n=50, seed=1):
    rng = np.random.default_rng(seed)
    polys = []
    for _ in range(n):
        cx, cy = rng.uniform(5, 58), rng.uniform(5, 58)
        r = rng.uniform(1.5, 4.0)
        k = int(rng.integers(5, 10))
        ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
        polys.append(np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=-1))
    return pd.DataFrame({
        "score": rng.uniform(0.1, 0.9, n), "polygon": polys,
        "category_id": np.zeros(n, int), "category_name": ["boulder"] * n,
        "is_within_slice": np.ones(n, bool),
    })


def _add_geometries_naive(in_raster, df):
    """Original per-polygon implementation, kept here as the reference oracle."""
    import geopandas as gpd
    with rio.open(in_raster) as src:
        in_crs = src.meta["crs"]
        geom = []
        for polygon in df.polygon.values:
            xs, ys = rio.transform.xy(src.transform, polygon[:, 1], polygon[:, 0])
            geom.append(Polygon(np.stack([xs, ys], axis=-1)))
        gdf = gpd.GeoDataFrame(df, geometry=geom, crs=in_crs.to_wkt())
        gdf["bbox"] = gdf.apply(lambda r: list(r.geometry.bounds), axis=1)
    return gdf


def test_add_geometries_matches_naive(small_raster):
    df = _make_df()
    fast = add_geometries(small_raster, df.copy())
    ref = _add_geometries_naive(small_raster, df.copy())
    # vectorized affine transform must be bit-identical to per-polygon rio.transform.xy
    assert fast.geometry.geom_equals_exact(ref.geometry, tolerance=0.0).all()
    assert np.allclose(np.array(fast.bbox.tolist()), np.array(ref.bbox.tolist()), atol=0.0)


def test_add_geometries_empty(small_raster):
    empty = _make_df(0)
    gdf = add_geometries(small_raster, empty)
    assert len(gdf) == 0
    assert "bbox" in gdf.columns


def test_bbox_column_is_geometry_bounds(small_raster):
    gdf = add_geometries(small_raster, _make_df())
    bounds = gdf.geometry.bounds
    bbox = np.array(gdf.bbox.tolist())
    assert np.array_equal(bbox, bounds[["minx", "miny", "maxx", "maxy"]].values)


def test_bboxes_to_shp_writes_box_geometry(small_raster, tmp_path):
    gdf = add_geometries(small_raster, _make_df())
    out = tmp_path / "bbox.shp"
    bboxes_to_shp(gdf, out)
    import geopandas as gpd
    written = gpd.read_file(out)
    assert len(written) == len(gdf)
    # each written geometry is the axis-aligned bbox of the corresponding bbox tuple
    for geom, bb in zip(written.geometry, gdf.bbox):
        assert np.allclose(geom.bounds, bb)


def test_binary_mask_to_polygon_square():
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[2:7, 2:7] = 1  # 5x5 solid square
    poly = binary_mask_to_polygon(mask)
    assert poly.ndim == 2 and poly.shape[1] == 2
    # contour traces the square's perimeter (skimage find_contours sits at the 0.5 isoline)
    assert poly[:, 0].min() >= 1.0 and poly[:, 0].max() <= 7.0
    assert poly[:, 1].min() >= 1.0 and poly[:, 1].max() <= 7.0
    # find_contours traces the 0.5 isoline around a 5x5 block -> ~5x5 square (corners clipped
    # by marching squares), area ~24.5.
    poly_shape = Polygon(poly)
    assert poly_shape.area == pytest.approx(24.5, abs=1.5)


def test_is_within_slice_edge_vs_interior():
    interior = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])
    assert is_within_slice(interior, 512, 512) is True
    edge = np.array([[-0.5, 10.0], [20.0, 10.0], [20.0, 20.0]])
    assert is_within_slice(edge, 512, 512) is False
