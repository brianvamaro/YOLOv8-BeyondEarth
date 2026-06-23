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
    add_geometries, binary_mask_to_polygon, binary_mask_to_polygon_cv,
    is_within_slice, bboxes_to_shp,
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


def _ellipse_orientation_deg(polygon_xy):
    """Replicate the orientation pipeline lightly: densify (~0.5 px) then fit a least-squares
    ellipse (skimage EllipseModel, as shptools.geometry.fitEllipse does); return long-axis
    orientation in [0, 180) degrees and the a/b ratio."""
    from shapely.geometry import Polygon as SPoly
    from shapely import segmentize
    from skimage.measure import EllipseModel
    poly = segmentize(SPoly(polygon_xy), 0.5)
    xy = np.asarray(poly.exterior.coords)
    m = EllipseModel()
    if not m.estimate(xy - xy.mean(axis=0)):
        return None, None
    _, _, a, b, theta = m.params
    return np.degrees(theta) % 180.0, max(a, b) / min(a, b)


def test_cv2_vs_skimage_orientation_regression():
    """cv2 contour extraction must not change measured boulder orientation vs skimage.

    Pins the Goal-2 bridge finding (median ~0.5 deg, max <4 deg on real boulders): on clean
    synthetic elongated ellipses, both contour methods recover the same orientation.
    """
    from skimage.draw import ellipse
    for ang in (15, 40, 65, 110, 150):
        img = np.zeros((140, 140), dtype=np.uint8)
        rr, cc = ellipse(70, 70, 14, 34, shape=img.shape, rotation=np.deg2rad(ang))
        img[rr, cc] = 1
        o_sk, ab_sk = _ellipse_orientation_deg(binary_mask_to_polygon(img))
        o_cv, ab_cv = _ellipse_orientation_deg(binary_mask_to_polygon_cv(img))
        assert o_sk is not None and o_cv is not None
        d = abs(o_sk - o_cv) % 180.0
        d = min(d, 180.0 - d)
        assert d < 8.0, f"angle {ang}: skimage {o_sk:.1f} vs cv2 {o_cv:.1f} differ by {d:.1f} deg"
        assert abs(ab_sk - ab_cv) < 0.3  # aspect ratio agreement


def test_is_within_slice_edge_vs_interior():
    interior = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])
    assert is_within_slice(interior, 512, 512) is True
    edge = np.array([[-0.5, 10.0], [20.0, 10.0], [20.0, 20.0]])
    assert is_within_slice(edge, 512, 512) is False
