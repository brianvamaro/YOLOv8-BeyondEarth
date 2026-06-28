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


def test_binary_mask_to_polygon_crop_is_translation_invariant():
    """The crop-to-bbox optimization must give the same contour (up to the placement offset)
    regardless of how much empty border surrounds the object."""
    obj = np.zeros((7, 7), dtype=np.float32)
    obj[1:6, 1:6] = 1.0  # 5x5 block with a 1px border
    small = binary_mask_to_polygon(obj)
    # same object embedded near the corner of a much larger (slice-size-like) mask
    big = np.zeros((256, 256), dtype=np.float32)
    big[40:45, 60:65] = 1.0  # 5x5 block at (row 40, col 60)
    big_poly = binary_mask_to_polygon(big)
    # contour is (x=col, y=row); strip the placement offset and compare to the small version
    rebased = big_poly - np.array([60 - 1, 40 - 1])  # account for the 1px border in `obj`
    assert small.shape == rebased.shape
    assert np.allclose(np.sort(small, axis=0), np.sort(rebased, axis=0), atol=0.0)


def test_threaded_matches_serial_result_to_df():
    """The ThreadPoolExecutor path in _result_to_df must produce byte-identical detections to
    the serial path (order-preserving), so threading is geometry-neutral."""
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    import torch
    from YOLOv8BeyondEarth.predict import _result_to_df

    rng = np.random.default_rng(3)
    n, sz = 40, 64
    masks = np.zeros((n, sz, sz), dtype=np.float32)
    boxes = np.zeros((n, 6), dtype=np.float32)
    for i in range(n):
        cy, cx = rng.integers(10, sz - 10), rng.integers(10, sz - 10)
        r = int(rng.integers(3, 7))
        masks[i, cy - r:cy + r, cx - r:cx + r] = 1.0
        boxes[i] = [0, 0, 0, 0, rng.uniform(0.2, 0.9), 0]
    result = SimpleNamespace(
        boxes=SimpleNamespace(data=torch.from_numpy(boxes)),
        masks=SimpleNamespace(data=torch.from_numpy(masks)),
    )
    model = SimpleNamespace(confidence_threshold=0.1, category_mapping={"0": "boulder"})
    kw = dict(detection_model=model, has_mask=True, shift_amount=(100, 200), slice_size=sz,
              min_area_threshold=6, downscale_pred=False, contour_method="skimage")
    serial = _result_to_df(result, executor=None, **kw)
    with ThreadPoolExecutor(max_workers=4) as ex:
        threaded = _result_to_df(result, executor=ex, **kw)
    assert len(serial) == len(threaded) and len(serial) > 0
    assert np.array_equal(serial.score.values, threaded.score.values)
    assert serial.is_within_slice.tolist() == threaded.is_within_slice.tolist()
    for ps, pt in zip(serial.polygon.values, threaded.polygon.values):
        assert np.array_equal(ps, pt)


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


def test_ellipse_orientation_recovers_known_angle():
    """boulder_orientations must recover the long-axis azimuth of synthetic ellipses, and
    grid_fraction must equal the uniform baseline for a uniform distribution."""
    import geopandas as gpd
    from shapely.affinity import scale, rotate
    from shapely.geometry import Point
    from YOLOv8BeyondEarth.orientation import (
        boulder_orientations, grid_fraction, grid_fraction_uniform)

    base = scale(Point(0, 0).buffer(1.0, quad_segs=64), xfact=4.0, yfact=1.0)  # 4:1 ellipse, major along x (East)
    geoms, expected = [], []
    for az in (0, 30, 60, 90, 135):
        # azimuth from North clockwise -> rotate the East-major ellipse by (90 - az) CCW
        geoms.append(rotate(base, 90 - az, origin=(0, 0)))
        expected.append(az % 180)
    gdf = gpd.GeoDataFrame({"id": range(len(geoms))}, geometry=geoms, crs="EPSG:32631")
    df = boulder_orientations(gdf, res=0.1, n_workers=1)
    assert len(df) == len(geoms)
    for got, exp in zip(df.angle180.values, expected):
        d = abs(got - exp) % 180.0
        d = min(d, 180.0 - d)
        assert d < 2.0, f"expected azimuth {exp}, got {got:.1f}"
    assert (df.aspect_ra > 3.5).all()  # 4:1 ellipses

    rng = np.random.default_rng(0)
    uniform = rng.uniform(0, 180, 200_000)
    assert abs(grid_fraction(uniform, tol_deg=10) - grid_fraction_uniform(10)) < 0.01


def test_structure_tensor_orientation_convention():
    """structure_tensor_orientation must use the same azimuth convention as the mask angle:
    N-S line -> 0, E-W line -> 90, NW-SE diagonal -> 135 (North=0, East=90, clockwise)."""
    from YOLOv8BeyondEarth.orientation import structure_tensor_orientation

    ns = np.zeros((41, 41)); ns[:, 20] = 1.0           # vertical line in image = North-South
    ew = np.zeros((41, 41)); ew[20, :] = 1.0           # horizontal line = East-West
    nwse = np.eye(41)                                   # row==col: top-left(NW)->bottom-right(SE)
    nesw = np.fliplr(np.eye(41))                        # NE-SW diagonal
    assert min(abs(structure_tensor_orientation(ns)[0] - g) for g in (0, 180)) < 1.0
    assert abs(structure_tensor_orientation(ew)[0] - 90) < 1.0
    assert abs(structure_tensor_orientation(nwse)[0] - 135) < 1.0
    assert abs(structure_tensor_orientation(nesw)[0] - 45) < 1.0
    # a clearly elongated bright feature is more coherent than uniform noise
    assert structure_tensor_orientation(ns)[1] > 0.5


def test_axial_distance_wraps_mod180():
    """axial_distance is the mod-180 distance in [0,90]: 10 vs 170 are 20 apart, 0 vs 90 are 90."""
    from YOLOv8BeyondEarth.orientation import axial_distance
    d = axial_distance([10.0, 170.0, 0.0, 135.0, 200.0], 0.0)  # 200 % 180 = 20
    np.testing.assert_allclose(d, [10.0, 10.0, 0.0, 45.0, 20.0], atol=1e-9)
    assert float(axial_distance([170.0], 10.0)[0]) == pytest.approx(20.0)


def test_refine_peak_and_bootstrap_ci_recover_known_mode():
    """refine_peak finds the mode of a concentrated axial distribution; the bootstrap CI is
    narrow and brackets it. Uses a wrapped-normal cluster at 135."""
    from YOLOv8BeyondEarth.orientation import refine_peak, bootstrap_peak_ci
    rng = np.random.default_rng(0)
    angles = (135.0 + rng.normal(0, 5, 20_000)) % 180.0
    pk = refine_peak(angles)
    assert abs(((pk - 135 + 90) % 180) - 90) < 1.5
    peak, lo, hi = bootstrap_peak_ci(angles, n_boot=200)
    assert lo <= peak <= hi
    assert (hi - lo) < 3.0                      # well-determined -> tight CI


def test_well_resolved_subset_filters_aspect_and_pixels():
    """well_resolved_subset keeps only elongated (aspect>1.35) AND well-resolved (diameter>=dpx_min
    px = diameter_m/res) boulders."""
    from YOLOv8BeyondEarth.orientation import well_resolved_subset
    df = pd.DataFrame({
        "aspect_ra":  [2.0, 1.2, 2.0, 2.0],
        "diameter_m": [10.0, 10.0, 1.0, 10.0],   # at res=0.5: dpx = 20, 20, 2, 20
        "angle180":   [10.0, 20.0, 30.0, 40.0],
    })
    out = well_resolved_subset(df, res=0.5, dpx_min=8)
    assert list(out.angle180) == [10.0, 40.0]    # row1 fails aspect, row2 fails dpx (2<8)


def test_meridian_convergence_zero_for_equirectangular(tmp_path):
    """Equirectangular (eqc) north-up products have zero meridian convergence by construction."""
    from rasterio.transform import from_origin
    from YOLOv8BeyondEarth.orientation import meridian_convergence
    path = tmp_path / "eqc.tif"
    transform = from_origin(0.0, 100000.0, 0.5, 0.5)
    with rio.open(path, "w", driver="GTiff", height=32, width=32, count=1, dtype="uint8",
                  crs="+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +R=3396190 +units=m +no_defs",
                  transform=transform) as dst:
        dst.write(np.zeros((1, 32, 32), dtype="uint8"))
    assert meridian_convergence(path) == 0.0


def test_is_within_slice_edge_vs_interior():
    interior = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])
    assert is_within_slice(interior, 512, 512) is True
    edge = np.array([[-0.5, 10.0], [20.0, 10.0], [20.0, 20.0]])
    assert is_within_slice(edge, 512, 512) is False
