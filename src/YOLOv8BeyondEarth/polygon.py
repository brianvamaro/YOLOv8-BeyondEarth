"""Mask -> polygon -> georeferenced geometry, and THE pixel-boundary convention.

Pixel-boundary convention (review item X-03)
--------------------------------------------
Every production geometry in this project — sizes, SFDs, area calibration, IoU, AP, orientation
cohorts — is traced by :func:`binary_mask_to_polygon`, i.e. **skimage marching squares at the 0.5
level set** of the mask. A pixel is a unit square centred on its integer index, so the boundary of
a solid block runs half a pixel outside the outermost pixel *centres*, and marching squares clips
the four corners at 45 degrees. A solid 5x5 block therefore encloses **~24.5**, not 25 (the 2%
shortfall is the corner clipping, and it shrinks as blobs grow).

:func:`binary_mask_to_polygon_cv` is a *different* convention: ``cv2.findContours`` follows
boundary pixel **centres**, so the same 5x5 block encloses **16.0** — a 36% area deficit and about
19% on equivalent diameter, worst for the smallest boulders. It is fast and its effect on
*orientation* was measured to be negligible (~0.5 deg median), which is what it was adopted for.
It is **not** area-calibrated and must not feed size, SFD, density or area-comparison work.
``tests/test_pixel_convention.py`` pins both numbers so the gap cannot drift unnoticed.

The mask-support level (:data:`MASK_SUPPORT_LEVEL`) is the same 0.5, so the area gate counts
exactly the pixels the traced polygon encloses (review item G1-06).
"""
import cv2
import numpy as np
import shapely
import skimage
import rasterio as rio
import geopandas as gpd

from shapely.geometry import (box, Polygon)

#: The level set traced by the production contour extractor, and therefore the threshold that
#: defines a pixel as "supported" for the area gate. Fractional masks (an ``INTER_AREA``-downscaled
#: prototype) must use the same number in both places or the gate rejects boulders the tracer
#: would happily have outlined.
MASK_SUPPORT_LEVEL = 0.5


def is_within_slice(polygon, slice_height, slice_width):
    """
    Returns True if the polygon is touching one of the edge of the slice else False
    """
    at_edge12 = np.any(np.any(polygon == -0.5, axis=0) == True)
    at_edge3 = np.any(polygon[:, 0] == slice_width - 0.5, axis=0)
    at_edge4 = np.any(polygon[:, 1] == slice_height - 0.5, axis=0)
    is_intersecting_edge = np.any(np.array([at_edge12, at_edge3, at_edge4]) == True)
    return (False if is_intersecting_edge else True)

def shift_polygon(polygon, shift_x, shift_y):
            return (np.stack([polygon[:, 0] + shift_x, polygon[:, 1] + shift_y], axis=-1))

def get_bbox_index(row, src):
    row_min, col_min = src.index(x=row.bbox[0], y=row.bbox[1])
    row_max, col_max = src.index(x=row.bbox[2], y=row.bbox[3])
    return [col_min, row_min, col_max, row_max]

def get_bbox_xy_shapely(bbox, src):
    """[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]
    but need to be reversed"""
    return [src.xy(bbox[1],bbox[0]),
            src.xy(bbox[3],bbox[0]),
            src.xy(bbox[3],bbox[2]),
            src.xy(bbox[1],bbox[2]),
            src.xy(bbox[1],bbox[0])]

def row_bbox(row):
    return(list(row.geometry.bounds))

def row_bbox_to_shapely(row):
    return(box(*row.bbox))

def add_geometries(in_raster, df):
    with rio.open(in_raster) as src:
        in_crs = src.meta["crs"]
        transform = src.transform

    polygons = df.polygon.values
    if len(polygons) == 0:
        gdf = gpd.GeoDataFrame(df, geometry=[], crs=in_crs.to_wkt())
        gdf["bbox"] = []
        return gdf

    # Apply the affine transform to every vertex of every polygon in a single vectorized
    # pass (equivalent to looping rio.transform.xy per polygon, which used pixel centers ->
    # the +0.5 offset). a,b,c / d,e,f are the affine coefficients; b and d are 0 for
    # north-up rasters but are kept for generality.
    vertex_counts = np.fromiter((len(p) for p in polygons), dtype=np.int64, count=len(polygons))
    stacked = np.concatenate(polygons, axis=0)
    cols = stacked[:, 0] + 0.5
    rows = stacked[:, 1] + 0.5
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    xs = a * cols + b * rows + c
    ys = d * cols + e * rows + f
    coords = np.stack([xs, ys], axis=-1)

    # Vectorized polygon construction from the stacked coords: build every linear ring in
    # one call (per-vertex ring index), then assemble polygons. Equivalent to the previous
    # per-polygon ``Polygon(coords)`` loop (shapely auto-closes open rings the same way), but
    # avoids one Python-level shapely call per boulder — matters at ~1e6 detections.
    ring_indices = np.repeat(np.arange(len(polygons)), vertex_counts)
    rings = shapely.linearrings(coords, indices=ring_indices)
    boulder_geometry = shapely.polygons(rings)

    gdf = gpd.GeoDataFrame(df, geometry=boulder_geometry, crs=in_crs.to_wkt())
    gdf["bbox"] = gdf.geometry.bounds.values.tolist()
    return gdf

def bboxes_to_shp(gdf, out_shp):
    if len(gdf) == 0:   # pyogrio cannot write a featureless layer; callers treat missing = empty
        return
    gdf_copy = gdf.rename(columns={"category_id": "cat_id", "category_name": "cat_name", "is_within_slice": "isin_slice"})
    # vectorized box construction from the [minx, miny, maxx, maxy] bbox column
    bbox_arr = np.asarray(gdf_copy.bbox.tolist(), dtype=float).reshape(-1, 4)
    gdf_copy["geometry"] = shapely.box(bbox_arr[:, 0], bbox_arr[:, 1], bbox_arr[:, 2], bbox_arr[:, 3])
    gdf_copy.drop(columns=['bbox', 'polygon']).to_file(out_shp)

def outlines_to_shp(gdf, out_shp):
    if len(gdf) == 0:   # pyogrio cannot write a featureless layer; callers treat missing = empty
        return
    gdf_copy = gdf.rename(columns={"category_id": "cat_id", "category_name": "cat_name", "is_within_slice": "isin_slice"})
    # errors="ignore": fused catalogs (fusion.fuse_predictions) may not carry bbox/polygon
    gdf_copy.drop(columns=['bbox', 'polygon'], errors="ignore").to_file(out_shp)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def _mask_bounds(binary_mask):
    """Bounding box ``(y0, y1, x0, x1)`` (half-open) of the nonzero region, or ``None`` if empty.

    Same bounds as ``np.nonzero(...)`` + min/max, but via ``any()`` row/column reductions and
    ``argmax`` — no per-pixel index arrays are materialized. That scan was the single biggest
    CPU line of the Goal-3 512 arm (26.8 s of a 128.5 s serial 4k-crop run vs find_contours'
    1.7 s; ``data/solutionDE/profile_512_arm.py``, 2026-07-11): nonzero cost scales with
    slice-size mask area per detection, which is exactly what the 512 arm quadruples.
    """
    rows = binary_mask.any(axis=1)
    if not rows.any():
        return None
    cols = binary_mask.any(axis=0)
    y0 = int(rows.argmax()); y1 = len(rows) - int(rows[::-1].argmax())
    x0 = int(cols.argmax()); x1 = len(cols) - int(cols[::-1].argmax())
    return y0, y1, x0, x1

def mask_support_area(binary_mask):
    """Number of pixels enclosed by the traced polygon, i.e. pixels at or above the 0.5 level.

    THE area definition for the pre-tracing gate (review item G1-06). The gate used to count only
    pixels *exactly* equal to 1, which is the same thing for a hard {0, 1} mask but wrong after an
    ``INTER_AREA`` downscale, where partially-covered pixels carry fractional values: a boulder
    could have plenty of support at the 0.5 level the tracer uses and still be discarded for
    having fewer than N *fully covered* pixels. Counting at :data:`MASK_SUPPORT_LEVEL` keeps the
    gate and the tracer talking about the same region.
    """
    return int(np.count_nonzero(np.asarray(binary_mask) >= MASK_SUPPORT_LEVEL))


def _enclosed_area(contour):
    """Absolute shoelace area of a closed ring given as (N, 2) coordinates."""
    if len(contour) < 3:
        return 0.0
    x, y = contour[:, 0], contour[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def binary_mask_to_polygon(binary_mask):
    """Trace a mask to a polygon at the 0.5 level set — the production convention.

    Args:
        binary_mask: a 2D numpy array; values >= 0.5 are inside the object. Hard {0, 1} masks and
            fractional (``INTER_AREA``-downscaled) masks are both accepted.

    Returns an ``(N, 2)`` float array of ``(x, y) = (col, row)`` vertices in the mask's own frame,
    or ``None`` if the mask is empty or no contour closes.

    **Component selection.** ``find_contours`` returns one ring per connected boundary — several
    if the mask is disconnected, plus one per hole. This keeps the ring enclosing the **greatest
    area** and discards the rest, so a mask that fragmented into a blob plus specks yields the
    blob. It used to keep the ring with the most *vertices*, which is not the same thing and fails
    in exactly the case that matters: a thin, ragged, disconnected artifact has a long perimeter
    and therefore many vertices, so it could replace the larger compact boulder beside it
    (review item X-03). Picking by area also correctly prefers an outer boundary over its own
    hole.
    """
    # Crop to the object's bounding box before tracing. ``find_contours`` scans the whole
    # array, so contouring a small boulder inside a full slice-size mask wastes most of the
    # work; cropping then shifting the contour back is geometry-identical (the 0.5 isoline is
    # local to the blob). Falls back to None for an all-zero mask.
    bounds = _mask_bounds(binary_mask)
    if bounds is None:
        return None
    y0, y1, x0, x1 = bounds
    sub = binary_mask[y0:y1, x0:x1]

    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(sub, pad_width=1, mode='constant', constant_values=0)
    contours = skimage.measure.find_contours(padded_binary_mask, MASK_SUPPORT_LEVEL)

    if not contours:
        return None
    contour_arr = max(contours, key=_enclosed_area) - 1  # largest ring; undo padding offset
    contour = np.flip(contour_arr, axis=1)  # (x, y) = (col, row) in the cropped frame
    contour[:, 0] += x0  # shift back to full-mask coordinates -> identical geometry
    contour[:, 1] += y0
    return contour

def binary_mask_to_polygon_cv(binary_mask):
    """Faster OpenCV alternative to :func:`binary_mask_to_polygon` — **orientation only**.

    Uses ``cv2.findContours`` (RETR_EXTERNAL, CHAIN_APPROX_SIMPLE) and keeps the largest external
    contour by enclosed area. Returns an (N, 2) float array of (x, y) = (col, row) coordinates, or
    ``None`` if no contour is found.

    **Different boundary convention — see the module docstring.** ``findContours`` follows
    boundary pixel *centres* rather than the 0.5 level set, so its polygons are inscribed: a solid
    5x5 block encloses 16.0 against skimage's ~24.5 (about -19% on equivalent diameter, and worse
    the smaller the boulder). The effect on *angles* is negligible (~0.5 deg median — see
    ``notebooks/Goal2_orientation_cv2_vs_skimage.ipynb``), which is the only axis it was validated
    on and the only one it is cleared for. It stays opt-in (``contour_method="cv2"``) and
    uncalibrated for area, so no size, SFD, density or area-comparison product may be built on it.
    """
    # Crop to the object's bounding box before tracing (see binary_mask_to_polygon): cheaper
    # and geometry-identical after shifting the contour back.
    bounds = _mask_bounds(binary_mask)
    if bounds is None:
        return None
    y0, y1, x0, x1 = bounds
    sub = binary_mask[y0:y1, x0:x1]

    # threshold at the same level the skimage tracer uses, so a fractional (downscaled) mask
    # defines the same inside/outside region under both tracers
    mask_uint8 = (np.asarray(sub) >= MASK_SUPPORT_LEVEL).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # largest by enclosed area; degenerate 1-px-wide blobs all measure 0, so fall back to the
    # longest outline rather than picking arbitrarily
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) == 0:
        best = max(contours, key=len)
    contour = best.squeeze(1).astype(float)
    contour[:, 0] += x0  # (x, y) cropped-frame -> full-mask coordinates
    contour[:, 1] += y0
    return contour

def check_mask_validity(binary_mask, min_area_threshold=4):

    # I need to be careful if I mix height, width...
    # at least two cells in height or width
    inside = np.asarray(binary_mask) >= MASK_SUPPORT_LEVEL   # same level the tracer uses
    rows = np.any(inside, axis=1)
    cols = np.any(inside, axis=0)

    # is the mask at least a pixel in width or height?
    if not np.any(rows) or not np.any(cols):
        wh_criteria = False
    else:
        wh_criteria = True

    # number of pixels
    area = mask_support_area(binary_mask)

    # number of blobs
    n_blobs = skimage.measure.label(inside).max()

    # is there any holes in the mask?
    padded_binary_mask = np.pad(inside, pad_width=1, mode='constant', constant_values=0)
    n_contours = len(skimage.measure.find_contours(padded_binary_mask, MASK_SUPPORT_LEVEL))
    if n_contours == 1:
        contour_criteria = True
    else:
        contour_criteria = False

    # we want at least 4 pixels, a width/height of at least two pixels
    # and only masks that have pixels that are interconnected, i.e., no multipolygons
    if area >= min_area_threshold and n_blobs == 1 and wh_criteria and contour_criteria:
        return True
    else:
        return False