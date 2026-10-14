"""Valid-data coverage: how much ground a scene actually observed (review item G3-07).

Every areal number — boulder density, SFD normalisation, abundance, cost per km^2 — is a count
divided by an area, and the area has to be the ground the sensor actually saw. It was the full
raster rectangle. For the HiRISE test scene that is wrong by **1.84x**: the product is a skewed
swath in a north-up bounding box, so 45.6% of the rectangle is acquisition fill and only 54.4% is
science data. Densities normalised by the rectangle are understated by that factor.

Nothing in the file says so. ``nodata`` is unset and ``mask_flag_enums`` reports ``all_valid``, so
a reader has to infer validity from the pixel values — and it is inferable: the scene has 1.5e6
pixels at exactly 0 against ~50 each at 1, 2, 3 and 4. A delta spike at zero with nothing beside
it is fill, not dark terrain.

Do **not** use ``rastertools.raster.true_footprint`` for this. It polygonises the valid pixels and
then takes the ``convex_hull``, which is fine for the edge-buffer job it exists for but fills every
interior gap — and this scene has 526 distinct zero regions, only half of which are the outer
wedges. The convex hull is a *containing* shape, not the observed area.
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio import features, windows

#: Assumed fill value. HiRISE and LRO NAC products both use 0 and declare no nodata.
DEFAULT_NODATA = 0


def valid_pixel_count(raster, nodata=DEFAULT_NODATA, decimate=1):
    """``(n_valid, n_total)`` pixels. Exact and memory-safe at ``decimate=1`` (windowed read).

    ``decimate > 1`` subsamples for a fast estimate; the ratio is unbiased for a region much
    larger than the decimation stride, which any scene is.
    """
    with rasterio.open(raster) as src:
        if decimate > 1:
            h, w = src.height // decimate, src.width // decimate
            arr = src.read(1, out_shape=(1, h, w))
            return int((arr != nodata).sum()), int(arr.size)
        n_valid = 0
        for _, window in src.block_windows(1):
            n_valid += int((src.read(1, window=window) != nodata).sum())
        return n_valid, int(src.width) * int(src.height)


def valid_area(raster, nodata=DEFAULT_NODATA, decimate=1):
    """Observed ground area in **map units squared**, from the valid-pixel count."""
    n_valid, n_total = valid_pixel_count(raster, nodata=nodata, decimate=decimate)
    with rasterio.open(raster) as src:
        px_area = abs(src.transform.a * src.transform.e)
        rect_area = px_area * src.width * src.height
    return {"valid_area": px_area * (n_valid / n_total) * (src.width * src.height),
            "rect_area": rect_area,
            "valid_fraction": n_valid / n_total if n_total else np.nan,
            "n_valid_px": n_valid, "n_total_px": n_total,
            "pixel_area": px_area, "decimate": int(decimate), "nodata": nodata}


def valid_footprint(raster, nodata=DEFAULT_NODATA, decimate=8, min_area_px=None):
    """Polygonal footprint of the valid data, **holes preserved** — no convex hull.

    Returns a GeoDataFrame with one row per connected valid region, in the raster's CRS. The
    geometry is decimated by default because full-resolution polygonisation of a ~2e9-px scene is
    not worth its cost for a coverage outline; ``decimate=1`` gives the exact boundary.

    ``min_area_px`` drops specks below that many *native* pixels (useful when decimation and
    noise conspire to produce single-pixel islands).
    """
    import geopandas as gpd
    from shapely.geometry import shape

    with rasterio.open(raster) as src:
        if decimate > 1:
            h, w = src.height // decimate, src.width // decimate
            arr = src.read(1, out_shape=(1, h, w))
            transform = src.transform * src.transform.scale(src.width / w, src.height / h)
        else:
            arr = src.read(1)
            transform = src.transform
        crs = src.crs
        px_area = abs(transform.a * transform.e)

    mask = (arr != nodata)
    geoms = [shape(geom) for geom, value in
             features.shapes(mask.astype(np.uint8), mask=mask, transform=transform) if value == 1]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    if min_area_px is not None and len(gdf):
        gdf = gdf[gdf.geometry.area >= min_area_px * px_area].reset_index(drop=True)
    return gdf


def valid_convex_hull(raster, nodata=DEFAULT_NODATA, block_rows=1024):
    """Convex hull of the valid pixels, streamed -- the polygon ``rastertools.raster.true_footprint`` returns.

    The prediction path needs this hull for its footprint-edge buffer (and *only* for that; it is not
    an observed area, see the module docstring). ``true_footprint`` read the whole raster, built an
    **int64** copy of the validity mask (``mask_array + 0``) and polygonised it at full resolution,
    only to keep the hull: ~10 bytes per scene pixel at its peak, ~70 GB at the 7 Gpx a 0.25 m/px
    HiRISE scene can reach (lever N2).

    The hull of a union of pixel squares is the hull of their corners, and in each row only the outer
    corners of the first and last valid pixel can be hull vertices. So this reads row blocks and keeps
    those four corners per row: memory is one block, not the scene. Corner coordinates go through the
    same affine as GDAL's polygoniser applies (``x = c + col*a``, exact for a north-up grid), and
    GEOS's convex hull depends only on the hull's vertex set, so the polygon is the same one --
    ``tests/test_windowed_slices.py`` compares it with ``true_footprint`` byte for byte. An all-fill
    raster returns an empty polygon.
    """
    from shapely.geometry import MultiPoint, Polygon

    cols, rows = [], []
    with rasterio.open(raster) as src:
        transform, width = src.transform, src.width
        for r0 in range(0, src.height, block_rows):
            h = min(block_rows, src.height - r0)
            valid = src.read(1, window=windows.Window(0, r0, width, h)) != nodata
            has = np.flatnonzero(valid.any(axis=1))
            if not has.size:
                continue
            first = valid[has].argmax(axis=1)
            last = width - 1 - valid[has, ::-1].argmax(axis=1)
            r = has + r0
            cols.append(np.concatenate([first, first, last + 1, last + 1]))
            rows.append(np.concatenate([r, r + 1, r, r + 1]))
    if not cols:
        return Polygon()
    x, y = transform * (np.concatenate(cols).astype(np.float64),
                        np.concatenate(rows).astype(np.float64))
    return MultiPoint(np.column_stack([x, y])).convex_hull


def coverage_report(raster, nodata=DEFAULT_NODATA, decimate=1):
    """Everything a density denominator needs, plus the provenance to defend it.

    The point of returning both areas is that a paper has to *say which one it used*: cost per
    processed km^2 and boulders per science-valid km^2 are both legitimate, and they differ here
    by 1.84x. Silence is what the review objected to, not the choice.
    """
    info = valid_area(raster, nodata=nodata, decimate=decimate)
    with rasterio.open(raster) as src:
        info.update(raster=str(raster), width=src.width, height=src.height,
                    crs=str(src.crs), res=(abs(src.transform.a), abs(src.transform.e)),
                    declared_nodata=src.nodata,
                    mask_flags=[str(f) for f in src.mask_flag_enums[0]])
    info["nodata_inferred"] = src.nodata is None
    info["overstatement_if_rectangle_used"] = (
        info["rect_area"] / info["valid_area"] if info["valid_area"] else np.nan)
    return info


def density_per_area(n_objects, area, area_units="m2", per="km2"):
    """Object density with the denominator stated, not assumed.

    ``area`` comes from :func:`valid_area` (science-valid) or the raster rectangle (processed);
    the returned dict names which, so a table can carry its own provenance.
    """
    scale = {"m2": 1.0, "km2": 1e6}[area_units]
    per_scale = {"m2": 1.0, "km2": 1e6}[per]
    area_m2 = area * scale
    if area_m2 <= 0:
        return {"density": np.nan, "n": int(n_objects), "area_m2": area_m2, "per": per}
    return {"density": n_objects / area_m2 * per_scale, "n": int(n_objects),
            "area_m2": area_m2, "per": per}
