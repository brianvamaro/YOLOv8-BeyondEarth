import numpy as np
import shapely
import skimage
import rasterio as rio
import geopandas as gpd

from shapely.geometry import (box, Polygon)


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

    boulder_geometry = []
    start = 0
    for n in vertex_counts:
        boulder_geometry.append(Polygon(coords[start:start + n]))
        start += n

    gdf = gpd.GeoDataFrame(df, geometry=boulder_geometry, crs=in_crs.to_wkt())
    gdf["bbox"] = gdf.geometry.bounds.values.tolist()
    return gdf

def bboxes_to_shp(gdf, out_shp):
    gdf_copy = gdf.rename(columns={"category_id": "cat_id", "category_name": "cat_name", "is_within_slice": "isin_slice"})
    # vectorized box construction from the [minx, miny, maxx, maxy] bbox column
    bbox_arr = np.asarray(gdf_copy.bbox.tolist(), dtype=float).reshape(-1, 4)
    gdf_copy["geometry"] = shapely.box(bbox_arr[:, 0], bbox_arr[:, 1], bbox_arr[:, 2], bbox_arr[:, 3])
    gdf_copy.drop(columns=['bbox', 'polygon']).to_file(out_shp)

def outlines_to_shp(gdf, out_shp):
    gdf_copy = gdf.rename(columns={"category_id": "cat_id", "category_name": "cat_name", "is_within_slice": "isin_slice"})
    gdf_copy.drop(columns=['bbox','polygon']).to_file(out_shp)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask):
    """Converts a binary mask to polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygon = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = skimage.measure.find_contours(padded_binary_mask, 0.5)

    # yolo can produce a mask where pixels are not interconnected
    # in this case the following line does not work
    contours = np.subtract(contours, 1)
    contour = np.flip(contours[0], axis=1) # should be interconnected
    return contour

def check_mask_validity(binary_mask, min_area_threshold=4):

    # I need to be careful if I mix height, width...
    # at least two cells in height or width
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)

    # is the mask at least a pixel in width or height?
    if not np.any(rows) or not np.any(cols):
        wh_criteria = False
    else:
        wh_criteria = True

    # number of pixels
    area = len(binary_mask[binary_mask == 1])

    # number of blobs
    n_blobs = skimage.measure.label(binary_mask).max()

    # is there any holes in the mask?
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    n_contours = len(skimage.measure.find_contours(padded_binary_mask, 0.5))
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